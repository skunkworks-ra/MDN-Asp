"""Conditional diffusion model of the sky posterior, p(sky | dirty or residual, PSF).

Plan: plan-diffusion-model.md. Two variants, same generator, U-Net and sampler. Current model m:
with --own_passes K, k (uniform 0..K) passes of the network's own draw, no gradient (A adds each
draw to m, B replaces m with it). Without it, A zeros and B zeros or (probability self_cond_p)
its own one-step estimate of the sky (self-conditioning).
  A  input residual (dirty - PSF * m) + PSF   -> target: sky - m (signed); m + draw is the sky
  B  input dirty + PSF + m                    -> target: full sky
Noise (eps) or v-prediction U-Net (--pred), cosine schedule, squared error; DDIM sampling.
Likelihood guidance (DPS) and the MAP climb are not in this file yet.

Sky, one 256 px patch, everything on the GPU:
  extended source at the centre, equal odds over 8 kinds: log-normal exp(s g) (alpha 2.5 s 1,
  alpha 2.5 s 2, alpha 4.5 s 2), ridge exp(-(g/w)^2) (alpha 3.5 w 0.1, alpha 4.5 w 0.1) inside
  an exp(-r^4) envelope; ring (wall sigma 1.5 beam sigmas); Gaussian (sigma_maj uniform
  0.5-42 px, sigma_min uniform 0.5-maj); point. Diameter log-uniform 32-256 px, axis ratio
  0.3-1, lumpiness 0.6 sqrt(u), PA uniform (envelope and ring).
  plus Poisson(25.16) AGN points at random pixels, fluxes drawn from T-RECS AGN
  (trecs_counts.npz), rescaled so the brightest point's dirty peak = the extended source's.
  Noise: dirty peak / sigma log-uniform 5-1000, PSF-shaped. Everything / max|dirty|.
  A's residual is renormalised by its own max|residual| inside the loss.

  pixi run -e gpu python -u scripts/train_tiled_asp.py --out_dir <dir> --variant A --preview 8
  pixi run -e gpu python -u scripts/train_tiled_asp.py --out_dir <dir> --variant A --overfit_batches 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from mdn_asp.data.psf_bank import load_corpus_psf_bank, load_g55_psf_bank
from mdn_asp.models.diffusion import (
    FILLS,
    KINDS,
    MAX_AGN,
    Diffusion,
    LDContextStem,
    ResBlock,
    SkyGenerator,
    UNet,
    _cg,
    _shift_zero_pad,
    beam_area_px,
    beam_sigma_px,
    build_diffusion_net,
    cosine_alpha_bar,
    crop_psf_centred,
    effective_beam_dofs,
    estimate_noise_cov_2w,
    estimate_window_noise_psd,
    fft_convolve_same,
    grf,
    timestep_embedding,
    unet_in_channels,
    window_noise_var,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the conditional sky diffusion model.")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--variant", choices=["A", "B", "P", "LD"], required=True,
                   help="P: unconditional prior over local sky shapes (plan-diffusion-local-proposal.md 3B); "
                        "LD: learned matching-pursuit dictionary (same plan, 3A)")
    p.add_argument("--ctx", type=int, default=256, help="LD: context window (px) of residual and PSF")
    p.add_argument("--ld_ctx_encoder", choices=("avgpool", "conv"), default="avgpool",
                   help="LD: context encoder mapping C x C (residual, PSF) to W x W (avgpool: 5 ch, conv: 11 ch)")
    p.add_argument("--ld_clean_prob", type=float, default=0.0,
                   help="LD: probability of simulating a mid-CLEAN partially peeled residual")
    p.add_argument("--n_extra", type=int, default=4,
                   help="LD: up to this many extra extended sources anywhere in the field (uniform 0..n)")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="P/LD: train on sky^(1/gamma) (unit-peak shapes), sample and raise to gamma (Drozdova 2024 Eq. 1)")
    p.add_argument("--zero_weight", type=float, default=0.0,
                   help="P/LD: extra loss weight on pixels where the target shape < zero_floor")
    p.add_argument("--zero_floor", type=float, default=1e-3)
    p.add_argument("--lam_chi2", type=float, default=0.0,
                   help="LD: weight of -J(g0_hat) / J_delta in the loss (plan 3A.2); 0 = diffusion loss only")
    p.add_argument("--chi2_tmax", type=int, default=200, help="LD: chi^2 term only for t < this")
    p.add_argument("--extra_flux", type=float, nargs=2, default=[0.1, 10.0],
                   help="LD: extra-source flux / main-source flux, log-uniform")
    p.add_argument("--shape_size", type=int, default=64, help="P: shape crop size (px)")
    p.add_argument("--crop_floor", type=float, default=0.1,
                   help="P: crop centres drawn uniformly among pixels with sky > this x the patch max")
    p.add_argument("--repo_root", type=str, default=os.environ.get("MDN_ASP_PSF_ROOT", "."))
    p.add_argument("--corpus_psf_dir", type=str, default=os.environ.get("MDN_ASP_CORPUS_PSF_DIR", ""))
    p.add_argument("--agn_npz", type=str, default=os.environ.get("MDN_ASP_AGN_NPZ", ""))
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val_size", type=int, default=64)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--val_every", type=int, default=2_000)
    p.add_argument("--checkpoint_every", type=int, default=10_000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", help="bf16 autocast (A100)")
    p.add_argument("--overfit_batches", type=int, default=0)
    p.add_argument("--overfit_extra_kind", choices=KINDS, default=None,
                   help="Append one patch of this kind to the first frozen batch.")
    p.add_argument("--preview", type=int, default=0, help="Write preview.png with this many samples and exit.")
    p.add_argument("--kinds", nargs="+", choices=KINDS, default=None, help="Draw only these kinds (default all)")
    # sky
    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--psf_size", type=int, default=512)
    p.add_argument("--diam_range", type=float, nargs=2, default=[32.0, 256.0])
    p.add_argument("--q_range", type=float, nargs=2, default=[0.3, 1.0])
    p.add_argument("--lump_max", type=float, default=0.6)
    p.add_argument("--gauss_sigma", type=float, nargs=2, default=[0.5, 42.0])
    p.add_argument("--ring_width_beams", type=float, default=1.5)
    p.add_argument("--agn_mean", type=float, default=25.16)
    p.add_argument("--snr_range", type=float, nargs=2, default=[5.0, 1000.0])
    # diffusion
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--pred", choices=["eps", "v"], default="eps",
                   help="Network output: noise (eps) or v = sqrt(abar) eps - sqrt(1 - abar) x0 (no 1/sqrt(abar) in x0)")
    p.add_argument("--self_cond_p", type=float, default=0.5,
                   help="B: probability the current-model input is the network's own estimate (else zeros)")
    p.add_argument("--sky_scale", type=float, default=0.0,
                   help="Multiply the target sky by this before diffusion; 0 = set from data (std 0.5)")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--init_ckpt", type=str, default="", help="Start from this checkpoint's weights")
    p.add_argument("--resume", type=str, default="",
                   help="Resume training from a ckpt_*.pt or best.pt checkpoint (restores model, opt, step, best).")
    p.add_argument("--own_passes", "--a_passes", dest="a_passes", type=int, default=0,
                   help="m = k passes of the network's own draw, k uniform in 0..own_passes. A adds each "
                        "draw to m, B replaces m with it. B: replaces the one-step self-conditioning. 0 = off")
    p.add_argument("--m_sample_steps", type=int, default=0, help="DDIM steps per pass for m (0 = sample_steps)")
    p.add_argument("--patience", type=int, default=0, help="Stop after this many vals without improvement (0 = off)")
    p.add_argument("--min_delta", type=float, default=0.0, help="Val loss must drop by this to count as improvement")
    p.add_argument("--val_select", choices=["mean", "k0", "worst_ratio"], default="mean",
                   help="Number best.pt and patience use: mean over all val batches; k = 0 batches only; or "
                        "the largest per-k ratio to that k's val loss at step 0 (no k may get worse)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _psf_bank(a):
    banks = [load_g55_psf_bank(repo_root=a.repo_root, target_size=a.psf_size, rotation_augment=True)]
    if a.corpus_psf_dir:
        banks.append(load_corpus_psf_bank(a.corpus_psf_dir, target_size=a.psf_size, rotation_augment=True))
    return torch.from_numpy(np.stack([b[i][0] for b in banks for i in range(len(b))]))


def _fmt(info):
    return "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in info.items())


def _show(ax, im, title, signed, vmax=None):
    m = float(np.abs(im).max()) or 1.0
    if vmax is not None:   # linear, shared scale
        ax.imshow(im, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
    elif signed:
        ax.imshow(im, origin="lower", cmap="RdBu_r", vmin=-m, vmax=m)
    else:
        ax.imshow(np.arcsinh(np.clip(im, 0, None) / (1e-2 * m)), origin="lower", cmap="inferno")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def preview(gen, a, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if a.variant == "LD":
        b = gen.sample_ld(a.preview)
        C, W = a.ctx, a.shape_size
        fig, ax = plt.subplots(a.preview, 5, figsize=(17, 3.4 * a.preview), squeeze=False)
        for i in range(a.preview):
            cy, cx = int(b["cy"][i]), int(b["cx"][i])
            for k, (im, ttl, sg) in enumerate([
                    (b["field"][i], f"field  {KINDS[int(b['kind'][i])]}  +{int(b['n_agn'][i])} extra", False),
                    (b["dirty"][i], f"noisy dirty  SNR {float(b['snr'][i]):.1f}", True)]):
                _show(ax[i][k], im.cpu().numpy(), ttl, sg)
                ax[i][k].add_patch(plt.Rectangle((cx - C / 2, cy - C / 2), C, C, fill=False, ec="c", lw=0.8))
            _show(ax[i][2], b["img"][i].cpu().numpy(), f"context {C} px / u", True)
            c0 = (C - W) / 2
            ax[i][2].add_patch(plt.Rectangle((c0, c0), W, W, fill=False, ec="k", lw=0.8))
            _show(ax[i][3], b["r_om"][i].cpu().numpy(), f"residual on Omega {W} px", True)
            _show(ax[i][4], b["sky"][i].cpu().numpy(), "target shape (sky x Hann, unit peak)", False)
        fig.tight_layout()
        fig.savefig(out_dir / "preview_ld.png", dpi=70)
        plt.close(fig)
        print(f"[diff] wrote {out_dir / 'preview_ld.png'}", flush=True)
        return
    b = gen.sample(a.preview)
    fig, ax = plt.subplots(a.preview, 2, figsize=(8, 4 * a.preview), squeeze=False)
    for i in range(a.preview):
        _show(ax[i][0], b["sky"][i].cpu().numpy(),
              f"target sky  {KINDS[int(b['kind'][i])]}  AGN {int(b['n_agn'][i])}", False)
        _show(ax[i][1], b["img"][i].cpu().numpy(),
              f"input dirty  SNR {float(b['snr'][i]):.0f}", True)
    fig.tight_layout()
    fig.savefig(out_dir / "preview.png", dpi=80)
    plt.close(fig)
    print(f"[diff] wrote {out_dir / 'preview.png'}", flush=True)


def draw_figure(diff, batch, a, path, n=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if a.variant == "P":
        return draw_figure_prior(diff, batch, path, n)
    n = min(n, batch["sky"].shape[0])
    sub = {k: v[:n] for k, v in batch.items()}
    d1, d2 = diff.sample(sub), diff.sample(sub)
    fig, ax = plt.subplots(n, 5, figsize=(20, 4 * n), squeeze=False)
    for i in range(n):
        tr = sub["sky"][i].cpu().numpy()
        _show(ax[i][0], sub["img"][i].cpu().numpy(), f"input  SNR {float(sub['snr'][i]):.0f}", True)
        _show(ax[i][1], tr, f"target  {KINDS[int(sub['kind'][i])]}", False, vmax=1.0)
        _show(ax[i][2], d1[i].cpu().numpy(), "draw 1", False, vmax=1.0)
        _show(ax[i][3], d2[i].cpu().numpy(), "draw 2", False, vmax=1.0)
        _show(ax[i][4], (d1[i] - d2[i]).cpu().numpy(), "draw 1 - draw 2", True)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)
    tn = (sub["sky"] ** 2).flatten(1).sum(1)
    err = ((d1 - sub["sky"]) ** 2).flatten(1).sum(1)[tn > 0] / tn[tn > 0]   # blank targets have no fractional error
    if a.variant == "LD":
        r, W = sub["r_om"], sub["r_om"].shape[-1]
        num = (d1 * r).sum((1, 2)).clamp_min(0)
        den = (d1 * fft_convolve_same(d1, sub["psf2"])).sum((1, 2)).clamp_min(1e-30)
        jr = num ** 2 / (2 * den) / (r[:, W // 2, W // 2] ** 2 / 2).clamp_min(1e-30)
        print("    draw J/J_delta: " + " ".join(f"{float(x):.2f}" for x in jr), flush=True)
    return float(err.mean()) if err.numel() > 0 else float("nan")


def draw_figure_prior(diff, batch, path, n=8):
    """P: unconditional draws next to the nearest training shape of the batch (min fractional
    squared error). Returns the mean of that min error over the draws."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = batch["sky"]
    sub = {k: v[:n] for k, v in batch.items()}
    d = diff.sample(sub)
    err = ((d[:, None] - tr[None]) ** 2).flatten(2).sum(2) / (tr[None] ** 2).flatten(2).sum(2).clamp_min(1e-30)
    e, j = err.min(1)
    fig, ax = plt.subplots(2, n, figsize=(2.5 * n, 5.5), squeeze=False)
    for i in range(d.shape[0]):
        _show(ax[0][i], d[i].cpu().numpy(), f"draw {i}", False)
        _show(ax[1][i], tr[int(j[i])].cpu().numpy(), f"nearest: shape {int(j[i])}  err {float(e[i]):.3g}", False)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)
    return float(e.mean())


def run(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    bank = _psf_bank(a)
    agn = np.load(a.agn_npz)["agn_flux"] if (a.agn_npz and Path(a.agn_npz).is_file()) else np.zeros(0, dtype=np.float32)
    if a.variant != "P" and a.agn_mean > 0 and len(agn) == 0:
        raise FileNotFoundError(
            "--agn_npz (or MDN_ASP_AGN_NPZ) pointing to an existing .npz file is required when --agn_mean > 0."
        )
    print(f"[diff] PSF bank {tuple(bank.shape)}  AGN fluxes {len(agn)}", flush=True)
    gen = SkyGenerator(bank, agn, a, dev, a.seed)
    if a.preview > 0:
        preview(gen, a, out_dir)
        return
    ref_ckpt = a.resume or a.init_ckpt
    if ref_ckpt:
        ck_scale = torch.load(ref_ckpt, map_location="cpu", weights_only=False)["args"]["sky_scale"]
        if a.sky_scale <= 0:
            a.sky_scale = ck_scale
            print(f"[diff] sky_scale {a.sky_scale:.4g} from {ref_ckpt}", flush=True)
        elif a.sky_scale != ck_scale:
            print(f"[diff] WARNING: --sky_scale {a.sky_scale:.4g} != {ck_scale:.4g} in {ref_ckpt}; "
                  f"the loaded weights were trained at the checkpoint's scale", flush=True)
    if a.sky_scale <= 0:
        with torch.no_grad():
            std = float(torch.stack([(gen.sample(a.batch_size)["sky"].clamp_min(0) ** (1.0 / a.gamma)
                                      if a.variant in ("P", "LD") else gen.sample(a.batch_size)["sky"]).std()
                                     for _ in range(8)]).median())
        a.sky_scale = 0.5 / std
        print(f"[diff] sky std {std:.4g} -> sky_scale {a.sky_scale:.4g}", flush=True)
    gpu_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
    with open(out_dir / "run_config.json", "w") as fh:
        json.dump({
            "args": vars(a),
            "gpu_name": gpu_name,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, fh, indent=2)

    val_gen = SkyGenerator(bank, agn, a, dev, a.seed + 10_000_000)
    val = [val_gen.sample(a.batch_size) for _ in range(max(1, a.val_size // a.batch_size))]
    if a.overfit_batches > 0:
        frozen = [gen.sample(a.batch_size) for _ in range(a.overfit_batches)]
        if a.overfit_extra_kind:
            extra = gen.sample(1, kind_override=a.overfit_extra_kind)
            frozen[0] = {k: torch.cat([v, extra[k]]) for k, v in frozen[0].items()}
        val = frozen
        torch.save({k: v.cpu() for k, v in frozen[0].items() if torch.is_tensor(v)}, out_dir / "frozen_batch.pt")
        nxt = lambda i: frozen[i % len(frozen)]
        print(f"[diff] OVERFIT GATE: {len(frozen)} frozen batches", flush=True)
    else:
        nxt = lambda i: gen.sample(a.batch_size)

    model = build_diffusion_net(a).to(dev)
    if a.init_ckpt and not a.resume:
        ck = torch.load(a.init_ckpt, map_location=dev, weights_only=False)
        for key in ("variant", "base_channels"):
            if ck["args"][key] != getattr(a, key):
                raise ValueError(f"--init_ckpt {key} {ck['args'][key]} != this run's {getattr(a, key)}")
        if ck["args"].get("ld_ctx_encoder", "avgpool") != getattr(a, "ld_ctx_encoder", "avgpool"):
            raise ValueError(
                f"--init_ckpt ld_ctx_encoder {ck['args'].get('ld_ctx_encoder', 'avgpool')} "
                f"!= this run's {getattr(a, 'ld_ctx_encoder', 'avgpool')}"
            )
        model.load_state_dict(ck["model"])
        print(f"[diff] init from {a.init_ckpt}", flush=True)
    diff = Diffusion(model, a, dev)

    def val_loss():
        """Same t, noise, k and draws for every val: only the weights change the number."""
        out = {}
        for i, b in enumerate(val):
            g = torch.Generator(device=dev).manual_seed(a.seed + 777 + i)
            kk = i % (a.a_passes + 1) if a.a_passes else 0
            k = torch.full((b["sky"].shape[0],), kk, device=dev) if a.a_passes else None
            out.setdefault(kk, []).append(float(diff.loss(b, gen=g, k=k)))
        per_k = {kk: float(np.mean(v)) for kk, v in sorted(out.items())}
        return float(np.mean([x for v in out.values() for x in v])), per_k

    def select(vl, per_k):
        if a.val_select == "k0":
            return per_k[0]
        if a.val_select == "worst_ratio":
            return max(per_k[kk] / base_k[kk] for kk in per_k)
        return vl

    if a.a_passes and len(val) < a.a_passes + 1:
        raise ValueError(f"{len(val)} val batches cannot cover k = 0..{a.a_passes}; raise --val_size")
    if a.val_select == "worst_ratio" and not (a.init_ckpt or a.resume):
        raise ValueError("--val_select worst_ratio needs --init_ckpt (the bar is the starting weights)")
    base_k = None
    if a.val_select == "worst_ratio":
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
            vl0, base_k = val_loss()
        model.train()
        print(f"  VAL step 0  loss={vl0:.4g}  " + "  ".join(f"k{kk}={v:.4g}" for kk, v in base_k.items()), flush=True)
    print(f"[diff] variant {a.variant}  params {sum(p.numel() for p in model.parameters()):,}  "
          f"bf16 {a.bf16}  T {a.T}  sample_steps {a.sample_steps}  gpu {gpu_name}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    log, best, stale, start_step = [], float("inf"), 0, 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        for key in ("variant", "base_channels"):
            if ck["args"][key] != getattr(a, key):
                raise ValueError(f"--resume {key} {ck['args'][key]} != this run's {getattr(a, key)}")
        if ck["args"].get("ld_ctx_encoder", "avgpool") != getattr(a, "ld_ctx_encoder", "avgpool"):
            raise ValueError(
                f"--resume ld_ctx_encoder {ck['args'].get('ld_ctx_encoder', 'avgpool')} "
                f"!= this run's {getattr(a, 'ld_ctx_encoder', 'avgpool')}"
            )
        model.load_state_dict(ck["model"])
        if "opt" in ck and ck["opt"] is not None:
            opt.load_state_dict(ck["opt"])
        start_step = int(ck.get("step", 0))
        best = float(ck.get("best", float("inf")))
        if (out_dir / "log.json").is_file():
            with open(out_dir / "log.json") as fh:
                log = [entry for entry in json.load(fh) if entry.get("step", 0) <= start_step]
        print(f"[diff] resumed from {a.resume} at step {start_step} (best={best:.4g})", flush=True)
    elif base_k is not None:
        log.append({"step": 0, "val_loss": vl0, "val_loss_k": base_k, "select": 1.0})
        best = 1.0
        torch.save({"step": 0, "best": best, "model": model.state_dict(), "args": vars(a)}, out_dir / "best.pt")

    t0 = time.time()
    step = start_step
    for step in range(start_step + 1, a.steps + 1):
        batch = nxt(step)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
            loss = diff.loss(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if a.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()
        if step % a.log_every == 0:
            info = {"loss": float(loss)}
            print(f"  step {step:7d}/{a.steps}  {_fmt(info)}  elapsed={(time.time() - t0) / 60:.1f}m", flush=True)
            log.append({"step": step, **info})
        if step % a.val_every == 0 or step == a.steps:
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
                vl, per_k = val_loss()
                sel = select(vl, per_k)
                err = draw_figure(diff, val[0], a, out_dir / f"draws_{step:07d}.png")
            model.train()
            print(f"  VAL step {step}  loss={vl:.4g}  " + "  ".join(f"k{kk}={v:.4g}" for kk, v in per_k.items())
                  + f"  select({a.val_select})={sel:.4g}  draw_frac_sq_err={err:.4g}", flush=True)
            log.append({"step": step, "val_loss": vl, "val_loss_k": per_k, "select": sel, "draw_frac_sq_err": err})
            if sel < best - a.min_delta:
                best, stale = sel, 0
                torch.save({"step": step, "best": best, "model": model.state_dict(), "args": vars(a)},
                           out_dir / "best.pt")
            else:
                stale += 1
            if a.patience and stale >= a.patience:
                print(f"[diff] early stop at step {step}: {stale} vals without improvement > {a.min_delta}", flush=True)
                break
        if step % a.checkpoint_every == 0:
            torch.save({"step": step, "best": best, "model": model.state_dict(), "opt": opt.state_dict(), "args": vars(a)},
                       out_dir / f"ckpt_{step:07d}.pt")
            with open(out_dir / "log.json", "w") as fh:
                json.dump(log, fh, indent=2)
    torch.save({"step": step, "best": best, "model": model.state_dict(), "args": vars(a)}, out_dir / "final.pt")
    with open(out_dir / "log.json", "w") as fh:
        json.dump(log, fh, indent=2)
    if dev.type == "cuda":
        print(f"[diff] done, best val loss {best:.4g}, peak GPU mem "
              f"{torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    run(parse_args())
