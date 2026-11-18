"""Full-scale training script for MDNAsp (flow_plan.md §Training).

Trains on GPU-generated cutout scenes with the G55 D-config
L-band PSF bank. Checkpoints saved every --checkpoint_every steps.
Validation NLL logged on a fixed held-out set.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.optim as optim

from mdn_asp.data.psf_bank import load_corpus_psf_bank, load_g55_psf_bank
from mdn_asp.models.mdn_asp import MDNAsp


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train MDNAsp on synthetic cutouts.")
    p.add_argument("--out_dir",    type=str, default="results/mdn_asp_train",
                   help="Output directory for checkpoints and logs.")
    p.add_argument("--repo_root",  type=str, default=os.environ.get("MDN_ASP_PSF_ROOT", "."),
                   help="Repo root containing data/g55/chunk_*/psf.fits.")
    p.add_argument("--steps",      type=int, default=50_000,
                   help="Total gradient steps (default: 50000).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--val_size",   type=int, default=1_000,
                   help="Fixed held-out validation scenes (default: 1000).")
    p.add_argument("--morphologies", type=str, default="point,blob",
                   help="Centred-source morphologies (default: 'point,blob').")
    p.add_argument("--extended_fraction", type=float, default=0.05,
                   help="Per-distractor extended probability (default: 0.05).")
    p.add_argument("--snr_min",    type=float, default=5.0,
                   help="Minimum convolved SNR for centred source (default: 5.0).")
    p.add_argument("--log_every",  type=int, default=100)
    p.add_argument("--val_every",  type=int, default=1_000)
    p.add_argument("--checkpoint_every", type=int, default=5_000)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--hidden",        type=int, default=256)
    p.add_argument("--n_components",  type=int, default=5)
    p.add_argument("--scale_relative", action="store_true",
                   help="input divided "
                        "by max|residual|, flux target = log(flux/scale), "
                        "sigma passed as sigma/scale. Makes the flux head "
                        "flux-equivariant (ARCHITECTURES.md section 4).")
    p.add_argument("--correlated_noise", action="store_true",
                   help="GPU generator: noise with Cov = sigma^2 * PSF, as in "
                        "a real dirty image, instead of white noise.")
    p.add_argument("--point_sigma_px", type=float, default=None,
                   help="GPU generator: size label for point sources (px). "
                        "Default keeps the legacy beam sigma (1.4 px).")
    p.add_argument("--blob_sigma_min_px", type=float, default=None,
                   help="GPU generator: lower bound of the blob size prior (px). "
                        "Default keeps the legacy beam sigma (1.4 px).")
    p.add_argument("--cutout", type=int, default=128,
                   help="Network input size (power of 2). The PSF bank is "
                        "cropped to the same size.")
    p.add_argument("--sigma_max_px", type=float, default=None,
                   help="Largest drawn source sigma (px). None = SIGMA_MAX_PX (11.2).")
    p.add_argument("--render_nsigma", type=float, default=None,
                   help="Truncate Gaussian sources at this many sigma. None = untruncated.")
    p.add_argument("--corpus_psf_dir", type=str,
                   default=os.environ.get("MDN_ASP_CORPUS_PSF_DIR") or None,
                   help="GPU generator: add corpus_field_*_psf.fits from this "
                        "directory to the G55 PSF bank.")
    p.add_argument("--pa_head", choices=("7d", "von_mises"), default="7d",
                   help="Position-angle head: '7d' legacy ambient Gaussian or 'von_mises' exact 6D product density.")
    p.add_argument("--uniform_point_pa", action="store_true",
                   help="Draw uniform PA in (-pi/2, pi/2] for circular/point targets instead of 0.0.")
    p.add_argument("--overfit_batches", type=int, default=0,
                   help="Gate mode: train on this many frozen batches and "
                        "report metrics on the same batches.")
    p.add_argument("--resume",        type=str, default=None,
                   help="Path to checkpoint .pt to resume from.")
    return p.parse_args(argv)


def _build_gpu_generator(psf_bank, args, device, seed):
    """GPUCutoutGenerator with the training field, noise and source settings."""
    import numpy as np
    from mdn_asp.data.gpu_cutout_generator import GPUCutoutGenerator
    banks = psf_bank if isinstance(psf_bank, list) else [psf_bank]
    bank = torch.from_numpy(np.stack([
        b[i][0] for b in banks for i in range(len(b))
    ]))
    return GPUCutoutGenerator(
        bank,
        field_size=512, cutout_size=args.cutout, sigma_noise=1e-4,
        n_sources=(5, 20), extended_fraction=args.extended_fraction,
        snr_min=args.snr_min,
        morphologies=tuple(m.strip() for m in args.morphologies.split(",")),
        scale_relative=args.scale_relative,
        correlated_noise=args.correlated_noise,
        point_sigma_px=args.point_sigma_px,
        blob_sigma_min_px=args.blob_sigma_min_px,
        sigma_max_px=args.sigma_max_px,
        render_nsigma=args.render_nsigma,
        uniform_point_pa=args.uniform_point_pa,
        device=device, seed=seed,
    )


def _infinite_gpu_batches(gen, batch_size):
    """Endless (img, cond, tgt) stream, so the training loop is unchanged."""
    while True:
        yield gen.sample(batch_size)


def eval_val(model, val_loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for img, cond, tgt in val_loader:
            img, cond, tgt = img.to(device), cond.to(device), tgt.to(device)
            params = model(img, cond)
            losses.append(model.nll_loss(params, tgt).item())
    model.train()
    return float(np.mean(losses))


# A target size below this is a point label (only meaningful when
# --point_sigma_px is sub-pixel; blobs start at the beam sigma).
_POINT_SIZE_CUT_PX = 0.7
# Label split between point (0.25 px) and blob (>= --blob_sigma_min_px) targets.
_POINT_LABEL_CUT_PX = 0.4
# Blobs below the real beam sigma (~2.1 px) are the super-resolution case.
_SUBBEAM_PX = 2.1


def eval_modes(model, batches, device, point_sigma_px=None):
    """Mode-estimate accuracy per kind, for the plan's G1/G2 gates."""
    model.eval()
    modes, tgts = [], []
    with torch.no_grad():
        for img, cond, tgt in batches:
            params = model(img.to(device), cond.to(device))
            modes.append(model.mode(params).cpu())
            tgts.append(tgt.cpu())
    model.train()
    m, t = torch.cat(modes), torch.cat(tgts)
    if point_sigma_px is not None:
        lp = math.log(point_sigma_px)
        is_pt = ((t[:, 3] - lp).abs() < 1e-4) & ((t[:, 4] - lp).abs() < 1e-4)
    else:
        is_pt = t[:, 3] < math.log(_POINT_LABEL_CUT_PX)
    pos_err = torch.hypot(m[:, 0] - t[:, 0], m[:, 1] - t[:, 1])
    dlnf = (m[:, 2] - t[:, 2]).abs()
    size_ratio = torch.exp(m[:, 3] - t[:, 3])
    # Per kind: point labels sit on the rounded pixel (offset exactly 0), so
    # pooled medians are dominated by the trivial point case.
    out = {"n_point": int(is_pt.sum()), "n_blob": int((~is_pt).sum())}
    if is_pt.any():
        out["point_pos_err_med_px"] = float(pos_err[is_pt].median())
        out["point_dlnflux_med"] = float(dlnf[is_pt].median())
        out["point_small_frac"] = float(
            (torch.exp(m[is_pt, 3]) < _POINT_SIZE_CUT_PX).float().mean())
    if (~is_pt).any():
        r = size_ratio[~is_pt]
        out["blob_pos_err_med_px"] = float(pos_err[~is_pt].median())
        out["blob_dlnflux_med"] = float(dlnf[~is_pt].median())
        out["blob_size_within_1p5_frac"] = float(
            ((r > 1 / 1.5) & (r < 1.5)).float().mean())
        sub = t[~is_pt, 3] < math.log(_SUBBEAM_PX)
        out["n_blob_subbeam"] = int(sub.sum())
        if sub.any():
            out["blob_subbeam_size_within_1p5_frac"] = float(
                ((r[sub] > 1 / 1.5) & (r[sub] < 1.5)).float().mean())
    return out


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Persist the full run configuration so every checkpoint is self-describing
    # (avoids the v2 provenance gap: morphologies/hyperparameters recoverable
    # from disk, not just from terminal scrollback).
    run_config = {
        "args": vars(args),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(out_dir / "run_config.json", "w") as fh:
        json.dump(run_config, fh, indent=2)
    print(f"[train] Wrote run config to {out_dir / 'run_config.json'}")

    print(f"[train] Loading PSF bank from {args.repo_root!r}/data/g55 ...")
    psf_bank = load_g55_psf_bank(
        repo_root=args.repo_root, target_size=args.cutout, rotation_augment=True,
    )
    print(f"[train] PSF bank size: {len(psf_bank)}")
    if args.corpus_psf_dir:
        corpus_bank = load_corpus_psf_bank(args.corpus_psf_dir, target_size=args.cutout,
                                           rotation_augment=True)
        print(f"[train] + corpus PSF bank: {len(corpus_bank)}")
        psf_bank = [psf_bank, corpus_bank]

    # Whole batches built on the device: no DataLoader, no workers, no host
    # image copies. The CPU path renders a 512 field per sample and keeps a
    # 128 crop, which measured 1.0 step/s with all 32 threads saturated and
    # the GPU at 1-2% (22.7 ms compute per ~1000 ms batch = ~2% duty cycle).
    gen = _build_gpu_generator(psf_bank, args, device, seed=args.seed)
    val_gen = _build_gpu_generator(psf_bank, args, device,
                                   seed=args.seed + 10_000_000)
    val_batches = [val_gen.sample(args.batch_size)
                   for _ in range(max(1, args.val_size // args.batch_size))]
    print(f"[train] GPU generator: {len(val_batches)} held-out val batches "
          f"(seed offset 10,000,000)")
    train_loader = _infinite_gpu_batches(gen, args.batch_size)
    val_loader = val_batches
    if args.overfit_batches > 0:
        frozen = [gen.sample(args.batch_size)
                  for _ in range(args.overfit_batches)]
        train_loader = (frozen[i % len(frozen)] for i in range(10 ** 12))
        val_loader = frozen
        print(f"[train] OVERFIT GATE: {len(frozen)} frozen batches")

    model = MDNAsp(
        base_channels=args.base_channels,
        hidden=args.hidden,
        n_components=args.n_components,
        cond_dim=5,
        cutout=args.cutout,
        pa_head=args.pa_head,
        scale_relative=args.scale_relative,
        point_sigma_px=args.point_sigma_px,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model parameters: {n_params:,}  pa_head={args.pa_head}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "opt" in ckpt:
            optimizer.load_state_dict(ckpt["opt"])
        start_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
        print(f"[train] Resumed from {args.resume} at step {start_step} (best_val_loss={best_val_loss:.3f})")

    log = []
    step = start_step
    t0 = time.time()

    print(f"[train] Training for {args.steps} steps, batch={args.batch_size}, "
          f"device={args.device}, morphologies={args.morphologies}")
    model.train()

    while step < args.steps:
        for img, cond, tgt in train_loader:
            if step >= args.steps:
                break

            img, cond, tgt = img.to(device), cond.to(device), tgt.to(device)
            optimizer.zero_grad()
            params = model(img, cond)
            loss = model.nll_loss(params, tgt)
            loss.backward()
            optimizer.step()

            step += 1
            loss_val = float(loss.item())

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                print(f"  step {step:6d}/{args.steps}  "
                      f"train_loss={loss_val:8.3f}  "
                      f"elapsed={elapsed/60:.1f}m")
                log.append({"step": step, "train_loss": loss_val})

            if step % args.val_every == 0:
                val_loss = eval_val(model, val_loader, device)
                gate = eval_modes(model, val_loader, device, point_sigma_px=args.point_sigma_px)
                print(f"  step {step:6d}  val_loss={val_loss:.3f}  "
                      + "  ".join(f"{k}={v:.3f}" if isinstance(v, float)
                                  else f"{k}={v}" for k, v in gate.items()))
                if log:
                    log[-1]["val_loss"] = val_loss
                    log[-1]["gate"] = gate
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {"step": step, "model": model.state_dict(),
                         "opt": optimizer.state_dict(),
                         "val_loss": val_loss, "best_val_loss": best_val_loss,
                         "scale_relative": args.scale_relative,
                         "args": vars(args)},
                        out_dir / "best.pt",
                    )

            if step % args.checkpoint_every == 0:
                torch.save(
                    {"step": step, "model": model.state_dict(),
                     "opt": optimizer.state_dict(),
                     "best_val_loss": best_val_loss,
                     "scale_relative": args.scale_relative,
                     "args": vars(args)},
                    out_dir / f"ckpt_{step:07d}.pt",
                )
                with open(out_dir / "log.json", "w") as fh:
                    json.dump(log, fh, indent=2)

    # Final save
    torch.save({"step": step, "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "scale_relative": args.scale_relative,
                "args": vars(args)},
               out_dir / "final.pt")
    with open(out_dir / "log.json", "w") as fh:
        json.dump(log, fh, indent=2)
    print(f"[train] Done. Best val_loss={best_val_loss:.3f}. "
          f"Checkpoints in {out_dir}/")


if __name__ == "__main__":
    run(parse_args())
