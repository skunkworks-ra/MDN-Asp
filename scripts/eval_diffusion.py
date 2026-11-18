"""Validation of a trained sky diffusion model (plan-diffusion-model.md, decision 8).

Fresh patches from the held-out generator stream (training seed + 10_000_000, as the
training VAL set), n_patches / 8 of each kind, n_draws draws per patch. Per patch (variants A/B):
  coverage 68 / 95: fraction of pixels whose true sky lies inside the central 68% / 95%
    range of the draws, over pixels where the noiseless dirty truth (PSF * sky) > 3 sigma;
  flux ratio: sum(mean of draws) / sum(true sky);
  data fit: rms(dirty - PSF * draw) / sigma, median over draws (about 1 if consistent);
  fit_mean: rms(dirty - PSF * mean of draws) / sigma (the mean is the model);
  a, c, fit_scaled: least-squares scale and offset of the mean, rms(dirty - PSF * (a mean + c)) / sigma.
Summaries per kind and per SNR bin (5-20, 20-100, 100-1000).
Also supports variant LD (Omega window shape/fit evaluation) and variant P (unconditional shape prior).

Writes metrics.json, coverage.png, flux.png, gallery_NN.png (8 patches per file: input,
truth, mean, std, (truth - mean) / std) for A/B; Omega metrics + gallery for LD; gallery.png
+ summary.json for P.

  pixi run -e gpu python -u scripts/eval_diffusion.py --ckpt <run>/best.pt --out_dir <dir> --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from mdn_asp.models.diffusion import (
    KINDS,
    Diffusion,
    SkyGenerator,
    UNet,
    build_diffusion_net,
    fft_convolve_same,
)
from train_tiled_asp import _psf_bank, _show

SNR_BINS = ((5.0, 20.0), (20.0, 100.0), (100.0, 1000.0))
LD_SNR_BINS = ((1.0, 5.0), (5.0, 20.0), (20.0, 100.0), (100.0, 1000.0))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Validate a trained sky diffusion model.")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--n_patches", type=int, default=256, help="Rounded down to a multiple of 8 (equal per kind).")
    p.add_argument("--n_draws", type=int, default=16)
    p.add_argument("--draw_batch", type=int, default=16, help="Draws per sampler call (lower to save GPU memory).")
    p.add_argument("--seed", type=int, default=None, help="Default: training seed + 10_000_000 (the VAL stream).")
    p.add_argument("--mask_nsigma", type=float, default=3.0)
    p.add_argument("--repo_root", type=str, default=None, help="Optional override for PSF repo_root.")
    p.add_argument("--corpus_psf_dir", type=str, default=None, help="Optional override for corpus_psf_dir.")
    p.add_argument("--agn_npz", type=str, default=None, help="Optional override for agn_npz.")
    return p.parse_args(argv)


def summarise(rows, key, metric_keys=("cov68", "cov95", "flux_ratio", "fit_rms", "fit_mean", "a", "c", "fit_scaled")):
    out = {}
    for name, sel in key:
        r = [x for x in rows if sel(x)]
        if not r:
            continue
        out[name] = {
            "n": len(r),
            **{k: float(np.nanmedian([x[k] for x in r])) for k in metric_keys},
        }
    return out


def _load_ckpt(ckpt_path: str, dev: torch.device) -> dict:
    p = Path(ckpt_path)
    if p.is_dir():
        from safetensors.torch import load_file
        with open(p / "config.json") as fh:
            cfg = json.load(fh)
        return {
            "step": cfg.get("step"),
            "args": cfg["args"],
            "model": load_file(str(p / "model.safetensors")),
        }
    return torch.load(ckpt_path, map_location=dev, weights_only=False)


def _load_agn(a) -> np.ndarray:
    agn_path = getattr(a, "agn_npz", "") or ""
    if agn_path and Path(agn_path).is_file():
        return np.load(agn_path)["agn_flux"]
    if a.variant == "P" or float(getattr(a, "agn_mean", 0.0)) <= 0.0:
        return np.zeros(0, dtype=np.float32)
    raise FileNotFoundError(
        f"AGN flux file {agn_path!r} not found (required for variant {a.variant} with agn_mean={a.agn_mean})."
    )


def _eval_prior(e, ck, a, diff: Diffusion, gen: SkyGenerator, out_dir: Path, dev: torch.device, seed: int):
    import matplotlib.pyplot as plt

    W = a.shape_size
    per_kind = max(1, e.n_patches // len(KINDS))
    held_by_kind: dict[str, torch.Tensor] = {}
    draws_by_kind: dict[str, torch.Tensor] = {}
    by_kind: dict[str, dict] = {}

    for k in KINDS:
        ref = gen.sample(per_kind, kind_override=k)["sky"]   # (per_kind, W, W)
        held_by_kind[k] = ref
        dummy = {
            "img": torch.zeros(e.n_draws, W, W, device=dev),
            "sigma": torch.ones(e.n_draws, device=dev),
        }
        with torch.no_grad():
            d = torch.cat([
                diff.sample({n: v[i:i + e.draw_batch] for n, v in dummy.items()})
                for i in range(0, e.n_draws, e.draw_batch)
            ])   # (n_draws, W, W)
        draws_by_kind[k] = d
        ref_energy = (ref ** 2).flatten(1).sum(1)
        ref_valid = ref[ref_energy > 1e-12]
        if ref_valid.shape[0] > 0:
            ref_norm = (ref_valid[None] ** 2).flatten(2).sum(2).clamp_min(1e-30)
            err_mat = ((d[:, None] - ref_valid[None]) ** 2).flatten(2).sum(2) / ref_norm
            nn_err = err_mat.min(1).values
            rec_err = err_mat.min(0).values
            nn_med, nn_mean, rec_med = float(nn_err.median()), float(nn_err.mean()), float(rec_err.median())
        else:
            nn_med, nn_mean, rec_med = float("nan"), float("nan"), float("nan")
        peaks = d.flatten(1).max(1).values
        pos_frac = float((d >= 0).float().mean())
        pos_flux = float(
            (d.clamp_min(0).flatten(1).sum(1) / d.abs().flatten(1).sum(1).clamp_min(1e-30)).median()
        )
        sup_frac = float((d > 0.05 * peaks.view(-1, 1, 1)).float().mean((1, 2)).median())
        by_kind[k] = {
            "n_held_out": int(per_kind),
            "n_draws": int(e.n_draws),
            "nn_frac_sq_err": nn_med,
            "nn_frac_sq_err_mean": nn_mean,
            "recall_frac_sq_err": rec_med,
            "peak_median": float(peaks.median()),
            "min_median": float(d.flatten(1).min(1).values.median()),
            "pos_frac": pos_frac,
            "pos_flux_frac": pos_flux,
            "support_frac": sup_frac,
        }
        print(
            f"  {k:16s}  nn_err {by_kind[k]['nn_frac_sq_err']:.3f}  "
            f"recall_err {by_kind[k]['recall_frac_sq_err']:.3f}  "
            f"peak {by_kind[k]['peak_median']:.3f}  pos_flux {by_kind[k]['pos_flux_frac']:.3f}  "
            f"support {by_kind[k]['support_frac']:.3f}",
            flush=True,
        )

    all_ref_raw = torch.cat([held_by_kind[k] for k in KINDS], dim=0)
    all_ref = all_ref_raw[(all_ref_raw ** 2).flatten(1).sum(1) > 1e-12]
    if all_ref.shape[0] == 0:
        all_ref = all_ref_raw
    all_draws = torch.cat([draws_by_kind[k] for k in KINDS], dim=0)
    all_ref_norm = (all_ref[None] ** 2).flatten(2).sum(2).clamp_min(1e-30)
    all_err = ((all_draws[:, None] - all_ref[None]) ** 2).flatten(2).sum(2) / all_ref_norm
    nn_all, _ = all_err.min(1)
    all_peaks = all_draws.flatten(1).max(1).values
    overall = {
        "n_held_out": int(all_ref.shape[0]),
        "n_draws": int(all_draws.shape[0]),
        "nn_frac_sq_err": float(nn_all.median()),
        "nn_frac_sq_err_mean": float(nn_all.mean()),
        "peak_median": float(all_peaks.median()),
        "pos_frac": float((all_draws >= 0).float().mean()),
        "pos_flux_frac": float(
            (all_draws.clamp_min(0).flatten(1).sum(1) / all_draws.abs().flatten(1).sum(1).clamp_min(1e-30)).median()
        ),
        "support_frac": float((all_draws > 0.05 * all_peaks.view(-1, 1, 1)).float().mean((1, 2)).median()),
    }

    # Gallery: one draw + its nearest held-out shape per kind (up to 8 columns).
    n_cols = len(KINDS)
    fig, ax = plt.subplots(2, n_cols, figsize=(2.6 * n_cols, 5.4), squeeze=False)
    for col, k in enumerate(KINDS):
        d0 = draws_by_kind[k][0]
        ref_k = held_by_kind[k]
        ek = ((d0[None] - ref_k) ** 2).flatten(1).sum(1) / (ref_k ** 2).flatten(1).sum(1).clamp_min(1e-30)
        emin, jmin = ek.min(0)
        _show(ax[0][col], d0.cpu().numpy(), f"prior draw ({k})", False)
        _show(ax[1][col], ref_k[int(jmin)].cpu().numpy(), f"nearest {k}  err {float(emin):.3g}", False)
    fig.tight_layout()
    fig.savefig(out_dir / "gallery.png", dpi=80)
    plt.close(fig)

    summary = {
        "ckpt": e.ckpt,
        "variant": "P",
        "step": ck.get("step"),
        "seed": seed,
        "n_draws": e.n_draws,
        "overall": overall,
        "by_kind": by_kind,
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[eval] overall: {overall}", flush=True)
    print(f"[eval] wrote {out_dir}", flush=True)


def _eval_ld(e, ck, a, diff: Diffusion, gen: SkyGenerator, out_dir: Path, dev: torch.device, seed: int):
    import matplotlib.pyplot as plt

    per_kind = max(1, e.n_patches // len(KINDS))
    rows, gal, means = [], [], []
    ld_keys = ("cov68", "cov95", "frac_sq_err", "J_over_J_delta", "fit_rms", "fit_mean")

    for k in KINDS:
        for _ in range(per_kind):
            b = gen.sample(1, kind_override=k)
            rep = {n: v.repeat_interleave(e.n_draws, 0) for n, v in b.items()}
            with torch.no_grad():
                d = torch.cat([
                    diff.sample({n: v[i:i + e.draw_batch] for n, v in rep.items()})
                    for i in range(0, e.n_draws, e.draw_batch)
                ])   # (n_draws, W, W)
            sky = b["sky"][0]
            img_om = b["r_om"][0]
            psf2 = b["psf2"]
            sig = float(b["sigma"][0])
            W = sky.shape[-1]

            mask = sky > 0.05
            q = torch.quantile(d.flatten(1), torch.tensor([0.025, 0.16, 0.84, 0.975], device=dev), dim=0)
            t = sky.flatten()
            m = mask.flatten()
            in68 = ((t >= q[1]) & (t <= q[2]))[m].float()
            in95 = ((t >= q[0]) & (t <= q[3]))[m].float()

            mean, std = d.mean(0), d.std(0)
            tn = float((sky ** 2).sum())
            frac_sq_err = float(((mean - sky) ** 2).sum() / tn) if tn > 0 else float("nan")

            Bg = fft_convolve_same(d, psf2.expand(e.n_draws, -1, -1))
            num = (d * img_om).sum((1, 2)).clamp_min(0.0)
            den = (d * Bg).sum((1, 2)).clamp_min(1e-30)
            ast = num / den
            jd = max(float(img_om[W // 2, W // 2] ** 2 / 2.0), 1e-30)
            j_ratio = float((num ** 2 / (2.0 * den) / jd).median())
            fit = float(((img_om[None] - ast.view(-1, 1, 1) * Bg).flatten(1).pow(2).mean(1).sqrt() / sig).median())

            Bgm = fft_convolve_same(mean[None], psf2)[0]
            ast_m = float((mean * img_om).sum().clamp_min(0.0) / (mean * Bgm).sum().clamp_min(1e-30))
            fit_mean = float((img_om - ast_m * Bgm).pow(2).mean().sqrt() / sig)

            row = {
                "kind": k,
                "snr": float(b["snr"][0]),
                "n_mask": int(m.sum()),
                "cov68": float(in68.mean()) if m.any() else float("nan"),
                "cov95": float(in95.mean()) if m.any() else float("nan"),
                "frac_sq_err": frac_sq_err,
                "J_over_J_delta": j_ratio,
                "fit_rms": fit,
                "fit_mean": fit_mean,
            }
            means.append(mean.cpu().numpy())
            rows.append(row)
            gal.append((
                img_om.cpu().numpy(),
                sky.cpu().numpy(),
                mean.cpu().numpy(),
                std.cpu().numpy(),
                ((sky - mean) / std.clamp_min(1e-30)).cpu().numpy(),
                row,
            ))
            print(
                f"  {len(rows):4d}  {k:16s} snr {row['snr']:7.1f}  cov68 {row['cov68']:.3f}  "
                f"cov95 {row['cov95']:.3f}  err {row['frac_sq_err']:.3f}  "
                f"J/Jd {row['J_over_J_delta']:.2f}  fit {row['fit_rms']:.3f}  fit_mean {row['fit_mean']:.3f}",
                flush=True,
            )

    np.save(out_dir / "means.npy", np.stack(means).astype(np.float32))
    by_kind = summarise(rows, [(k, lambda x, k=k: x["kind"] == k) for k in KINDS], metric_keys=ld_keys)
    by_snr = summarise(
        rows,
        [(f"{lo:g}-{hi:g}", lambda x, lo=lo, hi=hi: lo <= x["snr"] < hi) for lo, hi in LD_SNR_BINS],
        metric_keys=ld_keys,
    )
    allp = summarise(rows, [("all", lambda x: True)], metric_keys=ld_keys)
    payload = {
        "ckpt": e.ckpt,
        "variant": "LD",
        "step": ck.get("step"),
        "seed": seed,
        "n_draws": e.n_draws,
        "summary_median": {"all": allp, "by_kind": by_kind, "by_snr": by_snr},
        "patches": rows,
    }
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(payload["summary_median"], fh, indent=2)

    if by_kind and by_snr:
        fig, axs = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [len(by_kind), len(by_snr)]})
        for ax, grp, title in ((axs[0], by_kind, "by kind"), (axs[1], by_snr, "by SNR bin")):
            names = list(grp)
            x = np.arange(len(names))
            ax.bar(x - 0.2, [grp[n]["cov68"] for n in names], 0.4, label="68% range", color="#4C72B0")
            ax.bar(x + 0.2, [grp[n]["cov95"] for n in names], 0.4, label="95% range", color="#DD8452")
            ax.axhline(0.68, color="#4C72B0", ls="--", lw=1)
            ax.axhline(0.95, color="#DD8452", ls="--", lw=1)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{n}\n(n={grp[n]['n']})" for n in names], fontsize=7)
            ax.set_ylim(0, 1)
            ax.set_ylabel("fraction of truth inside range (median)")
            ax.set_title(f"LD coverage {title}, {e.n_draws} draws, pixels with sky > 0.05", fontsize=9)
        axs[0].legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(out_dir / "coverage.png", dpi=90)
        plt.close(fig)

    for g in range(0, len(gal), 8):
        chunk = gal[g:g + 8]
        fig, ax = plt.subplots(len(chunk), 5, figsize=(20, 4 * len(chunk)), squeeze=False)
        for i, (im, tr, mn, sd, z, row) in enumerate(chunk):
            _show(ax[i][0], im, f"r_om  SNR {row['snr']:.1f}", True)
            _show(ax[i][1], tr, f"truth  {row['kind']}", False, vmax=1.0)
            _show(ax[i][2], mn, f"mean of {e.n_draws}  err {row['frac_sq_err']:.2f}", False, vmax=1.0)
            _show(ax[i][3], sd, f"std  J/Jd {row['J_over_J_delta']:.2f}", False)
            _show(ax[i][4], z, f"(truth - mean) / std  cov68 {row['cov68']:.2f}", True)
        fig.tight_layout()
        fig.savefig(out_dir / f"gallery_{g // 8:02d}.png", dpi=60)
        if g == 0:
            fig.savefig(out_dir / "gallery.png", dpi=60)
        plt.close(fig)

    print(f"[eval] all: {allp}", flush=True)
    print(f"[eval] wrote {out_dir}", flush=True)


def main(e):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(e.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(e.device)
    ck = _load_ckpt(e.ckpt, dev)
    a = argparse.Namespace(**ck["args"])
    if e.repo_root is not None:
        a.repo_root = e.repo_root
    elif not (Path(a.repo_root) / "data" / "g55").is_dir() and (_REPO_ROOT / "data" / "g55").is_dir():
        a.repo_root = str(_REPO_ROOT)
    if e.corpus_psf_dir is not None:
        a.corpus_psf_dir = e.corpus_psf_dir
    if e.agn_npz is not None:
        a.agn_npz = e.agn_npz

    model = build_diffusion_net(a).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    diff = Diffusion(model, a, dev)
    seed = a.seed + 10_000_000 if e.seed is None else e.seed
    gen = SkyGenerator(_psf_bank(a), _load_agn(a), a, dev, seed)
    print(f"[eval] {e.ckpt}  variant {a.variant}  step {ck.get('step')}  seed {seed}", flush=True)

    if a.variant == "P":
        _eval_prior(e, ck, a, diff, gen, out_dir, dev, seed)
        return
    if a.variant == "LD":
        _eval_ld(e, ck, a, diff, gen, out_dir, dev, seed)
        return

    per_kind = max(1, e.n_patches // len(KINDS))
    rows, gal, means = [], [], []
    for k in KINDS:
        for _ in range(per_kind):
            b = gen.sample(1, kind_override=k)
            rep = {n: v.repeat_interleave(e.n_draws, 0) for n, v in b.items()}
            with torch.no_grad():
                d = torch.cat([diff.sample({n: v[i:i + e.draw_batch] for n, v in rep.items()})
                               for i in range(0, e.n_draws, e.draw_batch)])   # (n_draws, S, S)
            sky, img, sig = b["sky"][0], b["img"][0], float(b["sigma"][0])
            mask = fft_convolve_same(b["sky"], b["psf"])[0] > e.mask_nsigma * sig
            q = torch.quantile(d.flatten(1), torch.tensor([0.025, 0.16, 0.84, 0.975], device=dev), dim=0)
            t = sky.flatten()
            m = mask.flatten()
            in68 = ((t >= q[1]) & (t <= q[2]))[m].float()
            in95 = ((t >= q[0]) & (t <= q[3]))[m].float()
            mean, std = d.mean(0), d.std(0)
            fit = ((img - fft_convolve_same(d, b["psf"].expand(e.n_draws, -1, -1))).flatten(1).pow(2).mean(1).sqrt() / sig).median()
            pm = fft_convolve_same(mean[None], b["psf"])[0]
            fit_mean = (img - pm).pow(2).mean().sqrt() / sig
            p1 = fft_convolve_same(torch.ones_like(mean)[None], b["psf"])[0]
            ac = torch.linalg.lstsq(torch.stack([pm.flatten(), p1.flatten()], 1).double(),
                                    img.flatten()[:, None].double()).solution[:, 0].float()
            fit_scaled = (img - ac[0] * pm - ac[1] * p1).pow(2).mean().sqrt() / sig
            row = {"kind": k, "snr": float(b["snr"][0]), "n_mask": int(m.sum()),
                   "cov68": float(in68.mean()) if m.any() else float("nan"),
                   "cov95": float(in95.mean()) if m.any() else float("nan"),
                   "flux_ratio": float(mean.sum() / sky.sum().clamp_min(1e-30)),
                   "fit_rms": float(fit), "fit_mean": float(fit_mean),
                   "a": float(ac[0]), "c": float(ac[1] / sky.abs().max()), "fit_scaled": float(fit_scaled)}
            means.append(mean.cpu().numpy())
            rows.append(row)
            gal.append((img.cpu().numpy(), sky.cpu().numpy(), mean.cpu().numpy(), std.cpu().numpy(),
                        ((sky - mean) / std.clamp_min(1e-30)).cpu().numpy(), row))
            print(f"  {len(rows):4d}  {k:16s} snr {row['snr']:7.1f}  cov68 {row['cov68']:.3f}  "
                  f"cov95 {row['cov95']:.3f}  flux {row['flux_ratio']:.3f}  fit {row['fit_rms']:.3f}  fit_mean {row['fit_mean']:.3f}  "
                  f"a {row['a']:.3f}  c/maxsky {row['c']:.2e}  fit_scaled {row['fit_scaled']:.3f}", flush=True)

    np.save(out_dir / "means.npy", np.stack(means).astype(np.float32))
    by_kind = summarise(rows, [(k, lambda x, k=k: x["kind"] == k) for k in KINDS])
    by_snr = summarise(rows, [(f"{lo:g}-{hi:g}", lambda x, lo=lo, hi=hi: lo <= x["snr"] < hi) for lo, hi in SNR_BINS])
    allp = summarise(rows, [("all", lambda x: True)])
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump({"ckpt": e.ckpt, "step": ck.get("step"), "seed": seed, "n_draws": e.n_draws,
                   "mask_nsigma": e.mask_nsigma, "summary_median": {"all": allp, "by_kind": by_kind, "by_snr": by_snr},
                   "patches": rows}, fh, indent=2)

    # Coverage per kind and per SNR bin (medians over patches), with the nominal lines.
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [len(by_kind), len(by_snr)]})
    for ax, grp, title in ((axs[0], by_kind, "by kind"), (axs[1], by_snr, "by SNR bin")):
        names = list(grp)
        x = np.arange(len(names))
        ax.bar(x - 0.2, [grp[n]["cov68"] for n in names], 0.4, label="68% range", color="#4C72B0")
        ax.bar(x + 0.2, [grp[n]["cov95"] for n in names], 0.4, label="95% range", color="#DD8452")
        ax.axhline(0.68, color="#4C72B0", ls="--", lw=1)
        ax.axhline(0.95, color="#DD8452", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}\n(n={grp[n]['n']})" for n in names], fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction of truth inside range (median)")
        ax.set_title(f"coverage {title}, {e.n_draws} draws, pixels with PSF*truth > {e.mask_nsigma:g} sigma",
                     fontsize=9)
    axs[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "coverage.png", dpi=90)
    plt.close(fig)

    # Flux ratio against SNR, one colour per kind.
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    for i, k in enumerate(KINDS):
        r = [x for x in rows if x["kind"] == k]
        ax.scatter([x["snr"] for x in r], [x["flux_ratio"] for x in r], s=12, color=cmap(i), label=k)
    ax.axhline(1.0, color="k", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SNR (dirty peak / sigma)")
    ax.set_ylabel("flux ratio, mean of draws / truth")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "flux.png", dpi=90)
    plt.close(fig)

    # Galleries, 8 patches per file.
    for g in range(0, len(gal), 8):
        chunk = gal[g:g + 8]
        fig, ax = plt.subplots(len(chunk), 5, figsize=(20, 4 * len(chunk)), squeeze=False)
        for i, (im, tr, mn, sd, z, row) in enumerate(chunk):
            _show(ax[i][0], im, f"input  SNR {row['snr']:.0f}", True)
            _show(ax[i][1], tr, f"truth  {row['kind']}", False)
            _show(ax[i][2], mn, f"mean of {e.n_draws}  flux {row['flux_ratio']:.2f}", False)
            _show(ax[i][3], sd, "std of draws", False)
            _show(ax[i][4], z, f"(truth - mean) / std  cov68 {row['cov68']:.2f}", True)
        fig.tight_layout()
        fig.savefig(out_dir / f"gallery_{g // 8:02d}.png", dpi=60)
        plt.close(fig)
    print(f"[eval] all: {allp}", flush=True)
    print(f"[eval] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main(parse_args())
