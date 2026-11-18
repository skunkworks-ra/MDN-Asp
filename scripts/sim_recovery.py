"""Known-flux recovery gate for the MDN-Asp imaging loop, image plane only.

A shell plus point sources of known flux is convolved with a real PSF, given
PSF-shaped noise, and cleaned with the same minor_cycle() run_imaging.py uses.
The major cycle is exact: residual = dirty - PSF * model with a PSF twice the
field, so every sidelobe of every component is subtracted. No mask anywhere.

Two stop rules, same loop otherwise:
  frozen  run_imaging.py as of 2026-09-19: global stop and minor-cycle floor
          both max(sidelobe * DIRTY peak, n_sigma * sigma), fixed for the run.
  clean   Cotton-Schwab: the minor cycle keeps its own sidelobe * CURRENT
          peak floor (minor_cycle.py), the global stop is n_sigma * sigma.

Kill criterion (stated before running): total model flux off the input by
more than 10%, or a residual below -5 sigma on the source support.

    pixi run -e gpu python scripts/sim_recovery.py \\
        --psf data/g55/chunk_full/psf.fits \\
        --out_dir /mnt/fast/mad_clean_runs/sim_recovery/g55_chunk_full
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from mdn_asp.minor_cycle import (
    measure_psf_sidelobe,
    minor_cycle,
    minor_cycle_batched,
    psf_convolve,
)
from mdn_asp.refit import refit_commits
from mdn_asp.models.mdn_asp import (
    checkpoint_is_scale_relative, checkpoint_sigma_min_px, load_mdn_checkpoint)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/asp_sbi/proposal_v2/best.pt")
    p.add_argument("--psf", required=True, help="FITS file or CASA image")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--field", type=int, default=512)
    p.add_argument("--modes", nargs="+", default=["frozen", "clean"],
                   choices=["frozen", "clean"])
    p.add_argument("--shell_flux", type=float, default=0.15, help="Jy")
    p.add_argument("--shell_radius_beams", type=float, default=12.0,
                   help="shell radius in beam sigmas")
    p.add_argument("--shell_width_beams", type=float, default=1.5,
                   help="radial Gaussian sigma of the shell wall, in beam sigmas")
    p.add_argument("--point_fluxes", type=float, nargs="*",
                   default=[0.005, 0.003, 0.002], help="Jy")
    p.add_argument("--peak_snr", type=float, default=235.0,
                   help="dirty peak / sigma; 235 is G55 (6.71e-3 / 2.85e-5); 0 = no noise")
    p.add_argument("--n_sigma", type=float, default=3.0)
    p.add_argument("--loop_gain", type=float, default=0.1)
    p.add_argument("--max_components", type=int, default=2000)
    p.add_argument("--max_major", type=int, default=20)
    p.add_argument("--divergence_tol", type=float, default=0.20)
    p.add_argument("--config_idx", type=int, default=3,
                   help="3 is what GPUCutoutGenerator trains with")
    p.add_argument("--sigma_min_px", type=float, default=None)
    p.add_argument("--sigma_max_px", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sky", default=None,
                   help="file.npz:key, use this sky instead of the shell + points")
    p.add_argument("--refit_exact", action="store_true",
                   help="refit every Aspen or diffusion tile after each minor cycle, as run_imaging.py")
    p.add_argument("--refit_gain", type=float, default=0.5)
    p.add_argument("--refit_method", choices=("cd", "nnls"), default="cd")
    p.add_argument("--refit_nullspace_lam", type=float, default=0.0)
    p.add_argument("--refit_nnls_iters", type=int, default=60)
    p.add_argument("--flux_prior_weight", type=float, default=0.0)
    p.add_argument("--mdn_density_mode", action="store_true")
    p.add_argument("--mdn_profile_amp", action="store_true")
    p.add_argument("--mdn_delta_fallback", action="store_true")
    p.add_argument("--uv_coupling_min", type=float, default=0.0)
    p.add_argument("--batched_peaks", action="store_true",
                   help="minor_cycle_batched: all local maxima per pass, one forward pass")
    # Diffusion matching pursuit (run_imaging.py diffusion_mp_cycle) instead of MDN-Asp.
    p.add_argument("--diffusion_ckpt", default=None,
                   help="variant P checkpoint (prior B); replaces --ckpt, as run_imaging.py --diff_mp")
    p.add_argument("--diff_n_draws", type=int, default=10,
                   help="B draws per step; 0 with --diff_ld_ckpt = A draws + delta only")
    p.add_argument("--diff_ld_ckpt", default=None, help="variant LD checkpoint (dictionary A)")
    p.add_argument("--diff_ld_draws", type=int, default=10)
    p.add_argument("--diff_ld_cg_iters", type=int, default=0)
    p.add_argument("--diff_taper", action="store_true")
    p.add_argument("--diff_q_space", action="store_true")
    p.add_argument("--diff_dof_pen", type=float, default=0.0)
    p.add_argument("--diff_briggs_psd", action="store_true")
    p.add_argument("--diff_scales", type=str, default="")
    p.add_argument("--diff_save_draws", type=int, default=0)
    p.add_argument("--diff_seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.diffusion_ckpt and args.diff_n_draws < 1 and not args.diff_ld_ckpt:
        p.error("--diff_n_draws 0 needs --diff_ld_ckpt")
    if args.diffusion_ckpt and args.peak_snr <= 0:
        p.error("--diffusion_ckpt needs noise (--peak_snr > 0): the posterior data step divides by sigma")
    if args.diffusion_ckpt and args.batched_peaks:
        p.error("--batched_peaks is MDN-Asp only")
    return args


def load_psf(path: str) -> np.ndarray:
    if path.endswith(".fits"):
        from astropy.io import fits
        return np.squeeze(fits.getdata(path)).astype(np.float32)
    from casatools import image as iatool
    ia = iatool()
    ia.open(path)
    d = ia.getchunk().squeeze().T.astype(np.float32)
    ia.close()
    return d


def crop_at_peak(psf: np.ndarray, size: int) -> np.ndarray:
    """size x size window with the PSF peak at (size//2, size//2), zero padded."""
    py, px = np.unravel_index(int(np.argmax(psf)), psf.shape)
    out = np.zeros((size, size), dtype=np.float32)
    r0, c0 = py - size // 2, px - size // 2
    sr0, sc0 = max(r0, 0), max(c0, 0)
    sr1, sc1 = min(r0 + size, psf.shape[0]), min(c0 + size, psf.shape[1])
    out[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = psf[sr0:sr1, sc0:sc1]
    return out


def beam_sigma_px(psf: np.ndarray) -> float:
    """Circular sigma from the area of the main lobe above half maximum."""
    from scipy.ndimage import label
    lab, _ = label(psf >= 0.5 * psf.max())
    cy, cx = np.unravel_index(int(np.argmax(psf)), psf.shape)
    fwhm = 2.0 * math.sqrt((lab == lab[cy, cx]).sum() / math.pi)
    return fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def gaussian_beam(S: int, sig: float) -> np.ndarray:
    y, x = np.mgrid[:S, :S] - S // 2
    return np.exp(-0.5 * (x * x + y * y) / sig ** 2).astype(np.float32)


def psf_shaped_noise(psf_field: np.ndarray, rng) -> np.ndarray:
    """Unit-variance noise with autocorrelation equal to the PSF.

    Same construction as GPUCutoutGenerator._psf_shaped_noise: white noise
    filtered by sqrt(max(Re B^, 0)), the PSD projection of a cropped PSF.
    """
    spec = np.clip(np.fft.fft2(np.fft.ifftshift(psf_field)).real, 0.0, None)
    w = rng.standard_normal(psf_field.shape)
    x = np.fft.ifft2(np.fft.fft2(w) * np.sqrt(spec)).real
    return (x / x.std()).astype(np.float32)


def make_sky(S: int, bsig: float, args, rng):
    """Shell (radial Gaussian wall, unit sum * flux) plus pixel point sources."""
    y, x = np.mgrid[:S, :S].astype(np.float64)
    c = S / 2.0
    r = np.hypot(x - c, y - c)
    R, w = args.shell_radius_beams * bsig, args.shell_width_beams * bsig
    shell = np.exp(-0.5 * ((r - R) / w) ** 2)
    shell *= args.shell_flux / shell.sum()
    sky = shell.astype(np.float32)
    points = []
    r_min, r_max = R + 10 * bsig, S / 2 - 3 * bsig
    for f in args.point_fluxes:
        rad, ang = rng.uniform(r_min, r_max), rng.uniform(0, 2 * np.pi)
        py, px = int(round(c + rad * np.sin(ang))), int(round(c + rad * np.cos(ang)))
        sky[py, px] += f
        points.append({"x": px, "y": py, "flux": f})
    return sky, shell.astype(np.float32), points


def load_diffusion_models(args):
    """(diffusion_mp_cycle, refit_tiles, B, B args, A, A args, generator) as run_imaging.py --diff_mp."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_imaging import diffusion_mp_cycle, load_diffusion, refit_tiles
    device = torch.device(args.device)
    diff, a = load_diffusion(args.diffusion_ckpt, device)
    diffA, aA = load_diffusion(args.diff_ld_ckpt, device) if args.diff_ld_ckpt else (None, None)
    gen = torch.Generator(device=device).manual_seed(args.diff_seed)
    return diffusion_mp_cycle, refit_tiles, diff, a, diffA, aA, gen


def diffusion_cycle(dm, residual, psf, sigma, threshold, sidelobe, args):
    """One diffusion matching-pursuit minor cycle; returns (update, n steps, stop, winner counts, tiles)."""
    mp_cycle, _, diff, a, diffA, aA, gen = dm
    ns = argparse.Namespace(**vars(args), diff_unmask_output=True)   # no mask in the sim
    mask = np.ones(residual.shape, dtype=bool)
    upd, _, n, mstop, tiles, recs = mp_cycle(
        diff, a, residual, psf, sigma, ns, torch.device(args.device),
        gen, mask, threshold, sidelobe, diffA, aA,
    )
    wins = {w: sum(q["winner"] == w for q in recs) for w in ("draw", "A", "delta")}
    return upd, n, mstop, wins, tiles


def run_loop(mode, dirty, psf, sigma, sidelobe, mdn, scale_relative, args, dm=None):
    if dm is not None:
        dm[-1].manual_seed(args.diff_seed)
    S = dirty.shape[0]
    model = np.zeros((S, S), dtype=np.float32)
    residual = dirty.copy()
    dirty_peak = float(dirty.max())
    noise_floor = args.n_sigma * sigma
    psf_floor = sidelobe * dirty_peak
    threshold = max(psf_floor, noise_floor) if mode == "frozen" else noise_floor
    cycles, stop = [], "max_major"
    all_tiles = []
    for major in range(args.max_major):
        res_peak = float(residual.max())
        if res_peak < threshold:
            stop = "below_threshold"
            break
        if dm is not None:
            upd, n, mstop, wins, tiles = diffusion_cycle(dm, residual, psf, sigma, threshold, sidelobe, args)
            if n == 0:
                stop = f"no_commits ({mstop})"
                break
            all_tiles.extend(tiles)
            new_model = (model + upd).astype(np.float32)
            rstats = None
            if args.refit_exact:
                _, refit_tiles_fn, *_ = dm
                work_res = (dirty - psf_convolve(new_model, psf)).astype(np.float32)
                delta, _, rstats = refit_tiles_fn(
                    all_tiles, work_res, psf, args.refit_gain, torch.device(args.device),
                    sigma=sigma, model=new_model,
                    method=args.refit_method,
                    nullspace_lam=args.refit_nullspace_lam,
                    nnls_iters=args.refit_nnls_iters,
                )
                new_model = (new_model + delta).astype(np.float32)
            model = new_model
            residual = (dirty - psf_convolve(model, psf)).astype(np.float32)
            new_peak = float(residual.max())
            cycles.append({"major": major, "n": n, "minor_stop": mstop, "winners": wins,
                           "flux_committed": float(upd.sum()), "model_total": float(model.sum()),
                           "peak_before": res_peak, "peak_after": new_peak,
                           "min_after": float(residual.min()), "refit": rstats})
            print(f"[{mode}] major {major}  n={n} {wins}  stop={mstop}  model={model.sum():.4f} Jy  "
                  f"peak {res_peak/sigma:.1f} -> {new_peak/sigma:.1f} sigma  "
                  f"min {residual.min()/sigma:.1f} sigma", flush=True)
            if new_peak > res_peak * (1.0 + args.divergence_tol):
                stop = "divergence"
                break
            continue
        if args.batched_peaks:
            upd, commits, mstop = minor_cycle_batched(
                residual=residual, psf=psf, sigma=sigma, config_idx=args.config_idx,
                model=mdn, loop_gain=args.loop_gain, sidelobe_level=sidelobe,
                abs_noise_floor=threshold, divergence_tol=args.divergence_tol,
                image_mask=None, sigma_max_px=args.sigma_max_px,
                sigma_min_px=args.sigma_min_px, max_components=args.max_components,
                scale_relative=scale_relative, cutout=mdn.cutout,
                density_mode=args.mdn_density_mode, device=args.device)
        else:
            upd, commits, mstop = minor_cycle(
                residual=residual, psf=psf, sigma=sigma, config_idx=args.config_idx,
                model=mdn, loop_gain=args.loop_gain, sidelobe_level=sidelobe,
                abs_noise_floor=threshold, divergence_tol=args.divergence_tol,
                image_mask=None, sigma_max_px=args.sigma_max_px,
                sigma_min_px=args.sigma_min_px, max_components=args.max_components,
                scale_relative=scale_relative, cutout=mdn.cutout,
                density_mode=args.mdn_density_mode,
                profile_amp=args.mdn_profile_amp,
                delta_fallback=args.mdn_delta_fallback,
                uv_coupling_min=args.uv_coupling_min,
                nullspace_lam=args.refit_nullspace_lam,
                device=args.device)
        if not commits:
            stop = f"no_commits ({mstop})"
            break
        rstats = None
        if args.refit_exact:
            # The major cycle here is exact, so this is also the residual
            # run_imaging.py --major_before_refit hands the refit.
            work_res = (dirty - psf_convolve(model + upd, psf)).astype(np.float32)
            _, upd, commits, rstats = refit_commits(
                work_res, upd, psf, commits, sigma=sigma,
                refit_gain=args.refit_gain, sigma_min_px=args.sigma_min_px,
                sigma_max_px=args.sigma_max_px,
                flux_prior_weight=args.flux_prior_weight, cutout=mdn.cutout,
                device=args.device,
                amp_method=args.refit_method,
                nullspace_lam=args.refit_nullspace_lam,
                nnls_iters=args.refit_nnls_iters,
            )
        model += upd
        residual = (dirty - psf_convolve(model, psf)).astype(np.float32)
        new_peak = float(residual.max())
        cycles.append({"major": major, "n": len(commits), "minor_stop": mstop,
                       "flux_committed": float(upd.sum()),
                       "model_total": float(model.sum()),
                       "peak_before": res_peak, "peak_after": new_peak,
                       "min_after": float(residual.min()), "refit": rstats})
        su = sigma or float("nan")
        print(f"[{mode}] major {major}  n={len(commits)}  stop={mstop}  "
              f"model={model.sum():.4f} Jy  peak {res_peak/su:.1f} -> "
              f"{new_peak/su:.1f} sigma  min {residual.min()/su:.1f} sigma",
              flush=True)
        if new_peak > res_peak * (1.0 + args.divergence_tol):
            stop = "divergence"
            break
    return model, residual, cycles, stop, threshold


def metrics(model, residual, truth, shell, points, beam, sigma, bsig):
    from scipy.ndimage import binary_dilation
    sigma = sigma or float("nan")
    smooth = psf_convolve(truth, beam)
    support = binary_dilation(smooth > 1e-3 * smooth.max(),
                              iterations=int(math.ceil(2 * bsig)))
    out = {
        "flux_true": float(truth.sum()),
        "flux_model": float(model.sum()),
        "flux_ratio": float(model.sum() / truth.sum()),
        "flux_model_on_support": float(model[support].sum()),
        "flux_model_off_support": float(model[~support].sum()),
        "shell_flux_true": float(shell.sum()),
        "resid_max_sigma": float(residual.max() / sigma),
        "resid_min_on_support_sigma": float(residual[support].min() / sigma),
        "resid_rms_off_support_sigma": float(residual[~support].std() / sigma),
        "resid_rms_on_support_sigma": float(residual[support].std() / sigma),
    }
    y, x = np.mgrid[:model.shape[0], :model.shape[1]]
    near_pts = np.zeros(model.shape, dtype=bool)
    for p in points:
        near = np.hypot(x - p["x"], y - p["y"]) <= 2 * bsig
        near_pts |= near
        p["flux_model_within_2sigma_beam"] = float(model[near].sum())
    shell_zone = support & ~near_pts
    out["shell_flux_model"] = float(model[shell_zone].sum())
    out["shell_flux_ratio"] = float(model[shell_zone].sum() / shell.sum())
    out["points"] = points
    out["pass"] = bool(abs(out["flux_ratio"] - 1.0) <= 0.10
                       and out["resid_min_on_support_sigma"] >= -5.0)
    return out


def plot(rows, sky, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(rows)
    fig, ax = plt.subplots(n, 5, figsize=(22, 4.4 * n), squeeze=False)
    vmax = float(sky.max())
    for i, (mode, restored, model, residual, m) in enumerate(rows):
        diff = sky - model
        dlim = float(np.abs(diff).max()) or 1.0
        rlim = float(np.abs(residual).max()) or 1.0
        panels = [
            (sky, f"true model  sum {sky.sum():.4g}", "inferno", 0, vmax),
            (model, f"recovered model  sum {model.sum():.4g}  "
                    f"ratio {m['flux_ratio']:.3f}", "inferno", 0, vmax),
            (diff, "true - recovered", "RdBu_r", -dlim, dlim),
            (residual, f"final residual  max {residual.max():.3g}  "
                       f"min {residual.min():.3g}", "RdBu_r", -rlim, rlim),
            (restored, "restored", "inferno", 0, float(restored.max())),
        ]
        for a, (im, title, cmap, lo, hi) in zip(ax[i], panels):
            h = a.imshow(im, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
            a.set_title(f"{mode}: {title}", fontsize=9)
            a.set_xticks([]); a.set_yticks([])
            fig.colorbar(h, ax=a, fraction=0.046)
    fig.tight_layout()
    fig.savefig(fname, dpi=110)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    if args.sky:
        path, key = args.sky.rsplit(":", 1)
        sky_in = np.load(path)[key].astype(np.float32)
        S = sky_in.shape[0]
    else:
        S = args.field

    psf_full = load_psf(args.psf)
    psf = crop_at_peak(psf_full, 2 * S)          # every offset in the field covered
    bsig = beam_sigma_px(psf)
    sidelobe = measure_psf_sidelobe(crop_at_peak(psf_full, 128))   # as run_imaging.py

    if args.sky:
        sky, shell, points = sky_in, sky_in, []
    else:
        sky, shell, points = make_sky(S, bsig, args, rng)
    clean_dirty = psf_convolve(sky, psf).astype(np.float32)
    if args.peak_snr > 0:
        sigma = float(clean_dirty.max()) / args.peak_snr
        dirty = clean_dirty + sigma * psf_shaped_noise(crop_at_peak(psf_full, S), rng)
    else:
        sigma, dirty = 0.0, clean_dirty
    print(f"[sim] psf {args.psf}  beam sigma {bsig:.2f} px  sidelobe {sidelobe:.3f}")
    print(f"[sim] true flux {sky.sum():.4f} Jy (shell {shell.sum():.4f})  "
          f"dirty peak {clean_dirty.max():.3e}  sigma {sigma:.3e}")

    mdn = dm = scale_relative = None
    if args.diffusion_ckpt:
        dm = load_diffusion_models(args)
        print(f"[sim] diffusion MP: B {args.diffusion_ckpt} ({args.diff_n_draws} draws)  "
              f"A {args.diff_ld_ckpt} ({args.diff_ld_draws if args.diff_ld_ckpt else 0} draws)")
    else:
        scale_relative = bool(checkpoint_is_scale_relative(args.ckpt))
        if args.sigma_min_px is None:
            args.sigma_min_px = checkpoint_sigma_min_px(args.ckpt)
        print(f"[sim] component width floor {args.sigma_min_px} px")
        mdn = load_mdn_checkpoint(args.ckpt, args.device)
    beam = gaussian_beam(S, bsig)

    report = {"args": vars(args), "beam_sigma_px": bsig, "sidelobe": sidelobe,
              "sigma": sigma, "scale_relative": scale_relative, "modes": {}}
    rows = []
    for mode in args.modes:
        model, residual, cycles, stop, thr = run_loop(
            mode, dirty, psf, sigma, sidelobe, mdn, scale_relative, args, dm)
        m = metrics(model, residual, sky, shell, [dict(p) for p in points],
                    beam, sigma, bsig)
        m.update({"stop": stop, "threshold": thr, "threshold_sigma": thr / (sigma or float("nan")),
                  "cycles": cycles})
        report["modes"][mode] = m
        restored = psf_convolve(model, beam) + residual
        rows.append((mode, restored, model, residual, m))
        np.savez_compressed(out / f"{mode}.npz", model=model, residual=residual,
                            restored=restored)
        print(f"[{mode}] stop={stop}  flux {m['flux_model']:.4f} / {m['flux_true']:.4f}"
              f" = {m['flux_ratio']:.3f}  shell {m['shell_flux_ratio']:.3f}  "
              f"off-support {m['flux_model_off_support']:.4f} Jy  "
              f"resid min {m['resid_min_on_support_sigma']:.1f} sigma  "
              f"PASS={m['pass']}", flush=True)

    np.savez_compressed(out / "truth.npz", sky=sky, dirty=dirty, psf=psf)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    plot(rows, sky, out / "compare.png")
    print(f"[sim] wrote {out}/report.json and compare.png")


if __name__ == "__main__":
    main()
