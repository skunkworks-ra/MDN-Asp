"""MDN-Asp imaging loop (general purpose).

Runs in this repo's pixi env (casatasks/casatools included):
  pixi run -e gpu python scripts/run_imaging.py --vis <ms> --imagename <name> [options]

Loop
----
1. tclean niter=0 (major cycle) -> .residual + .psf images
2. minor_cycle(): find residual peak, crop 128x128 centred on peak,
   MDN forward pass, subtract PSF response, repeat until threshold
3. Accumulate model_update into the CASA .model image
4. tclean niter=0 (calcres=True) -> updated .residual
5. Repeat until residual_peak < threshold

Uncertainty image (optional)
-----------------------------
If --n_posterior_samples > 0, after the loop draw N samples from the MDN
mixture posterior for each committed Aspen, render each sample's flux std
as a Gaussian at the same position/shape, and take the pixel-wise maximum
across all components. Written as <imagename>.uncertainty (CASA image).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

# --- locate MAD-Clean repo ---
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
from casatasks import tclean
from casatools import image as iatool

from mdn_asp.minor_cycle import (minor_cycle, psf_convolve, render_aspen,
                                 measure_psf_sidelobe, measure_psf_sidelobe_first)
from mdn_asp.models.mdn_asp import (
    MDNAsp, checkpoint_is_scale_relative, checkpoint_sigma_min_px, checkpoint_train_arg,
    load_mdn_checkpoint)
from mdn_asp.constants import unstandardise_log_flux
from mdn_asp.constants import BEAM_SIGMA_PX, SIGMA_MAX_PX

_PSF_CROP = 128


def dump_components(comp_dir, major, pre, post):
    """Write every component of one major cycle: majorNN_commit.json (as the
    network emitted it, before the refit), majorNN_mixture.json (raw mixture,
    same order) and, if a refit ran, majorNN_refit.json (after it, same order)."""
    comp_dir.mkdir(parents=True, exist_ok=True)
    stem = comp_dir / f"major{major:02d}"
    scalars = ("cid", "born_in_major", "cx", "cy", "flux", "sig_maj", "sig_min",
               "pa", "cut_scale", "peak_row", "peak_col", "peak_val", "sigma_local")
    with open(f"{stem}_commit.json", "w") as f:
        json.dump([{"i": i, **{k: getattr(c, k) for k in scalars},
                    "mode6d": list(c.mode6d)} for i, c in enumerate(pre)], f)
    with open(f"{stem}_mixture.json", "w") as f:
        json.dump([{"i": i, "logits": c.params.logits[0].tolist(),
                    "mu": c.params.mu[0].tolist(),
                    "log_std": c.params.log_std[0].tolist()}
                   for i, c in enumerate(pre)], f)
    if post is not None:
        with open(f"{stem}_refit.json", "w") as f:
            json.dump([{"i": i, **{k: getattr(c, k) for k in
                                   ("cx", "cy", "flux", "sig_maj", "sig_min", "pa")}}
                       for i, c in enumerate(post)], f)


def dump_state(comp_dir, major, all_commits):
    """majorNN_state.json: every component so far with its current (post-refit)
    parameters. Together with the commit and mixture files this is what
    --resume needs to rebuild all_commits."""
    comp_dir.mkdir(parents=True, exist_ok=True)
    keys = ("cid", "born_in_major", "cx", "cy", "flux", "sig_maj", "sig_min", "pa")
    with open(comp_dir / f"major{major:02d}_state.json", "w") as f:
        json.dump([{k: getattr(c, k) for k in keys} for c in all_commits], f)


def load_components(comp_dir, n_cycles):
    """Rebuild all_commits from the last majorNN_state.json plus the commit and
    mixture files of the cycles that created each component. [] if there is no
    state file (runs made before component recording)."""
    from mdn_asp.minor_cycle import AspenCommit
    from mdn_asp.models.mdn_asp import MixParams
    path = comp_dir / f"major{n_cycles - 1:02d}_state.json"
    if not path.exists():
        return []
    with open(path) as f:
        state = json.load(f)
    born = {}
    for b in sorted({s["born_in_major"] for s in state}):
        with open(comp_dir / f"major{b:02d}_commit.json") as f:
            com = json.load(f)
        with open(comp_dir / f"major{b:02d}_mixture.json") as f:
            mix = json.load(f)
        for c, m in zip(com, mix):
            born[c["cid"]] = (c, m)
    t = lambda v: torch.tensor([v], dtype=torch.float32)
    out = []
    for s in state:
        c, m = born[s["cid"]]
        out.append(AspenCommit(
            cx=s["cx"], cy=s["cy"], flux=s["flux"],
            sig_maj=s["sig_maj"], sig_min=s["sig_min"], pa=s["pa"],
            params=MixParams(logits=t(m["logits"]), mu=t(m["mu"]),
                             log_std=t(m["log_std"])),
            cut_scale=c["cut_scale"], peak_row=c["peak_row"],
            peak_col=c["peak_col"], peak_val=c["peak_val"],
            sigma_local=c["sigma_local"], mode6d=tuple(c["mode6d"]),
            born_in_major=s["born_in_major"], cid=s["cid"]))
    return out


def check_model_consistency(all_commits, model_img, nsigma=None):
    """Print sum and max abs difference between the rendered components and
    the .model image. Approximate: each component is drawn in a 4 sigma window."""
    H, W = model_img.shape
    img = np.zeros((H, W), np.float32)
    for c in all_commits:
        r = int(min(max(4 * c.sig_maj, 8), max(H, W)))
        r0, r1 = max(0, int(c.cy) - r), min(H, int(c.cy) + r + 1)
        c0, c1 = max(0, int(c.cx) - r), min(W, int(c.cx) + r + 1)
        img[r0:r1, c0:c1] += render_aspen(c.cx - c0, c.cy - r0, c.flux, c.sig_maj,
                                          c.sig_min, c.pa, (r1 - r0, c1 - c0), nsigma=nsigma)
    print(f"[img] Resume check: {len(all_commits)} components render to "
          f"{img.sum():.6f} Jy, .model has {model_img.sum():.6f} Jy, "
          f"max abs difference {np.abs(img - model_img).max():.3e} Jy/pixel "
          f"(model max {model_img.max():.3e})")


def dump_tiles(out_dir: Path, all_tiles: list[dict]) -> None:
    """Persist diffusion tiles so ``--resume`` can continue ``refit_tiles`` and ``amp_error``."""
    if not all_tiles:
        return
    Wn = max(max(t["y1"] - t["y0"], t["x1"] - t["x0"]) for t in all_tiles)
    u = np.zeros((len(all_tiles), Wn, Wn), dtype=np.float32)
    hs, ws = [], []
    for i, t in enumerate(all_tiles):
        th, tw = t["u"].shape
        u[i, :th, :tw] = t["u"]
        hs.append(th)
        ws.append(tw)
    np.savez_compressed(
        out_dir / "tiles_state.npz",
        u=u,
        h=np.array(hs, dtype=np.int32),
        w=np.array(ws, dtype=np.int32),
        a=np.array([t["a"] for t in all_tiles], dtype=np.float64),
        y0=np.array([t["y0"] for t in all_tiles], dtype=np.int32),
        y1=np.array([t["y1"] for t in all_tiles], dtype=np.int32),
        x0=np.array([t["x0"] for t in all_tiles], dtype=np.int32),
        x1=np.array([t["x1"] for t in all_tiles], dtype=np.int32),
        **({"sv": np.stack([np.pad(t["sv"], ((0, Wn - t["sv"].shape[0]), (0, Wn - t["sv"].shape[1])))
                            for t in all_tiles]).astype(np.float32)}
           if all("sv" in t for t in all_tiles) else {}),
    )


def load_tiles(out_dir: Path) -> list[dict]:
    path = out_dir / "tiles_state.npz"
    if not path.exists():
        return []
    z = np.load(path)
    out = []
    for i in range(len(z["a"])):
        th, tw = int(z["h"][i]), int(z["w"][i])
        out.append({
            "y0": int(z["y0"][i]),
            "y1": int(z["y1"][i]),
            "x0": int(z["x0"][i]),
            "x1": int(z["x1"][i]),
            "u": z["u"][i, :th, :tw].astype(np.float32),
            "a": float(z["a"][i]),
            **({"sv": z["sv"][i, :th, :tw].astype(np.float32)} if "sv" in z.files else {}),
        })
    return out


def _sha256(path, chunk=1 << 20):
    import hashlib
    p = Path(path)
    if not p.exists():
        return f"hf:{path}"
    if p.is_dir():
        h = hashlib.sha256()
        for sub in sorted(x for x in p.rglob("*") if x.is_file()):
            h.update(str(sub.relative_to(p)).encode("utf-8"))
            with open(sub, "rb") as f:
                for b in iter(lambda: f.read(chunk), b""):
                    h.update(b)
        return h.hexdigest()
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _git_info():
    import subprocess
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=_REPO, capture_output=True,
                                        text=True, check=True).stdout.strip()
        return {"commit": run("rev-parse", "HEAD"),
                "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except Exception as e:
        return {"error": repr(e)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MDN-Asp major/minor cycle imaging loop.")

    # Required
    p.add_argument("--vis",       required=True, help="Input Measurement Set.")
    p.add_argument("--imagename", required=True, help="Output image name stem (no extension).")

    # Paths
    p.add_argument("--ckpt",    default="results/mdn_asp_v1/best.pt")
    p.add_argument("--out_dir", default="results/mdn_asp")
    p.add_argument("--scale_relative", default=None,
                   action=argparse.BooleanOptionalAction,
                   help="Read the flux head as log(flux/max|residual|) rather "
                        "than the legacy fixed-constant standardisation. "
                        "Default: auto-detect from the checkpoint. Must match "
                        "how the model was TRAINED -- a mismatch is a silent "
                        "flux error that grows with dynamic range.")

    # tclean imaging params
    p.add_argument("--imsize",    type=int,   default=512)
    p.add_argument("--cell",      default="2.5arcsec")
    p.add_argument("--field",     default="",    help="tclean field selection. Empty string = all fields; pass e.g. '2~8' or '3C391*' to restrict to science targets.")
    p.add_argument("--spw",       default="",    help="tclean spw selection.")
    p.add_argument("--gridder",      default="mosaic")
    p.add_argument("--wprojplanes",  type=int, default=-1,
                   help="Number of w-projection planes. -1 = auto (tclean default). Only used when --gridder=wproject.")
    p.add_argument("--specmode",  default="mfs")
    p.add_argument("--robust",    type=float, default=0.5)

    # Mask
    p.add_argument("--mask", type=str, default=None,
                   help="CASA mask image path. If not set, a centred circle of "
                        "--mask_radius_px is used.")
    p.add_argument("--mask_radius_px", type=int, default=200,
                   help="Circular mask radius in px from image centre, used "
                        "when --mask is not given. Set <= 0 for the full "
                        "image. Restricts where the loop cleans and where the "
                        "peak and noise are measured; tclean still images the "
                        "full field so sidelobes subtract correctly.")

    p.add_argument("--fit_unmasked", action="store_true",
                   help="Fit against the UNMASKED residual. The mask still "
                        "restricts where components are placed (as in CLEAN), "
                        "but the data handed to the proposal, the refit and "
                        "the joint solve is the true residual, not one zeroed "
                        "outside the mask. The Aspen profile g_k is then the "
                        "only weight in <g_k, r>.")

    # Loop control
    p.add_argument("--n_sigma",          type=float, default=3.0,
                   help="Stopping threshold in sigma: noise_floor = n_sigma * "
                        "sigma_rms, re-estimated each major cycle. The final "
                        "threshold is max(psf_floor, noise_floor).")
    p.add_argument("--no_psf_floor",     action="store_true",
                   help="Global stop is n_sigma * sigma_rms only (Cotton-Schwab). "
                        "The minor cycle still stops at sidelobe_level * its own "
                        "current peak. Default: global stop also >= the fixed "
                        "dirty_peak * sidelobe_level.")
    p.add_argument("--resume",           action="store_true",
                   help="Continue from the .model, .residual, .psf and log.json "
                        "already in out_dir: skip the initial tclean, keep the "
                        "dirty-image dirty_peak/psf_floor from log.json, and run "
                        "--max_major MORE major cycles.")
    p.add_argument("--global_sigma",     action="store_true",
                   help="Condition the MDN on the off-source sigma_rms instead "
                        "of the cutout MAD, which residual source structure inflates.")
    p.add_argument("--loop_gain",        type=float, default=0.1)
    p.add_argument("--max_major",        type=int,   default=20)
    p.add_argument("--max_components",   type=int,   default=1000,
                   help="Max minor-cycle components per major cycle.")
    p.add_argument("--refit_exact",      action="store_true",
                   help="After each minor cycle, refit every Aspen against the "
                        "exact likelihood by add-back (Asp-SBI phase 2).")
    p.add_argument("--major_before_refit", action="store_true",
                   help="With --refit_exact: run a major cycle after the minor "
                        "cycle and refit against that residual instead of the "
                        "image-plane estimate, then a second major cycle.")
    p.add_argument("--refit_all",        action="store_true",
                   help="With --refit_exact: refit every component found so far "
                        "(oldest first) at each major cycle, not only the new "
                        "ones. The model image is written from the refit result.")
    p.add_argument("--size_bias",        type=float, default=0.0,
                   help="With --refit_exact: add size_bias * ln(sqrt(sig_maj*sig_min)) "
                        "to the likelihood when ranking candidate shapes. Positive "
                        "favours larger components. 0 = plain likelihood.")
    p.add_argument("--sidelobe_level", type=float, default=None,
                   help="Use this PSF sidelobe level instead of measuring it. "
                        "Sets the minor-cycle floor (level x peak) and the PSF floor.")
    p.add_argument("--flux_clip", action="store_true",
                   help="MDN minor cycle: clip each full flux estimate to [0, |residual| max at the cycle start] "
                        "(the overflow guard removed in ab673a1).")
    p.add_argument("--cap_to_pixel", action="store_true",
                   help="Scale a component down so its full-flux dirty peak does not "
                        "exceed the residual at the pixel it was picked from.")
    p.add_argument("--sidelobe_abs", action="store_true",
                   help="Sidelobe level = peak of the first positive sidelobe (not |PSF min|), "
                        "and the minor-cycle sidelobe floor uses max|residual| (negative "
                        "holes count), not the positive peak.")
    p.add_argument("--stop_below_noise", action="store_true",
                   help="Stop the minor cycle when the network's full flux estimate, "
                        "convolved with the PSF, peaks below 1 x sigma_rms.")
    p.add_argument("--refit_zero_rejected", action="store_true",
                   help="With --refit_exact: a component with no positive-amplitude "
                        "shape steps its flux toward 0 by --refit_gain, instead of "
                        "keeping its old flux.")
    p.add_argument("--error_map",        action="store_true",
                   help="With --refit_exact: at the end write <imagename>.errormap, the "
                        "per-pixel std of the model under the refit posterior "
                        "(shape weights include --size_bias; independent Aspen).")
    p.add_argument("--error_map_joint",  action="store_true",
                   help="With --refit_exact: also write <imagename>.errormap_joint, the shape spread of "
                        "--error_map plus the JOINT amplitude covariance of all active Aspen "
                        "(mdn_asp.refit.joint_amplitude_error), so overlapping Aspen are not independent.")
    p.add_argument("--flux_prior_weight", type=float, default=0.0,
                   help="MAP instead of ML amplitudes in --refit_exact. Scales the proposal's flux-posterior "
                        "sigma (larger = weaker prior). 0 = off, unregularised "
                        "ML, bit-identical to before.")
    p.add_argument("--refit_repropose", action="store_true",
                   help="With --refit_exact: run the MDN again on each Aspen's add-back "
                        "residual and add the fresh mixture shapes to the refit candidates.")
    p.add_argument("--refit_gradient", action="store_true",
                   help="With --refit_all: after the discrete refit, a joint Adam fit of "
                        "position, sizes, PA and log flux of every Aspen on the full-field chi^2.")
    p.add_argument("--grad_iters",  type=int,   default=200, help="--refit_gradient Adam steps per major cycle.")
    p.add_argument("--grad_lr_pos", type=float, default=0.05, help="--refit_gradient learning rate on positions (px).")
    p.add_argument("--grad_lr_log", type=float, default=0.02,
                   help="--refit_gradient learning rate on log sizes, PA and log flux.")
    p.add_argument("--grad_objective", choices=("image", "vis"), default="image",
                   help="--refit_gradient objective: image-plane L2 (exact transpose B^T) or visibility PSD quadratic.")
    p.add_argument("--refit_gain",       type=float, default=0.5,
                   help="Damping on the refit amplitude step (1.0 = jump to the "
                        "exact profile maximum).")
    p.add_argument("--refit_method", choices=("cd", "nnls"), default="cd",
                   help="Amplitude refit solver for Aspens and diffusion tiles: "
                        "'cd' (sequential coordinate descent) or 'nnls' (joint preconditioned FISTA NNLS).")
    p.add_argument("--refit_nullspace_lam", type=float, default=0.0,
                   help="Tikhonov nullspace regularisation weight (scaled by beam area) "
                        "to damp missing-short-spacing ker(B) ghosts.")
    p.add_argument("--refit_nnls_iters", type=int, default=60,
                   help="Max FISTA iterations for --refit_method nnls.")
    p.add_argument("--mdn_density_mode", action="store_true",
                   help="MDN minor cycle: select peak posterior density component pi_k / prod(sigma_k) "
                        "instead of highest weight pi_k.")
    p.add_argument("--mdn_profile_amp", action="store_true",
                   help="MDN minor cycle: use closed-form exact profile likelihood amplitude <g, r>/<g, B g>.")
    p.add_argument("--mdn_delta_fallback", action="store_true",
                   help="MDN minor cycle: fall back to a point component when its profile likelihood beats the Aspen's.")
    p.add_argument("--uv_coupling_min", type=float, default=0.0,
                   help="MDN minor cycle: fall back to a point component when <g, B g>/<g, B_main g> < uv_coupling_min.")
    p.add_argument("--sigma_min_px",     type=float, default=None,
                   help="Override MDN sigma clip floor (px). Default: BEAM_SIGMA_PX (1.4px).")
    p.add_argument("--sigma_max_px",     type=float, default=None,
                   help="Override MDN sigma_maj clip ceiling (px). Default: the checkpoint's "
                        "training --sigma_max_px, else SIGMA_MAX_PX (~11.2px).")
    p.add_argument("--render_nsigma",    type=float, default=None,
                   help="Truncate every rendered component at this many sigma. Default: "
                        "the checkpoint's training --render_nsigma, else untruncated.")
    p.add_argument("--config_idx",       type=int,   default=3,
                   help="MDN conditioning one-hot. Training always used 3, so keep 3.")
    p.add_argument("--divergence_tol",   type=float, default=0.20,
                   help="Stop if residual peak rises by more than this fraction between major cycles.")
    p.add_argument("--global_threshold", type=float, default=None,
                   help="Override computed stopping threshold (Jy/beam).")

    # Posterior uncertainty
    p.add_argument("--n_posterior_samples", type=int, default=0,
                   help="Draw N samples from MDN mixture to build uncertainty image. 0 = off.")

    # Diffusion sky model (replaces the MDN minor cycle when set)
    p.add_argument("--diffusion_ckpt", type=str, default=None,
                   help="train_tiled_asp.py checkpoint (variant A). Replaces the MDN minor "
                        "cycle: each step cuts the 256 px tile centred on the residual peak, "
                        "scales it by its own peak, draws, keeps the central 128 px and "
                        "subtracts its PSF response (loop gain 1). Stops as minor_cycle().")
    p.add_argument("--diff_n_draws", type=int, default=1,
                   help="Draws per tile per major cycle.")
    p.add_argument("--diff_update", choices=("draw", "mean"), default="draw",
                   help="Model update from the draws: the first draw, or the mean of all.")
    p.add_argument("--diff_mask_k", type=float, default=None,
                   help="Draw mask: keep a pixel only where mean > k * std over the draws "
                        "(needs --diff_n_draws >= 2). Default: off.")
    p.add_argument("--diff_batch", type=int, default=16,
                   help="Tile draws per sampler call (lower to save GPU memory).")
    p.add_argument("--diff_seed", type=int, default=0)
    p.add_argument("--diff_unmask_output", action="store_true",
                   help="Do not multiply the committed draw by the mask; the mask only "
                        "limits where tiles are centred (as MDN components).")
    p.add_argument("--diff_guide", type=float, default=None,
                   help="Likelihood guidance in the sampler: fraction of the exact line-search "
                        "step along the residual (see Diffusion.sample). Default: off.")
    p.add_argument("--diff_gain", type=float, default=1.0,
                   help="Loop gain on each diffusion tile commit (1 = whole draw mean).")
    p.add_argument("--diff_taper", action="store_true",
                   help="--diff_mp: multiply every candidate by the W x W Hann taper before scoring and commit.")
    p.add_argument("--diff_rebuild_drop", type=float, default=None,
                   help="--diff_mp: rebuild the whole model when the off-source sigma has dropped by this fraction "
                        "since the last rebuild (first reference: cycle 0), and always once at the stopping threshold. "
                        "Rebuild = pull back r0 = r + B x, empty model, matching pursuit on r0 at the current sigma "
                        "with no sidelobe floor. After the threshold rebuild: normal cycles, no more rebuilds, until "
                        "below threshold; the rebuilt model is kept only if its residual sum of squares is lower "
                        "than before that rebuild. Default: off.")
    p.add_argument("--diff_ensemble", type=int, default=0,
                   help="--diff_mp --diff_rebuild_drop: K members in all. Member 0 is the run itself; members 1..K-1 "
                        "restart from the model before the final rebuild with --diff_seed + k and redo the final rebuild "
                        "and the cycles after it. Writes <imagename>.ens_mean / .ens_std and ensemble/*.npy; "
                        "the .model left at the end is member 0's. 0 = off.")
    p.add_argument("--diff_save_atoms", action="store_true",
                   help="--diff_mp: write atoms_majorNN.npz per cycle (committed atoms u, padded to shape_size at the "
                        "top left, amplitude a, window y0 y1 x0 x1), before the refit; and majorNN_start_residual.npy, "
                        "_start_model.npy (what the cycle starts from), majorNN_residual.npy, _model.npy (after it).")
    p.add_argument("--diff_save_draws", type=int, default=0,
                   help="--diff_mp: write draws_majorNN.npz with the first N steps of each cycle: residual window, "
                        "every candidate (B draws, A draws, delta; as scored), J, winner index, a*, peak. 0 = off.")
    p.add_argument("--diff_error", action="store_true",
                   help="Write <imagename>.diff_err: per-pixel std of the diffusion model, the shape posterior of "
                        "each step (variance of loop_gain a*_i g_i over all candidates, weights exp(J_i / sigma^2), "
                        "rescaled by the final tile amplitude) plus the JOINT amplitude covariance of all tiles "
                        "(mdn_asp.refit.joint_amplitude_error). Steps are independent in the shape part.")
    p.add_argument("--diff_amp_error", action="store_true",
                   help="Tile refit: write <imagename>.amp_error, sqrt(sum_k (u_k sigma / sqrt(<u_k, B u_k>))^2) over the "
                        "final tiles (amplitude noise error, shapes fixed, tiles taken as independent).")
    p.add_argument("--refit_noise_lam", type=float, default=None,
                   help="Tile refit: amplitude per tile minimising chi^2 / sigma^2 + lam R(r) on its window "
                        "(R: residual autocorrelation vs sigma^2 PSF, lags <= --refit_noise_lag), undamped grid. "
                        "Default: off (the damped amplitude refit).")
    p.add_argument("--refit_noise_lag", type=int, default=16)
    p.add_argument("--cycle_report", action="store_true",
                   help="--diff_mp --refit_exact: print noise-like numbers of the residual after the minor cycle "
                        "and after the refit (extremes in and off the mask, residual autocorrelation vs sigma^2 PSF, "
                        "sum of squares). Prints only; no stop uses them.")
    p.add_argument("--diff_mp", action="store_true",
                   help="Matching pursuit with a variant P checkpoint: --diff_n_draws posterior draws "
                        "+ the delta per step, argmax J, commit --loop_gain x a* g (diffusion_mp_cycle).")
    p.add_argument("--diff_ld_ckpt", type=str, default=None,
                   help="With --diff_mp: conditional dictionary A (variant LD) checkpoint; its draws join "
                        "the candidates of each step (plan-diffusion-local-proposal.md 3A.5). "
                        "With --diff_n_draws 0: A draws + the delta only, no B draws.")
    p.add_argument("--diff_ld_draws", type=int, default=10, help="A draws per step.")
    p.add_argument("--diff_ld_cg_iters", type=int, default=0,
                   help="Gauss-Newton CG likelihood guidance steps per DDIM step for LD dictionary draws (0 = off).")
    p.add_argument("--diff_q_space", action="store_true",
                   help="--diff_mp: sample shape posterior in positive softplus q-space with exact Hann-tapered forward operator.")
    p.add_argument("--diff_dof_pen", type=float, default=0.0,
                   help="--diff_mp: AIC complexity penalty per effective beam DOF (e.g. 2.0 = standard AIC; 0 = off).")
    p.add_argument("--diff_briggs_psd", action="store_true",
                   help="--diff_mp: score candidates using empirical off-source Briggs noise autocorrelation instead of sigma^2 * PSF.")
    p.add_argument("--diff_scales", type=str, default="",
                   help="--diff_mp: comma-separated sub-window crop sizes (e.g. '16,32,64') added as multi-scale candidates.")
    p.add_argument("--rebuild_metric", choices=("image_l2", "vis_chi2"), default="image_l2",
                   help="Rebuild acceptance metric: 'image_l2' (residual sum of squares) or 'vis_chi2' (visibility quadratic).")

    p.add_argument("--pblimit", type=float, default=0.05,
                   help="Primary beam gain limit below which pixels are masked. "
                        "Set small (e.g. 0.05) to suppress PB masking over the image.")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    if args.refit_all and not args.refit_exact:
        p.error("--refit_all needs --refit_exact")
    if args.refit_repropose and not args.refit_exact:
        p.error("--refit_repropose needs --refit_exact")
    if args.refit_gradient and not args.refit_all:
        p.error("--refit_gradient needs --refit_all")
    if args.error_map and not args.refit_exact:
        p.error("--error_map needs --refit_exact")
    if args.diff_mask_k is not None and args.diff_n_draws < 2:
        p.error("--diff_mask_k needs --diff_n_draws >= 2")
    if args.diff_mp and args.diff_n_draws < 1 and not args.diff_ld_ckpt:
        p.error("--diff_mp with --diff_n_draws 0 needs --diff_ld_ckpt")
    return args


# ---------------------------------------------------------------------------
# CASA image helpers
# ---------------------------------------------------------------------------

def read_casa_image(imagepath: str) -> np.ndarray:
    """Read a CASA image -> 2D float32 (ny, nx) row-major."""
    ia = iatool()
    ia.open(imagepath)
    chunk = ia.getchunk()   # (nx, ny, npol, nchan) Fortran order
    ia.close()
    return chunk.squeeze().T.astype(np.float32)


def write_casa_image(imagepath: str, data: np.ndarray) -> None:
    """Write (ny, nx) float32 array into an existing CASA image."""
    ia = iatool()
    ia.open(imagepath)
    arr = data.T.astype(np.float32)[:, :, np.newaxis, np.newaxis]
    ia.putchunk(arr)
    ia.close()


def copy_and_write_casa_image(src: str, dst: str, data: np.ndarray) -> None:
    """Copy a CASA image (for coordinate system) then overwrite pixel data."""
    import shutil
    if Path(dst).exists():   # --resume rewrites images from the earlier run
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    write_casa_image(dst, data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mask(imsize: int, radius_px: int) -> np.ndarray:
    cy, cx = imsize // 2, imsize // 2
    Y, X = np.ogrid[:imsize, :imsize]
    return ((X - cx)**2 + (Y - cy)**2) <= radius_px**2


def crop_psf_centred(psf: np.ndarray, size: int) -> np.ndarray:
    py, px = np.unravel_index(int(np.argmax(psf)), psf.shape)
    half = size // 2
    r0, c0 = py - half, px - half
    H, W = psf.shape
    out = np.zeros((size, size), dtype=np.float32)
    sr0, sr1 = max(0, r0), min(H, r0 + size)
    sc0, sc1 = max(0, c0), min(W, c0 + size)
    dr0, dc0 = sr0 - r0, sc0 - c0
    out[dr0:dr0+(sr1-sr0), dc0:dc0+(sc1-sc0)] = psf[sr0:sr1, sc0:sc1]
    return out


def measure_noise_rms(residual: np.ndarray, mask: np.ndarray) -> float:
    """Robust noise estimate from the lower half of the pixel distribution.

    Blanked pixels must be excluded or the estimate collapses to exactly zero
    whenever the imaged field is much larger than the primary beam: ``tclean``
    writes identical zeros outside ``pblimit``, and a median over an array that
    is mostly zeros is zero. Measured on G55 at 1280 px / 8 arcsec (2026-08-10):
    93% of the residual is exactly 0, the 400-600 px annulus is entirely 0, and
    this returned 0.0 while the true noise inside the beam was ~2e-4 (median
    |x| = 6.9e-5). A zero here silently disables the noise floor -- both the
    major-cycle threshold and the ``n_sigma_stop * sigma`` floor inside
    ``minor_cycle`` -- so the loop cleans on the absolute threshold alone.
    3C391 never hit it: at 480 px / 2.5 arcsec the field sits inside the PB.
    """
    pixels = residual[mask].ravel()
    pixels = pixels[np.isfinite(pixels)]
    # Drop blanked pixels: with pblimit > 0, tclean writes identical zeros
    # outside the primary beam and a median over a mostly-zero array is zero.
    # Prefer pblimit < 0 (no PB masking) -- then there is nothing to drop.
    nz = pixels[pixels != 0.0]
    if nz.size >= 100:
        pixels = nz
    if pixels.size == 0:
        return 0.0

    # One-sided MAD from the LOWER half of the distribution. Sky emission is
    # positive, so the lower half is noise-dominated while the upper half
    # carries source and sidelobe flux. Measured on G55 (2026-08-10): the
    # two-sided MAD over the full image gives 2.87e-5 against 1.78e-5 here,
    # i.e. ~61% high, because sigma-clipping does not remove an extended
    # source that fills much of the field. Off-source annuli bracket it
    # (2.60e-5 at 500-600 px). NOTE this is still a CONFUSION floor, not
    # thermal noise -- std/MAD is 2.3 on this residual, so the tails are
    # sidelobes, not Gaussian.
    med = float(np.median(pixels))
    lower = pixels[pixels <= med]
    if lower.size == 0:
        return 0.0
    return 1.4826 * float(np.median(med - lower))


# ---------------------------------------------------------------------------
# Posterior uncertainty image
# ---------------------------------------------------------------------------

def build_uncertainty_image(all_commits: list, shape: tuple[int, int],
                             n_samples: int, device: torch.device,
                             scale_relative: bool = False,
                             nsigma: float | None = None) -> np.ndarray:
    """Pixel-wise max of per-Aspen flux std, rendered at each component's position.

    For each committed Aspen draw N samples from its MDN mixture posterior,
    compute the std of the resulting flux values, render that std as a Gaussian
    at the Aspen's position and shape, then take the pixel-wise maximum across
    all components. This gives a conservative per-pixel uncertainty map in Jy.
    """
    uncertainty = np.zeros(shape, dtype=np.float32)

    for commit in all_commits:
        mp = commit.params   # MixParams: logits (1,K), mu (1,K,7), log_std (1,K,7)
        logits  = mp.logits.squeeze(0)          # (K,)
        mu      = mp.mu.squeeze(0)              # (K, 7)
        log_std = mp.log_std.squeeze(0)         # (K, 7)

        # Sample component indices from the categorical
        probs = torch.softmax(logits, dim=0)
        idx = torch.multinomial(probs.unsqueeze(0).expand(n_samples, -1),
                                num_samples=1).squeeze(1)  # (N,)

        # Sample from selected Gaussian components
        mu_sel      = mu[idx]       # (N, 7)
        std_sel     = torch.exp(log_std[idx])  # (N, 7)
        samples     = mu_sel + std_sel * torch.randn_like(std_sel)  # (N, 7)

        # Decode flux dimension (index 2 is log_flux_std)
        log_flux_std_samples = samples[:, 2].numpy()
        # Same inverse as the mode readout in minor_cycle. In scale-relative
        # mode the head emits log(flux/scale), so the only inverse is adding
        # log(scale) back; cut_scale is 1.0 for absolute-flux checkpoints.
        if scale_relative:
            cut_scale = float(getattr(commit, "cut_scale", 1.0) or 1.0)
            log_flux_vals = log_flux_std_samples + np.log(cut_scale)
        else:
            log_flux_vals = np.array(
                [unstandardise_log_flux(float(z)) for z in log_flux_std_samples])
        # Overflow guard only; do not cap at 1 Jy (bright sources exceed 1 Jy).
        log_flux_vals = np.clip(log_flux_vals, -30.0, 20.0)
        flux_samples = np.exp(log_flux_vals)

        # Robust IQR-based std -- immune to posterior tail outliers
        p16, p84 = np.percentile(flux_samples, [16.0, 84.0])
        flux_std = float((p84 - p16) / 2.0)
        if flux_std <= 0 or not np.isfinite(flux_std):
            continue

        # Render the flux std as a Gaussian at this Aspen's position/shape
        component_uncertainty = render_aspen(
            commit.cx, commit.cy,
            flux_std,
            commit.sig_maj, commit.sig_min, commit.pa,
            shape, nsigma=nsigma,
        )
        uncertainty = np.maximum(uncertainty, component_uncertainty)

    return uncertainty


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Diffusion sky model
# ---------------------------------------------------------------------------

def load_diffusion(ckpt: str, device: torch.device):
    """(Diffusion, training args) from a train_tiled_asp.py checkpoint, as eval_diffusion.py."""
    from mdn_asp.models.diffusion import Diffusion, build_diffusion_net
    if Path(ckpt).is_dir():   # HF folder from convert_checkpoint_to_hf.py --diffusion
        from safetensors.torch import load_file
        with open(Path(ckpt) / "config.json") as f:
            ck = {"args": json.load(f)["args"], "model": load_file(str(Path(ckpt) / "model.safetensors"))}
    else:
        ck = torch.load(ckpt, map_location=device, weights_only=False)
    a = argparse.Namespace(**ck["args"])
    net = build_diffusion_net(a).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return Diffusion(net, a, device), a


@torch.no_grad()
def diffusion_minor_cycle(diff, a, residual, psf, sigma, args, device, gen, mask,
                          threshold, sidelobe_level):
    """CLEAN-style minor cycle with the diffusion model.

    Each step: global positive peak of the residual (inside mask), a.patch px
    tile centred on it, scaled by its own max|value| (the peak, as in
    training), mean or first of --diff_n_draws draws scaled back, central
    patch/2 px kept (zeroed outside mask), added to the update and its PSF
    response subtracted from the residual in the image plane. Loop gain 1.
    Stops when the peak falls below max(threshold, sidelobe_level x start
    peak), rises above --divergence_tol, or after
    --max_components steps.
    Each step commits --diff_gain x the kept crop; the unit-gain crop and its
    place are recorded as a tile so the refit can rescale it later.
    Returns (update, max draw std per pixel or None, n steps, stop reason, tiles),
    tiles = list of dicts {y0, y1, x0, x1, u (unit shape crop), a (amplitude)}.
    """
    from mdn_asp.minor_cycle import PSFConvolver
    T, half, crop = a.patch, a.patch // 2, a.patch // 4
    H, W = residual.shape
    res = residual.astype(np.float32).copy()
    update = np.zeros((H, W), np.float32)
    std_img = np.zeros((H, W), np.float32) if args.diff_n_draws > 1 else None
    convolve = PSFConvolver(psf, (H, W), device)
    psf_big = torch.from_numpy(crop_psf_centred(psf, a.psf_size)).to(device)
    c0 = (a.psf_size - T) // 2
    psf_cut = psf_big[c0:c0 + T, c0:c0 + T]
    n = args.diff_n_draws
    tiles = []

    def peak_of(r):
        s = np.where(mask & (r > 0), r, 0.0)
        i = int(np.argmax(s))
        return divmod(i, W), float(s.flat[i])

    (_, _), start_peak = peak_of(res)
    floor = max(threshold, sidelobe_level * start_peak)   # as minor_cycle(abs_noise_floor=threshold)
    prev, steps, stop = start_peak, 0, "max_components"
    while steps < args.max_components:
        (r, c), pk = peak_of(res)
        if pk <= 0.0 or pk < floor:
            stop = "noise_floor" if pk <= 0.0 or pk < threshold else "sidelobe_floor"
            break
        if pk > prev * (1.0 + args.divergence_tol):
            stop = "divergence"
            break
        prev = pk
        pad = np.zeros((H + T, W + T), np.float32)
        pad[half:half + H, half:half + W] = res
        tile = torch.from_numpy(pad[r:r + T, c:c + T].copy()).to(device)   # peak at (half, half)
        scale = tile.abs().max().clamp_min(1e-30)
        batch = {"img": (tile / scale)[None].expand(n, -1, -1),
                 "sigma": (torch.tensor(sigma, device=device) / scale).repeat(n),
                 "psf": psf_big.expand(n, -1, -1), "psf_cut": psf_cut.expand(n, -1, -1)}
        d = torch.cat([diff.sample({k: v[i:i + args.diff_batch] for k, v in batch.items()}, gen=gen,
                                   guide=getattr(args, "diff_guide", None))
                       for i in range(0, n, args.diff_batch)]) * scale
        u = d[0] if args.diff_update == "draw" else d.mean(0)
        if args.diff_mask_k is not None:
            u = u * (d.mean(0) > args.diff_mask_k * d.std(0))
        u = u[half - crop:half + crop, half - crop:half + crop].cpu().numpy()
        # Place the central crop, clipped at the image edge.
        r0, c0_ = r - crop, c - crop
        y0, x0 = max(r0, 0), max(c0_, 0)
        y1, x1 = min(r0 + 2 * crop, H), min(c0_ + 2 * crop, W)
        shape = u[y0 - r0:y1 - r0, x0 - c0_:x1 - c0_]
        if not args.diff_unmask_output:
            shape = shape * mask[y0:y1, x0:x1]
        step = np.zeros((H, W), np.float32)
        step[y0:y1, x0:x1] = args.diff_gain * shape
        tiles.append({"y0": y0, "y1": y1, "x0": x0, "x1": x1,
                      "u": shape.astype(np.float32), "a": float(args.diff_gain)})
        update += step
        res -= convolve(step)
        if std_img is not None:
            sd = d.std(0)[half - crop:half + crop, half - crop:half + crop].cpu().numpy()
            np.maximum(std_img[y0:y1, x0:x1], sd[y0 - r0:y1 - r0, x0 - c0_:x1 - c0_],
                       out=std_img[y0:y1, x0:x1])
        steps += 1
        print(f"[img]     tile {steps}  peak {pk:.4e} at ({r},{c})  added {step.sum():.4f} Jy", flush=True)
    return update, std_img, steps, stop, tiles


@torch.no_grad()
def diffusion_mp_cycle(diff, a, residual, psf, sigma, args, device, gen, mask,
                       threshold, sidelobe_level, diffA=None, aA=None, draws_out=None,
                       noise_mask=None):
    """Matching-pursuit minor cycle with diffusion atoms (plan-diffusion-local-proposal.md 5).

    Each step: positive residual peak p inside the mask; W x W window of the residual
    (W = a.shape_size) centred on p; --diff_n_draws posterior draws of variant P
    (Diffusion.sample_shape_posterior) plus the delta at p; per candidate g (masked unless
    --diff_unmask_output) a* = <g, r> / <g, B g> and J = <g, r>^2 / (2 <g, B g>), B g with
    the 2W PSF crop (exact on the window); drop a* <= 0, take argmax J; commit
    --loop_gain x a* g and subtract its full-field PSF response. Stops as
    diffusion_minor_cycle. Same returns, plus per-step records (winner, J ratios).
    diffA (variant LD): --diff_ld_draws more candidates per step from A, conditioned on the
    a.ctx px residual and PSF centred on p, scaled by the context's max |r| as in training."""
    from mdn_asp.minor_cycle import PSFConvolver
    from mdn_asp.data.gpu_cutout_generator import fft_convolve_same
    from mdn_asp.models.diffusion import (
        beam_area_px,
        effective_beam_dofs,
        estimate_noise_cov_2w,
        window_noise_var,
    )
    if a.variant != "P":
        raise ValueError("--diff_mp needs a variant P checkpoint")
    Wn, h = a.shape_size, a.shape_size // 2
    H, W = residual.shape
    res = residual.astype(np.float32).copy()
    update = np.zeros((H, W), np.float32)
    std_img = np.zeros((H, W), np.float32) if args.diff_n_draws > 1 else None
    convolve = PSFConvolver(psf, (H, W), device)
    psf2 = torch.from_numpy(crop_psf_centred(psf, 2 * Wn)).to(device)   # peak at (Wn, Wn)
    mpad = np.zeros((H + Wn, W + Wn), bool)
    mpad[h:h + H, h:h + W] = mask
    delta = torch.zeros(1, Wn, Wn, device=device)
    delta[0, h, h] = 1.0
    hw = torch.hann_window(Wn, periodic=False, device=device)
    hann = hw[:, None] * hw[None, :]

    q_space = bool(getattr(args, "diff_q_space", False))
    dof_pen = float(getattr(args, "diff_dof_pen", 0.0) or 0.0)
    use_briggs = bool(getattr(args, "diff_briggs_psd", False))
    ld_cg_iters = int(getattr(args, "diff_ld_cg_iters", 0) or 0)
    null_lam = float(getattr(args, "refit_nullspace_lam", 0.0) or 0.0)
    scales_raw = str(getattr(args, "diff_scales", "") or "")
    sub_scales = sorted({int(s) for s in scales_raw.split(",") if s.strip() and 2 <= int(s) < Wn})
    sub_tapers = []
    for sw in sub_scales:
        sh = sw // 2
        tw = torch.hann_window(sw, periodic=False, device=device)
        emb = torch.zeros((Wn, Wn), dtype=torch.float32, device=device)
        emb[h - sh:h - sh + sw, h - sh:h - sh + sw] = tw[:, None] * tw[None, :]
        sub_tapers.append(emb)

    beam_area = beam_area_px(psf)
    mu_null = max(null_lam, 0.0) * beam_area
    cov_2w = None
    if use_briggs:
        nmask = noise_mask if noise_mask is not None else np.ones((H, W), dtype=bool)
        cov_2w = estimate_noise_cov_2w(res, nmask, Wn, device)

    tiles, recs = [], []
    if diffA is not None:
        if aA.variant != "LD" or aA.shape_size != Wn:
            raise ValueError("--diff_ld_ckpt needs a variant LD checkpoint with the same shape_size")
        Cn, hc = aA.ctx, aA.ctx // 2
        psf_ctx = torch.from_numpy(crop_psf_centred(psf, Cn)).to(device)
        cpad = np.zeros((H + Cn, W + Cn), np.float32)

    def peak_of(r):
        s = np.where(mask & (r > 0), r, 0.0)
        i = int(np.argmax(s))
        return divmod(i, W), float(s.flat[i])

    (_, _), start_peak = peak_of(res)
    floor = max(threshold, sidelobe_level * start_peak)
    prev, steps, stop = start_peak, 0, "max_components"
    while steps < args.max_components:
        (r, c), pk = peak_of(res)
        if pk <= 0.0 or pk < floor:
            stop = "noise_floor" if pk <= 0.0 or pk < threshold else "sidelobe_floor"
            break
        if pk > prev * (1.0 + args.divergence_tol):
            stop = "divergence"
            break
        prev = pk
        pad = np.zeros((H + Wn, W + Wn), np.float32)
        pad[h:h + H, h:h + W] = res
        rw = torch.from_numpy(pad[r:r + Wn, c:c + Wn].copy()).to(device)   # peak at (h, h)
        if args.diff_n_draws > 0:
            dB = diff.sample_shape_posterior(
                rw / pk, psf2, sigma / pk, args.diff_n_draws, gen=gen,
                q_space=q_space, taper=(args.diff_taper and q_space),
            ).float()
            if args.diff_taper and not q_space:
                dB = dB * hann
        else:   # A alone
            dB = torch.zeros(0, Wn, Wn, device=device)
        nA = 0
        dA = torch.zeros(0, Wn, Wn, device=device)
        if diffA is not None:
            cpad[hc:hc + H, hc:hc + W] = res
            ctx = torch.from_numpy(cpad[r:r + Cn, c:c + Cn].copy()).to(device)   # peak at (hc, hc)
            u = ctx.abs().max().clamp_min(1e-30)
            nA = args.diff_ld_draws
            ld_batch = {
                "img": (ctx / u).expand(nA, -1, -1),
                "sigma": (torch.tensor(float(sigma), device=device) / u).repeat(nA),
                "psf_cut": psf_ctx.expand(nA, -1, -1),
            }
            if ld_cg_iters > 0:
                dA = diffA.sample_ld_posterior(
                    ld_batch, psf2, rw / u, float(sigma) / float(u),
                    gen=gen, cg_iters=ld_cg_iters, taper=False,
                ).float()
            else:
                dA = diffA.sample(ld_batch, gen=gen).float()
            if args.diff_taper:
                dA = dA * hann

        if sub_tapers:
            if len(dB) > 0:
                dB = torch.cat([dB] + [dB * st for st in sub_tapers], dim=0)
            if len(dA) > 0:
                dA = torch.cat([dA] + [dA * st for st in sub_tapers], dim=0)
                nA = len(dA)

        d = torch.cat([dB, dA], dim=0)
        delta_cand = delta * hann if (args.diff_taper and not q_space) else delta
        cand = torch.cat([d, delta_cand], dim=0)
        if not args.diff_unmask_output:
            cand = cand * torch.from_numpy(mpad[r:r + Wn, c:c + Wn]).to(device)
        num = (cand * rw).sum((1, 2))
        raw_den = (cand * fft_convolve_same(cand, psf2.expand(len(cand), -1, -1))).sum((1, 2))
        den = raw_den + (mu_null * (cand * cand).sum((1, 2)) if mu_null > 0.0 else 0.0)
        ok = (num > 0) & (den > 0)

        if (use_briggs and cov_2w is not None) or dof_pen > 0.0:
            sig2 = max(float(sigma) ** 2, 1e-30)
            if use_briggs and cov_2w is not None:
                n_var = window_noise_var(cand, cov_2w).clamp_min(1e-30)
            else:
                n_var = (sig2 * den).clamp_min(1e-30)
            j_nats = 0.5 * (num ** 2) / n_var
            if dof_pen > 0.0:
                k_eff = torch.ones_like(j_nats)
                if len(cand) > 1:
                    k_eff[:-1] = effective_beam_dofs(cand[:-1], beam_area)
                j_score = j_nats - 0.5 * dof_pen * k_eff
            else:
                j_score = j_nats
            J = torch.where(ok, j_score * sig2, torch.full_like(num, -1e30))
        else:
            J = torch.where(ok, num ** 2 / (2 * den.clamp_min(1e-30)), torch.full_like(num, -1.0))

        k = int(J.argmax())
        if not bool(ok[k]):
            stop = "noise_floor"
            break
        g = cand[k].cpu().numpy()
        ast = float(num[k] / den[k].clamp_min(1e-30))
        if draws_out is not None and steps < args.diff_save_draws:
            draws_out.append({"rw": rw.cpu().numpy(), "cand": cand.cpu().numpy(), "J": J.cpu().numpy(),
                              "k": k, "a": ast, "y": r, "x": c, "nA": nA})
        r0, c0_ = r - h, c - h
        y0, x0 = max(r0, 0), max(c0_, 0)
        y1, x1 = min(r0 + Wn, H), min(c0_ + Wn, W)
        shape = g[y0 - r0:y1 - r0, x0 - c0_:x1 - c0_]
        step = np.zeros((H, W), np.float32)
        step[y0:y1, x0:x1] = args.loop_gain * ast * shape
        tiles.append({"y0": y0, "y1": y1, "x0": x0, "x1": x1,
                      "u": shape.astype(np.float32), "a": float(args.loop_gain * ast)})
        if getattr(args, "diff_error", False):
            # Shape posterior of this step over all candidates: weights exp(J / sigma^2) on the usable ones,
            # committed term gamma a*_i g_i; per-pixel variance, stored relative to the committed amplitude^2
            # so a later amplitude refit rescales it.
            a_all = num / den.clamp_min(1e-30)
            use = ok & (a_all > 0)
            w = torch.softmax(torch.where(use, J / max(float(sigma) ** 2, 1e-30),
                                          torch.full_like(J, -float("inf"))), 0)
            c_all = args.loop_gain * a_all[:, None, None] * cand
            m1 = (w[:, None, None] * c_all).sum(0)
            sv = ((w[:, None, None] * c_all * c_all).sum(0) - m1 * m1).clamp_min(0.0)
            a_c = args.loop_gain * ast
            tiles[-1]["sv"] = (sv[y0 - r0:y1 - r0, x0 - c0_:x1 - c0_].cpu().numpy() / max(a_c, 1e-30) ** 2
                               ).astype(np.float32)
        update += step
        res -= convolve(step)
        if std_img is not None and len(dB) >= 2:
            sd = dB[:args.diff_n_draws].std(0).cpu().numpy()
            np.maximum(std_img[y0:y1, x0:x1], sd[y0 - r0:y1 - r0, x0 - c0_:x1 - c0_],
                       out=std_img[y0:y1, x0:x1])
        jd = float(J[-1])
        denom_jd = jd if abs(jd) > 1e-30 else 1e-30
        nB = len(cand) - 1 - nA
        winner = "delta" if k == len(cand) - 1 else ("draw" if k < nB else "A")
        recs.append({
            "y": r, "x": c, "peak": pk, "winner": winner,
            "J_win_over_delta": float(J[k]) / denom_jd if jd > 0 else float(J[k] - jd),
            "J_best_draw_over_delta": (
                float(J[:nB].max()) / denom_jd if (nB and jd > 0)
                else (float(J[:nB].max() - jd) if nB else None)
            ),
        })
        if nA:
            recs[-1]["J_best_A_over_delta"] = (
                float(J[nB:nB + nA].max()) / denom_jd if jd > 0
                else float(J[nB:nB + nA].max() - jd)
            )
        steps += 1
        jb = recs[-1]["J_best_draw_over_delta"] if nB else recs[-1]["J_best_A_over_delta"]
        print(f"[img]     step {steps}  peak {pk:.4e} at ({r},{c})  {winner}  J/J_delta {recs[-1]['J_win_over_delta']:.2f}  "
              f"best {'draw' if nB else 'A'} J/J_delta {jb:.2f}  added {step.sum():.4f} Jy", flush=True)
    return update, std_img, steps, stop, tiles, recs


@torch.no_grad()
def diffusion_grid_draw(diff, a, image, psf, sigma, args, device, gen, mask):
    """Variant B with m = 0: one draw set of p(sky | dirty, PSF) over the field.

    Non-overlapping patch/2 px crops on a grid; each comes from the central
    crop of an a.patch px tile of `image` centred on it, scaled by the tile's
    own max|value|. Crops with no mask pixel are skipped. Each crop is
    committed at --diff_gain (masked unless --diff_unmask_output) and recorded
    as a tile for refit_tiles. Same returns as diffusion_minor_cycle."""
    T, half, crop = a.patch, a.patch // 2, a.patch // 4
    H, W = image.shape
    update = np.zeros((H, W), np.float32)
    std_img = np.zeros((H, W), np.float32) if args.diff_n_draws > 1 else None
    psf_big = torch.from_numpy(crop_psf_centred(psf, a.psf_size)).to(device)
    c0 = (a.psf_size - T) // 2
    psf_cut = psf_big[c0:c0 + T, c0:c0 + T]
    pad = np.zeros((H + T, W + T), np.float32)
    pad[half:half + H, half:half + W] = image
    n, tiles = args.diff_n_draws, []
    for y0 in range(0, H, 2 * crop):
        for x0 in range(0, W, 2 * crop):
            y1, x1 = min(y0 + 2 * crop, H), min(x0 + 2 * crop, W)
            if not mask[y0:y1, x0:x1].any():
                continue
            r, c = y0 + crop, x0 + crop   # tile centre
            tile = torch.from_numpy(pad[r:r + T, c:c + T].copy()).to(device)
            scale = tile.abs().max().clamp_min(1e-30)
            batch = {"img": (tile / scale)[None].expand(n, -1, -1),
                     "sigma": (torch.tensor(sigma, device=device) / scale).repeat(n),
                     "psf": psf_big.expand(n, -1, -1), "psf_cut": psf_cut.expand(n, -1, -1)}
            d = torch.cat([diff.sample({k: v[i:i + args.diff_batch] for k, v in batch.items()}, gen=gen)
                           for i in range(0, n, args.diff_batch)]) * scale
            u = d[0] if args.diff_update == "draw" else d.mean(0)
            if args.diff_mask_k is not None:
                u = u * (d.mean(0) > args.diff_mask_k * d.std(0))
            shape = u[half - crop:half - crop + (y1 - y0), half - crop:half - crop + (x1 - x0)].cpu().numpy()
            if not args.diff_unmask_output:
                shape = shape * mask[y0:y1, x0:x1]
            update[y0:y1, x0:x1] += args.diff_gain * shape
            tiles.append({"y0": y0, "y1": y1, "x0": x0, "x1": x1,
                          "u": shape.astype(np.float32), "a": float(args.diff_gain)})
            if std_img is not None:
                std_img[y0:y1, x0:x1] = d.std(0)[half - crop:half - crop + (y1 - y0),
                                                 half - crop:half - crop + (x1 - x0)].cpu().numpy()
            print(f"[img]     tile ({y0},{x0})  added {args.diff_gain * shape.sum():.4f} Jy", flush=True)
    return update, std_img, len(tiles), "grid", tiles


def _noise_term(residual, model, psf, sigma, L, Wn, device):
    """R(w) for (n, Wn, Wn) residual windows: sum over lags |tau| <= L of
    (C_w(tau) / sigma^2 - C_noise(tau))^2 / Var_noise(tau), where C_w is the unbiased
    window autocorrelation normalised by exact pixel overlap count N_overlap(tau) and
    C_noise is the empirical empty-sky Briggs noise autocorrelation (falling back to PSF
    only if < 4 empty windows exist). None if no empty window exists."""
    from scipy.ndimage import distance_transform_edt
    H, W = residual.shape
    h = Wn // 2
    far = distance_transform_edt(model <= 0) > 2 * Wn
    cy, cx = np.nonzero(far[h:H - h, h:W - h])
    if len(cy) == 0:
        return None
    i = np.random.default_rng(0).choice(len(cy), min(64, len(cy)), replace=False)
    nw = torch.from_numpy(
        np.stack([residual[y:y + Wn, c:c + Wn] for y, c in zip(cy[i], cx[i])]).astype(np.float32)
    ).to(device)
    sig2 = max(float(sigma) ** 2, 1e-30)

    full_ones = torch.ones((1, Wn, Wn), dtype=torch.float32, device=device)
    F_ones = torch.fft.rfft2(full_ones, s=(2 * Wn, 2 * Wn))
    ov_full = torch.fft.fftshift(
        torch.fft.irfft2(F_ones.abs() ** 2, s=(2 * Wn, 2 * Wn)), dim=(-2, -1)
    )[:, Wn - L:Wn + L + 1, Wn - L:Wn + L + 1].clamp_min(1.0)

    def acorr(w, valid_hw=None):
        F = torch.fft.rfft2(w, s=(2 * Wn, 2 * Wn))
        raw = torch.fft.fftshift(
            torch.fft.irfft2(F.abs() ** 2, s=(2 * Wn, 2 * Wn)), dim=(-2, -1)
        )[:, Wn - L:Wn + L + 1, Wn - L:Wn + L + 1]
        if valid_hw is None or valid_hw == (Wn, Wn):
            return raw / ov_full
        vh, vw = valid_hw
        m = torch.zeros((1, Wn, Wn), dtype=torch.float32, device=device)
        m[:, :vh, :vw] = 1.0
        Fm = torch.fft.rfft2(m, s=(2 * Wn, 2 * Wn))
        ov = torch.fft.fftshift(
            torch.fft.irfft2(Fm.abs() ** 2, s=(2 * Wn, 2 * Wn)), dim=(-2, -1)
        )[:, Wn - L:Wn + L + 1, Wn - L:Wn + L + 1].clamp_min(1.0)
        return raw / ov

    nw_c = acorr(nw) / sig2
    if len(nw) >= 4:
        target = nw_c.mean(0)
        var_n = nw_c.var(0, unbiased=True).clamp_min(1e-12)
    else:
        target = torch.from_numpy(crop_psf_centred(psf, 2 * L + 1)).to(device)
        var_n = nw_c.var(0, unbiased=False).clamp_min(1e-12)
    return lambda w, valid_hw=None: (((acorr(w, valid_hw=valid_hw) / sig2 - target) ** 2) / var_n).sum((1, 2))


def report_noise_like(tag, res, mask, noise_mask, model, psf, Wn, lag, device):
    """Print, for one residual image: extremes in the mask and off it (Jy/beam and in sigma of the off-source
    rms), sum of squares, and R of _noise_term on non-overlapping Wn windows centred in the mask.
    R is a sum over (2 lag + 1)^2 lags, so it is about that number for a noise-like window."""
    sig = measure_noise_rms(res, noise_mask)
    ins, off = res[mask], res[noise_mask]
    print(f"[img]   noise check, {tag}: off-source rms {sig:.4e}", flush=True)
    print(f"[img]     in mask  ({ins.size} px): max {ins.max():.4e} ({ins.max() / sig:+.1f} sigma)  "
          f"min {ins.min():.4e} ({ins.min() / sig:+.1f} sigma)", flush=True)
    print(f"[img]     off mask ({off.size} px): max {off.max():.4e} ({off.max() / sig:+.1f} sigma)  "
          f"min {off.min():.4e} ({off.min() / sig:+.1f} sigma)", flush=True)
    print(f"[img]     sum of squares: full image {float((res.astype(np.float64) ** 2).sum()):.6e}  "
          f"in mask {float((ins.astype(np.float64) ** 2).sum()):.6e}", flush=True)
    Rf = _noise_term(res, model, psf, sig, lag, Wn, device)
    if Rf is None:
        print("[img]     autocorrelation: no empty windows far from the model", flush=True)
        return
    H, W = res.shape
    wins = [res[y:y + Wn, x:x + Wn] for y in range(0, H - Wn + 1, Wn) for x in range(0, W - Wn + 1, Wn)
            if mask[y + Wn // 2, x + Wn // 2]]
    if not wins:
        print("[img]     autocorrelation: no window centred in the mask", flush=True)
        return
    R = Rf(torch.from_numpy(np.stack(wins).astype(np.float32)).to(device)).cpu().numpy()
    print(f"[img]     autocorrelation R over {len(wins)} windows of {Wn} px, lags <= {lag} "
          f"(noise-like is about {(2 * lag + 1) ** 2}): median {np.median(R):.1f}  min {R.min():.1f}  max {R.max():.1f}",
          flush=True)


def refit_tiles(tiles, residual, psf, refit_gain, device, noise_lam=None, sigma=None, model=None, lag=16,
                method="cd", nullspace_lam=0.0, nnls_iters=60):
    """Amplitude refit of committed diffusion tiles.

    ``method='nnls'`` (when ``noise_lam is None``) solves the joint proximal Tikhonov-damped
    non-negative least squares problem over all tiles simultaneously via preconditioned FISTA
    (``mdn_asp.refit.solve_nnls_amplitudes``), preventing collinear atom shattering.
    ``method='cd'`` runs sequential coordinate descent in commit order.
    Updates ``tiles[i]['a']`` in place and returns ``(model change image, residual after, stats)``."""
    from mdn_asp.minor_cycle import PSFConvolver, _extract_psf_main_lobe
    from mdn_asp.refit import solve_nnls_amplitudes

    if method == "nnls" and noise_lam is None:
        atoms = [(t["y0"], t["y1"], t["x0"], t["x1"], t["u"]) for t in tiles]
        a0 = [float(t["a"]) for t in tiles]
        a_new, delta, r_after, st = solve_nnls_amplitudes(
            atoms, a0, residual, psf,
            refit_gain=refit_gain,
            nullspace_lam=nullspace_lam,
            n_iter=nnls_iters,
            device=str(device),
        )
        for t, an in zip(tiles, a_new):
            t["a"] = float(an)
        stats = {
            "n_tiles": int(st["n_atoms"]),
            "n_down": int(st["n_down"]),
            "n_zeroed": int(st["n_zeroed"]),
            "amp_ratio_med": st["amp_ratio_med"],
            "chi2_before": st["chi2_before"],
            "chi2_after": st["chi2_after"],
            "model_change": st["model_change"],
            "method": "nnls",
            "fista_iters": int(st["fista_iters"]),
        }
        return delta, r_after, stats

    H, W = residual.shape
    convolve = PSFConvolver(psf, (H, W), device)
    omega_beam = _extract_psf_main_lobe(psf)[2] if nullspace_lam > 0.0 else 1.0
    mu_null = max(float(nullspace_lam), 0.0) * float(omega_beam)
    Rf = None
    if noise_lam is not None:
        Wn = max(max(t["y1"] - t["y0"], t["x1"] - t["x0"]) for t in tiles) if tiles else 64
        Rf = _noise_term(residual, model, psf, sigma, lag, Wn, device) if noise_lam > 0 else None
        if noise_lam > 0 and Rf is None:
            print("[img]   tile refit: no empty windows for the noise term; lam ignored", flush=True)
    r = residual.astype(np.float32).copy()
    delta = np.zeros((H, W), np.float32)
    chi2_before = float((r.astype(np.float64) ** 2).sum())
    n_down = n_zero = 0
    ratios = []
    for t in tiles:
        img = np.zeros((H, W), np.float32)
        img[t["y0"]:t["y1"], t["x0"]:t["x1"]] = t["u"]
        Bu = convolve(img)
        den = float((img * Bu).sum()) + (mu_null * float((img * img).sum()) if mu_null > 0.0 else 0.0)
        if den <= 0 or t["a"] <= 0 and float((img * r).sum()) <= 0:
            continue
        rk_num = float((img * r).sum()) + t["a"] * float((img * Bu).sum())
        a_star = rk_num / den
        if noise_lam is None:
            a_new = max(0.0, t["a"] + refit_gain * (a_star - t["a"]))
        else:
            rk = r + t["a"] * Bu
            gu = float((img * rk).sum())
            ast = max(gu / den, 0.0)
            grid = np.array(sorted(set(np.linspace(0.0, 1.5 * ast, 16).tolist() + [t["a"]])))
            obj = -(2 * grid * gu - grid ** 2 * den) / (2 * max(float(sigma) ** 2, 1e-30))
            if Rf is not None:
                ys, xs = slice(t["y0"], t["y1"]), slice(t["x0"], t["x1"])
                rw = torch.from_numpy(rk[ys, xs].copy()).to(device)
                bw = torch.from_numpy(Bu[ys, xs].copy()).to(device)
                vh, vw = rw.shape
                gt = torch.from_numpy(grid.astype(np.float32)).to(device)
                stack = rw[None] - gt.view(-1, 1, 1) * bw[None]
                stack = torch.nn.functional.pad(stack, (0, Wn - vw, 0, Wn - vh))
                obj = obj + noise_lam * Rf(stack, valid_hw=(vh, vw)).cpu().numpy()
            a_grid = float(grid[int(np.argmin(obj))])
            a_new = max(0.0, t["a"] + refit_gain * (a_grid - t["a"]))
        da = a_new - t["a"]
        delta += da * img
        r -= da * Bu
        ratios.append(a_new / max(t["a"], 1e-30))
        n_down += a_new < t["a"]
        n_zero += a_new == 0.0 and t["a"] > 0.0
        t["a"] = a_new
    stats = {"n_tiles": len(tiles), "n_down": int(n_down), "n_zeroed": int(n_zero),
             "amp_ratio_med": float(np.median(ratios)) if ratios else None,
             "chi2_before": chi2_before, "chi2_after": float((r.astype(np.float64) ** 2).sum()),
             "model_change": float(delta.sum()), "method": "cd"}
    return delta, r, stats


def amp_error_image(tiles, psf, sigma, shape, device):
    """Per tile Var(a_k) = sigma^2 / <u_k, B u_k> (dirty-image noise covariance sigma^2 x PSF, shape fixed);
    returns sqrt(sum_k (u_k sd(a_k))^2), tiles treated as independent."""
    from mdn_asp.minor_cycle import PSFConvolver
    convolve = PSFConvolver(psf, shape, device)
    var = np.zeros(shape, np.float64)
    for t in tiles:
        if t["a"] <= 0:
            continue
        img = np.zeros(shape, np.float32)
        img[t["y0"]:t["y1"], t["x0"]:t["x1"]] = t["u"]
        den = float((img * convolve(img)).sum())
        if den > 0:
            var[t["y0"]:t["y1"], t["x0"]:t["x1"]] += (t["u"].astype(np.float64) ** 2) * sigma ** 2 / den
    return np.sqrt(var).astype(np.float32)


def save_cycle_png(path, residual, model, update, std):
    """Residual, model, this cycle's update and draw std, arcsinh stretch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("residual", residual, "RdBu_r", True), ("model", model, "inferno", False),
              ("this cycle's update", update, "inferno", False)]
    if std is not None:
        panels.append(("draw std (max over tiles)", std, "inferno", False))
    fig, axs = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    for ax, (t, im, cm, signed) in zip(axs, panels):
        s = 1e-2 * max(float(np.abs(im).max()), 1e-30)
        v = np.arcsinh(im / s)
        lim = float(np.abs(v).max()) or 1.0
        ax.imshow(v, origin="lower", cmap=cm, vmin=-lim if signed else min(0.0, float(v.min())), vmax=lim)
        ax.set_title(f"{t}  [{im.min():.3g}, {im.max():.3g}]", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    imgname = str(out_dir / args.imagename)
    S = args.imsize

    # --- load MDN ---
    diff = diff_a = None
    if args.diffusion_ckpt:
        diff, diff_a = load_diffusion(args.diffusion_ckpt, device)
        if diff_a.variant == "B" and args.resume:
            sys.exit("[img] variant B draws once from the dirty image (m = 0); --resume is not supported")
        diff_gen = torch.Generator(device=device).manual_seed(args.diff_seed)
        print(f"[img] Diffusion checkpoint: {args.diffusion_ckpt}  variant {diff_a.variant}  "
              f"patch {diff_a.patch}  sample steps {diff_a.sample_steps}  draws/tile {args.diff_n_draws}  "
              f"update {args.diff_update}  draw mask k {args.diff_mask_k}")
    diffA = diffA_a = None
    if args.diff_ld_ckpt:
        diffA, diffA_a = load_diffusion(args.diff_ld_ckpt, device)
        print(f"[img] Dictionary A checkpoint: {args.diff_ld_ckpt}  variant {diffA_a.variant}  "
              f"ctx {diffA_a.ctx}  draws/step {args.diff_ld_draws}")
    if not args.diffusion_ckpt:   # MDN path only; the else had slipped onto the --diff_ld_ckpt block
        mdn = load_mdn_checkpoint(args.ckpt, device)
        print(f"[img] Loaded checkpoint: {args.ckpt}")

        # Flux-head convention. Auto-detect, but never guess in silence: an
        # unmarked checkpoint predates the marker and is absolute-flux.
        scale_relative = args.scale_relative
        if scale_relative is None:
            detected = checkpoint_is_scale_relative(args.ckpt)
            if detected is None:
                scale_relative = False
                print("[img] WARNING: checkpoint records no flux convention "
                      "(pre-2026-08-10). Assuming ABSOLUTE flux. If this model was "
                      "trained with --scale_relative, pass it explicitly.")
            else:
                scale_relative = detected
        print(f"[img] Flux head: "
              f"{'scale-relative log(flux/scale)' if scale_relative else 'absolute'}")
        if args.sigma_min_px is None:
            args.sigma_min_px = checkpoint_sigma_min_px(args.ckpt)
        if args.sigma_max_px is None:
            args.sigma_max_px = checkpoint_train_arg(args.ckpt, "sigma_max_px")
        if args.render_nsigma is None:
            args.render_nsigma = checkpoint_train_arg(args.ckpt, "render_nsigma")
        print(f"[img] Cutout {mdn.cutout} px, width ceiling "
              f"{args.sigma_max_px if args.sigma_max_px is not None else 'SIGMA_MAX_PX'} px, "
              f"render truncation {args.render_nsigma} sigma")
        print(f"[img] Component width floor: "
              f"{args.sigma_min_px if args.sigma_min_px is not None else 'BEAM_SIGMA_PX (not recorded)'} px")

    if args.mask:
        casa_mask = args.mask
        mask = read_casa_image(args.mask).astype(bool)
    elif args.mask_radius_px > 0:
        # Restrict cleaning (and the noise/peak measurements) to the centre,
        # where the source actually is. Applies to the numpy-side loop only --
        # tclean itself still images the full field, so the sidelobes of what
        # we clean are still subtracted correctly.
        casa_mask = ""
        mask = make_mask(S, args.mask_radius_px)
    else:
        casa_mask = ""
        mask = np.ones((S, S), dtype=bool)
    print(f"[img] CASA mask: {casa_mask if casa_mask else '(none)'}")

    # Noise must be measured OFF source. Using the clean mask for both is
    # self-defeating: on G55 the sigma inside r=150 px is 1.15e-4 versus
    # 2.18e-5 over the full field, because the circle contains the SNR. A
    # "1 sigma" stop measured on-source would then be shallower than a 3 sigma
    # stop measured off-source. When the clean region is restricted, estimate
    # the noise on its complement.
    if mask.all():
        noise_mask = mask
        print("[img] Noise region: full image (clean region unrestricted)")
    else:
        noise_mask = ~mask
        print(f"[img] Noise region: OUTSIDE clean mask "
              f"({noise_mask.sum()} px, {100*noise_mask.mean():.1f}% of image)")

    # --- initial tclean: PSF + dirty residual ---
    prev_log = None
    if args.resume:
        with open(out_dir / "log.json") as f:
            prev_log = json.load(f)
        print(f"[img] Resuming after {len(prev_log['major_cycles'])} major cycles "
              f"(skipping initial tclean)")
    else:
      print("[img] Initial tclean (niter=0, calcpsf=True) ...")
      tclean(
          vis=args.vis, imagename=imgname,
          field=args.field, spw=args.spw,
          specmode=args.specmode, gridder=args.gridder,
          wprojplanes=args.wprojplanes,
          imsize=[S, S], cell=[args.cell, args.cell],
          stokes='I', weighting='briggs', robust=args.robust,
          niter=0, mask=casa_mask,
          calcpsf=True, calcres=True, pbcor=False, pblimit=args.pblimit,
      )

    psf      = read_casa_image(imgname + ".psf")
    residual = read_casa_image(imgname + ".residual")
    dirty_npy = out_dir / "dirty.npy"
    if args.resume:
        dirty = np.load(dirty_npy).astype(np.float32) if dirty_npy.exists() else None
    else:
        dirty = residual.copy()
        np.save(dirty_npy, dirty.astype(np.float32))
    psf_cut  = crop_psf_centred(psf, _PSF_CROP)

    sidelobe_level = (measure_psf_sidelobe_first(psf_cut) if args.sidelobe_abs
                      else measure_psf_sidelobe(psf_cut))
    if args.sidelobe_level is not None:
        sidelobe_level = args.sidelobe_level
    dirty_peak     = (float(prev_log["dirty_peak"]) if prev_log
                      else float(np.where(mask, residual, 0.0).max()))
    sigma_rms      = measure_noise_rms(residual, noise_mask)
    psf_floor      = dirty_peak * sidelobe_level
    noise_floor    = args.n_sigma * sigma_rms
    threshold      = noise_floor if args.no_psf_floor else max(psf_floor, noise_floor)

    print(f"[img] dirty_peak={dirty_peak:.4f} Jy/beam")
    print(f"[img] PSF sidelobe level={sidelobe_level:.3f}  PSF floor={psf_floor:.4f} Jy/beam")
    print(f"[img] Off-source RMS={sigma_rms:.4e} Jy/beam  "
          f"{args.n_sigma:g}-sigma={noise_floor:.4e} Jy/beam")
    if args.global_threshold is not None:
        noise_floor = args.global_threshold
        threshold   = args.global_threshold
        print(f"[img] Global stop threshold={threshold:.4e} Jy/beam  (user override)")
    else:
        print(f"[img] Global stop threshold={threshold:.4e} Jy/beam  "
              f"(limited by {'noise' if args.no_psf_floor or noise_floor >= psf_floor else 'PSF sidelobe'})")

    gpu_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    log = prev_log if prev_log else {
        "vis": args.vis, "imagename": args.imagename,
        "dirty_peak": dirty_peak, "sidelobe_level": sidelobe_level,
        "sigma_rms": sigma_rms, "psf_floor": psf_floor,
        "noise_floor": noise_floor, "threshold": threshold,
        "sigma_max_px": args.sigma_max_px,
        "fit_unmasked": bool(args.fit_unmasked),
        "flux_prior_weight": args.flux_prior_weight,
        "gpu_name": gpu_name,
        "args": vars(args), "ckpt_sha256": _sha256(args.diffusion_ckpt or args.ckpt), "git": _git_info(),
        "major_cycles": [],
    }
    if prev_log:
        run_info = {"args": vars(args), "gpu_name": gpu_name,
                    "ckpt_sha256": _sha256(args.diffusion_ckpt or args.ckpt),
                    "git": _git_info()}
        if prev_log.get("ckpt_sha256") not in (None, run_info["ckpt_sha256"]):
            print("[img] WARNING: checkpoint differs from the one this run started with")
        log.setdefault("resumes", []).append(run_info)

    # Every component so far, with its current parameters. Refit in full when
    # --refit_all; otherwise only used for the uncertainty image.
    all_commits = []
    all_tiles = []     # diffusion tiles, for the tile refit
    post_records = {}   # cid -> latest refit candidate posterior (for --error_map)
    if args.resume:
        if diff is not None:
            all_tiles = load_tiles(out_dir)
            if all_tiles:
                print(f"[img] Restored {len(all_tiles)} diffusion tiles from {out_dir / 'tiles_state.npz'}")
        else:
            all_commits = load_components(out_dir / "components", len(log["major_cycles"]))
            if all_commits:
                print(f"[img] Restored {len(all_commits)} components from "
                      f"{out_dir / 'components'}")
                check_model_consistency(all_commits, read_casa_image(imgname + ".model"),
                                        nsigma=args.render_nsigma)
            elif args.refit_all:
                sys.exit("[img] --refit_all --resume needs components/majorNN_state.json "
                         "from the earlier run; none found.")
            else:
                print("[img] WARNING: no component state files; the uncertainty image "
                      "will cover only the resumed cycles.")

    # --- major cycle loop ---
    start = len(log["major_cycles"])
    if diff is not None and start > 0:
        diff_gen = torch.Generator(device=device).manual_seed(args.diff_seed + start * 1009)

    def _rebuild_score(res_img: np.ndarray, mod_img: np.ndarray) -> float:
        if args.rebuild_metric == "vis_chi2" and dirty is not None:
            return -0.5 * float(
                np.sum(mod_img.astype(np.float64) * (dirty.astype(np.float64) + res_img.astype(np.float64)))
            )
        return float((res_img.astype(np.float64) ** 2).sum())
    def run_major_cycle():
        tclean(
            vis=args.vis, imagename=imgname,
            field=args.field, spw=args.spw,
            specmode=args.specmode, gridder=args.gridder,
        wprojplanes=args.wprojplanes,
            imsize=[S, S], cell=[args.cell, args.cell],
            stokes='I', weighting='briggs', robust=args.robust,
            niter=0, mask=casa_mask,
            calcpsf=False, calcres=True, restart=True, pbcor=False, pblimit=args.pblimit,
        )

    sigma_ref = None        # --diff_rebuild_drop: off-source sigma at the last rebuild (cycle 0 first)
    final_rebuild = False   # set when the stopping threshold is first reached: one last rebuild
    final_state = None      # model before that rebuild and its residual sum of squares; kept if the rebuild ends worse
    rebuild_state = None    # same, for the last rebuild of any kind (drop or final); restored if the run ends worse
    ens_models, ens_seeds, member, major_next, main_end = [], [args.diff_seed], 0, start, None   # --diff_ensemble
    while True:   # --diff_ensemble: one pass per member
        for major in range(major_next, major_next + args.max_major):
            residual = read_casa_image(imgname + ".residual")
            # The noise floor is re-estimated per cycle so it tracks the true
            # noise level as cleaning progresses. The PSF floor is GLOBAL and
            # fixed: dirty_peak * sidelobe_level is the hard stop below which the
            # residual is indistinguishable from the cleaned source's sidelobes.
            # It must NOT be re-anchored to the decaying residual peak -- since
            # sidelobe_level < 1, res_peak*sidelobe_level < res_peak always, so a
            # chased PSF floor can never fire the stop and cleaning goes deeper
            # than the global criterion. A user-supplied --global_threshold still
            # overrides and stays fixed, by design.
            sigma_rms = measure_noise_rms(residual, noise_mask)
            if args.global_threshold is None:
                noise_floor = args.n_sigma * sigma_rms
            res_peak = float(np.where(mask, residual, 0.0).max())
            res_min = float(np.where(mask, residual, 0.0).min())
            threshold = noise_floor if args.no_psf_floor else max(psf_floor, noise_floor)
            print(f"\n[img] Major cycle {major}  residual_peak={res_peak:.4e}  "
                  f"residual_min={res_min:.4e} ({res_min / sigma_rms:.1f} sigma)  "
                  f"sigma_rms={sigma_rms:.4e}  threshold={threshold:.4e}  "
                  f"(psf_floor={psf_floor:.4e}  noise_floor={noise_floor:.4e})")

            if res_peak < threshold and not final_rebuild:
                if (diff is not None and args.diff_mp and args.diff_rebuild_drop is not None and major > start
                        and final_state is None):
                    final_rebuild = True
                    print("[img] Below threshold -- one last rebuild at this sigma, then normal cycles to the threshold.")
                else:
                    print("[img] Below threshold -- stopping.")
                    break

            if diff is not None:
                model_img = read_casa_image(imgname + ".model")
                # Network input as the MDN path: residual x mask unless --fit_unmasked.
                fit_residual = (residual.astype(np.float32) if args.fit_unmasked
                                else np.where(mask, residual, 0.0).astype(np.float32))
                mp_recs = None
                rebuild = False
                if args.diff_mp and args.diff_rebuild_drop is not None:
                    if sigma_ref is None:
                        sigma_ref = sigma_rms
                    rebuild = final_rebuild or (final_state is None and major > start
                                                and sigma_rms <= (1.0 - args.diff_rebuild_drop) * sigma_ref)
                if rebuild:
                    rebuild_state = {"model": model_img.copy(), "chi2": _rebuild_score(residual, model_img),
                                     "major": major, "tiles": [dict(t) for t in all_tiles]}
                if final_rebuild:
                    final_state = rebuild_state
                if rebuild:
                    # Pull back to exact dirty image when available (avoiding mosaic image-plane PSF mismatch);
                    # fall back to image-plane PSF pullback only when dirty was not cached.
                    if dirty is not None:
                        fit_residual = (dirty.astype(np.float32) if args.fit_unmasked
                                        else np.where(mask, dirty, 0.0).astype(np.float32))
                    else:
                        from mdn_asp.minor_cycle import PSFConvolver
                        fit_residual = (fit_residual + PSFConvolver(psf, fit_residual.shape, device)(model_img)).astype(np.float32)
                    print(f"[img]   REBUILD: sigma {sigma_rms:.4e} vs {sigma_ref:.4e} at the last rebuild"
                          f"{' (final, at the stopping threshold)' if final_rebuild else ''}; model {model_img.sum():.4f} Jy pulled back, "
                          f"r0 peak {fit_residual.max():.4e}", flush=True)
                    sigma_ref = sigma_rms
                    model_img = np.zeros_like(model_img)
                    all_tiles = []
                if args.diff_save_atoms:   # what this cycle starts from, for step-by-step replays
                    np.save(out_dir / f"major{major:02d}_start_residual.npy", fit_residual.astype(np.float32))
                    np.save(out_dir / f"major{major:02d}_start_model.npy", model_img.astype(np.float32))
                if args.diff_mp:
                    draws_out = [] if args.diff_save_draws > 0 else None
                    upd, std, n_steps, stop_reason, tiles, mp_recs = diffusion_mp_cycle(
                        diff, diff_a, fit_residual, psf, sigma_rms, args, device, diff_gen, mask,
                        threshold, 0.0 if rebuild else sidelobe_level, diffA, diffA_a, draws_out,
                        noise_mask=noise_mask)
                    if draws_out:
                        np.savez(out_dir / f"draws_major{major:02d}.npz",
                                 **{k: np.stack([q[k] for q in draws_out]) for k in ("rw", "cand", "J")},
                                 **{k: np.array([q[k] for q in draws_out]) for k in ("k", "a", "y", "x", "nA")})
                    nd = sum(q["winner"] == "draw" for q in mp_recs)
                    nA = sum(q["winner"] == "A" for q in mp_recs)
                    print(f"[img]   matching pursuit: {nd} of {n_steps} steps won by a B draw, {nA} by an A draw, "
                          f"median J winner / J delta {np.median([q['J_win_over_delta'] for q in mp_recs]) if mp_recs else float('nan'):.3f}")
                    with open(out_dir / f"major{major:02d}_mp_steps.json", "w") as f:
                        json.dump(mp_recs, f)
                    if args.diff_save_atoms and tiles:   # committed atoms of this cycle, before the refit
                        Wn = diff_a.shape_size
                        u = np.zeros((len(tiles), Wn, Wn), np.float32)
                        for i, t in enumerate(tiles):
                            u[i, :t["y1"] - t["y0"], :t["x1"] - t["x0"]] = t["u"]
                        np.savez(out_dir / f"atoms_major{major:02d}.npz", u=u, a=np.array([t["a"] for t in tiles]),
                                 **{k: np.array([t[k] for t in tiles]) for k in ("y0", "y1", "x0", "x1")})
                elif diff_a.variant == "B":   # m = 0: draw once from the dirty image, then refit only
                    upd, std, n_steps, stop_reason, tiles = (diffusion_grid_draw(
                        diff, diff_a, fit_residual, psf, sigma_rms, args, device, diff_gen, mask)
                        if major == 0 else (np.zeros_like(model_img), None, 0, "refit_only", []))
                else:
                    upd, std, n_steps, stop_reason, tiles = diffusion_minor_cycle(
                        diff, diff_a, fit_residual, psf, sigma_rms, args, device, diff_gen, mask,
                        threshold, sidelobe_level)
                if n_steps == 0 and not (diff_a.variant == "B" and args.refit_exact and all_tiles):
                    print(f"[img] No tiles placed (minor_cycle_stop={stop_reason}) -- stopping.")
                    break
                new_model = (model_img + upd).astype(np.float32)
                print(f"[img]   diffusion {diff_a.variant}: {n_steps} tiles  stop={stop_reason}  "
                      f"update {upd.sum():.4f} Jy  model {model_img.sum():.4f} -> {new_model.sum():.4f} Jy")
                refit_stats = None
                if args.refit_exact:
                    all_tiles.extend(tiles)
                    if args.major_before_refit:
                        write_casa_image(imgname + ".model", new_model)
                        run_major_cycle()
                        work_res = read_casa_image(imgname + ".residual").astype(np.float32)
                        write_casa_image(imgname + ".model", model_img)
                        if args.cycle_report:
                            report_noise_like("after minor cycle, before refit", work_res, mask, noise_mask,
                                              new_model, psf,
                                              max(max(t["y1"] - t["y0"], t["x1"] - t["x0"]) for t in all_tiles),
                                              args.refit_noise_lag, device)
                    else:
                        from mdn_asp.minor_cycle import PSFConvolver
                        work_res = fit_residual - PSFConvolver(psf, fit_residual.shape, device)(upd)
                    if not args.fit_unmasked:
                        work_res = np.where(mask, work_res, 0.0).astype(np.float32)
                    if args.diff_save_atoms:   # before the refit: residual and model, amplitudes of every atom so far
                        np.save(out_dir / f"major{major:02d}_before_refit_residual.npy", work_res.astype(np.float32))
                        np.save(out_dir / f"major{major:02d}_before_refit_model.npy", new_model.astype(np.float32))
                        a_before = np.array([t["a"] for t in all_tiles])
                    delta, _, refit_stats = refit_tiles(
                        all_tiles, work_res, psf, args.refit_gain, device,
                        noise_lam=args.refit_noise_lam, sigma=sigma_rms,
                        model=new_model, lag=args.refit_noise_lag,
                        method=args.refit_method,
                        nullspace_lam=args.refit_nullspace_lam,
                        nnls_iters=args.refit_nnls_iters,
                    )
                    new_model = (new_model + delta).astype(np.float32)
                    if args.diff_save_atoms:   # amplitudes of every atom so far, before and after (shapes: atoms_major*.npz)
                        np.savez(out_dir / f"atoms_refit_major{major:02d}.npz", a_before=a_before,
                                 a_after=np.array([t["a"] for t in all_tiles]),
                                 **{k: np.array([t[k] for t in all_tiles]) for k in ("y0", "y1", "x0", "x1")})
                    print(f"[img]   tile refit: {refit_stats}  model -> {new_model.sum():.4f} Jy")
                else:
                    all_tiles.extend(tiles)
                write_casa_image(imgname + ".model", new_model)
                run_major_cycle()
                new_residual = read_casa_image(imgname + ".residual")
                new_peak = float(np.where(mask, new_residual, 0.0).max())
                new_min = float(np.where(mask, new_residual, 0.0).min())
                save_cycle_png(out_dir / f"major{major:02d}.png", new_residual, new_model, upd, std)
                if args.diff_save_atoms:
                    np.save(out_dir / f"major{major:02d}_residual.npy", new_residual.astype(np.float32))
                    np.save(out_dir / f"major{major:02d}_model.npy", new_model.astype(np.float32))
                log["major_cycles"].append({
                    "major": major, "residual_peak_before": res_peak, "residual_min_before": res_min,
                    "threshold": threshold, "sigma_rms": sigma_rms,
                    "n_tiles": n_steps, "minor_stop": stop_reason, "update_sum": float(upd.sum()),
                    "model_sum": float(new_model.sum()), "diff_gain": args.diff_gain,
                    "refit": refit_stats, "rebuild": bool(args.diff_mp and rebuild),
                    "residual_peak_after": new_peak, "residual_min_after": new_min, "ensemble_member": member,
                })
                with open(out_dir / "log.json", "w") as f:
                    json.dump(log, f, indent=2)
                dump_tiles(out_dir, all_tiles)
                print(f"[img]   residual_peak after major cycle: {new_peak:.4e}  residual_min: {new_min:.4e}")
                if args.cycle_report and args.refit_exact:
                    report_noise_like("after refit", new_residual, mask, noise_mask, new_model, psf,
                                      max(max(t["y1"] - t["y0"], t["x1"] - t["x0"]) for t in all_tiles),
                                      args.refit_noise_lag, device)
                if final_rebuild:
                    final_rebuild = False
                    print("[img] Final rebuild done -- normal cycles (no more rebuilds) until below threshold.")
                    continue
                if new_peak > res_peak * (1.0 + args.divergence_tol) and not (args.diff_mp and rebuild):
                    print(f"[img] Residual peak rose by >{args.divergence_tol*100:.0f}% "
                          f"({res_peak:.4e} -> {new_peak:.4e}) -- stopping.")
                    break
                continue

            # Two jobs, kept separate. `image_mask` restricts WHERE components may
            # be placed (what CLEAN does). Zeroing the residual outside the mask is
            # a second, different thing: it falsifies the data the proposal and the
            # exact-likelihood fits see. <g_k, r> is already flux weighted by the
            # Aspen profile itself, so the binary mask is a redundant second weight
            # that also truncates g_k's own support.
            fit_residual = (residual.astype(np.float32) if args.fit_unmasked
                            else np.where(mask, residual, 0.0).astype(np.float32))
            model_update, commits, stop_reason = minor_cycle(
                residual=fit_residual, psf=psf,
                sigma=sigma_rms, config_idx=args.config_idx, model=mdn,
                loop_gain=args.loop_gain, sidelobe_level=sidelobe_level,
                abs_noise_floor=threshold,
                divergence_tol=args.divergence_tol, image_mask=mask,
                sigma_max_px=args.sigma_max_px,
                sigma_min_px=args.sigma_min_px,
                max_components=args.max_components,
                scale_relative=scale_relative,
                global_sigma=args.global_sigma,
                cutout=mdn.cutout,
                render_nsigma=args.render_nsigma, stop_below_noise=args.stop_below_noise,
                sidelobe_abs_peak=args.sidelobe_abs, cap_to_pixel=args.cap_to_pixel, flux_clip=args.flux_clip,
                density_mode=args.mdn_density_mode,
                profile_amp=args.mdn_profile_amp,
                delta_fallback=args.mdn_delta_fallback,
                uv_coupling_min=args.uv_coupling_min,
                nullspace_lam=args.refit_nullspace_lam,
                device=args.device,
            )
            if not commits:
                print(f"[img] No components accepted (minor_cycle_stop={stop_reason}) -- stopping.")
                break
            commits = [replace(c, born_in_major=major, cid=len(all_commits) + i)
                       for i, c in enumerate(commits)]
            commits_pre = list(commits)
            n_new = len(commits)

            refit_stats = None
            if args.refit_exact and args.major_before_refit:
                # Major cycle on model + this minor cycle's update, so the refit
                # sees the residual from the visibilities.
                model_before = read_casa_image(imgname + ".model")
                write_casa_image(imgname + ".model", model_before + model_update)
                run_major_cycle()
                true_res = read_casa_image(imgname + ".residual").astype(np.float32)
                work_res = (true_res if args.fit_unmasked
                            else np.where(mask, true_res, 0.0).astype(np.float32))
                write_casa_image(imgname + ".model", model_before)
                print(f"[img]   major cycle before refit: residual peak "
                      f"{float(np.where(mask, true_res, 0.0).max()):.4e}")
            elif args.refit_exact:
                # The minor cycle does not return its working residual, so rebuild
                # it with the same operator it used: r = masked_residual - B m.
                work_res = (fit_residual
                            - psf_convolve(model_update, psf)).astype(np.float32)

            model_full = None
            recs = []
            if args.refit_exact:
                from mdn_asp.refit import refit_commits
                repropose = (dict(model=mdn, config_idx=args.config_idx,
                                  scale_relative=scale_relative, global_sigma=args.global_sigma)
                             if args.refit_repropose else None)
                if args.refit_all:
                    # work_res is the residual of the whole model (old + this
                    # cycle's update), so the older components can be refit against
                    # it. The refit returns the whole model image.
                    model_full = read_casa_image(imgname + ".model") + model_update
                    work_res, model_full, refit_list, rstats = refit_commits(
                        work_res, model_full, psf, all_commits + commits,
                        sigma=sigma_rms, refit_gain=args.refit_gain,
                        sigma_min_px=args.sigma_min_px, sigma_max_px=args.sigma_max_px,
                        flux_prior_weight=args.flux_prior_weight,
                        size_bias=args.size_bias,
                        zero_rejected=args.refit_zero_rejected, cutout=mdn.cutout,
                        render_nsigma=args.render_nsigma, collect=recs, device=device,
                        repropose=repropose,
                        amp_method=args.refit_method,
                        nullspace_lam=args.refit_nullspace_lam,
                        nnls_iters=args.refit_nnls_iters,
                    )
                    if args.refit_gradient:
                        from mdn_asp.refit import refit_gradient
                        work_res, model_full, refit_list, gstats = refit_gradient(
                            work_res, model_full, psf, refit_list, sigma=sigma_rms,
                            n_iter=args.grad_iters, lr_pos=args.grad_lr_pos, lr_log=args.grad_lr_log,
                            sigma_min_px=args.sigma_min_px, sigma_max_px=args.sigma_max_px,
                            max_window=mdn.cutout, render_nsigma=args.render_nsigma, device=device,
                            objective=args.grad_objective,
                        )
                        rstats["gradient"] = gstats
                        print(f"[img]   gradient refit: {gstats}")
                    all_commits = refit_list
                    commits = refit_list[-n_new:]
                else:
                    work_res, model_update, commits, rstats = refit_commits(
                        work_res, model_update, psf, commits,
                        sigma=sigma_rms, refit_gain=args.refit_gain,
                        sigma_min_px=args.sigma_min_px, sigma_max_px=args.sigma_max_px,
                        flux_prior_weight=args.flux_prior_weight,
                        size_bias=args.size_bias,
                        zero_rejected=args.refit_zero_rejected, cutout=mdn.cutout,
                        render_nsigma=args.render_nsigma, collect=recs, device=device,
                        repropose=repropose,
                        amp_method=args.refit_method,
                        nullspace_lam=args.refit_nullspace_lam,
                        nnls_iters=args.refit_nnls_iters,
                    )
                print(f"[img]   refit: {rstats['n_refit']} Aspen  "
                      f"shape changed {rstats['n_shape_changed']}  "
                      f"rejected {rstats['n_rejected']} (zeroed {rstats['n_zeroed']})  "
                      f"size ratio med {rstats['size_ratio_med']:.2f}  "
                      f"beyond x1.5 {rstats['frac_size_beyond_1p5']:.2f}  "
                      f"flux ratio med {rstats['flux_ratio_med']:.2f}  "
                      f"chi2 {rstats['chi2_before']:.4e} -> {rstats['chi2_after']:.4e}")
                refit_stats = rstats
                post_records.update({r["cid"]: r for r in recs})
                # Every candidate the refit scored, so the shape choice can be checked later.
                (out_dir / "components").mkdir(parents=True, exist_ok=True)
                with open(out_dir / "components" / f"major{major:02d}_cands.json", "w") as f:
                    json.dump(recs, f)

            if model_full is None:
                all_commits.extend(commits)
            total_flux = sum(c.flux for c in commits)
            print(f"[img]   {len(commits)} components committed  "
                  f"total_flux_commit={total_flux:.4f} Jy  minor_cycle_stop={stop_reason}")

            model_img = read_casa_image(imgname + ".model")
            model_prev = model_img.copy()
            model_total_before = float(model_img.sum())
            if model_full is not None:
                model_img = model_full.astype(np.float32)
            else:
                model_img += model_update
            model_total_after = float(model_img.sum())
            print(f"[img]   model total flux: {model_total_before:.4f} -> {model_total_after:.4f} Jy")
            write_casa_image(imgname + ".model", model_img)

            _check = read_casa_image(imgname + ".model")
            print(f"[img]   model sum before tclean: {_check.sum():.6f} Jy")

            run_major_cycle()

            _check2 = read_casa_image(imgname + ".model")
            print(f"[img]   model sum after tclean:  {_check2.sum():.6f} Jy")

            new_residual = read_casa_image(imgname + ".residual")

            new_peak = float(np.where(mask, new_residual, 0.0).max())
            new_min = float(np.where(mask, new_residual, 0.0).min())
            # Per-cycle arrays and PNG, as the diffusion loop, for side-by-side comparison.
            np.save(out_dir / f"major{major:02d}_residual.npy", new_residual.astype(np.float32))
            np.save(out_dir / f"major{major:02d}_model.npy", model_img.astype(np.float32))
            save_cycle_png(out_dir / f"major{major:02d}.png", new_residual, model_img,
                           model_img - model_prev, None)

            # Per-commit diagnostics so the committed component distribution is
            # measurable, not just the one-line med/max summary (SURVEY.md 3.8).
            sig_majs = np.array([c.sig_maj for c in commits])
            sig_mins = np.array([c.sig_min for c in commits])
            pas      = np.array([c.pa      for c in commits])
            fluxes   = np.array([c.flux    for c in commits])
            log["major_cycles"].append({
                "major": major, "residual_peak_before": res_peak,
                "refit": refit_stats,
                "threshold": threshold, "n_components": len(commits),
                "n_components_total": len(all_commits),
                "residual_peak_after": new_peak,
                "residual_min_before": res_min, "residual_min_after": new_min,
                "sigma_rms": sigma_rms, "model_sum": model_total_after,
                "sigma_max_px": args.sigma_max_px,
                "sig_maj_med": float(np.median(sig_majs)) if commits else None,
                "sig_maj_p90": float(np.percentile(sig_majs, 90)) if commits else None,
                "sig_maj_max": float(sig_majs.max()) if commits else None,
                "sig_min_med": float(np.median(sig_mins)) if commits else None,
                "sig_min_max": float(sig_mins.max()) if commits else None,
                "n_at_ceiling": int((sig_majs >= (args.sigma_max_px or 11.2) - 1e-3).sum()) if commits else 0,
                "ellip_med": float(np.median(sig_majs / np.maximum(sig_mins, 1e-6))) if commits else None,
                "ellip_max": float((sig_majs / np.maximum(sig_mins, 1e-6)).max()) if commits else None,
                "sig_maj": sig_majs.tolist() if commits else [],
                "sig_min": sig_mins.tolist() if commits else [],
                "pa":      pas.tolist()      if commits else [],
                "flux":    fluxes.tolist()   if commits else [],
            })
            with open(out_dir / "log.json", "w") as f:
                json.dump(log, f, indent=2)
            dump_components(out_dir / "components", major, commits_pre,
                            commits if args.refit_exact else None)
            dump_state(out_dir / "components", major, all_commits)

            print(f"[img]   residual_peak after major cycle: {new_peak:.4e}  "
                  f"residual_min: {new_min:.4e}")

            if new_peak > res_peak * (1.0 + args.divergence_tol):
                print(f"[img] Residual peak rose by >{args.divergence_tol*100:.0f}% "
                      f"({res_peak:.4e} -> {new_peak:.4e}) -- stopping.")
                break

        if rebuild_state is not None:
            # Keep the last rebuild (drop or final) only if its rebuild metric is lower than before it.
            chi2_new = _rebuild_score(
                read_casa_image(imgname + ".residual"),
                read_casa_image(imgname + ".model"),
            )
            keep = chi2_new < rebuild_state["chi2"]
            print(f"[img] Final rebuild check: rebuild score ({args.rebuild_metric}) {chi2_new:.6e} vs {rebuild_state['chi2']:.6e} "
                  f"before the rebuild (cycle {rebuild_state['major']}) -- {'keep rebuilt model' if keep else 'restore old model'}")
            if not keep:
                write_casa_image(imgname + ".model", rebuild_state["model"])
                run_major_cycle()
                all_tiles = [dict(t) for t in rebuild_state["tiles"]]
                dump_tiles(out_dir, all_tiles)
            log.setdefault("final_rebuild_checks", []).append({"member": member, "seed": ens_seeds[member],
                                                               **{"chi2_rebuilt": chi2_new, "chi2_before": rebuild_state["chi2"],
                                          "major": rebuild_state["major"], "kept": bool(keep)}})

        if not args.diff_ensemble or final_state is None:
            if args.diff_ensemble:
                print("[img] --diff_ensemble: no final rebuild happened -- no ensemble.")
            break
        ens_models.append(read_casa_image(imgname + ".model").astype(np.float32))
        if member == 0:
            main_end = {"model": ens_models[0], "tiles": [dict(t) for t in all_tiles]}
        if member + 1 >= args.diff_ensemble:
            break
        member += 1
        ens_seeds.append(args.diff_seed + member)
        diff_gen = torch.Generator(device=device).manual_seed(ens_seeds[member])
        write_casa_image(imgname + ".model", final_state["model"])
        run_major_cycle()
        all_tiles = [dict(t) for t in final_state["tiles"]]
        final_rebuild, major_next = True, major + 1
        print(f"\n[img] === ensemble member {member} of {args.diff_ensemble - 1}, seed {ens_seeds[member]}: "
              f"from the model before the final rebuild (cycle {final_state['major']}) ===", flush=True)

    if len(ens_models) > 1:
        ens = np.stack(ens_models)
        (out_dir / "ensemble").mkdir(exist_ok=True)
        for k, m in enumerate(ens_models):
            np.save(out_dir / "ensemble" / f"model_member{k:02d}_seed{ens_seeds[k]}.npy", m)
        copy_and_write_casa_image(imgname + ".model", imgname + ".ens_mean", ens.mean(0).astype(np.float32))
        copy_and_write_casa_image(imgname + ".model", imgname + ".ens_std", ens.std(0, ddof=1).astype(np.float32))
        print(f"[img] Ensemble: {len(ens_models)} members, model flux {[round(float(m.sum()), 4) for m in ens_models]} Jy; "
              f"{imgname}.ens_mean, {imgname}.ens_std; .model back to member 0")
        log["ensemble"] = {"seeds": ens_seeds, "model_flux": [float(m.sum()) for m in ens_models]}
        write_casa_image(imgname + ".model", main_end["model"])
        run_major_cycle()
        all_tiles = main_end["tiles"]
        dump_tiles(out_dir, all_tiles)

    # --- final reconstruction score ---
    R = read_casa_image(imgname + ".residual")
    M = read_casa_image(imgname + ".model")
    sig_off = measure_noise_rms(R, noise_mask)
    final = {
        "model_total_flux": float(M.sum()),
        "sigma_off_source": sig_off,
        "residual_peak_in_mask": float(R[mask].max()),
        "residual_min_in_mask": float(R[mask].min()),
        "residual_rms_in_mask": float(np.sqrt(np.mean(R[mask].astype(np.float64) ** 2))),
        "residual_rms_outside_mask": (float(np.sqrt(np.mean(R[~mask].astype(np.float64) ** 2)))
                                      if (~mask).any() else None),
        # ~1 means the residual in the mask is noise-like, larger means structure is left
        "residual_power_ratio_in_mask": (float(np.mean(R[mask].astype(np.float64) ** 2) / sig_off ** 2)
                                         if sig_off > 0 else None),
    }
    if dirty is not None:
        # 0.5 * (chi2_vis(0) - chi2_vis(M)) = (M.D - 0.5 M.B M)/sigma^2 = 0.5 * M.(D + R)/sigma^2
        # for the linear operator R = D - B M. APPROXIMATE for a mosaic (PB weight normalisation).
        dot = float(np.sum(M.astype(np.float64) * (dirty.astype(np.float64) + R)))
        final["model_dot_dirty_plus_residual"] = dot
        final["delta_chi2_image_approx"] = 0.5 * dot / sig_off ** 2 if sig_off > 0 else None
    active_cids = {c.cid for c in all_commits if c.flux > 0.0}
    active_records = [r for cid, r in post_records.items() if cid in active_cids] if all_commits else list(post_records.values())
    if args.error_map and active_records:
        from mdn_asp.refit import error_map_from_records
        emap = error_map_from_records(active_records, (S, S),
                                      size_bias=args.size_bias, max_half=mdn.cutout // 2,
                                      nsigma=args.render_nsigma)
        emap_path = imgname + ".errormap"
        copy_and_write_casa_image(imgname + ".model", emap_path, emap)
        final["errormap_max"] = float(emap.max())
        final["errormap_median_in_mask"] = float(np.median(emap[mask]))
        final["errormap_n_aspen"] = len(active_records)
        print(f"[img] Error map: {emap_path}  (max={emap.max():.4e}  "
              f"median in mask={np.median(emap[mask]):.4e} Jy/pixel, "
              f"{len(active_records)} Aspen)")
    if args.error_map_joint and active_records:
        from mdn_asp.refit import error_map_from_records, joint_amplitude_error
        shape_sd = error_map_from_records(active_records, (S, S), size_bias=args.size_bias,
                                          max_half=mdn.cutout // 2, nsigma=args.render_nsigma,
                                          include_amp=False)
        jatoms, jamps = [], []
        for c in all_commits:
            if c.flux <= 0.0:
                continue
            rr = int(min(max(4 * c.sig_maj, 8), mdn.cutout // 2))
            r0, r1 = max(0, int(c.cy) - rr), min(S, int(c.cy) + rr + 1)
            c0, c1 = max(0, int(c.cx) - rr), min(S, int(c.cx) + rr + 1)
            g = render_aspen(c.cx - c0, c.cy - r0, 1.0, c.sig_maj, c.sig_min, c.pa,
                             (r1 - r0, c1 - c0), nsigma=args.render_nsigma)
            jatoms.append((r0, r1, c0, c1, g.astype(np.float32)))
            jamps.append(float(c.flux))
        amp_sd, jst = joint_amplitude_error(jatoms, jamps, (S, S), psf, sig_off, device=str(device))
        ejoint = np.sqrt(shape_sd.astype(np.float64) ** 2 + amp_sd.astype(np.float64) ** 2).astype(np.float32)
        copy_and_write_casa_image(imgname + ".model", imgname + ".errormap_joint", ejoint)
        final["errormap_joint_max"] = float(ejoint.max())
        final["errormap_joint_stats"] = jst
        print(f"[img] Joint error map: {imgname}.errormap_joint  (max {ejoint.max():.4e} Jy/pixel, "
              f"{jst['n_active']} Aspen, {jst['n_floored']} floored eigenvalues, cond {jst['cond']:.3e})")
    if getattr(args, "diff_error", False) and all_tiles:
        from mdn_asp.refit import joint_amplitude_error
        act_t = [t for t in all_tiles if t["a"] > 0.0]
        shape_var = np.zeros((S, S), np.float64)
        n_sv = 0
        for t in act_t:
            if "sv" in t:
                shape_var[t["y0"]:t["y1"], t["x0"]:t["x1"]] += t["sv"].astype(np.float64) * t["a"] ** 2
                n_sv += 1
        amp_sd, jst = joint_amplitude_error([(t["y0"], t["y1"], t["x0"], t["x1"], t["u"]) for t in act_t],
                                            [t["a"] for t in act_t], (S, S), psf, sig_off, device=str(device))
        derr = np.sqrt(shape_var + amp_sd.astype(np.float64) ** 2).astype(np.float32)
        copy_and_write_casa_image(imgname + ".model", imgname + ".diff_err", derr)
        final["diff_err_max"] = float(derr.max())
        final["diff_err_stats"] = dict(jst, n_tiles_with_shape_var=n_sv)
        print(f"[img] Diffusion error: {imgname}.diff_err  (max {derr.max():.4e} Jy/pixel, {len(act_t)} tiles, "
              f"{n_sv} with shape variance, {jst['n_floored']} floored eigenvalues, cond {jst['cond']:.3e})")
    if args.diff_amp_error and all_tiles:
        aerr = amp_error_image([t for t in all_tiles if t["a"] > 0.0], psf, sig_off, (S, S), device)
        copy_and_write_casa_image(imgname + ".model", imgname + ".amp_error", aerr)
        final["amp_error_max"] = float(aerr.max())
        final["amp_error_median_in_mask"] = float(np.median(aerr[mask]))
        print(f"[img] Amplitude error: {imgname}.amp_error  (max {aerr.max():.4e}, median in mask "
              f"{np.median(aerr[mask]):.4e} Jy/pixel, {len(all_tiles)} tiles, sigma {sig_off:.4e})")
    log["final"] = final
    with open(out_dir / "log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"[img] Final: residual rms in mask {final['residual_rms_in_mask']:.4e}  "
          f"power ratio {final['residual_power_ratio_in_mask']}  "
          f"model {final['model_total_flux']:.4f} Jy  "
          f"dchi2_approx {final.get('delta_chi2_image_approx')}")

    # --- posterior uncertainty image ---
    active_commits = [c for c in all_commits if c.flux > 0.0]
    if args.n_posterior_samples > 0 and active_commits:
        print(f"\n[img] Building uncertainty image from {args.n_posterior_samples} posterior samples "
              f"over {len(active_commits)} active components ...")
        uncertainty = build_uncertainty_image(
            active_commits, shape=(S, S),
            n_samples=args.n_posterior_samples,
            device=device,
            scale_relative=scale_relative,
            nsigma=args.render_nsigma,
        )
        unc_path = imgname + ".uncertainty"
        copy_and_write_casa_image(imgname + ".model", unc_path, uncertainty)
        print(f"[img] Uncertainty image: {unc_path}  "
              f"(peak={uncertainty.max():.4e}  median={np.median(uncertainty[mask]):.4e} Jy)")

    print(f"\n[img] Done. Log at {out_dir}/log.json")
    print(f"[img] Model image:    {imgname}.model")
    print(f"[img] Final residual: {imgname}.residual")


if __name__ == "__main__":
    main()
