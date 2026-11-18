"""Exact-likelihood refit of committed Aspen (Asp-SBI phase 2).

The minor cycle commits one Aspen per peak from the MDN mode, with a loop
gain, and never revisits it.  Asp itself refits the active set every iteration
so components can morph as more flux comes out.  This module does that refit.

Method, per Aspen k (coordinate descent over the active set):

1. Add its own PSF response back into the residual: ``r_k = r + B A_k``.  The
   likelihood in A_k alone is then exact -- nothing of A_k is left in ``r_k``.
2. The amplitude is linear given the shape.  For a unit-sum shape ``g``,

       a*(g)  = <g, r_k> / <g, B g>
       J(g)   = <g, r_k>^2 / (2 sigma^2 <g, B g>)

   ``J`` is the profile log-likelihood: section 2 of the plan, log p(d|s) =
   (1/sigma^2) (s^T d - 1/2 s^T B s), maximised over ``a``.  No gradient
   descent is needed -- only the shape is searched.
3. The shape candidates are the stored MDN mixture means (all K components,
   not just the mode) plus size rescalings of the current shape.  The proposal
   proposes; the exact likelihood picks.
4. Position stays fixed at the peak, as in Bhatnagar & Cornwell 2004 p.4.
5. ``<g, r_k> <= 0`` means the best amplitude is negative: the candidate is
   rejected, so positivity is kept by construction as in the greedy pass.

Approximation, deliberate: everything is computed on a cutout around the Aspen
with the PSF cropped to the same size, so sidelobes beyond the cutout are not
counted in ``<g, B g>``.  This is the same truncation the minor cycle already
uses to build its input, and it keeps the refit O(n_components) FFTs of the
cutout rather than of the field.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from mdn_asp.constants import BEAM_SIGMA_PX, SIGMA_MAX_PX
from mdn_asp.minor_cycle import (
    AspenCommit,
    PSFConvolver,
    _canonical_axes,
    _extract_psf_main_lobe,
    render_aspen,
)

# Size rescalings of the current shape, tried alongside the mixture means.
_SCALE_LADDER = (0.6, 0.8, 1.0, 1.25, 1.6)


def _crop_psf(psf: np.ndarray, size: int) -> np.ndarray:
    """Centred PSF crop that keeps the peak at (size//2, size//2)."""
    h, w = psf.shape
    cy, cx = h // 2, w // 2
    half = size // 2
    out = np.zeros((size, size), dtype=np.float32)
    r0, r1 = max(cy - half, 0), min(cy + half, h)
    c0, c1 = max(cx - half, 0), min(cx + half, w)
    out[half - (cy - r0): half + (r1 - cy),
        half - (cx - c0): half + (c1 - cx)] = psf[r0:r1, c0:c1]
    return out


def _mixture_shapes(commit: AspenCommit) -> list[tuple[float, float, float]]:
    """(sig_maj, sig_min, pa) of every mixture component of the proposal."""
    from mdn_asp.models.mdn_asp import MDNAsp
    mu = commit.params.mu                       # (1, K, 7)
    if mu is None or mu.ndim != 3:
        return []
    out = []
    for k in range(mu.shape[1]):
        v = MDNAsp._decode_7d_to_6d(mu[:, k, :])     # (1, 6)
        v = v.squeeze(0).detach().cpu().numpy()
        out.append((float(np.exp(v[3])), float(np.exp(v[4])), float(v[5])))
    return out


def _decode_shapes(mu) -> list[tuple[float, float, float]]:
    """(sig_maj, sig_min, pa) of every mixture component in ``mu`` (1, K, 7)."""
    from mdn_asp.models.mdn_asp import MDNAsp
    out = []
    for k in range(mu.shape[1]):
        v = MDNAsp._decode_7d_to_6d(mu[:, k, :]).squeeze(0).detach().cpu().numpy()
        out.append((float(np.exp(v[3])), float(np.exp(v[4])), float(v[5])))
    return out


def _repropose(model, res_cut: np.ndarray, psf_cut: np.ndarray,
               sigma: float, cfg: dict) -> list[tuple[float, float, float]]:
    """Fresh MDN mixture shapes from ``res_cut``, the (cutout, cutout) residual
    centred on the Aspen's pixel with its own response already added back.
    Conditioning as minor_cycle builds it."""
    import torch
    from mdn_asp.models.mdn_asp import make_cond
    sigma_local = (sigma if cfg.get("global_sigma", False)
                   else float(1.4826 * np.median(np.abs(res_cut))))
    if not np.isfinite(sigma_local) or sigma_local <= 0:
        sigma_local = sigma
    if cfg.get("scale_relative", False):
        cut_scale = float(np.abs(res_cut).max())
        if not np.isfinite(cut_scale) or cut_scale <= 0.0:
            cut_scale = 1.0
        res_cut = res_cut / np.float32(cut_scale)
        sigma_local = sigma_local / cut_scale
    dev = next(model.parameters()).device
    cond = make_cond(torch.tensor([sigma_local], dtype=torch.float32),
                     torch.tensor([cfg.get("config_idx", 3)], dtype=torch.long)).to(dev)
    img = torch.from_numpy(np.stack([res_cut, psf_cut], axis=0)[None].astype(np.float32)).to(dev)
    with torch.no_grad():
        params = model(img, cond)
    return _decode_shapes(params.mu)


def _candidates(commit: AspenCommit, sigma_min: float, sigma_max: float,
                extra: list | None = None) -> list[tuple[float, float, float]]:
    cands = [(commit.sig_maj, commit.sig_min, commit.pa)]
    cands += _mixture_shapes(commit)
    cands += extra or []
    for f in _SCALE_LADDER:
        if f != 1.0:
            cands.append((commit.sig_maj * f, commit.sig_min * f, commit.pa))
    out, seen = [], set()
    for smaj, smin, pa in cands:
        if not (np.isfinite(smaj) and np.isfinite(smin) and np.isfinite(pa)):
            continue
        smaj, smin, pa = _canonical_axes(float(smaj), float(smin), float(pa), sigma_min, sigma_max)
        key = (round(smaj, 4), round(smin, 4), round(pa, 4))
        if key not in seen:
            seen.add(key)
            out.append((smaj, smin, float(pa)))
    return out


def candidate_log_prior(commit: AspenCommit, cands: list, cur: tuple, sigma_min: float,
                        sigma_max: float) -> list[float]:
    """Log prior weight of each candidate shape, from the network's mixture weights.

    A candidate that is a mixture mean gets that component's log weight. One that is not (the size
    rescalings of the current shape, re-proposed shapes) gets the log weight of the mixture mean
    nearest the current shape ``cur`` in (log sig_maj, log sig_min). All zero when there is no mixture.
    """
    import torch
    logits = commit.params.logits
    shapes = _mixture_shapes(commit)
    if logits is None or not shapes:
        return [0.0] * len(cands)
    lw = torch.log_softmax(logits.reshape(-1).double(), 0).numpy()
    canon = [_canonical_axes(a, b, p, sigma_min, sigma_max) for a, b, p in shapes]
    mix: dict = {}
    for (a, b, p), l in zip(canon, lw):
        key = (round(a, 4), round(b, 4), round(p, 4))
        mix[key] = float(np.logaddexp(mix.get(key, -np.inf), l))
    d = [(math.log(a) - math.log(cur[0])) ** 2 + (math.log(b) - math.log(cur[1])) ** 2 for a, b, _ in canon]
    l_near = float(lw[int(np.argmin(d))])
    return [mix.get((round(a, 4), round(b, 4), round(p, 4)), l_near) for a, b, p in cands]


def flux_prior(commit: AspenCommit) -> tuple[float, float]:
    """(mean, std) in Jy of the proposal's flux posterior for one Aspen.

    The MDN emits dim 2 as a *standardised log* flux. Collapse the mixture to
    its first two moments in that space, then push through the same inverse the
    mode readout uses (``cut_scale`` in scale-relative mode, the fixed affine
    otherwise) and convert to Jy by the delta method.

    Returned mean is the UNDAMPED full flux -- the same quantity the greedy
    commit multiplied by ``loop_gain`` -- so it is a prior on the amplitude the
    exact likelihood is also estimating, not on the damped commit.
    """
    import torch
    from mdn_asp.constants import LOG_FLUX_SCALE, unstandardise_log_flux

    mu, log_std, logits = commit.params.mu, commit.params.log_std, commit.params.logits
    if mu is None or mu.ndim != 3:
        return float("nan"), float("nan")
    w = torch.softmax(logits, dim=-1)[0]            # (K,)
    m_k = mu[0, :, 2].double()                      # standardised log flux
    s_k = log_std[0, :, 2].double().exp()
    m = float((w.double() * m_k).sum())
    v = float((w.double() * (s_k ** 2 + m_k ** 2)).sum()) - m * m
    v = max(v, 0.0)

    if commit.cut_scale != 1.0:
        log_mean, log_var = m + math.log(commit.cut_scale), v
    else:
        log_mean = unstandardise_log_flux(m)
        log_var = v * LOG_FLUX_SCALE ** 2

    mean = math.exp(log_mean)
    # delta method: sd(exp(X)) ~= exp(E[X]) * sd(X)
    return mean, mean * math.sqrt(log_var)


def solve_nnls_amplitudes(
    atoms: list[tuple[int, int, int, int, np.ndarray]],
    a0: np.ndarray | list[float],
    residual: np.ndarray,
    psf: np.ndarray,
    *,
    refit_gain: float = 1.0,
    nullspace_lam: float = 0.0,
    n_iter: int = 60,
    prior_mu: np.ndarray | None = None,
    prior_ridge: np.ndarray | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Joint proximal Tikhonov-damped NNLS amplitude solver via preconditioned FISTA.

    Minimises over ``a >= 0``:
        0.5 <S(a) - S(a0), B_+ (S(a) - S(a0))> - <S(a) - S(a0), r0>
        + 0.5 * mu_null * ||S(a)||_2^2
        + 0.5 * tau * sum_i d_i (a_i - a0_i)^2
        + 0.5 * sum_i prior_ridge_i * (a_i - prior_mu_i)^2

    where ``tau = (1 - refit_gain) / max(refit_gain, 1e-6)``, ``d_i = <u_i, B_+ u_i>``,
    and ``mu_null = nullspace_lam * omega_beam``. Requires 1 full-field PSD FFT per FISTA
    iteration and reduces identically to ``max(0, a0 + refit_gain * (a* - a0))`` for M=1.
    """
    import torch
    from mdn_asp.data.gpu_cutout_generator import fft_convolve_same

    H, W = residual.shape
    a0_np = np.asarray(a0, dtype=np.float64).copy()
    M = len(atoms)
    chi2_before = float((residual.astype(np.float64) ** 2).sum())
    if M == 0:
        return (
            a0_np,
            np.zeros((H, W), dtype=np.float32),
            residual.astype(np.float32).copy(),
            {
                "n_atoms": 0, "n_down": 0, "n_zeroed": 0, "amp_ratio_med": None,
                "chi2_before": chi2_before, "chi2_after": chi2_before,
                "model_change": 0.0, "fista_iters": 0,
            },
        )

    dev = torch.device(device)
    conv = PSFConvolver(psf, (H, W), device=dev)
    omega_beam = _extract_psf_main_lobe(psf)[2] if nullspace_lam > 0.0 else 1.0
    mu_null = float(max(nullspace_lam, 0.0)) * float(omega_beam)
    gamma = float(np.clip(refit_gain, 1e-6, 1.0))
    tau = (1.0 - gamma) / gamma

    # Build sparse flat synthesis arrays for O(nnz) S(a) and U^T w via index_add_.
    id_chunks, pix_chunks, val_chunks = [], [], []
    u_norm2 = np.zeros(M, dtype=np.float64)
    buckets: dict[int, list[int]] = {}
    for i, (y0, y1, x0, x1, u_arr) in enumerate(atoms):
        u_f = np.asarray(u_arr, dtype=np.float32)
        if u_f.size == 0 or y1 <= y0 or x1 <= x0:
            continue
        u_norm2[i] = float((u_f.astype(np.float64) ** 2).sum())
        nz_r, nz_c = np.nonzero(u_f)
        if nz_r.size == 0:
            continue
        vals = u_f[nz_r, nz_c]
        p_idx = (nz_r + int(y0)) * W + (nz_c + int(x0))
        id_chunks.append(np.full(vals.shape, i, dtype=np.int64))
        pix_chunks.append(p_idx.astype(np.int64))
        val_chunks.append(vals)
        wb = max(8, int(2 ** math.ceil(math.log2(max(y1 - y0, x1 - x0, 1)))))
        buckets.setdefault(wb, []).append(i)

    if not id_chunks:
        a_out = np.zeros_like(a0_np) if mu_null > 0 else a0_np
        return (
            a_out,
            np.zeros((H, W), dtype=np.float32),
            residual.astype(np.float32).copy(),
            {
                "n_atoms": M, "n_down": 0, "n_zeroed": 0, "amp_ratio_med": 1.0,
                "chi2_before": chi2_before, "chi2_after": chi2_before,
                "model_change": 0.0, "fista_iters": 0,
            },
        )

    atom_ids = torch.from_numpy(np.concatenate(id_chunks)).to(dev)
    pix_idx = torch.from_numpy(np.concatenate(pix_chunks)).to(dev)
    pix_val = torch.from_numpy(np.concatenate(val_chunks)).to(dev)

    # Exact per-atom diagonal d_i = <u_i, B_+ u_i> via bucketed PSD kernel crops.
    d_diag = torch.zeros(M, dtype=torch.float32, device=dev)
    with torch.no_grad():
        for wb, idxs in buckets.items():
            k_psd = conv.psd_kernel_crop(2 * wb)
            step = max(1, int(2 ** 20 // max(wb * wb, 1)))
            for s0 in range(0, len(idxs), step):
                sub = idxs[s0:s0 + step]
                batch = np.zeros((len(sub), wb, wb), dtype=np.float32)
                for bi, ai in enumerate(sub):
                    y0, y1, x0, x1, u_arr = atoms[ai]
                    batch[bi, :y1 - y0, :x1 - x0] = u_arr
                bt = torch.from_numpy(batch).to(dev)
                bu = fft_convolve_same(bt, k_psd.unsqueeze(0).expand(len(sub), -1, -1))
                d_sub = (bt * bu).sum(dim=(1, 2)).clamp_min(0.0)
                d_diag[torch.as_tensor(sub, device=dev, dtype=torch.long)] = d_sub

        a0_t = torch.from_numpy(a0_np.astype(np.float32)).to(dev).clamp_min(0.0)
        r0_t = torch.from_numpy(np.ascontiguousarray(residual, dtype=np.float32)).to(dev)
        u_n2_t = torch.from_numpy(u_norm2.astype(np.float32)).to(dev)
        p_mu_t = (
            torch.from_numpy(np.asarray(prior_mu, dtype=np.float32)).to(dev)
            if prior_mu is not None else torch.zeros_like(a0_t)
        )
        p_rd_t = (
            torch.from_numpy(np.asarray(prior_ridge, dtype=np.float32)).to(dev).clamp_min(0.0)
            if prior_ridge is not None else torch.zeros_like(a0_t)
        )

        def synth(a_vec: torch.Tensor) -> torch.Tensor:
            flat = torch.zeros(H * W, dtype=torch.float32, device=dev)
            flat.index_add_(0, pix_idx, pix_val * a_vec[atom_ids])
            return flat.view(H, W)

        def trans(img: torch.Tensor) -> torch.Tensor:
            out = torch.zeros(M, dtype=torch.float32, device=dev)
            flat_img = img.reshape(-1)
            out.index_add_(0, atom_ids, pix_val * flat_img[pix_idx])
            return out

        m0_t = synth(a0_t)
        # Diagonal preconditioner h_i so each coordinate has unit diagonal curvature.
        h_diag = (1.0 + tau) * d_diag + mu_null * u_n2_t + p_rd_t
        active = (h_diag > 1e-12) & (u_n2_t > 0.0)
        scale = torch.where(active, torch.rsqrt(h_diag.clamp_min(1e-12)), torch.zeros_like(h_diag))
        z0 = torch.where(active, a0_t / scale.clamp_min(1e-30), torch.zeros_like(a0_t))

        def eval_obj_and_grad(z_vec: torch.Tensor) -> tuple[float, torch.Tensor]:
            a_vec = scale * z_vec
            m = synth(a_vec)
            dm = m - m0_t
            q = conv.convolve_psd_torch(dm)
            da = a_vec - a0_t
            quad_vis = 0.5 * float((dm * q).sum())
            lin_vis = float((dm * r0_t).sum())
            reg_null = 0.5 * mu_null * float((m * m).sum()) if mu_null > 0.0 else 0.0
            reg_prox = 0.5 * tau * float((d_diag * da * da).sum()) if tau > 0.0 else 0.0
            reg_prior = (
                0.5 * float((p_rd_t * (a_vec - p_mu_t) ** 2).sum())
                if prior_ridge is not None else 0.0
            )
            obj = quad_vis - lin_vis + reg_null + reg_prox + reg_prior

            w_img = q - r0_t
            if mu_null > 0.0:
                w_img = w_img + mu_null * m
            g_a = trans(w_img)
            if tau > 0.0:
                g_a = g_a + tau * d_diag * da
            if prior_ridge is not None:
                g_a = g_a + p_rd_t * (a_vec - p_mu_t)
            return obj, scale * g_a

        # For M=1 the preconditioned 1D quadratic has exact curvature 1.0, solved in 1 step.
        L = 1.0
        if int(active.sum()) > 1:
            # 3 power-method iterations to estimate initial Lipschitz bound L >= 1.
            v = torch.ones_like(z0) * active.float()
            v = v / v.norm().clamp_min(1e-12)
            for _ in range(3):
                av = scale * v
                mv = synth(av)
                qv = conv.convolve_psd_torch(mv)
                wv = qv + (mu_null * mv if mu_null > 0.0 else 0.0)
                hv = scale * (trans(wv) + (tau * d_diag + p_rd_t) * av)
                nrm = float(hv.norm())
                if nrm <= 1e-12:
                    break
                L = max(1.0, nrm)
                v = hv / nrm

        f0, _ = eval_obj_and_grad(z0)
        z = z0.clone()
        y = z0.clone()
        t_k = 1.0
        best_f = f0
        best_z = z0.clone()
        iters_run = 0

        for it in range(max(1, int(n_iter))):
            iters_run = it + 1
            fy, gy = eval_obj_and_grad(y)
            # Backtracking line search for guaranteed monotone convex majorisation.
            z_next = (y - gy / L).clamp_min(0.0)
            f_next = fy
            for _ in range(12):
                z_cand = (y - gy / L).clamp_min(0.0)
                dz = z_cand - y
                dz_n2 = float((dz * dz).sum())
                if dz_n2 <= 1e-24:
                    z_next = z_cand
                    f_next = fy
                    break
                fc, _ = eval_obj_and_grad(z_cand)
                quad_upper = fy + float((gy * dz).sum()) + 0.5 * L * dz_n2
                if fc <= quad_upper + 1e-6 * max(abs(fy), 1.0):
                    z_next = z_cand
                    f_next = fc
                    break
                L *= 2.0
            else:
                z_next = y.clone()
                f_next = fy

            if f_next < best_f:
                best_f = f_next
                best_z = z_next.clone()

            step_vec = z_next - z
            if float((step_vec * step_vec).sum()) <= 1e-16 * max(float((z * z).sum()), 1.0):
                z = z_next
                break

            # O'Donoghue-Candes gradient restart when momentum opposes negative gradient.
            if float(((y - z_next) * step_vec).sum()) > 0.0:
                t_k = 1.0
                y = z_next.clone()
            else:
                t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_k * t_k))
                y = z_next + ((t_k - 1.0) / t_next) * step_vec
                t_k = t_next
            z = z_next

        a_new_t = (scale * best_z).clamp_min(0.0)
        # Preserve inactive zero-curvature atoms only if nullspace damping is off.
        if mu_null == 0.0:
            a_new_t = torch.where(active, a_new_t, a0_t)
        delta_m_t = synth(a_new_t) - m0_t
        r_after_t = r0_t - conv.convolve_torch(delta_m_t)

        a_new = a_new_t.cpu().numpy().astype(np.float64)
        delta_model = delta_m_t.cpu().numpy().astype(np.float32)
        r_after = r_after_t.cpu().numpy().astype(np.float32)

    chi2_after = float((r_after.astype(np.float64) ** 2).sum())
    ratios = [float(an / max(ao, 1e-30)) for an, ao in zip(a_new, a0_np) if ao > 0.0 or an > 0.0]
    n_down = int(np.sum(a_new < a0_np - 1e-12))
    n_zero = int(np.sum((a_new <= 1e-12) & (a0_np > 1e-12)))
    stats = {
        "n_atoms": M,
        "n_down": n_down,
        "n_zeroed": n_zero,
        "amp_ratio_med": float(np.median(ratios)) if ratios else None,
        "chi2_before": chi2_before,
        "chi2_after": chi2_after,
        "model_change": float(delta_model.sum()),
        "fista_iters": iters_run,
    }
    return a_new, delta_model, r_after, stats


def refit_commits(
    residual: np.ndarray,
    model_update: np.ndarray,
    psf: np.ndarray,
    commits: list[AspenCommit],
    *,
    sigma: float = 1.0,
    refit_gain: float = 0.5,
    cutout: int = 128,
    sigma_min_px: float | None = None,
    sigma_max_px: float | None = None,
    exact_residual: bool = True,
    exact_every: int = 64,
    flux_prior_weight: float = 0.0,
    size_bias: float = 0.0,
    zero_rejected: bool = False,
    render_nsigma: float | None = None,
    collect: list | None = None,
    device: str = "cpu",
    repropose: dict | None = None,
    amp_method: str = "cd",
    nullspace_lam: float = 0.0,
    nnls_iters: int = 60,
) -> tuple[np.ndarray, np.ndarray, list[AspenCommit], dict]:
    """Refit every commit against the exact likelihood, by add-back.

    ``residual`` and ``model_update`` are the working residual and the model
    this minor cycle added; both are returned updated, so the caller can hand
    the model to the major cycle unchanged in meaning.

    ``refit_gain`` damps the amplitude step (1.0 = jump straight to the exact
    maximum).  The shape is taken whole, since it is chosen by the same
    objective that sets the amplitude.

    ``size_bias`` adds ``size_bias * ln(sqrt(sig_maj * sig_min))`` to J when the
    candidate shapes are ranked (a power-law prior p(sigma) ~ sigma^size_bias).
    Positive favours larger components; 0 is the plain profile likelihood. J
    assumes independent pixel noise, so the value is a relative slider, not
    calibrated nats.

    ``zero_rejected``: when no candidate has a positive amplitude, step the
    flux toward 0 by ``refit_gain`` (the positivity-constrained optimum is 0).
    Off keeps the old flux, so over-subtracted flux is never taken back.

    ``render_nsigma`` truncates every rendered Aspen as the minor cycle does;
    with ``cutout >= 2 * render_nsigma * sigma_max`` each Aspen's whole
    footprint is inside its window, so the windowed bookkeeping is exact.

    ``repropose``: dict with ``model`` (the MDNAsp) and the minor cycle's
    conditioning (``config_idx``, ``scale_relative``, ``global_sigma``). Each
    Aspen's add-back residual is fed to the network again at the Aspen's pixel
    and the fresh mixture shapes join the candidates, so a shape proposed when
    the surroundings were still full of flux can be replaced. None = off.

    ``collect``, if a list, receives one dict per refit Aspen (cid, cx, cy,
    sig2, and every valid candidate as (sig_maj, sig_min, pa, J, den, a_star))
    -- the raw material for ``error_map_from_records``.

    ``exact_residual`` re-derives the residual from the model with the FULL psf
    every ``exact_every`` Aspen and once at the end, one FFT each.  Without it the per-Aspen
    windowed subtraction leaves each component's out-of-window sidelobes in the
    residual: measured on a 60-component synthetic shell (G55 PSF), the
    internal residual drifts 10% from the exact one after one sweep and 137%
    after five, and every Aspen later in the sweep fits a residual that is too
    positive.
    """
    smin = BEAM_SIGMA_PX if sigma_min_px is None else float(sigma_min_px)
    smax = SIGMA_MAX_PX if sigma_max_px is None else float(sigma_max_px)
    residual_start = residual
    model_start = model_update
    residual = residual.copy()
    model_update = model_update.copy()
    H, W = residual.shape
    c = min(int(cutout), H, W)
    half = c // 2
    psf_cut = _crop_psf(psf, min(2 * c, max(psf.shape)))
    conv_cut = PSFConvolver(psf_cut, (c, c), device)
    use_nnls = (amp_method == "nnls")
    conv_full = PSFConvolver(psf, (H, W), device) if (exact_residual or use_nnls) else None
    sig2 = float(sigma) ** 2 if sigma > 0 else 1.0
    omega_beam = _extract_psf_main_lobe(psf)[2] if nullspace_lam > 0.0 else 1.0
    mu_null = float(max(nullspace_lam, 0.0)) * float(omega_beam)
    if repropose is not None:
        from mdn_asp.minor_cycle import _crop_psf_centred
        rp_psf = _crop_psf_centred(psf, repropose["model"].cutout)

    out: list[AspenCommit] = []
    n_shape_changed = 0
    n_rejected = 0
    n_zeroed = 0
    n_prior_applied = 0
    size_ratios: list[float] = []
    flux_ratios: list[float] = []
    chi2_before = chi2_after = 0.0

    def _exact() -> np.ndarray:
        delta_total = (model_update - model_start).astype(np.float32)
        return (residual_start - conv_full(delta_total)).astype(np.float32)

    nnls_atoms: list[tuple[int, int, int, int, np.ndarray]] = []
    nnls_prior_mu = np.zeros(len(commits), dtype=np.float64)
    nnls_prior_rd = np.zeros(len(commits), dtype=np.float64)
    shape_delta_model = np.zeros((H, W), dtype=np.float32) if use_nnls else None

    for n_done, commit in enumerate(commits):
        if (not use_nnls) and exact_residual and exact_every > 0 and n_done and n_done % exact_every == 0:
            # Intra-sweep correction. Without it every later Aspen fits a
            # residual still holding the earlier ones' out-of-window sidelobes.
            residual = _exact()
        r0 = int(round(commit.cy)) - half
        c0 = int(round(commit.cx)) - half
        r0 = int(np.clip(r0, 0, H - c))
        c0 = int(np.clip(c0, 0, W - c))
        win = (slice(r0, r0 + c), slice(c0, c0 + c))
        cy_l = commit.cy - r0
        cx_l = commit.cx - c0

        # 1. add-back
        old_img = render_aspen(cx_l, cy_l, commit.flux, commit.sig_maj,
                               commit.sig_min, commit.pa, (c, c), nsigma=render_nsigma)
        r_k = residual[win] + conv_cut(old_img)
        chi2_before += float((residual[win] ** 2).sum())

        # 2-3. profile likelihood over the candidate shapes
        best = None
        best_score = -math.inf
        cand_log = []
        extra = None
        if repropose is not None:
            # Full-frame residual with this Aspen added back inside its window,
            # so the network sees what the refit fits.
            from mdn_asp.minor_cycle import _safe_crop
            rc, cc = int(round(commit.cy)), int(round(commit.cx))
            n = repropose["model"].cutout
            q0, p0 = rc - n // 2, cc - n // 2          # cutout origin, as minor_cycle
            res_cut = _safe_crop(residual, q0, q0 + n, p0, p0 + n)
            ir0, ir1 = max(q0, r0), min(q0 + n, r0 + c)
            ic0, ic1 = max(p0, c0), min(p0 + n, c0 + c)
            res_cut[ir0 - q0:ir1 - q0, ic0 - p0:ic1 - p0] = r_k[ir0 - r0:ir1 - r0, ic0 - c0:ic1 - c0]
            extra = _repropose(repropose["model"], res_cut, rp_psf, float(sigma), repropose)
        cands = _candidates(commit, smin, smax, extra)
        lprior = candidate_log_prior(commit, cands, (commit.sig_maj, commit.sig_min), smin, smax)
        gs = np.stack([render_aspen(cx_l, cy_l, 1.0, smaj, smin_, pa, (c, c),
                                    nsigma=render_nsigma) for smaj, smin_, pa in cands]
                      or [np.zeros((c, c), np.float32)])[:len(cands)]
        Bgs = conv_cut(gs)    # all candidates in one batched FFT
        for (smaj, smin_, pa), lp, g, Bg in zip(cands, lprior, gs, Bgs):
            num = float((g * r_k).sum())
            den = float((g * Bg).sum()) + (mu_null * float((g * g).sum()) if mu_null > 0.0 else 0.0)
            if num <= 0.0 or den <= 0.0:
                continue
            J = num * num / (2.0 * sig2 * den)
            score = J + size_bias * 0.5 * math.log(smaj * smin_)
            cand_log.append((smaj, smin_, float(pa), J, den, num / den, lp))
            if best is None or score > best_score:
                best_score = score
                best = (J, smaj, smin_, pa, num / den, g)

        # MAP rather than ML amplitude. With a Gaussian prior N(mu_a, tau^2) on
        # the amplitude the profile optimum picks up a ridge term:
        #   a = (num + sig2*mu_a/tau^2) / (den + sig2/tau^2)
        # which is the same sig2/tau^2 added to the diagonal of H in the joint
        # solve. flux_prior_weight scales tau (larger = weaker prior); 0 keeps
        # the unregularised ML path bit-identical.
        if flux_prior_weight > 0.0:
            mu_a, tau = flux_prior(commit)
            if np.isfinite(mu_a) and np.isfinite(tau) and tau > 0.0:
                tau = tau * flux_prior_weight
                ridge = sig2 / (tau * tau)
                if use_nnls:
                    nnls_prior_mu[n_done] = max(mu_a, 0.0)
                    nnls_prior_rd[n_done] = ridge
                    n_prior_applied += 1
                elif best is not None:
                    _J, smaj, smin_, pa, _a_ml, g = best
                    Bg = conv_cut(g)
                    num = float((g * r_k).sum())
                    den = float((g * Bg).sum()) + (mu_null * float((g * g).sum()) if mu_null > 0.0 else 0.0)
                    a_map = (num + ridge * mu_a) / (den + ridge)
                    best = (_J, smaj, smin_, pa, max(a_map, 0.0), g)
                    n_prior_applied += 1

        if collect is not None and cand_log:
            collect.append({"cid": commit.cid, "cx": commit.cx, "cy": commit.cy,
                            "sig2": sig2, "cands": cand_log,
                            "cur": (commit.sig_maj, commit.sig_min, commit.pa),
                            "flux_cur": commit.flux})

        if use_nnls:
            old_unit = gs[0]
            if best is None:
                n_rejected += 1
                smaj, smin_, pa, g_sel = commit.sig_maj, commit.sig_min, commit.pa, old_unit
            else:
                _J, smaj, smin_, pa, _a_star, g_sel = best
                if abs(smaj - commit.sig_maj) > 1e-3 or abs(smin_ - commit.sig_min) > 1e-3:
                    n_shape_changed += 1
            size_ratios.append(smaj / max(commit.sig_maj, 1e-6))
            if (abs(smaj - commit.sig_maj) > 1e-9 or abs(smin_ - commit.sig_min) > 1e-9
                    or abs(pa - commit.pa) > 1e-9) and commit.flux != 0.0:
                new_base_img = render_aspen(cx_l, cy_l, commit.flux, smaj, smin_, pa, (c, c),
                                            nsigma=render_nsigma)
                shape_delta_model[win] += (new_base_img - old_img).astype(np.float32)
            nnls_atoms.append((r0, r0 + c, c0, c0 + c, g_sel))
            out.append(replace(commit, sig_maj=smaj, sig_min=smin_, pa=float(pa)))
            continue

        if best is None:
            n_rejected += 1
            if zero_rejected and commit.flux > 0.0:
                flux_new = float(commit.flux * (1.0 - refit_gain))
                delta = render_aspen(cx_l, cy_l, flux_new, commit.sig_maj, commit.sig_min,
                                     commit.pa, (c, c), nsigma=render_nsigma) - old_img
                residual[win] -= conv_cut(delta)
                model_update[win] += delta
                n_zeroed += 1
                commit = replace(commit, flux=flux_new)
            out.append(commit)
            chi2_after += float((residual[win] ** 2).sum())
            continue

        _J, smaj, smin_, pa, a_star, g = best
        flux_new = float(max(commit.flux + refit_gain * (a_star - commit.flux), 0.0))
        new_img = render_aspen(cx_l, cy_l, flux_new, smaj, smin_, pa, (c, c),
                               nsigma=render_nsigma)

        delta = new_img - old_img
        residual[win] -= conv_cut(delta)
        model_update[win] += delta
        chi2_after += float((residual[win] ** 2).sum())

        if abs(smaj - commit.sig_maj) > 1e-3 or abs(smin_ - commit.sig_min) > 1e-3:
            n_shape_changed += 1
        size_ratios.append(smaj / max(commit.sig_maj, 1e-6))
        flux_ratios.append(flux_new / max(commit.flux, 1e-30))
        out.append(replace(commit, flux=flux_new, sig_maj=smaj,
                           sig_min=smin_, pa=float(pa)))

    if use_nnls and commits:
        model_base = (model_start + shape_delta_model).astype(np.float32)
        res_base = (
            (residual_start - conv_full(shape_delta_model)).astype(np.float32)
            if float(np.abs(shape_delta_model).max()) > 0.0
            else residual_start.astype(np.float32)
        )
        a0_vec = np.array([max(c_old.flux, 0.0) for c_old in commits], dtype=np.float64)
        a_new, delta_nnls, r_after, nnls_st = solve_nnls_amplitudes(
            nnls_atoms,
            a0_vec,
            res_base,
            psf,
            refit_gain=refit_gain,
            nullspace_lam=nullspace_lam,
            n_iter=nnls_iters,
            prior_mu=nnls_prior_mu if n_prior_applied > 0 else None,
            prior_ridge=nnls_prior_rd if n_prior_applied > 0 else None,
            device=str(device),
        )
        model_update = (model_base + delta_nnls).astype(np.float32)
        residual = (residual_start - conv_full((model_update - model_start).astype(np.float32))).astype(np.float32)
        n_zeroed = int(nnls_st["n_zeroed"])
        out = [replace(c_cur, flux=float(an)) for c_cur, an in zip(out, a_new)]
        flux_ratios = [float(an / max(ao, 1e-30)) for an, ao in zip(a_new, a0_vec)]
        chi2_before = float((residual_start.astype(np.float64) ** 2).sum())
        chi2_after = float((residual.astype(np.float64) ** 2).sum())
    elif exact_residual:
        # Both chi^2 over the whole field, so they are comparable; the running
        # sums above are over per-Aspen windows, which overlap.
        residual = _exact()
        chi2_before = float((residual_start ** 2).sum())
        chi2_after = float((residual ** 2).sum())

    stats = {
        "n_refit": len(commits),
        "n_prior_applied": n_prior_applied,
        "flux_prior_weight": float(flux_prior_weight),
        "exact_residual": bool(exact_residual),
        "n_shape_changed": n_shape_changed,
        "n_rejected": n_rejected,
        "n_zeroed": n_zeroed,
        "size_ratio_med": float(np.median(size_ratios)) if size_ratios else float("nan"),
        # The plan's phase-2 measurement: how often the exact likelihood
        # disagrees with the proposal by more than x1.5 in size.
        "frac_size_beyond_1p5": (
            float(np.mean([(r > 1.5) or (r < 1 / 1.5) for r in size_ratios]))
            if size_ratios else float("nan")),
        "flux_ratio_med": float(np.median(flux_ratios)) if flux_ratios else float("nan"),
        "chi2_before": chi2_before,
        "chi2_after": chi2_after,
    }
    return residual, model_update, out, stats


def joint_amplitude_error(atoms: list, a, shape: tuple[int, int], psf: np.ndarray, sigma: float,
                          device: str = "cpu", eig_floor: float = 1e-10) -> tuple[np.ndarray, dict]:
    """Per-pixel std (Jy/pixel) of the model sum_k a_k u_k from the JOINT amplitude posterior, shapes fixed.

    With dirty-image noise covariance sigma^2 B, the Fisher matrix of the active amplitudes (a_k > 0) is
    F = U^T B_+ U / sigma^2, Cov(a) = F^-1 and Var(model(x)) = u(x)^T F^-1 u(x), u(x) the atom values at x.
    Overlapping atoms are anticorrelated through F; the per-atom sigma^2 / <u_k, B u_k> ignores that.
    Eigenvalues of F below eig_floor * max are floored there (directions the data do not constrain keep a
    large variance, they are not dropped); the count is in the stats.
    atoms: (y0, y1, x0, x1, u), u in Jy/pixel per unit amplitude, on an image of the given shape."""
    import torch
    from mdn_asp.minor_cycle import PSFConvolver

    H, W = shape
    a = np.asarray(a, dtype=np.float64)
    act = [i for i in range(len(atoms)) if a[i] > 0.0 and np.asarray(atoms[i][4]).size > 0]
    stats = {"n_atoms": len(atoms), "n_active": len(act), "n_floored": 0, "cond": None}
    if not act:
        return np.zeros((H, W), np.float32), stats
    dev = torch.device(device)
    conv = PSFConvolver(psf, (H, W), device=dev)
    M = len(act)
    ids, pix, val = [], [], []
    for n, i in enumerate(act):
        y0, y1, x0, x1, u = atoms[i]
        u = np.asarray(u, dtype=np.float32)
        r, c = np.nonzero(u)
        ids.append(np.full(r.size, n, np.int64)); pix.append((r + y0) * W + (c + x0)); val.append(u[r, c])
    ids = torch.from_numpy(np.concatenate(ids)).to(dev)
    pix = torch.from_numpy(np.concatenate(pix).astype(np.int64)).to(dev)
    val = torch.from_numpy(np.concatenate(val)).to(dev)
    nnz = int(val.numel())
    blk = max(1, min(M, int(2 ** 26 // max(nnz, H * W))))

    def synth(coef):            # coef (M, b) -> images (b, H, W)
        b = coef.shape[1]
        flat = torch.zeros(b * H * W, dtype=torch.float64, device=dev)
        off = torch.arange(b, device=dev, dtype=torch.long)[:, None] * (H * W)
        flat.index_add_(0, (off + pix[None, :]).reshape(-1), (coef[ids].T * val[None, :]).reshape(-1))
        return flat.view(b, H, W)

    F = torch.zeros((M, M), dtype=torch.float64, device=dev)
    with torch.no_grad():
        for j0 in range(0, M, blk):
            j1 = min(M, j0 + blk)
            e = torch.zeros((M, j1 - j0), dtype=torch.float64, device=dev)
            e[torch.arange(j0, j1, device=dev), torch.arange(j1 - j0, device=dev)] = 1.0
            bu = conv.convolve_psd_torch(synth(e).float()).double().reshape(j1 - j0, -1)
            g = torch.zeros((j1 - j0, M), dtype=torch.float64, device=dev)
            g.index_add_(1, ids, bu[:, pix] * val[None, :].double())
            F[:, j0:j1] = g.T
        F = 0.5 * (F + F.T) / float(sigma) ** 2
        lam, V = torch.linalg.eigh(F)
        lmax = float(lam.max())
        floor = eig_floor * lmax
        stats["n_floored"] = int((lam < floor).sum())
        stats["cond"] = float(lmax / max(float(lam.clamp_min(floor).min()), 1e-300))
        L = V * lam.clamp_min(floor).rsqrt()[None, :]          # Cov = L L^T
        var = torch.zeros((H, W), dtype=torch.float64, device=dev)
        for j0 in range(0, M, blk):
            img = synth(L[:, j0:min(M, j0 + blk)])
            var += (img * img).sum(0)
    return var.sqrt().float().cpu().numpy(), stats


def error_map_from_records(records: list, shape: tuple[int, int],
                           size_bias: float = 0.0, max_half: int = 64,
                           nsigma: float | None = None, include_amp: bool = True) -> np.ndarray:
    """Per-pixel std (Jy/pixel) of the model image under the refit posterior.

    Per Aspen, the shape candidates ``c`` carry weights p_c ~ pi_c exp(J_c +
    size_bias * ln sqrt(sig_maj sig_min)) (so the bias is part of the
    posterior), pi_c the network's mixture weight (column 7, candidate_log_prior; records
    without it get a flat prior), and given a shape the amplitude is N(a_c, sig2 / <g_c, B g_c>).
    With unit-sum shapes g_c the per-Aspen variance at pixel x is

        sum_c p_c (s_c^2 + a_c^2) g_c(x)^2 - (sum_c p_c a_c g_c(x))^2

    and Aspen are summed as independent. include_amp=False drops s_c^2 (shape spread only), for use with
    joint_amplitude_error, which carries the amplitude part with the overlaps. Overlapping Aspen are in fact
    anticorrelated, so the map is conservative (too large), not too small.
    Each Aspen is drawn in a window of +-max_half pixels, normalised inside it.
    """
    H, W = shape
    var = np.zeros((H, W), np.float64)
    for rec in records:
        cands = np.array(rec["cands"], dtype=np.float64)   # smaj, smin, pa, J, den, a[, log prior]
        smaj, smin_, pa, J, den, a = cands.T[:6]
        score = J + size_bias * 0.5 * np.log(smaj * smin_)
        if cands.shape[1] > 6:
            score = score + cands[:, 6]
        p = np.exp(score - score.max())
        p /= p.sum()
        s2 = rec["sig2"] / den
        r = int(min(max(4 * smaj.max(), 8), max_half))
        r0, r1 = max(0, int(rec["cy"]) - r), min(H, int(rec["cy"]) + r + 1)
        c0, c1 = max(0, int(rec["cx"]) - r), min(W, int(rec["cx"]) + r + 1)
        m1 = np.zeros((r1 - r0, c1 - c0))
        m2 = np.zeros_like(m1)
        for k in range(len(p)):
            g = render_aspen(rec["cx"] - c0, rec["cy"] - r0, 1.0, smaj[k], smin_[k],
                             pa[k], m1.shape, nsigma=nsigma).astype(np.float64)
            m1 += p[k] * a[k] * g
            m2 += p[k] * ((s2[k] if include_amp else 0.0) + a[k] ** 2) * g * g
        var[r0:r1, c0:c1] += m2 - m1 * m1
    return np.sqrt(np.maximum(var, 0.0)).astype(np.float32)


def _render_windows(p, r0, c0, w, H, W, nsig, smin, smax):
    """Differentiable unit-sum Aspen renders, one (w, w) window each, as
    render_aspen: truncated at ``nsig`` (None = untruncated in the window) and
    normalised over the in-frame pixels. Returns (vals (n, w, w), flat index
    (n, w, w) with H*W for out-of-frame pixels)."""
    import torch
    dev = p["cx"].device
    ar = torch.arange(w, device=dev, dtype=torch.float32)
    R = r0[:, None, None].float() + ar[None, :, None]
    C = c0[:, None, None].float() + ar[None, None, :]
    smaj = p["lsmaj"].exp().clamp(smin, smax)[:, None, None]
    smin_ = p["lsmin"].exp().clamp(smin, smax)[:, None, None]
    pa = p["pa"][:, None, None]
    dc, dr = C - p["cx"][:, None, None], R - p["cy"][:, None, None]
    u = dc * torch.cos(pa) + dr * torch.sin(pa)
    v = -dc * torch.sin(pa) + dr * torch.cos(pa)
    e = 0.5 * ((u / smaj) ** 2 + (v / smin_) ** 2)
    inside = (R >= 0) & (R < H) & (C >= 0) & (C < W)
    keep = inside if nsig is None else inside & (e <= 0.5 * nsig * nsig)
    g = torch.exp(-e) * keep
    g = g / g.sum(dim=(1, 2), keepdim=True).clamp_min(1e-30)
    idx = torch.where(inside, R.long() * W + C.long(), torch.full_like(R, H * W, dtype=torch.long))
    return p["lflux"].exp()[:, None, None] * g, idx


def refit_gradient(
    residual: np.ndarray,
    model: np.ndarray,
    psf: np.ndarray,
    commits: list[AspenCommit],
    *,
    sigma: float = 1.0,
    n_iter: int = 200,
    lr_pos: float = 0.05,
    lr_log: float = 0.02,
    sigma_min_px: float | None = None,
    sigma_max_px: float | None = None,
    max_window: int = 256,
    render_nsigma: float | None = None,
    objective: str = "image",
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, list[AspenCommit], dict]:
    """Joint gradient refit of every committed Aspen, as Asp's optimiser does.

    Free per Aspen: position (px), log sizes, PA and log flux (so flux stays
    positive). Loss: chi^2 = |r|^2 / (2 sigma^2) over the field, with
    r = residual - B (M(theta) - M(theta_0)), the same operator the discrete
    refit uses. Adam, ``lr_pos`` on positions and ``lr_log`` on the rest.

    Memory: the gradient is taken in two passes per step. All windows are
    rendered without grad to form r, then dchi2/dM = -B r / sigma^2 (B is
    symmetric about its peak) is gathered back into each window and pushed
    through a re-render, chunk by chunk. Windows are fixed at the start
    (2 x 1.6 x the render extent of the starting size, power of two, at most
    ``max_window``) and centred on the starting pixel.

    The best iterate by chi^2 is kept. The returned model and residual are
    rebuilt with render_aspen and the full-field operator, so the bookkeeping
    is exact whatever the optimiser did; if that exact chi^2 is not below the
    start, nothing is changed.

    Aspen with zero flux are left out (log flux undefined).
    """
    import torch
    smin = BEAM_SIGMA_PX if sigma_min_px is None else float(sigma_min_px)
    smax = SIGMA_MAX_PX if sigma_max_px is None else float(sigma_max_px)
    dev = torch.device(device)
    H, W = residual.shape
    sig2 = float(sigma) ** 2 if sigma > 0 else 1.0
    ext = render_nsigma if render_nsigma is not None else 5.0
    conv = PSFConvolver(psf, (H, W), device)

    act = [i for i, c in enumerate(commits) if c.flux > 0.0]
    if not act:
        return residual, model, commits, {"n_active": 0}
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
    c_ = [commits[i] for i in act]
    p0 = {"cx": t([c.cx for c in c_]), "cy": t([c.cy for c in c_]),
          "lsmaj": t([math.log(max(c.sig_maj, 1e-6)) for c in c_]),
          "lsmin": t([math.log(max(c.sig_min, 1e-6)) for c in c_]),
          "pa": t([c.pa for c in c_]), "lflux": t([math.log(c.flux) for c in c_])}
    p = {k: v.clone().requires_grad_(True) for k, v in p0.items()}

    # Fixed windows, bucketed by size so each bucket is one batched render.
    need = np.array([2.0 * 1.6 * ext * max(c.sig_maj, c.sig_min) + 3.0 for c in c_])
    wsz = np.clip(2 ** np.ceil(np.log2(np.maximum(need, 8.0))), 8, max_window).astype(int)
    buckets = []
    for w in np.unique(wsz):
        ids = np.nonzero(wsz == w)[0]
        step = max(1, int(2 ** 22 // (w * w)))       # ~4M window pixels per chunk
        for s in range(0, len(ids), step):
            sel = torch.as_tensor(ids[s:s + step], device=dev)
            r0 = torch.round(p0["cy"][sel]).long() - int(w) // 2
            c0 = torch.round(p0["cx"][sel]).long() - int(w) // 2
            buckets.append((sel, r0, c0, int(w)))

    def render_all(params, grad_img=None):
        """Model of the active Aspen (flat, H*W). With ``grad_img``, also
        backprop <grad_img, M> into ``params`` chunk by chunk."""
        flat = torch.zeros(H * W + 1, device=dev)
        g_ext = None if grad_img is None else torch.cat([grad_img.reshape(-1), grad_img.new_zeros(1)])
        for sel, r0, c0, w in buckets:
            sub = {k: v[sel] for k, v in params.items()}
            if grad_img is None:
                with torch.no_grad():
                    vals, idx = _render_windows(sub, r0, c0, w, H, W, render_nsigma, smin, smax)
            else:
                vals, idx = _render_windows(sub, r0, c0, w, H, W, render_nsigma, smin, smax)
                (vals * g_ext[idx]).sum().backward()
                vals = vals.detach()
            flat.index_add_(0, idx.reshape(-1), vals.reshape(-1))
        return flat[:-1].reshape(H, W)

    R0 = torch.from_numpy(residual.astype(np.float32)).to(dev)
    with torch.no_grad():
        M0 = render_all(p0)
    opt = torch.optim.Adam([{"params": [p["cx"], p["cy"]], "lr": lr_pos},
                            {"params": [p["lsmaj"], p["lsmin"], p["pa"], p["lflux"]], "lr": lr_log}])
    chi2_0 = 0.0 if objective == "vis" else float((R0 ** 2).sum()) / (2 * sig2)
    best_chi2, best = chi2_0, {k: v.detach().clone() for k, v in p0.items()}
    for _ in range(n_iter):
        opt.zero_grad()
        with torch.no_grad():
            M = render_all(p)
            dm = M - M0
            if objective == "vis":
                q = conv.convolve_psd_torch(dm)
                chi2 = float((0.5 * (dm * q).sum() - (dm * R0).sum()) / sig2)
                gM = (q - R0) / sig2
            else:
                r = R0 - conv.convolve_torch(dm)
                chi2 = float((r ** 2).sum()) / (2 * sig2)
                gM = -conv.adjoint_torch(r) / sig2
            if chi2 < best_chi2:
                best_chi2, best = chi2, {k: v.detach().clone() for k, v in p.items()}
        render_all(p, grad_img=gM)
        opt.step()
    with torch.no_grad():
        M = render_all(p)
        dm = M - M0
        if objective == "vis":
            q = conv.convolve_psd_torch(dm)
            chi2 = float((0.5 * (dm * q).sum() - (dm * R0).sum()) / sig2)
        else:
            chi2 = float(((R0 - conv.convolve_torch(dm)) ** 2).sum()) / (2 * sig2)
        if chi2 < best_chi2:
            best_chi2, best = chi2, {k: v.detach().clone() for k, v in p.items()}

    # Exact rebuild with render_aspen, windowed, as everywhere else.
    b = {k: v.cpu().numpy().astype(np.float64) for k, v in best.items()}
    new = list(commits)
    delta = np.zeros((H, W), np.float64)
    moved = []
    for j, i in enumerate(act):
        old = commits[i]
        raw_sj = float(math.exp(np.clip(b["lsmaj"][j], -20.0, 20.0)))
        raw_sn = float(math.exp(np.clip(b["lsmin"][j], -20.0, 20.0)))
        sj, sn, pa = _canonical_axes(raw_sj, raw_sn, float(b["pa"][j]), smin, smax)
        c = replace(old, cx=float(b["cx"][j]), cy=float(b["cy"][j]), sig_maj=sj,
                    sig_min=sn, pa=pa, flux=float(math.exp(np.clip(b["lflux"][j], -30.0, 20.0))))
        for q, sgn in ((old, -1.0), (c, 1.0)):
            rr = (render_nsigma or 8.0) * max(q.sig_maj, q.sig_min) + 2
            r0, r1 = max(0, int(q.cy - rr)), min(H, int(q.cy + rr) + 2)
            c0, c1 = max(0, int(q.cx - rr)), min(W, int(q.cx + rr) + 2)
            if r1 > r0 and c1 > c0:
                delta[r0:r1, c0:c1] += sgn * render_aspen(
                    q.cx - c0, q.cy - r0, q.flux, q.sig_maj, q.sig_min, q.pa,
                    (r1 - r0, c1 - c0), nsigma=render_nsigma)
        new[i] = c
        moved.append(math.hypot(c.cx - old.cx, c.cy - old.cy))
    res_new = (residual - conv(delta.astype(np.float32))).astype(np.float32)
    chi2_exact_0 = float((residual.astype(np.float64) ** 2).sum()) / (2 * sig2)
    chi2_exact = float((res_new.astype(np.float64) ** 2).sum()) / (2 * sig2)
    stats = {"n_active": len(act), "chi2_start": chi2_exact_0, "chi2_opt": best_chi2,
             "chi2_end": chi2_exact, "accepted": chi2_exact < chi2_exact_0,
             "move_med_px": float(np.median(moved)), "move_max_px": float(np.max(moved))}
    if not stats["accepted"]:
        return residual, model, commits, stats
    return res_new, (model + delta).astype(np.float32), new, stats
