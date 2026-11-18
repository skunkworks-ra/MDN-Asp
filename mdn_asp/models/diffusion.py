"""Sky diffusion models, synthetic scene generator, posterior samplers, and noise-PSD helpers.

Supports four training/inference variants sharing one U-Net backbone and cosine schedule:
  A   residual-conditioned full-patch sky posterior: input (residual, PSF) -> sky - m
  B   dirty-conditioned full-patch sky posterior: input (dirty, PSF, m) -> sky
  P   unconditional local sky shape prior on W x W Hann-tapered windows (with DDS CG sampler)
  LD  context-conditioned local matching-pursuit dictionary atom generator on W x W windows
      given C x C context (residual, PSF), using either avgpool (5 ch) or strided conv stem (11 ch)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mdn_asp.data.gpu_cutout_generator import fft_convolve_same

FILLS = (("ln", 2.5, 1.0), ("ln", 2.5, 2.0), ("ln", 4.5, 2.0), ("ridge", 3.5, 0.1), ("ridge", 4.5, 0.1))
KINDS = [f"{k}_a{a}_{p}" for k, a, p in FILLS] + ["ring", "gauss", "point"]
MAX_AGN = 64


# ---------------------------------------------------------------------------
# Spatial & PSF helpers
# ---------------------------------------------------------------------------

def crop_psf_centred(psf: np.ndarray, size: int) -> np.ndarray:
    """Crop or zero-pad 2D PSF array to (size, size) with its peak at (size // 2, size // 2)."""
    py, px = np.unravel_index(int(np.argmax(psf)), psf.shape)
    half = size // 2
    r0, c0 = py - half, px - half
    H, W = psf.shape
    out = np.zeros((size, size), dtype=np.float32)
    sr0, sr1 = max(0, r0), min(H, r0 + size)
    sc0, sc1 = max(0, c0), min(W, c0 + size)
    if sr1 <= sr0 or sc1 <= sc0:
        return out
    dr0, dc0 = sr0 - r0, sc0 - c0
    out[dr0:dr0 + (sr1 - sr0), dc0:dc0 + (sc1 - sc0)] = psf[sr0:sr1, sc0:sc1]
    return out


def _shift_zero_pad(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Shift a (..., H, W) tensor by (dy, dx) with zero padding (no periodic wrap-around)."""
    dy, dx = int(dy), int(dx)
    H, W = x.shape[-2], x.shape[-1]
    out = torch.zeros_like(x)
    if abs(dy) >= H or abs(dx) >= W:
        return out
    if dy >= 0:
        sy0, sy1, dy0, dy1 = 0, H - dy, dy, H
    else:
        sy0, sy1, dy0, dy1 = -dy, H, 0, H + dy
    if dx >= 0:
        sx0, sx1, dx0, dx1 = 0, W - dx, dx, W
    else:
        sx0, sx1, dx0, dx1 = -dx, W, 0, W + dx
    out[..., dy0:dy1, dx0:dx1] = x[..., sy0:sy1, sx0:sx1]
    return out


def grf(alpha, S, gen, dev):
    """Unit-variance Gaussian random fields with P(k) ~ k^-alpha. alpha (B,)."""
    B = alpha.shape[0]
    w = torch.randn(B, S, S, generator=gen, device=dev)
    ky = torch.fft.fftfreq(S, device=dev).view(1, S, 1)
    kx = torch.fft.fftfreq(S, device=dev).view(1, 1, S)
    k = torch.sqrt(kx ** 2 + ky ** 2)
    k[:, 0, 0] = 1.0
    amp = k ** (-alpha.view(B, 1, 1) / 2)
    amp[:, 0, 0] = 0.0
    f = torch.fft.ifft2(torch.fft.fft2(w) * amp).real
    f = f - f.mean(dim=(1, 2), keepdim=True)
    return f / f.std(dim=(1, 2), keepdim=True)


def beam_sigma_px(psf: np.ndarray) -> float:
    """Circular sigma from the main-lobe area above half maximum (as sim_recovery.py)."""
    from scipy.ndimage import label
    lab, _ = label(psf >= 0.5 * psf.max())
    cy, cx = np.unravel_index(int(np.argmax(psf)), psf.shape)
    fwhm = 2.0 * math.sqrt((lab == lab[cy, cx]).sum() / math.pi)
    return fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def beam_area_px(psf: np.ndarray) -> float:
    """Gaussian main-lobe beam area (2 * pi * sigma_maj * sigma_min = Area_FWHM / ln 2) in pixels."""
    from scipy.ndimage import label
    psf_arr = np.asarray(psf)
    pmax = float(psf_arr.max())
    if pmax <= 0.0:
        return 1.0
    py, px = np.unravel_index(int(np.argmax(psf_arr)), psf_arr.shape)
    lab, _ = label(psf_arr >= 0.5 * pmax)
    n_half = float((lab == lab[py, px]).sum())
    return max(1.0, n_half / math.log(2.0))


# ---------------------------------------------------------------------------
# Briggs Noise PSD & AIC/DOF Atom Scoring Helpers
# ---------------------------------------------------------------------------

def estimate_noise_cov_2w(
    residual: np.ndarray,
    noise_mask: np.ndarray,
    Wn: int,
    device,
    max_windows: int = 64,
) -> torch.Tensor | None:
    """Estimate PSD-projected empirical noise covariance kernel on lags (-Wn, Wn)^2.

    Extracts up to max_windows (Wn, Wn) windows strictly inside noise_mask, computes
    the unbiased empirical 2D autocorrelation tapered smoothly beyond lag Wn // 2,
    and projects its Fourier spectrum onto the non-negative cone so g^T Sigma_d g >= 0
    for every candidate shape g on (Wn, Wn). Returns (1, 2 * Wn, 2 * Wn) tensor on
    device with zero-lag peak at (Wn, Wn), or None if fewer than 4 valid windows fit.
    """
    res_arr = np.asarray(residual, dtype=np.float32)
    valid_pix = np.asarray(noise_mask, dtype=bool) & np.isfinite(res_arr)
    H, W = res_arr.shape
    if H < Wn or W < Wn or not valid_pix.any():
        return None
    # Exclude primary-beam zero-blanked regions when tclean pblimit > 0 writes exact zeros.
    if float((res_arr[valid_pix] == 0.0).mean()) > 0.05:
        valid_pix = valid_pix & (res_arr != 0.0)

    # Exact integral-image check: every pixel in [r0:r0 + Wn, c0:c0 + Wn] must be valid.
    cs = np.pad(valid_pix.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    box = cs[Wn:, Wn:] - cs[:-Wn, Wn:] - cs[Wn:, :-Wn] + cs[:-Wn, :-Wn]
    ok_top_left = box == (Wn * Wn)
    total_valid = int(ok_top_left.sum())
    if total_valid < 4:
        return None

    # Prefer half-overlapping grid stride so windows span the full empty-sky area;
    # fall back to finer strides when the mask has tight geometry.
    coords = None
    for stride in (max(1, Wn // 2), max(1, Wn // 4), 1):
        sub = ok_top_left[::stride, ::stride]
        rr, cc = np.nonzero(sub)
        if len(rr) >= min(max_windows, 4):
            coords = np.stack([rr * stride, cc * stride], axis=1)
            break
    if coords is None:
        rr, cc = np.nonzero(ok_top_left)
        coords = np.stack([rr, cc], axis=1)
    if len(coords) < 4:
        return None
    if len(coords) > max_windows:
        sel = np.linspace(0, len(coords) - 1, max_windows, dtype=int)
        coords = coords[sel]

    dev = torch.device(device)
    windows = np.stack([res_arr[r0:r0 + Wn, c0:c0 + Wn] for r0, c0 in coords], axis=0)
    nw = torch.from_numpy(windows).to(device=dev, dtype=torch.float32)
    nw = nw - nw.mean()

    M = 2 * Wn
    F_nw = torch.fft.rfft2(nw, s=(M, M))
    raw_cov = torch.fft.irfft2(F_nw.abs().pow(2).mean(0), s=(M, M))
    ones = torch.ones((Wn, Wn), device=dev, dtype=torch.float32)
    n_ov = torch.fft.irfft2(torch.fft.rfft2(ones, s=(M, M)).abs().pow(2), s=(M, M)).clamp_min(1.0)
    cov_centred = torch.fft.fftshift(raw_cov / n_ov, dim=(-2, -1))

    # Flat-top cosine (Tukey) lag taper: 1 on |tau| <= Wn // 2, smooth to 0 at |tau| = Wn.
    tau = (torch.arange(M, device=dev, dtype=torch.float32) - float(Wn)).abs()
    half_lag = float(Wn // 2)
    denom = max(float(Wn) - half_lag, 1.0)
    t1d = torch.where(
        tau <= half_lag,
        torch.ones_like(tau),
        torch.where(
            tau >= float(Wn),
            torch.zeros_like(tau),
            0.5 * (1.0 + torch.cos(math.pi * (tau - half_lag) / denom)),
        ),
    )
    cov_tapered = cov_centred * (t1d[:, None] * t1d[None, :])

    spec = torch.fft.fft2(torch.fft.ifftshift(cov_tapered, dim=(-2, -1))).real.clamp_min(0.0)
    c_psd = torch.fft.fftshift(torch.fft.ifft2(spec).real, dim=(-2, -1))
    c00 = float(cov_tapered[Wn, Wn])
    p00 = float(c_psd[Wn, Wn])
    if c00 > 0.0 and p00 > 0.0:
        c_psd = c_psd * (c00 / p00)
    return c_psd.unsqueeze(0)


estimate_window_noise_psd = estimate_noise_cov_2w


def window_noise_var(cand: torch.Tensor, cov_2w: torch.Tensor) -> torch.Tensor:
    """Quadratic form g^T Sigma_d g for (N, Wn, Wn) candidates using (1, 2*Wn, 2*Wn) C_psd."""
    if cand.numel() == 0:
        return torch.zeros(0, device=cand.device, dtype=cand.dtype)
    c = cov_2w.to(device=cand.device, dtype=cand.dtype).expand(cand.shape[0], -1, -1)
    return (cand * fft_convolve_same(cand, c)).sum((1, 2)).clamp_min(0.0)


def effective_beam_dofs(cand: torch.Tensor, beam_area: float) -> torch.Tensor:
    """Extra beam degrees of freedom max(0, Area_eff(g_+) / beam_area - 1) per candidate."""
    g_pos = cand.clamp_min(0.0)
    l1_sq = g_pos.sum((1, 2)).pow(2)
    l2_sq = g_pos.pow(2).sum((1, 2)).clamp_min(1e-30)
    area_eff = torch.where(l1_sq > 0, l1_sq / l2_sq, torch.zeros_like(l1_sq))
    return (area_eff / max(float(beam_area), 1.0) - 1.0).clamp_min(0.0)


# ---------------------------------------------------------------------------
# Sky generator
# ---------------------------------------------------------------------------

class SkyGenerator:
    def __init__(self, psf_bank, agn_flux, a, device, seed):
        self.dev = torch.device(device)
        self.psf_bank = psf_bank.to(self.dev).float()
        self.beam = torch.tensor([beam_sigma_px(p.cpu().numpy()) for p in psf_bank], device=self.dev)
        if agn_flux is None:
            self.agn = torch.zeros(0, dtype=torch.float32, device=self.dev)
        else:
            self.agn = torch.as_tensor(agn_flux, dtype=torch.float32, device=self.dev).flatten()
        self.a = a
        default_agn = 0.0 if len(self.agn) == 0 else 25.16
        for key, val in (
            ("patch", 256), ("ctx", 256), ("shape_size", 64), ("n_extra", 4),
            ("extra_flux", [0.1, 10.0]), ("crop_floor", 0.1),
            ("diam_range", [32.0, 256.0]), ("q_range", [0.3, 1.0]),
            ("lump_max", 0.6), ("gauss_sigma", [0.5, 42.0]),
            ("ring_width_beams", 1.5), ("agn_mean", default_agn),
            ("snr_range", [5.0, 1000.0]), ("ld_clean_prob", 0.0),
        ):
            if not hasattr(a, key):
                setattr(a, key, val)
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)

    def _u(self, lo, hi, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.dev) * (hi - lo) + lo

    def _logu(self, lo, hi, *shape):
        return torch.exp(self._u(math.log(lo), math.log(hi), *shape))

    def _psf_shaped_noise(self, psf_cut):
        """Unit-variance noise with the PSF as autocorrelation (as GPUCutoutGenerator)."""
        B, c, _ = psf_cut.shape
        n = 2 * c
        pad = torch.zeros(B, n, n, device=self.dev)
        pad[:, c // 2:c // 2 + c, c // 2:c // 2 + c] = psf_cut
        spec = torch.fft.fft2(torch.fft.ifftshift(pad, dim=(-2, -1))).real.clamp_min(0.0)
        w = torch.randn(B, n, n, generator=self.gen, device=self.dev)
        x = torch.fft.ifft2(torch.fft.fft2(w) * spec.sqrt()).real
        x = x / spec.mean(dim=(-2, -1)).clamp_min(1e-30).sqrt().view(B, 1, 1)
        return x[:, c // 2:c // 2 + c, c // 2:c // 2 + c]

    def _extended(self, kind, beam):
        a, dev, S = self.a, self.dev, self.a.patch
        B = kind.shape[0]
        yy, xx = torch.meshgrid(torch.arange(S, device=dev, dtype=torch.float32),
                                torch.arange(S, device=dev, dtype=torch.float32), indexing="ij")
        R = (self._logu(*a.diam_range, B) / 2).view(B, 1, 1)
        q = self._u(*a.q_range, B).view(B, 1, 1)
        lump = (a.lump_max * torch.sqrt(self._u(0.0, 1.0, B))).view(B, 1, 1)
        pa = self._u(0.0, math.pi, B).view(B, 1, 1)
        c, s = torch.cos(pa), torch.sin(pa)
        up = (xx - S / 2) * c + (yy - S / 2) * s          # px along the major axis
        vp = (-(xx - S / 2) * s + (yy - S / 2) * c) / q   # px, minor axis stretched to circular
        rho = torch.sqrt(up ** 2 + vp ** 2)

        # Fills inside a lumpy envelope.
        fk = kind.clamp(max=len(FILLS) - 1)
        ridge = torch.tensor([f[0] == "ridge" for f in FILLS], device=dev)[fk].view(B, 1, 1)
        alpha = torch.tensor([f[1] for f in FILLS], device=dev)[fk]
        par = torch.tensor([f[2] for f in FILLS], device=dev)[fk].view(B, 1, 1)
        g = grf(alpha, S, self.gen, dev)
        fill = torch.where(ridge, torch.exp(-(g / par) ** 2), torch.exp(par * g))
        g5 = grf(torch.full((B,), 5.0, device=dev), S, self.gen, dev)
        env = torch.exp(-((rho / R) * torch.exp(-lump * g5)) ** 4)
        sky = fill * env

        # Ring: radial Gaussian wall of sigma ring_width_beams x beam sigma at radius R.
        w = (a.ring_width_beams * beam).view(B, 1, 1)
        sky = torch.where((kind == 5).view(B, 1, 1), torch.exp(-0.5 * ((rho - R) / w) ** 2), sky)
        # Gaussian.
        smaj = self._u(*a.gauss_sigma, B).view(B, 1, 1)
        smin = a.gauss_sigma[0] + torch.rand(B, 1, 1, generator=self.gen, device=dev) * (smaj - a.gauss_sigma[0])
        gau = torch.exp(-0.5 * ((up / smaj) ** 2 + (vp * q / smin) ** 2))
        sky = torch.where((kind == 6).view(B, 1, 1), gau, sky)
        # Point at the centre pixel.
        pt = torch.zeros_like(sky)
        pt[:, S // 2, S // 2] = 1.0
        sky = torch.where((kind == 7).view(B, 1, 1), pt, sky)
        return sky / sky.flatten(1).sum(1).clamp_min(1e-30).view(B, 1, 1)

    def _centred_psf(self, psfs: torch.Tensor, size: int) -> torch.Tensor:
        """Crop or zero-pad (B, P, P) PSFs around (P // 2, P // 2) to (B, size, size)."""
        B, P, _ = psfs.shape
        if P == size:
            return psfs
        if P > size:
            c0 = (P - size) // 2
            return psfs[:, c0:c0 + size, c0:c0 + size]
        p0 = (size - P) // 2
        p1 = size - P - p0
        return F.pad(psfs, (p0, p1, p0, p1))

    def sample(self, B, kind_override=None):
        if self.a.variant == "LD":
            return self.sample_ld(B, kind_override=kind_override)
        a, dev, S = self.a, self.dev, self.a.patch
        idx = torch.randint(0, len(self.psf_bank), (B,), generator=self.gen, device=dev)
        psfs = self.psf_bank[idx]
        psf_cut = self._centred_psf(psfs, S)
        if getattr(a, "kinds", None):
            allowed = torch.tensor([KINDS.index(k) for k in a.kinds], device=dev)
            kind = allowed[torch.randint(0, len(allowed), (B,), generator=self.gen, device=dev)]
        else:
            kind = torch.randint(0, len(KINDS), (B,), generator=self.gen, device=dev)
        if kind_override is not None:
            kind = torch.full_like(kind, KINDS.index(kind_override))
        ext = self._extended(kind, self.beam[idx])
        ext_peak = fft_convolve_same(ext, psfs).flatten(1).max(1).values.clamp_min(1e-30)

        # AGN points: Poisson count, T-RECS fluxes, brightest point's dirty peak = ext_peak.
        agn_mean = float(getattr(a, "agn_mean", 0.0))
        if len(self.agn) > 0:
            n = torch.poisson(torch.full((B,), agn_mean, device=dev), generator=self.gen).long().clamp(max=MAX_AGN)
            valid = torch.arange(MAX_AGN, device=dev).view(1, -1) < n.view(-1, 1)
            fl = self.agn[torch.randint(0, len(self.agn), (B, MAX_AGN), generator=self.gen, device=dev)] * valid
            fl = fl / fl.max(1, keepdim=True).values.clamp_min(1e-30) * ext_peak.view(B, 1)
            pix = torch.randint(0, S * S, (B, MAX_AGN), generator=self.gen, device=dev)
            pts = torch.zeros(B, S * S, device=dev).scatter_add_(1, pix, fl).view(B, S, S)
            sky = ext + pts
        else:
            n = torch.zeros(B, dtype=torch.long, device=dev)
            sky = ext

        dirty = fft_convolve_same(sky, psfs)
        snr = self._logu(*a.snr_range, B)
        sigma = dirty.flatten(1).max(1).values.clamp_min(1e-30) / snr
        img = dirty + sigma.view(B, 1, 1) * self._psf_shaped_noise(psf_cut)
        scale = img.flatten(1).abs().max(1).values.clamp_min(1e-30).view(B, 1, 1)
        if a.variant == "P":
            return self._shapes(sky, kind, n)
        return {"img": img / scale, "psf_cut": psf_cut, "psf": psfs, "sky": sky / scale,
                "sigma": sigma / scale.view(B), "snr": snr, "kind": kind, "n_agn": n}

    def _crop(self, x, cy, cx, size):
        """(B, H, W) -> (B, size, size) centred on (cy, cx) per batch element (must fit)."""
        B, h = x.shape[0], size // 2
        off = torch.arange(size, device=self.dev) - h
        rows, cols = cy.view(B, 1) + off, cx.view(B, 1) + off
        return x[torch.arange(B, device=self.dev).view(B, 1, 1), rows.unsqueeze(2), cols.unsqueeze(1)]

    def sample_ld(self, B, kind_override=None):
        """LD training pair (plan 3A.4). Field a.patch px: one extended source placed so the
        context fits, plus 0..n_extra extended sources anywhere (sidelobes leak into the context)
        and optional T-RECS AGN point distractors. Shifts use zero padding (never periodic wrap).
        With ld_clean_prob > 0, simulates mid-CLEAN residuals by subtracting partial models from
        distractors and the main source before convolving and picking the residual peak.
        Returns context residual / u, Omega residual / u, PSF over the context and 2W, the target
        shape (remaining positive sky x Hann on Omega, unit peak), sigma / u."""
        a, dev, S, C, W = self.a, self.dev, self.a.patch, self.a.ctx, self.a.shape_size
        ld_clean_prob = float(getattr(a, "ld_clean_prob", 0.0))
        agn_mean = float(getattr(a, "agn_mean", 0.0))

        idx = torch.randint(0, len(self.psf_bank), (B,), generator=self.gen, device=dev)
        psfs = self.psf_bank[idx]
        allowed = torch.tensor([KINDS.index(k) for k in a.kinds] if getattr(a, "kinds", None)
                               else list(range(len(KINDS))), device=dev)
        pick = lambda: allowed[torch.randint(0, len(allowed), (B,), generator=self.gen, device=dev)]
        kind = pick()
        if kind_override is not None:
            kind = torch.full_like(kind, KINDS.index(kind_override))

        lo, hi = C // 2 - S // 2, S // 2 - C // 2   # shifts that keep the main source's centre in the central region
        if lo > hi:
            lo, hi = 0, 0
        sh = torch.randint(lo, hi + 1, (B, 2), generator=self.gen, device=dev)
        main_src = torch.stack([_shift_zero_pad(s, int(dy), int(dx))
                                for s, (dy, dx) in zip(self._extended(kind, self.beam[idx]), sh)])

        n = torch.randint(0, a.n_extra + 1, (B,), generator=self.gen, device=dev)
        ext_sum = torch.zeros_like(main_src)
        ext_clean = torch.zeros_like(main_src)
        for j in range(a.n_extra):
            ext = self._extended(pick(), self.beam[idx]) * self._logu(*a.extra_flux, B).view(B, 1, 1)
            shj = torch.randint(-S // 2, S // 2, (B, 2), generator=self.gen, device=dev)
            ext = torch.stack([_shift_zero_pad(s, int(dy), int(dx)) for s, (dy, dx) in zip(ext, shj)])
            ext_j = ext * (j < n).float().view(B, 1, 1)
            ext_sum = ext_sum + ext_j
            if ld_clean_prob > 0.0:
                ext_clean = ext_clean + ext_j * self._u(0.3, 1.08, B, 1, 1)

        sky_ext = main_src + ext_sum
        pts = torch.zeros_like(main_src)
        pts_clean = torch.zeros_like(main_src)
        if agn_mean > 0.0 and len(self.agn) > 0:
            ext_peak = fft_convolve_same(sky_ext, psfs).flatten(1).max(1).values.clamp_min(1e-30)
            n_pts = torch.poisson(torch.full((B,), agn_mean, device=dev), generator=self.gen).long().clamp(max=MAX_AGN)
            valid = torch.arange(MAX_AGN, device=dev).view(1, -1) < n_pts.view(-1, 1)
            fl = self.agn[torch.randint(0, len(self.agn), (B, MAX_AGN), generator=self.gen, device=dev)] * valid
            fl = fl / fl.max(1, keepdim=True).values.clamp_min(1e-30) * ext_peak.view(B, 1)
            pix = torch.randint(0, S * S, (B, MAX_AGN), generator=self.gen, device=dev)
            pts = torch.zeros(B, S * S, device=dev).scatter_add_(1, pix, fl).view(B, S, S)
            if ld_clean_prob > 0.0:
                f_pt = self._u(0.3, 1.08, B, MAX_AGN)
                pts_clean = torch.zeros(B, S * S, device=dev).scatter_add_(1, pix, fl * f_pt).view(B, S, S)

        sky = sky_ext + pts
        use_clean = torch.zeros(B, dtype=torch.bool, device=dev)
        if ld_clean_prob > 0.0:
            do_clean = torch.rand(B, generator=self.gen, device=dev) < ld_clean_prob
            f_main = self._u(0.0, 0.7, B, 1, 1)
            main_blur = F.avg_pool2d(main_src[:, None], kernel_size=5, stride=1, padding=2)[:, 0]
            p_norm = (main_blur / main_blur.flatten(1).max(1).values.clamp_min(1e-30).view(B, 1, 1)).clamp(0.0, 1.0)
            main_clean = f_main * main_src * (0.5 + 0.5 * p_norm)
            m_cand = main_clean + ext_clean + pts_clean
            rem_pos_max = (sky - m_cand).clamp_min(0.0).flatten(1).max(1).values
            use_clean = do_clean & (rem_pos_max > 0.0)
            m_clean = torch.where(use_clean.view(B, 1, 1), m_cand, torch.zeros_like(sky))
            sky_rem = sky - m_clean
            sky_pos = sky_rem.clamp_min(0.0)
        else:
            sky_rem = sky
            sky_pos = sky

        dirty = fft_convolve_same(sky_rem, psfs)
        h = C // 2
        snr = self._logu(*a.snr_range, B)
        dirty_peak = dirty.flatten(1).max(1).values
        dirty_scale = torch.where(dirty_peak > 0, dirty_peak, dirty.flatten(1).abs().max(1).values).clamp_min(1e-30)
        sigma = dirty_scale / snr
        psf_noise = self._centred_psf(psfs, S)
        img = dirty + sigma.view(B, 1, 1) * self._psf_shaped_noise(psf_noise)
        i = img.flatten(1).argmax(1)   # global peak, as the loop; zero padding past the field edge
        cy, cx = i // S + h, i % S + h
        img_p = F.pad(img, (h, h, h, h))
        sky_pos_p = F.pad(sky_pos, (h, h, h, h))
        ctx = self._crop(img_p, cy, cx, C)
        u = ctx.flatten(1).abs().max(1).values.clamp_min(1e-30).view(B, 1, 1)
        hw = torch.hann_window(W, periodic=False, device=dev)
        taper2d = hw[:, None] * hw[None, :]
        g = self._crop(sky_pos_p, cy, cx, W) * taper2d
        if ld_clean_prob > 0.0:
            sky_uncl_p = F.pad(sky, (h, h, h, h))
            g_uncl = self._crop(sky_uncl_p, cy, cx, W) * taper2d
            fallback = use_clean & (g.flatten(1).max(1).values <= 0.0)
            g = torch.where(fallback.view(B, 1, 1), g_uncl, g)
        g = g / g.flatten(1).max(1).values.clamp_min(1e-30).view(B, 1, 1)
        c0 = (C - W) // 2
        return {"sky": g, "img": ctx / u, "r_om": (ctx / u)[:, c0:c0 + W, c0:c0 + W],
                "psf_cut": self._centred_psf(psfs, C), "psf2": self._centred_psf(psfs, 2 * W),
                "sigma": sigma / u.view(B), "snr": snr, "kind": kind, "n_agn": n,
                "field": sky_rem, "dirty": img, "cy": cy - h, "cx": cx - h}

    def _shapes(self, sky, kind, n):
        """P: shape_size crop centred at a random pixel with sky > crop_floor x max (the crop
        must fit), Hann taper, unit peak. No PSF, no noise."""
        a, dev = self.a, self.dev
        B, S, _ = sky.shape
        W, h = a.shape_size, a.shape_size // 2
        ok = sky > a.crop_floor * sky.flatten(1).max(1).values.view(B, 1, 1)
        ok[:, :h], ok[:, S - h:], ok[:, :, :h], ok[:, :, S - h:] = False, False, False, False
        w = ok.flatten(1).float()
        w[w.sum(1) == 0, (S // 2) * S + S // 2] = 1.0   # nothing fits: centre
        i = torch.multinomial(w, 1, generator=self.gen)[:, 0]
        off = torch.arange(W, device=dev) - h
        rows, cols = (i // S).view(B, 1) + off, (i % S).view(B, 1) + off
        c = sky[torch.arange(B, device=dev).view(B, 1, 1), rows.unsqueeze(2), cols.unsqueeze(1)]
        hw = torch.hann_window(W, periodic=False, device=dev)
        g = c * (hw[:, None] * hw[None, :])
        g = g / g.flatten(1).max(1).values.clamp_min(1e-30).view(B, 1, 1)
        one = torch.ones(B, device=dev)
        return {"sky": g, "img": torch.zeros_like(g), "sigma": one, "snr": one, "kind": kind, "n_agn": n}


# ---------------------------------------------------------------------------
# U-Net & Context Encoder
# ---------------------------------------------------------------------------

def timestep_embedding(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class ResBlock(nn.Module):
    def __init__(self, ic, oc, edim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(math.gcd(32, ic), ic), nn.Conv2d(ic, oc, 3, padding=1)
        self.e = nn.Linear(edim, oc)
        self.n2, self.c2 = nn.GroupNorm(math.gcd(32, oc), oc), nn.Conv2d(oc, oc, 3, padding=1)
        self.skip = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x))) + self.e(e)[..., None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class LDContextStem(nn.Module):
    """Strided convolutional encoder mapping (B, 2, C, C) context (img, psf) to (B, 8, W, W)."""

    def __init__(self, ctx_factor: int = 4, out_channels: int = 8):
        super().__init__()
        if ctx_factor < 1 or (ctx_factor & (ctx_factor - 1)) != 0:
            raise ValueError(f"ctx_factor must be a positive power of 2, got {ctx_factor}")
        self.ctx_factor = ctx_factor
        self.out_channels = out_channels
        if ctx_factor == 1:
            self.net = nn.Sequential(
                nn.Conv2d(2, out_channels, 3, padding=1),
                nn.SiLU(),
            )
        else:
            n_stages = int(round(math.log2(ctx_factor)))
            layers: list[nn.Module] = []
            in_c = 2
            for s in range(n_stages):
                oc = out_channels if s == n_stages - 1 else 16
                layers.extend([
                    nn.Conv2d(in_c, oc, 3, stride=2, padding=1),
                    nn.GroupNorm(min(4, oc), oc),
                    nn.SiLU(),
                ])
                in_c = oc
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """Noise-prediction U-Net; conditioning images as extra input channels, the noise
    level t and log(sigma) of the input image through the embedding."""

    def __init__(self, in_ch, base=32, mults=(1, 2, 4, 4, 8), ld_ctx_stem=False, ctx_factor=4):
        super().__init__()
        edim = 4 * base
        self.base = base
        self.ctx_stem = LDContextStem(ctx_factor=ctx_factor) if ld_ctx_stem else None
        self.emb = nn.Sequential(nn.Linear(base + 1, edim), nn.SiLU(), nn.Linear(edim, edim))
        ch = [base * m for m in mults]
        self.inp = nn.Conv2d(in_ch, ch[0], 3, padding=1)
        self.down = nn.ModuleList()
        c = ch[0]
        for i, oc in enumerate(ch):
            self.down.append(nn.ModuleList([
                ResBlock(c, oc, edim),
                ResBlock(oc, oc, edim),
                nn.Conv2d(oc, oc, 3, stride=2, padding=1) if i < len(ch) - 1 else nn.Identity(),
            ]))
            c = oc
        self.mid = nn.ModuleList([ResBlock(c, c, edim), ResBlock(c, c, edim)])
        self.up = nn.ModuleList()
        for i, oc in reversed(list(enumerate(ch))):
            self.up.append(nn.ModuleList([ResBlock(c + oc, oc, edim), ResBlock(oc, oc, edim)]))
            c = oc
        self.out = nn.Sequential(nn.GroupNorm(math.gcd(32, c), c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t, log_sigma):
        e = self.emb(torch.cat([timestep_embedding(t, self.base), log_sigma[:, None]], -1))
        h = self.inp(x)
        skips = []
        for r1, r2, dn in self.down:
            h = r2(r1(h, e), e)
            skips.append(h)
            h = dn(h)
        for r in self.mid:
            h = r(h, e)
        for (r1, r2), s in zip(self.up, reversed(skips)):
            if h.shape[-1] != s.shape[-1]:
                h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
            h = r2(r1(torch.cat([h, s], 1), e), e)
        return self.out(h)


def unet_in_channels(a) -> int:
    """Number of input channels to UNet.inp for training args Namespace a."""
    if a.variant == "LD":
        return 11 if getattr(a, "ld_ctx_encoder", "avgpool") == "conv" else 5
    return {"P": 1, "A": 3, "B": 4}[a.variant]


def build_diffusion_net(a) -> UNet:
    """Construct the UNet matching args Namespace a (including legacy avgpool vs conv LD)."""
    ld_stem = (a.variant == "LD" and getattr(a, "ld_ctx_encoder", "avgpool") == "conv")
    f = max(1, getattr(a, "ctx", 256) // getattr(a, "shape_size", 64))
    return UNet(unet_in_channels(a), getattr(a, "base_channels", 32), ld_ctx_stem=ld_stem, ctx_factor=f)


# ---------------------------------------------------------------------------
# Diffusion & Posterior Guidance
# ---------------------------------------------------------------------------

def cosine_alpha_bar(T, dev):
    s = 0.008
    t = torch.linspace(0, T, T + 1, device=dev) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return (f[1:] / f[0]).clamp(1e-5, 0.9999)


def _cg(A, b, x, iters):
    """Batched conjugate gradient for symmetric positive definite A on (n, H, W)."""
    r = b - A(x)
    p, rs = r.clone(), (r * r).sum((1, 2), keepdim=True)
    for _ in range(iters):
        Ap = A(p)
        al = rs / (p * Ap).sum((1, 2), keepdim=True).clamp_min(1e-30)
        x, r = x + al * p, r - al * Ap
        rn = (r * r).sum((1, 2), keepdim=True)
        p, rs = r + rn / rs.clamp_min(1e-30) * p, rn
    return x


class Diffusion:
    def __init__(self, model, a, dev):
        self.m, self.a = model, a
        self.ab = cosine_alpha_bar(a.T, dev)
        for key in ("a_passes", "m_sample_steps"):   # args of checkpoints older than these flags
            if not hasattr(a, key):
                setattr(a, key, 0)
        for key, v in (
            ("gamma", 1.0), ("zero_weight", 0.0), ("zero_floor", 1e-3),
            ("lam_chi2", 0.0), ("chi2_tmax", 200), ("sample_steps", 50),
            ("ld_ctx_encoder", "avgpool"), ("ld_clean_prob", 0.0),
        ):
            if not hasattr(a, key):
                setattr(a, key, v)
        if not hasattr(a, "pred"):
            a.pred = "eps"

    def condition(self, batch, m):
        """Network conditioning for current model m (generator units). Returns
        (input image, sigma, units): A sees the residual renormalised by its own
        max, so its target sky - m is in units of that max; B sees dirty + m."""
        if self.a.variant in ("P", "LD"):
            return batch["img"], batch["sigma"], torch.ones_like(batch["sigma"]).view(-1, 1, 1)
        if self.a.variant == "A":
            res = batch["img"] - fft_convolve_same(m, batch["psf"])
            u = res.flatten(1).abs().max(1).values.clamp_min(1e-30).view(-1, 1, 1)
            return res / u, batch["sigma"] / u.view(-1), u
        return batch["img"], batch["sigma"], torch.ones_like(batch["sigma"]).view(-1, 1, 1)

    def eps(self, x_t, t, img, sigma, batch, m):
        if self.a.variant == "P":   # unconditional: x_t only, no sigma
            return self.m(x_t[:, None], t, torch.zeros_like(sigma))[:, 0]
        if self.a.variant == "LD":  # x_t, r and PSF on Omega, plus context r and PSF on C
            W, C = x_t.shape[-1], img.shape[-1]
            c0, f = (C - W) // 2, C // W
            psf = batch["psf_cut"]
            if getattr(self.m, "ctx_stem", None) is not None:
                ctx_feat = self.m.ctx_stem(torch.stack([img, psf], 1))
                if ctx_feat.shape[-2:] != (W, W):
                    ctx_feat = F.interpolate(ctx_feat, size=(W, W), mode="bilinear", align_corners=False)
                ch = torch.cat([
                    torch.stack([x_t, img[:, c0:c0 + W, c0:c0 + W], psf[:, c0:c0 + W, c0:c0 + W]], 1),
                    ctx_feat,
                ], dim=1)
                return self.m(ch, t, torch.log(sigma))[:, 0]
            ch = [x_t, img[:, c0:c0 + W, c0:c0 + W], psf[:, c0:c0 + W, c0:c0 + W],
                  F.avg_pool2d(img[:, None], f)[:, 0], F.avg_pool2d(psf[:, None], f)[:, 0]]
            return self.m(torch.stack(ch, 1), t, torch.log(sigma))[:, 0]
        ch = [x_t[:, None], img[:, None], batch["psf_cut"][:, None]]
        if self.a.variant == "B":
            ch.append(m[:, None] * self.a.sky_scale)
        return self.m(torch.cat(ch, 1), t, torch.log(sigma))[:, 0]

    def split(self, x_t, t, out):
        """Network output -> (x0, eps) for --pred eps or v."""
        ab = self.ab[t].view(-1, 1, 1)
        if self.a.pred == "v":
            return ab.sqrt() * x_t - (1 - ab).sqrt() * out, (1 - ab).sqrt() * x_t + ab.sqrt() * out
        return (x_t - (1 - ab).sqrt() * out) / ab.sqrt(), out

    def x0_from_eps(self, x_t, t, e):
        return self.split(x_t, t, e)[0]

    def _noised(self, x0, t, gen=None):
        ab = self.ab[t].view(-1, 1, 1)
        n = torch.randn(x0.shape, device=x0.device, generator=gen)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * n, n

    def target(self, batch, m, u):
        if self.a.variant in ("P", "LD") and self.a.gamma != 1.0:
            return batch["sky"].clamp_min(0) ** (1.0 / self.a.gamma)
        return (batch["sky"] - m) / u if self.a.variant == "A" else batch["sky"]

    def own_passes(self, batch, k, gen=None):
        """m after k[i] passes of the network's own draw, no gradient (A: m + draw, B: draw)."""
        m = torch.zeros_like(batch["sky"])
        for j in range(1, int(k.max()) + 1 if len(k) else 1):
            idx = (k >= j).nonzero()[:, 0]
            sub = {n: v[idx] for n, v in batch.items()}
            m[idx] = self.sample(sub, m=m[idx], steps=self.a.m_sample_steps or None, gen=gen)
        return m

    def loss(self, batch, gen=None, k=None):
        """gen, k given: deterministic (validation). k = own passes per patch."""
        a = self.a
        sky_ref = batch["sky"] if "sky" in batch else batch["shape"]
        B = sky_ref.shape[0]
        dev = sky_ref.device
        m = torch.zeros_like(sky_ref)
        passes = a.a_passes
        if passes > 0:
            if k is None:
                k = torch.randint(0, passes + 1, (B,), device=dev, generator=gen)
            m = self.own_passes(batch, k, gen)
        p_sc = a.self_cond_p if a.variant == "B" and passes == 0 else 0.0  # A: never; B: off with own passes
        use = torch.rand(B, device=dev, generator=gen) < p_sc
        if use.any():
            # B only: own one-step estimate of the sky from m = 0.
            with torch.no_grad():
                img0, sig0, u0 = self.condition(batch, m)
                t2 = torch.randint(0, a.T, (B,), device=dev, generator=gen)
                x2, _ = self._noised(self.target(batch, m, u0) * a.sky_scale, t2, gen)
                est = self.x0_from_eps(x2, t2, self.eps(x2, t2, img0, sig0, batch, m)) / a.sky_scale * u0
            m = torch.where(use.view(B, 1, 1), est.float(), m)
        img, sig, u = self.condition(batch, m)
        x0 = self.target(batch, m, u) * a.sky_scale
        t = torch.randint(0, a.T, (B,), device=dev, generator=gen)
        x_t, noise = self._noised(x0, t, gen)
        if a.pred == "v":
            ab = self.ab[t].view(-1, 1, 1)
            noise = ab.sqrt() * noise - (1 - ab).sqrt() * x0
        out = self.eps(x_t, t, img, sig, batch, m).float()
        if a.variant in ("P", "LD") and a.zero_weight > 0:
            sky = sky_ref
            w = 1.0 + a.zero_weight * (sky < a.zero_floor * sky.flatten(1).max(1).values.view(-1, 1, 1)).float()
            loss = (w * (out - noise) ** 2).mean()
        else:
            loss = F.mse_loss(out, noise)
        if a.variant == "LD" and a.lam_chi2 > 0:
            use = t < a.chi2_tmax
            if use.any():
                g = self.split(x_t, t, out)[0][use] / a.sky_scale
                r, P = batch["r_om"][use], batch["psf2"][use]
                num = (g * r).sum((1, 2)).clamp_min(0)
                den = (g * fft_convolve_same(g, P)).sum((1, 2)).clamp_min(1e-30)
                W = r.shape[-1]
                jd = (r[:, W // 2, W // 2] ** 2 / 2).clamp_min(1e-30)
                loss = loss - a.lam_chi2 * (num ** 2 / (2 * den) / jd).mean() * use.float().mean()
        return loss

    @torch.no_grad()
    def sample(self, batch, m=None, steps=None, gen=None, guide=None, clamp0=False):
        """DDIM (eta 0). Returns the full sky estimate in generator units (A: m + draw).

        guide: likelihood guidance (DPS-like, no gradient through the network).
        At each step the network's x0 estimate s is moved along its own residual
        r = img - PSF * s by steepest descent with exact line search on
        1/2 s^T B s - s^T img: s <- s + guide * a * r, a = <r, r> / <r, B r>
        (guide 1 = the exact minimiser along r), then eps is recomputed from
        x_t and the new x0 so the DDIM update stays consistent. The step is clipped so the
        sky stays >= 0 (A: m + u s >= 0). None = off."""
        a = self.a
        B = batch["img"].shape[0]
        S = a.shape_size if a.variant == "LD" else batch["img"].shape[-1]
        dev = batch["img"].device
        m = torch.zeros(B, S, S, device=dev) if m is None else m
        img, sig, u = self.condition(batch, m)
        ts = torch.linspace(a.T - 1, 0, steps or a.sample_steps, device=dev).long()
        x = torch.randn(B, S, S, device=dev, generator=gen)
        for i, t in enumerate(ts):
            tt = t.repeat(B)
            x0, e = self.split(x, tt, self.eps(x, tt, img, sig, batch, m))
            if guide is not None:
                s = x0.float() / a.sky_scale
                r = img - fft_convolve_same(s, batch["psf"])
                Br = fft_convolve_same(r, batch["psf"])
                step = ((r * r).flatten(1).sum(1) / (r * Br).flatten(1).sum(1).clamp_min(1e-30)).view(-1, 1, 1)
                s = s + guide * step * r
                s = torch.maximum(s, -m / u) if a.variant == "A" else s.clamp_min(0)   # sky m + u s >= 0
                x0 = (s * a.sky_scale).to(x0.dtype)
                ab = self.ab[t]
                e = (x - ab.sqrt() * x0) / (1 - ab).sqrt().clamp_min(1e-8)
            if clamp0:   # P/LD test: x0 >= 0 at every step, eps made consistent
                x0 = x0.clamp_min(0)
                ab = self.ab[t]
                e = (x - ab.sqrt() * x0) / (1 - ab).sqrt().clamp_min(1e-8)
            if i + 1 < len(ts):
                abn = self.ab[ts[i + 1]]
                x = abn.sqrt() * x0 + (1 - abn).sqrt() * e
            else:
                x = x0
        x = x.float() / a.sky_scale
        if a.variant in ("P", "LD") and a.gamma != 1.0:
            x = x.clamp_min(0) ** a.gamma
        return m + x * u if a.variant == "A" else x

    def _q_space_cg_step(self, x0, ab, R, Bop, sig2, w, cg_iters, data_var, allow_probe):
        """One Gauss-Newton CG update directly in q-space with taper w inside g(q)."""
        a = self.a
        n = x0.shape[0]
        s = (1 - ab) / ab
        rho2 = data_var * s / (data_var + s) / a.sky_scale ** 2
        q0 = x0.float() / a.sky_scale
        phi0 = F.softplus(q0, beta=12.0)
        g0 = w * (phi0 ** a.gamma if a.gamma != 1.0 else phi0)
        dphi = torch.sigmoid(12.0 * q0)
        D = w * (a.gamma * phi0.clamp_min(1e-6) ** (a.gamma - 1.0) * dphi if a.gamma != 1.0 else dphi)
        ast = ((g0 * R).sum((1, 2)) / (g0 * Bop(g0)).sum((1, 2)).clamp_min(1e-30)).clamp_min(0).view(n, 1, 1)
        if allow_probe:
            g_probe = w * R.clamp_min(0)
            ast_probe = (
                (g_probe * R).sum((1, 2)) / (g_probe * Bop(g_probe)).sum((1, 2)).clamp_min(1e-30)
            ).clamp_min(0).view(n, 1, 1)
            ast = torch.where(ast > 0, ast, 0.5 * ast_probe)
        A_op = lambda v: v / rho2 + (ast ** 2 / sig2) * D * Bop(D * v)
        rhs = (ast / sig2) * D * (R - ast * Bop(g0))
        dq = _cg(A_op, rhs, torch.zeros_like(q0), cg_iters)
        q = q0 + dq
        return (q * a.sky_scale).to(x0.dtype)

    @torch.no_grad()
    def sample_shape_posterior(
        self,
        r,
        psf2,
        sigma,
        n,
        steps=None,
        gen=None,
        cg_iters=5,
        data_var=0.25,
        q_space=False,
        taper=False,
    ):
        """P: n draws of a shape g (W x W) from p(g) exp(J(g) / sigma^2), J(g) = <g, r>^2 / (2 <g, B g>)
        (plan-diffusion-local-proposal.md 3B.4). r: (W, W) residual window; psf2: (2W, 2W) PSF,
        peak at (W, W), so B g on the window is exact; sigma: noise of r, same units.

        When q_space=False (default), runs the legacy sky-space CG update so existing checkpoints
        reproduce bit-for-bit. When q_space=True, runs Gauss-Newton CG directly in network q-space
        with smooth non-negative stretch softplus(q, beta=12)^gamma and optional Hann taper w
        inside the forward operator (preventing CG from placing flux on window borders)."""
        a = self.a
        if a.variant != "P":
            raise ValueError("sample_shape_posterior needs a variant P checkpoint")
        dev, W = r.device, r.shape[-1]
        R = r.float().expand(n, W, W)
        P = psf2.float().expand(n, -1, -1)
        Bop = lambda g: fft_convolve_same(g, P)
        zero, one = torch.zeros(n, W, W, device=dev), torch.ones(n, device=dev)
        ts = torch.linspace(a.T - 1, 0, steps or a.sample_steps, device=dev).long()
        x = torch.randn(n, W, W, device=dev, generator=gen)

        if not q_space:
            for i, t in enumerate(ts):
                x0, _ = self.split(x, t.repeat(n), self.eps(x, t.repeat(n), zero, one, None, None))
                ab = self.ab[t]
                s = (1 - ab) / ab
                rho2 = data_var * s / (data_var + s) / a.sky_scale ** 2
                q0 = x0.float() / a.sky_scale             # network space (sky^(1/gamma) if gamma != 1)
                g0 = q0.clamp_min(0) ** a.gamma if a.gamma != 1.0 else q0
                ast = ((g0 * R).sum((1, 2)) / (g0 * Bop(g0)).sum((1, 2)).clamp_min(1e-30)).clamp_min(0).view(n, 1, 1)
                if a.gamma != 1.0:   # prior uncertainty carried to sky space: d sky = gamma q^(gamma-1) d q,
                    # floored at the gamma 1 value so pixels where the draw is 0 can still take flux
                    rho2 = torch.maximum(rho2 * (a.gamma * q0.clamp_min(0) ** (a.gamma - 1)) ** 2, rho2)
                A = lambda v: v / rho2 + ast ** 2 * Bop(v) / sigma ** 2
                g = _cg(A, g0 / rho2 + ast * R / sigma ** 2, g0, cg_iters)
                q = g.clamp_min(0) ** (1.0 / a.gamma) if a.gamma != 1.0 else g
                x0 = q * a.sky_scale
                e = (x - ab.sqrt() * x0) / (1 - ab).sqrt().clamp_min(1e-8)
                if i + 1 < len(ts):
                    abn = self.ab[ts[i + 1]]
                    x = abn.sqrt() * x0 + (1 - abn).sqrt() * e
                else:
                    x = x0
            x = x.float() / a.sky_scale
            out = x.clamp_min(0) ** a.gamma if a.gamma != 1.0 else x
            if taper:
                hw = torch.hann_window(W, periodic=False, device=dev)
                out = out * (hw[:, None] * hw[None, :])
            return out

        if taper:
            hw = torch.hann_window(W, periodic=False, device=dev)
            w = hw[:, None] * hw[None, :]
        else:
            w = torch.ones((W, W), device=dev, dtype=torch.float32)
        sig_t = torch.as_tensor(sigma, dtype=torch.float32, device=dev)
        if sig_t.ndim > 0:
            sig_t = sig_t.view(-1, 1, 1)
        sig2 = (sig_t ** 2).clamp_min(1e-30)

        for i, t in enumerate(ts):
            x0, _ = self.split(x, t.repeat(n), self.eps(x, t.repeat(n), zero, one, None, None))
            ab = self.ab[t]
            if cg_iters > 0:
                x0 = self._q_space_cg_step(x0, ab, R, Bop, sig2, w, cg_iters, data_var, i < len(ts) // 2)
            e = (x - ab.sqrt() * x0) / (1 - ab).sqrt().clamp_min(1e-8)
            if i + 1 < len(ts):
                abn = self.ab[ts[i + 1]]
                x = abn.sqrt() * x0 + (1 - abn).sqrt() * e
            else:
                x = x0
        phi_f = F.softplus(x.float() / a.sky_scale, beta=12.0)
        return w * (phi_f ** a.gamma if a.gamma != 1.0 else phi_f)

    @torch.no_grad()
    def sample_ld_posterior(
        self,
        batch,
        psf2,
        r_om,
        sigma,
        steps=None,
        gen=None,
        cg_iters=0,
        data_var=0.25,
        taper=True,
    ):
        """LD: DDIM draws conditioned on batch, optionally refined at each step by q-space
        Gauss-Newton CG against (r_om, psf2, sigma) when cg_iters > 0."""
        a = self.a
        if a.variant != "LD":
            raise ValueError("sample_ld_posterior needs a variant LD checkpoint")
        if cg_iters <= 0:
            return self.sample(batch, steps=steps, gen=gen)

        B = batch["img"].shape[0]
        W = a.shape_size
        dev = batch["img"].device
        m = torch.zeros(B, W, W, device=dev)
        img, sig_cond, _ = self.condition(batch, m)
        R = r_om.float().unsqueeze(0).expand(B, W, W) if r_om.ndim == 2 else r_om.float().expand(B, W, W)
        P = psf2.float().unsqueeze(0).expand(B, -1, -1) if psf2.ndim == 2 else psf2.float().expand(B, -1, -1)
        Bop = lambda g: fft_convolve_same(g, P)
        if taper:
            hw = torch.hann_window(W, periodic=False, device=dev)
            w = hw[:, None] * hw[None, :]
        else:
            w = torch.ones((W, W), device=dev, dtype=torch.float32)
        sig_t = torch.as_tensor(sigma, dtype=torch.float32, device=dev)
        if sig_t.ndim > 0:
            sig_t = sig_t.view(-1, 1, 1)
        sig2 = (sig_t ** 2).clamp_min(1e-30)

        ts = torch.linspace(a.T - 1, 0, steps or a.sample_steps, device=dev).long()
        x = torch.randn(B, W, W, device=dev, generator=gen)
        for i, t in enumerate(ts):
            tt = t.repeat(B)
            x0, _ = self.split(x, tt, self.eps(x, tt, img, sig_cond, batch, m))
            ab = self.ab[t]
            x0 = self._q_space_cg_step(x0, ab, R, Bop, sig2, w, cg_iters, data_var, i < len(ts) // 2)
            e = (x - ab.sqrt() * x0) / (1 - ab).sqrt().clamp_min(1e-8)
            if i + 1 < len(ts):
                abn = self.ab[ts[i + 1]]
                x = abn.sqrt() * x0 + (1 - abn).sqrt() * e
            else:
                x = x0
        phi_f = F.softplus(x.float() / a.sky_scale, beta=12.0)
        return w * (phi_f ** a.gamma if a.gamma != 1.0 else phi_f)


__all__ = [
    "FILLS",
    "KINDS",
    "MAX_AGN",
    "fft_convolve_same",
    "crop_psf_centred",
    "_shift_zero_pad",
    "grf",
    "beam_sigma_px",
    "beam_area_px",
    "estimate_noise_cov_2w",
    "estimate_window_noise_psd",
    "window_noise_var",
    "effective_beam_dofs",
    "SkyGenerator",
    "timestep_embedding",
    "ResBlock",
    "LDContextStem",
    "UNet",
    "unet_in_channels",
    "build_diffusion_net",
    "cosine_alpha_bar",
    "Diffusion",
    "_cg",
]
