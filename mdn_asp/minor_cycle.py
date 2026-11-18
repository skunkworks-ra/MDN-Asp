"""MDN-Asp minor cycle.

Public API
----------
render_aspen(aspen6d, shape, cx, cy) -> np.ndarray
    Place an Aspen Gaussian into a model image of given shape.

minor_cycle(residual, psf, sigma, config_idx, model, ...) -> (model_update, aspen_list)
    Run the iterative MDN-Asp minor cycle and return the model image update
    plus the list of committed (aspen6d, posterior) pairs.

The caller (CASA or a test harness) owns:
  - adding model_update into its running model image
  - major-cycle PSF convolution and vis-domain subtract
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.signal import fftconvolve

from mdn_asp.constants import BEAM_SIGMA_PX, standardise_log_flux
from mdn_asp.models.mdn_asp import MDNAsp, MixParams, make_cond

__all__ = [
    "render_aspen",
    "minor_cycle",
    "minor_cycle_batched",
    "AspenCommit",
    "psf_convolve",
    "PSFConvolver",
    "_canonical_axes",
    "_extract_psf_main_lobe",
    "measure_psf_sidelobe",
    "measure_psf_sidelobe_first",
]

_CUTOUT = 128


def psf_convolve(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Linear convolution with the PSF peak at (h//2, w//2), CASA's convention.

    scipy's ``mode="same"`` crops at ``(h-1)//2``, so on an even-sized PSF the
    response lands one pixel down-right of the source (measured 2026-09-12:
    delta at (64,64) with a 128 PSF peaks at (65,65); same on the 480 and 1280
    CASA PSFs).
    """
    h, w = psf.shape
    H, W = image.shape
    full = fftconvolve(image, psf, mode="full")
    return full[h // 2:h // 2 + H, w // 2:w // 2 + W]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class AspenCommit:
    """One committed Aspen component."""
    cx:       float        # absolute column position in full image
    cy:       float        # absolute row position in full image
    flux:     float        # committed flux (Jy), after loop_gain scaling
    sig_maj:  float        # major-axis sigma (px)
    sig_min:  float        # minor-axis sigma (px)
    pa:       float        # position angle (rad)
    params:   MixParams    # raw MDN mixture parameters (before mode selection)
    # Cutout scale used to normalise this component's input, in Jy. 1.0 in
    # absolute-flux mode. Anything decoding ``params[..., 2]`` (the flux
    # dimension) afterwards -- the posterior uncertainty image, notably -- must
    # apply the SAME inverse the mode readout does, or it decodes a
    # dimensionless log-ratio as an absolute log-flux.
    cut_scale: float = 1.0
    # Recorded for the component logs only; nothing in the loop reads these.
    peak_row:    int = -1                # residual peak pixel the cutout was centred on
    peak_col:    int = -1
    peak_val:    float = float("nan")    # working residual at that pixel (Jy/beam)
    sigma_local: float = float("nan")    # noise value fed to the network (after scale division)
    # Raw mode, before size clipping: (x_off, y_off, log_flux_std, log_sig_maj, log_sig_min, pa)
    mode6d:      tuple = ()
    born_in_major: int = -1              # major cycle that created it
    cid:           int = -1              # chronological id over the whole run


# ---------------------------------------------------------------------------
# render_aspen
# ---------------------------------------------------------------------------

def render_aspen(
    cx: float, cy: float,
    flux: float,
    sig_maj: float, sig_min: float,
    pa: float,
    shape: tuple[int, int],
    nsigma: float | None = None,
) -> np.ndarray:
    """Render one Aspen as a 2D Gaussian model image (Jy/pixel).

    Parameters
    ----------
    cx, cy   : centre position (col, row) in the full image
    flux     : total integrated flux (Jy)
    sig_maj  : major-axis sigma (px)
    sig_min  : minor-axis sigma (px)
    pa       : position angle of major axis (rad, CCW from +col)
    shape    : (H, W) of the output image
    nsigma   : if set, zero the profile outside the nsigma ellipse before
               normalising, so the flux stays inside a known footprint

    Returns
    -------
    (H, W) float32 array, sums to `flux`.
    """
    H, W = shape
    cos_pa = math.cos(pa)
    sin_pa = math.sin(pa)
    # With truncation only the bounding box of the nsigma ellipse is nonzero,
    # so evaluate there instead of over the whole frame.
    r0, r1, c0, c1 = 0, H, 0, W
    if nsigma is not None:
        a, b = nsigma * max(sig_maj, 1e-6), nsigma * max(sig_min, 1e-6)
        hx = math.hypot(a * cos_pa, b * sin_pa)
        hy = math.hypot(a * sin_pa, b * cos_pa)
        r0, r1 = max(0, math.floor(cy - hy)), min(H, math.ceil(cy + hy) + 1)
        c0, c1 = max(0, math.floor(cx - hx)), min(W, math.ceil(cx + hx) + 1)
        if r1 <= r0 or c1 <= c0:
            return np.zeros((H, W), dtype=np.float32)
    rows = np.arange(r0, r1, dtype=np.float64)
    cols = np.arange(c0, c1, dtype=np.float64)
    C, R = np.meshgrid(cols, rows)
    dc = C - cx
    dr = R - cy
    u =  dc * cos_pa + dr * sin_pa   # along major axis
    v = -dc * sin_pa + dr * cos_pa   # along minor axis
    exponent = 0.5 * ((u / max(sig_maj, 1e-6)) ** 2
                    + (v / max(sig_min, 1e-6)) ** 2)
    g = np.exp(-exponent)
    if nsigma is not None:
        g[exponent > 0.5 * nsigma * nsigma] = 0.0
    g_sum = g.sum()
    if g_sum > 0:
        g *= flux / g_sum
    g = np.clip(g, -3.4e38, 3.4e38).astype(np.float32)
    if (r0, r1, c0, c1) == (0, H, 0, W):
        return g
    out = np.zeros((H, W), dtype=np.float32)
    out[r0:r1, c0:c1] = g
    return out


def _canonical_axes(
    raw_smaj: float,
    raw_smin: float,
    pa: float,
    smin: float,
    smax: float,
) -> tuple[float, float, float]:
    """Canonicalise Gaussian axes so ``sig_maj >= sig_min`` without distorting shape.

    When ``raw_smin > raw_smaj``, swapping the axes and rotating ``pa`` by ``pi/2``
    produces the exact same 2D Gaussian ellipse rather than collapsing it to a circle.
    """
    if raw_smin > raw_smaj:
        raw_smaj, raw_smin = raw_smin, raw_smaj
        pa = pa + 0.5 * math.pi
    pa = 0.5 * math.atan2(math.sin(2.0 * pa), math.cos(2.0 * pa))
    sig_maj = float(np.clip(raw_smaj, smin, smax))
    sig_min = float(np.clip(raw_smin, smin, sig_maj))
    return sig_maj, sig_min, float(pa)


class PSFConvolver:
    """``psf_convolve`` against a fixed PSF and image shape, as a torch FFT with
    the PSF spectrum cached, so each call is one forward and one inverse FFT on
    ``device``."""

    def __init__(self, psf: np.ndarray, shape: tuple[int, int],
                 device: str | torch.device = "cpu"):
        h, w = psf.shape
        self.H, self.W = shape
        self.n = (self.H + h, self.W + w)
        self.r0, self.c0 = h // 2, w // 2
        self.device = torch.device(device)
        psf_t = torch.from_numpy(np.ascontiguousarray(psf, dtype=np.float32)).to(self.device)
        p = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        p[:h, :w] = psf_t
        self.psf_f = torch.fft.rfft2(p)

        # Symmetric PSD spectrum B_+ on >= 2x padded grid for convex NNLS solves.
        self.psd_n = (max(self.n[0], 2 * self.H), max(self.n[1], 2 * self.W))
        p_psd = torch.zeros(self.psd_n, dtype=torch.float32, device=self.device)
        p_psd[:h, :w] = psf_t
        p_rolled = torch.roll(p_psd, (-self.r0, -self.c0), dims=(0, 1))
        self.psf_psd_f = torch.fft.rfft2(p_rolled).real.clamp_min(0.0)
        self._psd_spatial: torch.Tensor | None = None

    def convolve_torch(self, x_t: torch.Tensor) -> torch.Tensor:
        lead = x_t.shape[:-2]
        x = torch.zeros(*lead, *self.n, dtype=torch.float32, device=self.device)
        x[..., :self.H, :self.W] = x_t.to(device=self.device, dtype=torch.float32)
        y = torch.fft.irfft2(torch.fft.rfft2(x) * self.psf_f, s=self.n)
        return y[..., self.r0:self.r0 + self.H, self.c0:self.c0 + self.W]

    def adjoint_torch(self, y_t: torch.Tensor) -> torch.Tensor:
        """Exact transpose B^T of ``convolve_torch`` under the real L2 inner product."""
        lead = y_t.shape[:-2]
        z = torch.zeros(*lead, *self.n, dtype=torch.float32, device=self.device)
        z[..., self.r0:self.r0 + self.H, self.c0:self.c0 + self.W] = y_t.to(
            device=self.device, dtype=torch.float32
        )
        x = torch.fft.irfft2(torch.fft.rfft2(z) * self.psf_f.conj(), s=self.n)
        return x[..., :self.H, :self.W]

    def convolve_psd_torch(self, x_t: torch.Tensor) -> torch.Tensor:
        """Apply symmetric positive semi-definite projection B_+ of the PSF."""
        lead = x_t.shape[:-2]
        x = torch.zeros(*lead, *self.psd_n, dtype=torch.float32, device=self.device)
        x[..., :self.H, :self.W] = x_t.to(device=self.device, dtype=torch.float32)
        y = torch.fft.irfft2(torch.fft.rfft2(x) * self.psf_psd_f, s=self.psd_n)
        return y[..., :self.H, :self.W]

    def psd_kernel_crop(self, size: int) -> torch.Tensor:
        """Centred ``(size, size)`` spatial kernel of ``B_+`` with peak at ``(size//2, size//2)``."""
        if self._psd_spatial is None:
            self._psd_spatial = torch.fft.irfft2(self.psf_psd_f, s=self.psd_n)
        half = size // 2
        if size <= self.psd_n[0] and size <= self.psd_n[1]:
            shifted = torch.roll(self._psd_spatial, (half, half), dims=(0, 1))
            return shifted[:size, :size].contiguous()
        out = torch.zeros((size, size), dtype=torch.float32, device=self.device)
        sh0, sh1 = self.psd_n[0] // 2, self.psd_n[1] // 2
        centered = torch.roll(self._psd_spatial, (sh0, sh1), dims=(0, 1))
        r0 = half - sh0
        c0 = half - sh1
        out[r0:r0 + self.psd_n[0], c0:c0 + self.psd_n[1]] = centered
        return out

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """(H, W) or a stack (..., H, W)."""
        x_t = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
        return self.convolve_torch(x_t).cpu().numpy()

    def adjoint(self, image: np.ndarray) -> np.ndarray:
        """Exact transpose B^T on (H, W) or stack (..., H, W)."""
        y_t = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
        return self.adjoint_torch(y_t).cpu().numpy()


# ---------------------------------------------------------------------------
# minor_cycle
# ---------------------------------------------------------------------------

def minor_cycle(
    residual:   np.ndarray,       # (H, W) float32, Jy/beam
    psf:        np.ndarray,       # (H, W) float32, peak = 1
    sigma:      float,            # noise estimate (Jy/beam)
    config_idx: int,              # 0=A 1=B 2=C 3=D
    model:      MDNAsp,
    *,
    loop_gain:   float = 0.1,
    n_sigma_stop: float = 3.0,
    abs_noise_floor: float | None = None,  # absolute floor (Jy/beam); overrides n_sigma_stop*sigma when set
    sidelobe_level: float = 0.2,
    divergence_tol: float = 0.05,
    max_components: int = 1000,
    image_mask: np.ndarray | None = None,  # (H, W) bool; peak search restricted to True pixels
    sigma_max_px: float | None = None,     # override SIGMA_MAX_PX clip ceiling
    sigma_min_px: float | None = None,     # override BEAM_SIGMA_PX clip floor
    scale_relative: bool = False,          # model trained with ScaleRelativeCutoutDataset
    global_sigma: bool = False,            # condition on `sigma`, not the cutout MAD
    cutout: int = _CUTOUT,                 # network input size (MDNAsp.cutout)
    render_nsigma: float | None = None,    # truncate each Aspen at this many sigma
    stop_below_noise: bool = False,        # stop when the full estimate's dirty peak < sigma
    sidelobe_abs_peak: bool = False,       # sidelobe floor from max|residual|, not the positive peak
    cap_to_pixel: bool = False,            # full estimate's dirty peak may not exceed the picked pixel
    flux_clip: bool = False,               # cap each full estimate's dirty peak to dirty_peak (Jy/beam)
    density_mode: bool = False,            # pick argmax pi_k / prod(sigma_k) instead of argmax pi_k
    profile_amp: bool = False,             # replace open-loop flux readout with exact profile ML amplitude
    delta_fallback: bool = False,          # fall back to point component when point likelihood exceeds Aspen's
    uv_coupling_min: float = 0.0,          # reject/fallback when <g, B g> / <g, B_main g> < uv_coupling_min
    nullspace_lam: float = 0.0,            # Tikhonov nullspace damping on profile denominator
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, list[AspenCommit], str]:
    """Run the MDN-Asp minor cycle.

    Returns
    -------
    model_update : (H, W) float32 -- accumulated Aspen model image (Jy/pixel).
                   Add this to CASA's running model image.
    commits      : list of AspenCommit, one per accepted component.
    stop_reason  : str -- one of "noise_floor", "sidelobe_floor", "divergence", "max_components",
                   "estimate_below_noise".
    """
    device = torch.device(device)
    model.eval()

    H, W = residual.shape
    residual = residual.copy().astype(np.float32)
    model_update = np.zeros((H, W), dtype=np.float32)
    commits: list[AspenCommit] = []

    noise_floor  = abs_noise_floor if abs_noise_floor is not None else n_sigma_stop * sigma
    dirty_peak   = float(np.abs(residual).max())
    # The sidelobe floor must come from the POSITIVE peak, in the same region
    # the peak search actually looks. Using max|residual| lets a negative bowl
    # set the stopping floor: measured on G55 2026-08-10, the residual bottomed
    # at -5.04e-3 while the positive peak was 3.17e-4, so the floor became
    # 0.063 * 5.04e-3 = 3.18e-4 and the minor cycle refused to commit anything
    # below 14.6 sigma -- the clean stalled with the source still in the map
    # (365 -> 12 -> 1 -> 0 components over four cycles).
    _pos = np.where(residual > 0, residual, 0.0)
    if image_mask is not None:
        _pos = np.where(image_mask, _pos, 0.0)
    pos_peak     = float(_pos.max())
    # With sidelobe_abs_peak a negative bowl also sets the floor: its sidelobes
    # ring through the field just like those of a positive error.
    if sidelobe_abs_peak:
        _abs = np.abs(residual) if image_mask is None else np.where(image_mask, np.abs(residual), 0.0)
        sidelobe_floor = sidelobe_level * float(_abs.max())
    else:
        sidelobe_floor = sidelobe_level * pos_peak
    prev_peak    = dirty_peak

    convolve = PSFConvolver(psf, (H, W), device)
    use_exact_guard = bool(profile_amp or delta_fallback or uv_coupling_min > 0.0)
    convolve_main = None
    omega_beam = 1.0
    if use_exact_guard:
        _, psf_main, omega_beam = _extract_psf_main_lobe(psf)
        if uv_coupling_min > 0.0:
            convolve_main = PSFConvolver(psf_main, (H, W), device)

    stop_reason = "max_components"
    while len(commits) < max_components:
        pos = np.where(residual > 0, residual, 0.0)
        search = pos if image_mask is None else np.where(image_mask, pos, 0.0)
        peak_idx = np.argmax(search)
        if float(search.flat[peak_idx]) <= 0.0:
            stop_reason = "noise_floor"
            break
        peak_row, peak_col = divmod(int(peak_idx), W)
        peak_val = float(residual[peak_row, peak_col])
        abs_peak = abs(peak_val)

        # Stopping rules
        if abs_peak < noise_floor:
            stop_reason = "noise_floor"
            break
        if abs_peak < sidelobe_floor:
            stop_reason = "sidelobe_floor"
            break
        if abs_peak > prev_peak * (1.0 + divergence_tol):
            stop_reason = "divergence"
            break
        prev_peak = abs_peak

        # --- Crop cutout ---
        r0 = peak_row - cutout // 2
        r1 = r0 + cutout
        c0 = peak_col - cutout // 2
        c1 = c0 + cutout
        res_cut = _safe_crop(residual, r0, r1, c0, c1)
        psf_cut = _crop_psf_centred(psf, cutout)

        # --- Conditioning ---
        # global_sigma: the cutout MAD is inflated by residual source structure
        # (G55 2026-09-25: 8.5x the off-source sigma, peak/MAD 2.4 -> zero flux).
        sigma_local = (sigma if global_sigma
                       else float(1.4826 * np.median(np.abs(res_cut))))
        if not np.isfinite(sigma_local) or sigma_local <= 0:
            sigma_local = sigma

        # Scale-relative models (ScaleRelativeCutoutDataset) see a dimensionless
        # cutout and emit log(flux/scale); the scale is multiplied back below.
        # ``cut_scale`` MUST be the same statistic the dataset used --
        # max|residual| over the cutout -- or the round trip is not the identity.
        cut_scale = 1.0
        if scale_relative:
            cut_scale = float(np.abs(res_cut).max())
            if not np.isfinite(cut_scale) or cut_scale <= 0.0:
                cut_scale = 1.0
            res_cut = res_cut / np.float32(cut_scale)
            sigma_local = sigma_local / cut_scale

        sig_t = torch.tensor([sigma_local], dtype=torch.float32)
        cfg_t = torch.tensor([config_idx],  dtype=torch.long)
        cond  = make_cond(sig_t, cfg_t).to(device)          # (1, 5)

        img_t = torch.from_numpy(
            np.stack([res_cut, psf_cut], axis=0)[None]       # (1, 2, H, W)
        ).to(device)

        # --- MDN forward pass ---
        with torch.no_grad():
            params = model(img_t, cond)

        mode6d = model.mode(params, density_mode=density_mode).squeeze(0).cpu().numpy()  # (6,)
        # mode6d: (x_off, y_off, log_flux_std, log_sig_maj, log_sig_min, pa)

        x_off      = float(mode6d[0])
        y_off      = float(mode6d[1])
        log_flux_std = float(mode6d[2])
        from mdn_asp.constants import BEAM_SIGMA_PX, SIGMA_MAX_PX
        _sigma_max = sigma_max_px if sigma_max_px is not None else SIGMA_MAX_PX
        _sigma_min = sigma_min_px if sigma_min_px is not None else BEAM_SIGMA_PX
        raw_smaj = float(np.exp(np.clip(mode6d[3], -20.0, 20.0)))
        raw_smin = float(np.exp(np.clip(mode6d[4], -20.0, 20.0)))
        sig_maj, sig_min, pa = _canonical_axes(
            raw_smaj, raw_smin, float(mode6d[5]), _sigma_min, _sigma_max
        )

        # Flux readout. In scale-relative mode the head emits log(flux/scale)
        # directly, so the ONLY inverse is multiplying the cutout scale back --
        # no fixed corpus constants are involved and there is nothing to go
        # stale outside the trained decade. In absolute mode the legacy
        # standardisation applies, and `np.exp` there is what overflows on
        # out-of-range fields (measured 2026-08-08 at sky peak 1.0 Jy).
        from mdn_asp.constants import unstandardise_log_flux
        if scale_relative:
            log_flux = log_flux_std + math.log(cut_scale)
        else:
            log_flux = unstandardise_log_flux(log_flux_std)
        log_flux = float(np.clip(log_flux, -30.0, 20.0))
        flux_full = float(math.exp(log_flux))

        # Absolute position in full image
        cx = float(peak_col) + x_off
        cy = float(peak_row) + y_off

        if use_exact_guard:
            g1 = render_aspen(cx, cy, 1.0, sig_maj, sig_min, pa, (H, W), nsigma=render_nsigma)
            Bg1 = convolve(g1)
            num = float((g1 * residual).sum())
            raw_den = float((g1 * Bg1).sum())
            den = raw_den + nullspace_lam * omega_beam * float((g1 * g1).sum())
            uv_ok = True
            if convolve_main is not None:
                den_main = float((g1 * convolve_main(g1)).sum())
                uv_coupling = raw_den / max(den_main, 1e-30)
                uv_ok = uv_coupling >= uv_coupling_min
            need_fallback = (not uv_ok) or (num <= 0.0) or (den <= 0.0)
            if (not need_fallback) and delta_fallback:
                g_pt = render_aspen(float(peak_col), float(peak_row), 1.0,
                                    _sigma_min, _sigma_min, 0.0, (H, W), nsigma=render_nsigma)
                Bg_pt = convolve(g_pt)
                num_pt = float((g_pt * residual).sum())
                den_pt = float((g_pt * Bg_pt).sum()) + nullspace_lam * omega_beam * float((g_pt * g_pt).sum())
                j_asp = (num * num / (2.0 * den)) if (num > 0.0 and den > 0.0) else -1.0
                j_pt = (num_pt * num_pt / (2.0 * den_pt)) if (num_pt > 0.0 and den_pt > 0.0) else -1.0
                if j_pt > j_asp:
                    need_fallback = True
            else:
                g_pt = Bg_pt = None
                num_pt = den_pt = 0.0

            if need_fallback:
                if g_pt is None:
                    g_pt = render_aspen(float(peak_col), float(peak_row), 1.0,
                                        _sigma_min, _sigma_min, 0.0, (H, W), nsigma=render_nsigma)
                    Bg_pt = convolve(g_pt)
                    num_pt = float((g_pt * residual).sum())
                    den_pt = float((g_pt * Bg_pt).sum()) + nullspace_lam * omega_beam * float((g_pt * g_pt).sum())
                if num_pt <= 0.0 or den_pt <= 0.0:
                    stop_reason = "estimate_below_noise"
                    break
                cx, cy = float(peak_col), float(peak_row)
                sig_maj, sig_min, pa = _sigma_min, _sigma_min, 0.0
                g1, Bg1, num, den = g_pt, Bg_pt, num_pt, den_pt
                flux_full = float(max(0.0, num / max(den, 1e-12)))
            elif profile_amp:
                flux_full = float(max(0.0, num / max(den, 1e-12)))

            flux_commit = loop_gain * flux_full
            aspen_img = (flux_commit * g1).astype(np.float32)
            psf_response = (flux_commit * Bg1).astype(np.float32)
        else:
            # Committed flux with loop gain
            flux_commit = loop_gain * flux_full
            # --- Accumulate model image ---
            aspen_img = render_aspen(cx, cy, flux_commit, sig_maj, sig_min, pa, (H, W),
                                     nsigma=render_nsigma)
            psf_response = convolve(aspen_img)

        # Dimensional overflow guard: cap the full-estimate dirty peak (Jy/beam)
        # to dirty_peak (Jy/beam) so extended components are not penalised by beam area.
        if flux_clip and dirty_peak > 0.0:
            full_peak = float(psf_response.max()) / max(loop_gain, 1e-12)
            if full_peak > dirty_peak:
                s = dirty_peak / full_peak
                aspen_img *= s
                psf_response *= s
                flux_commit *= s
                flux_full *= s

        # A positive component's own dirty peak cannot exceed the data where it
        # was picked: scale the full estimate down to the picked pixel.
        if cap_to_pixel:
            full_peak = float(psf_response.max()) / loop_gain
            if full_peak > peak_val > 0:
                s = peak_val / full_peak
                aspen_img *= s; psf_response *= s
                flux_commit *= s; flux_full *= s

        # The full estimate (before loop gain) would not show above the noise.
        if stop_below_noise and float(psf_response.max()) / loop_gain < sigma:
            stop_reason = "estimate_below_noise"
            break

        model_update += aspen_img

        # --- Subtract PSF response from working residual ---
        residual -= psf_response

        commits.append(AspenCommit(
            cx=cx, cy=cy,
            flux=flux_commit,
            sig_maj=sig_maj, sig_min=sig_min, pa=pa,
            params=MixParams(
                logits=params.logits.cpu(),
                mu=params.mu.cpu(),
                log_std=params.log_std.cpu(),
            ),
            cut_scale=cut_scale,
            peak_row=peak_row, peak_col=peak_col, peak_val=peak_val,
            sigma_local=float(sigma_local),
            mode6d=tuple(float(v) for v in mode6d),
        ))

    if commits:
        sig_majs = np.array([c.sig_maj for c in commits])
        sig_mins = np.array([c.sig_min for c in commits])
        fluxes   = np.array([c.flux    for c in commits])
        cxs      = np.array([c.cx      for c in commits])
        cys      = np.array([c.cy      for c in commits])
        print(f"  [minor] n={len(commits)}  "
              f"sig_maj med/max={np.median(sig_majs):.2f}/{sig_majs.max():.2f}px  "
              f"sig_min med/max={np.median(sig_mins):.2f}/{sig_mins.max():.2f}px  "
              f"flux med/max={np.median(fluxes):.4f}/{fluxes.max():.4f} Jy  "
              f"cx spread={cxs.std():.1f}px  cy spread={cys.std():.1f}px  "
              f"stop={stop_reason}")

    return model_update, commits, stop_reason


# ---------------------------------------------------------------------------
# minor_cycle_batched
# ---------------------------------------------------------------------------

def minor_cycle_batched(
    residual:   np.ndarray,       # (H, W) float32, Jy/beam
    psf:        np.ndarray,       # (H, W) float32, peak = 1
    sigma:      float,            # noise estimate (Jy/beam)
    config_idx: int,              # 0=A 1=B 2=C 3=D
    model:      MDNAsp,
    *,
    loop_gain:   float = 0.1,
    n_sigma_stop: float = 3.0,
    abs_noise_floor: float | None = None,
    sidelobe_level: float = 0.2,
    divergence_tol: float = 0.05,
    max_components: int = 1000,
    image_mask: np.ndarray | None = None,
    sigma_max_px: float | None = None,
    sigma_min_px: float | None = None,
    scale_relative: bool = False,
    global_sigma: bool = False,
    cutout: int = _CUTOUT,
    render_nsigma: float | None = None,
    density_mode: bool = False,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, list[AspenCommit], str]:
    """Clark-style minor cycle: each pass takes every local maximum (3x3) of
    the positive residual above max(sidelobe_level x the pass peak, noise
    floor), runs all their cutouts through the MDN in one forward pass,
    commits every mode at ``loop_gain`` and subtracts them together.

    Stop rules as ``minor_cycle``, applied to the pass peak. Same returns.
    """
    from scipy.ndimage import maximum_filter
    from mdn_asp.constants import SIGMA_MAX_PX, unstandardise_log_flux

    device = torch.device(device)
    model.eval()
    H, W = residual.shape
    residual = residual.copy().astype(np.float32)
    model_update = np.zeros((H, W), dtype=np.float32)
    commits: list[AspenCommit] = []

    noise_floor = abs_noise_floor if abs_noise_floor is not None else n_sigma_stop * sigma
    _sigma_max = sigma_max_px if sigma_max_px is not None else SIGMA_MAX_PX
    _sigma_min = sigma_min_px if sigma_min_px is not None else BEAM_SIGMA_PX
    psf_cut = _crop_psf_centred(psf, cutout)
    convolve = PSFConvolver(psf, (H, W), device)

    def _search(res):
        s = np.where(res > 0, res, 0.0)
        return s if image_mask is None else np.where(image_mask, s, 0.0)

    sidelobe_floor = sidelobe_level * float(_search(residual).max())
    prev_peak = float(np.abs(residual).max())
    stop_reason = "max_components"
    while len(commits) < max_components:
        search = _search(residual)
        peak = float(search.max())
        if peak <= 0.0 or peak < noise_floor:
            stop_reason = "noise_floor"
            break
        if peak < sidelobe_floor:
            stop_reason = "sidelobe_floor"
            break
        if peak > prev_peak * (1.0 + divergence_tol):
            stop_reason = "divergence"
            break
        prev_peak = peak

        # Every local maximum above this pass's floor, brightest first.
        floor = max(sidelobe_level * peak, noise_floor)
        is_max = (search == maximum_filter(search, size=3)) & (search >= floor)
        rows, cols = np.nonzero(is_max)
        order = np.argsort(-search[rows, cols])[:max_components - len(commits)]
        rows, cols = rows[order], cols[order]

        imgs, sigs, scales = [], [], []
        for pr, pc in zip(rows, cols):
            r0, c0 = pr - cutout // 2, pc - cutout // 2
            res_cut = _safe_crop(residual, r0, r0 + cutout, c0, c0 + cutout)
            sigma_local = (sigma if global_sigma
                           else float(1.4826 * np.median(np.abs(res_cut))))
            if not np.isfinite(sigma_local) or sigma_local <= 0:
                sigma_local = sigma
            cut_scale = 1.0
            if scale_relative:
                cut_scale = float(np.abs(res_cut).max())
                if not np.isfinite(cut_scale) or cut_scale <= 0.0:
                    cut_scale = 1.0
                res_cut = res_cut / np.float32(cut_scale)
                sigma_local = sigma_local / cut_scale
            imgs.append(np.stack([res_cut, psf_cut], axis=0))
            sigs.append(sigma_local)
            scales.append(cut_scale)

        n = len(imgs)
        cond = make_cond(torch.tensor(sigs, dtype=torch.float32),
                         torch.full((n,), config_idx, dtype=torch.long)).to(device)
        img_t = torch.from_numpy(np.stack(imgs, axis=0)).to(device)
        with torch.no_grad():
            params = model(img_t, cond)
        modes = model.mode(params, density_mode=density_mode).cpu().numpy()  # (n, 6)
        logits, mu, log_std = (t.cpu() for t in params)

        pass_update = np.zeros((H, W), dtype=np.float32)
        for i in range(n):
            m6 = modes[i]
            raw_smaj = float(np.exp(np.clip(m6[3], -20.0, 20.0)))
            raw_smin = float(np.exp(np.clip(m6[4], -20.0, 20.0)))
            sig_maj, sig_min, pa = _canonical_axes(
                raw_smaj, raw_smin, float(m6[5]), _sigma_min, _sigma_max
            )
            log_flux = (float(m6[2]) + math.log(scales[i]) if scale_relative
                        else unstandardise_log_flux(float(m6[2])))
            log_flux = float(np.clip(log_flux, -30.0, 20.0))
            flux_commit = loop_gain * float(math.exp(log_flux))
            pr, pc = int(rows[i]), int(cols[i])
            cx, cy = float(pc) + float(m6[0]), float(pr) + float(m6[1])
            pass_update += render_aspen(cx, cy, flux_commit, sig_maj, sig_min, pa,
                                        (H, W), nsigma=render_nsigma)
            commits.append(AspenCommit(
                cx=cx, cy=cy, flux=flux_commit,
                sig_maj=sig_maj, sig_min=sig_min, pa=pa,
                params=MixParams(logits=logits[i:i + 1], mu=mu[i:i + 1],
                                 log_std=log_std[i:i + 1]),
                cut_scale=scales[i],
                peak_row=pr, peak_col=pc, peak_val=float(residual[pr, pc]),
                sigma_local=float(sigs[i]),
                mode6d=tuple(float(v) for v in m6),
            ))

        model_update += pass_update
        residual -= convolve(pass_update)

    if commits:
        fluxes = np.array([c.flux for c in commits])
        print(f"  [minor batched] n={len(commits)}  "
              f"flux med/max={np.median(fluxes):.4f}/{fluxes.max():.4f} Jy  "
              f"stop={stop_reason}")

    return model_update, commits, stop_reason


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_crop(arr, r0, r1, c0, c1):
    H, W = arr.shape
    out = np.zeros((r1 - r0, c1 - c0), dtype=arr.dtype)
    sr0 = max(0, r0); sr1 = min(H, r1)
    sc0 = max(0, c0); sc1 = min(W, c1)
    if sr1 <= sr0 or sc1 <= sc0:
        return out
    dr0 = sr0 - r0
    dc0 = sc0 - c0
    out[dr0:dr0+(sr1-sr0), dc0:dc0+(sc1-sc0)] = arr[sr0:sr1, sc0:sc1]
    return out


def _crop_psf_centred(psf, size):
    if psf.shape == (size, size):
        return psf.copy()
    py, px = np.unravel_index(int(np.argmax(psf)), psf.shape)
    half = size // 2
    return _safe_crop(psf, py - half, py - half + size,
                           px - half, px - half + size)


def _extract_psf_main_lobe(psf: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Isolate the positive central main lobe of a PSF (excluding outer spokes/rings).

    Returns ``(main_mask, psf_main, omega_beam)`` where ``psf_main`` is peak-normalised
    and zero outside ``main_mask``, and ``omega_beam = max(psf_main.sum(), 1.0)``.
    """
    from scipy.ndimage import label

    pmax = float(psf.max())
    if pmax <= 0.0 or not np.isfinite(pmax):
        mask = np.ones(psf.shape, dtype=bool)
        return mask, np.zeros_like(psf, dtype=np.float32), 1.0
    p = psf.astype(np.float64) / pmax
    py, px = np.unravel_index(int(np.argmax(p)), p.shape)

    lab50, _ = label(p >= 0.5)
    core = (lab50 == lab50[py, px]) if lab50[py, px] > 0 else np.zeros_like(p, dtype=bool)
    core[py, px] = True
    yy, xx = np.indices(p.shape, dtype=np.float64)
    dy, dx = yy - py, xx - px
    cy_pts, cx_pts = dy[core], dx[core]
    if cy_pts.size >= 4:
        w = p[core]
        w = w / max(float(w.sum()), 1e-30)
        syy = max(float((w * cy_pts * cy_pts).sum()), 0.25)
        sxx = max(float((w * cx_pts * cx_pts).sum()), 0.25)
        syx = float((w * cy_pts * cx_pts).sum())
        det = max(syy * sxx - syx * syx, 0.0625)
        iyy, ixx, iyx = sxx / det, syy / det, -syx / det
        r_ell = np.sqrt(np.maximum(iyy * dy * dy + 2.0 * iyx * dy * dx + ixx * dx * dx, 0.0))
        r_bin = np.floor(r_ell * 2.0).astype(int)
    else:
        r_bin = np.hypot(dy, dx).astype(int)

    counts = np.bincount(r_bin.ravel())
    prof = np.bincount(r_bin.ravel(), p.ravel()) / np.maximum(counts, 1)
    r_bound = None
    for i in range(1, len(prof)):
        if prof[i] <= 0.0:
            r_bound = i
            break
        if prof[i] < 0.15 and i + 1 < len(prof) and prof[i + 1] > prof[i] + 1e-4:
            r_bound = i + 1
            break

    lab_pos, _ = label(p > 0.0)
    pos_conn = (lab_pos == lab_pos[py, px]) if lab_pos[py, px] > 0 else (p > 0.0)
    if r_bound is None:
        main_mask = pos_conn
    else:
        main_mask = pos_conn & (r_bin < r_bound)
        main_mask[py, px] = True

    psf_main = np.where(main_mask, p, 0.0).astype(np.float32)
    omega = max(float(psf_main.sum()), 1.0)
    return main_mask, psf_main, omega


def measure_psf_sidelobe(psf: np.ndarray) -> float:
    """Peak sidelobe amplitude (both negative bowls and positive outer spokes/rings)."""
    pmax = float(psf.max())
    if pmax <= 0.0 or not np.isfinite(pmax):
        return float(abs(psf.min()))
    main_mask, _, _ = _extract_psf_main_lobe(psf)
    neg_lobe = abs(float(psf.min())) / pmax
    pos_lobe = (float(psf[~main_mask].max()) / pmax) if (~main_mask).any() else 0.0
    return float(max(neg_lobe, pos_lobe))


def measure_psf_sidelobe_first(psf: np.ndarray) -> float:
    """Peak of the first positive sidelobe: past the main lobe's first null
    (azimuthally averaged profile <= 0), the first ring where it is positive
    again. Returns the largest PSF pixel in that ring, peak = 1."""
    py, px = np.unravel_index(int(np.argmax(psf)), psf.shape)
    yy, xx = np.indices(psf.shape)
    r = np.hypot(yy - py, xx - px).astype(int)
    prof = np.bincount(r.ravel(), psf.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    n = len(prof)
    r_null = next((i for i in range(1, n) if prof[i] <= 0), None)
    r_up = next((i for i in range(r_null, n) if prof[i] > 0), None) if r_null else None
    if r_up is None:
        raise ValueError("PSF profile has no null followed by a positive sidelobe in the crop")
    r_down = next((i for i in range(r_up, n) if prof[i] <= 0), n)
    ring = (r >= r_up) & (r < r_down)
    return float(psf[ring].max()) / float(psf.max())
