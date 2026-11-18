"""Batched on-GPU equivalent of ``CutoutDataset`` for MDN-Asp training.

Why this exists
---------------
``CutoutDataset`` builds one sample per worker process in numpy: a 512x512
field with 5-20 rendered sources, two FFT convolutions, then a 128x128 crop --
15/16 of the rendered pixels are thrown away. Measured 2026-08-10 on a 32-thread
box with 24 workers: ~1.0 step/s achieved, every core saturated, GPU at 1-2%.
The GPU needs 22.7 ms/step and waits ~1000 ms for a batch, so its duty cycle is
~2%. Single-process per-sample cost is 14 ms but the workload does not scale
across cores -- 5-20 float32 512x512 intermediates per sample is
memory-bandwidth bound, not compute bound, so 4 -> 24 workers barely moved it.

This module builds whole batches on the device instead: no DataLoader, no
workers, no host-to-device copy of image data.

Fidelity contract
-----------------
The CENTRED source carries the 6D label, so its distributions match
``CutoutDataset`` exactly: kind uniform over ``morphologies``, position
uniform in ``[cutout/2+4, field-cutout/2-4]``, flux log-uniform on
``[1e-4, 1e-1]`` Jy, and for a blob ``sig_maj ~ U(BEAM, SIGMA_MAX)``,
``sig_min ~ U(BEAM, sig_maj)``, ``pa ~ U(0, pi)``, rendered unit-sum then
scaled by flux (so the label IS the integrated flux).

DISTRACTORS are context only -- their targets are never labels, they exist to
put confusion and PSF sidelobes into the window. Count, positions and the
truncated-power-law fluxes match; the rare extended distractor (5% of ~12
sources) uses an elongated-Gaussian stand-in for ``render_filament`` rather
than reproducing its exact profile. This is a deliberate, stated
approximation, not an oversight.

All extended profiles are normalised to UNIT SUM before flux scaling, matching
``extended_sky``. Note ``gpu_sky_generator`` normalises by PEAK instead, which
is why its renderers are not reused here -- that would change photometry.
"""
from __future__ import annotations

import math

import torch

from mdn_asp.constants import BEAM_SIGMA_PX, SIGMA_MAX_PX
from mdn_asp.constants import LOG_FLUX_OFFSET as _LFO
from mdn_asp.constants import LOG_FLUX_SCALE as _LFS
from mdn_asp.models.mdn_asp import N_CONFIGS

_FLUX_MIN_JY = 1e-4
_FLUX_MAX_JY = 1e-1
_DN_DS_SLOPE = -1.6
_EDGE_MARGIN = 16
_AMOR_K_MAX = 6          # amorphous source: 2..6 Gaussian components
_AMOR_SPREAD_PX = 6.0    # per-axis component offset sigma (px), clamped at 3x


def fft_convolve_same(sky: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
    """Batched ``scipy.signal.fftconvolve(..., mode='same')``.

    ``sky`` is (B, H, W), ``psf`` is (B, h, w); h, w need not equal H, W (the
    field is 512 while the PSF bank is 128). Returns (B, H, W).

    Pads to a fast length: the linear length H+h-1 is 639 for these shapes and
    a naive 2H-1=1023 (= 3*11*31) transform cost 271 ms/call on CPU versus
    8.6 ms for scipy, which was 65% of sample-generation time. The crop offset
    is ``h//2``, the PSF peak (CASA convention), matching minor_cycle.psf_convolve.
    scipy's ``(h-1)//2`` put the response 1 px down-right of the source on the
    even 128 PSFs, so models trained here learned an offset deployment lacks.
    """
    B, H, W = sky.shape
    _, h, w = psf.shape
    fH, fW = _next_fast_len(H + h - 1), _next_fast_len(W + w - 1)
    out = torch.fft.irfft2(torch.fft.rfft2(sky, s=(fH, fW))
                           * torch.fft.rfft2(psf, s=(fH, fW)), s=(fH, fW))
    y0, x0 = h // 2, w // 2
    return out[:, y0:y0 + H, x0:x0 + W]


def _next_fast_len(n: int) -> int:
    """Smallest 5-smooth integer >= n (2,3,5 factors only), as scipy uses."""
    if n <= 1:
        return 1
    best = 2 * n
    p2 = 1
    while p2 < best:
        p23 = p2
        while p23 < best:
            p235 = p23
            while p235 < n:
                p235 *= 5
            best = min(best, p235)
            p23 *= 3
        p2 *= 2
    return best


class GPUCutoutGenerator:
    """Generates (img, cond, target) batches for MDN-Asp entirely on device.

    ``sample(B)`` returns the same three tensors the training loop already
    consumes from its DataLoader:
      img    : (B, 2, cutout, cutout)  channels = [residual, psf]
      cond   : (B, 5)                  [sigma_local, config one-hot]
      target : (B, 6)                  [x_off, y_off, log_flux*, log_sig_maj,
                                        log_sig_min, pa]
    ``log_flux*`` is ``log(flux/scale)`` when ``scale_relative``, else the
    legacy ``standardise_log_flux``.
    """

    def __init__(
        self,
        psf_bank: torch.Tensor,          # (N, h, w) peak-normalised
        *,
        field_size: int = 512,
        cutout_size: int = 128,
        sigma_noise: float = 1e-4,
        n_sources: tuple[int, int] = (5, 20),
        extended_fraction: float = 0.05,
        snr_min: float = 5.0,
        morphologies: tuple[str, ...] = ("point", "blob"),
        config_idx: int = 3,
        scale_relative: bool = True,
        correlated_noise: bool = False,
        point_sigma_px: float | None = None,
        blob_sigma_min_px: float | None = None,
        sigma_max_px: float | None = None,
        render_nsigma: float | None = None,
        uniform_point_pa: bool = False,
        slot_chunk: int = 4,
        device: str | torch.device = "cuda",
        seed: int = 42,
    ) -> None:
        self.device = torch.device(device)
        self.psf_bank = psf_bank.to(self.device).float()
        self.H = self.W = field_size
        self.cutout = cutout_size
        self.sigma_noise = float(sigma_noise)
        self.n_src_min, self.n_src_max = n_sources
        self.ext_frac = float(extended_fraction)
        self.snr_min = float(snr_min)
        self.morphologies = tuple(morphologies)
        self.config_idx = int(config_idx)
        self.scale_relative = bool(scale_relative)
        # Real dirty-image noise has Cov = sigma^2 B (the PSF), not sigma^2 I.
        self.correlated_noise = bool(correlated_noise)
        # Size label for point sources. None keeps the legacy beam-sigma label;
        # a sub-pixel value labels points as sigma -> 0 (unresolved).
        self.point_sigma_px = (BEAM_SIGMA_PX if point_sigma_px is None
                               else float(point_sigma_px))
        # Lower bound of the labelled blob size. None keeps the beam sigma.
        self.blob_sigma_min_px = (BEAM_SIGMA_PX if blob_sigma_min_px is None
                                  else float(blob_sigma_min_px))
        # Upper bound of every drawn size. None keeps SIGMA_MAX_PX.
        self.sigma_max_px = (SIGMA_MAX_PX if sigma_max_px is None
                             else float(sigma_max_px))
        # Truncate Gaussian (kind 0) profiles at this many sigma, as
        # minor_cycle.render_aspen does at deployment. None = untruncated.
        self.render_nsigma = None if render_nsigma is None else float(render_nsigma)
        self.uniform_point_pa = bool(uniform_point_pa)
        # Slot chunking bounds the (B, N, H, W) intermediate. At B=64, N=20,
        # 512^2 float32 that tensor is 1.34 GB, which does not fit alongside a
        # desktop on an 11 GB card.
        self.slot_chunk = int(slot_chunk)
        # Max extended sources rendered at once, at (M, H, W) float32.
        self._ext_chunk = 128
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        ar = torch.arange(field_size, device=self.device, dtype=torch.float32)
        self._yg = ar.view(1, 1, field_size, 1)
        self._xg = ar.view(1, 1, 1, field_size)

    # -- helpers ---------------------------------------------------------

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _uniform(self, lo: float, hi: float, *shape: int) -> torch.Tensor:
        return self._rand(*shape) * (hi - lo) + lo

    def _power_law_flux(self, *shape: int) -> torch.Tensor:
        """Inverse-CDF draw from p(S) ~ S^slope, matching point_sky."""
        a = _DN_DS_SLOPE + 1.0
        u = self._rand(*shape)
        lo, hi = _FLUX_MIN_JY ** a, _FLUX_MAX_JY ** a
        return (lo + u * (hi - lo)) ** (1.0 / a)

    def _unit_sum(self, prof: torch.Tensor) -> torch.Tensor:
        """Normalise each (B, N, H, W) profile to unit sum, as extended_sky."""
        tot = prof.flatten(2).sum(dim=2).clamp_min(1e-30)
        return prof / tot.view(*prof.shape[:2], 1, 1)

    # -- scene construction ---------------------------------------------

    def _distractor_sky(self, B: int) -> torch.Tensor:
        """Point-dominated confusion field. Context only, never labelled."""
        H, W, dev = self.H, self.W, self.device
        n_src = torch.randint(self.n_src_min, self.n_src_max + 1, (B,),
                              generator=self.gen, device=dev)
        N = int(n_src.max().item())
        slot = torch.arange(N, device=dev).view(1, N)
        valid = slot < n_src.view(B, 1)

        lo, hi = _EDGE_MARGIN, self.H - _EDGE_MARGIN
        cx = self._uniform(lo, hi, B, N)
        cy = self._uniform(lo, hi, B, N)
        flux = self._power_law_flux(B, N)
        is_ext = self._rand(B, N) < self.ext_frac

        sky = torch.zeros(B, H, W, device=dev)

        # Points (95%): scatter-add, no large intermediate.
        pm = valid & ~is_ext
        xi = cx.round().long().clamp_(0, W - 1)
        yi = cy.round().long().clamp_(0, H - 1)
        sky.view(B, H * W).scatter_add_(
            1, yi * W + xi, flux * pm.to(flux.dtype))

        # Extended (5%): chunked over slots to bound memory.
        em = valid & is_ext
        if bool(em.any()):
            kind = torch.randint(0, 3, (B, N), generator=self.gen, device=dev)
            sig_maj = self._uniform(BEAM_SIGMA_PX, self.sigma_max_px, B, N)
            sig_min = BEAM_SIGMA_PX + self._rand(B, N) * (sig_maj - BEAM_SIGMA_PX)
            pa = self._uniform(0.0, math.pi, B, N)
            thick = self._uniform(BEAM_SIGMA_PX * 0.5, BEAM_SIGMA_PX, B, N)
            sel = em.nonzero(as_tuple=False)          # (M, 2) -> [batch, slot]
            bsel, ssel = sel[:, 0], sel[:, 1]
            # Chunk over the SELECTED sources, not over slots, so memory is
            # bounded by how many extended sources there actually are.
            for m0 in range(0, len(bsel), self._ext_chunk):
                m1 = min(m0 + self._ext_chunk, len(bsel))
                b, t = bsel[m0:m1], ssel[m0:m1]
                prof = self._render_extended_flat(
                    kind[b, t], cx[b, t], cy[b, t], sig_maj[b, t],
                    sig_min[b, t], pa[b, t], thick[b, t])
                sky.index_add_(0, b, prof * flux[b, t].view(-1, 1, 1))
        return sky

    def _render_extended_flat(self, kind, cx, cy, sig_maj, sig_min, pa, thick):
        """Render M extended sources as a compact (M, H, W) batch.

        Takes only the slots that are ACTUALLY extended. Rendering the full
        (B, N, H, W) slot grid instead costs 20x more: at B=64, N=20 only ~5%
        of the 1280 slots are extended, but a masked chunk still evaluates
        every slot in the chunk. Measured: 240 ms vs ~12 ms per batch, i.e.
        91% of total generation time.
        """
        M = kind.shape[0]
        # Chunk internally so no caller can OOM by passing a large batch: each
        # (M, H, W) float32 intermediate is 1.26 GB at M=1200, 512^2, and
        # several exist at once.
        if M > self._ext_chunk:
            return torch.cat([
                self._render_extended_flat(
                    kind[i:i + self._ext_chunk], cx[i:i + self._ext_chunk],
                    cy[i:i + self._ext_chunk], sig_maj[i:i + self._ext_chunk],
                    sig_min[i:i + self._ext_chunk], pa[i:i + self._ext_chunk],
                    thick[i:i + self._ext_chunk])
                for i in range(0, M, self._ext_chunk)], dim=0)
        cx4, cy4 = cx.view(M, 1, 1), cy.view(M, 1, 1)
        xg, yg = self._xg.view(1, 1, self.W), self._yg.view(1, self.H, 1)
        dx, dy = xg - cx4, yg - cy4
        c, s = torch.cos(pa).view(M, 1, 1), torch.sin(pa).view(M, 1, 1)
        u, v = dx * c + dy * s, -dx * s + dy * c
        smaj, smin = sig_maj.view(M, 1, 1), sig_min.view(M, 1, 1)

        prof = torch.zeros(M, self.H, self.W, device=self.device)
        k = kind.view(M, 1, 1)
        if bool((kind == 0).any()):
            q = (u / smaj) ** 2 + (v / smin) ** 2
            g = torch.exp(-0.5 * q)
            if self.render_nsigma is not None:
                g = torch.where(q > self.render_nsigma ** 2, torch.zeros_like(g), g)
            prof = torch.where(k == 0, g, prof)
        if bool((kind == 1).any()):
            r = (dx ** 2 + dy ** 2).sqrt()
            ssig = (thick.view(M, 1, 1) / 2.0).clamp_min(1e-3)
            prof = torch.where(
                k == 1, torch.exp(-0.5 * ((r - smaj) / ssig) ** 2), prof)
        if bool((kind == 2).any()):
            prof = torch.where(
                k == 2,
                torch.exp(-0.5 * ((u / (smaj * 2.0)) ** 2 + (v / smin) ** 2)),
                prof)
        tot = prof.flatten(1).sum(dim=1).clamp_min(1e-30)
        return prof / tot.view(M, 1, 1)

    def _render_extended_chunk(self, kind, cx, cy, flux, sig_maj, sig_min,
                               pa, thick, mask) -> torch.Tensor:
        """(B, n, H, W) intermediate for one slot chunk; returns (B, H, W)."""
        cx4, cy4 = cx.unsqueeze(-1).unsqueeze(-1), cy.unsqueeze(-1).unsqueeze(-1)
        dx, dy = self._xg - cx4, self._yg - cy4
        c, s = torch.cos(pa).view(*pa.shape, 1, 1), torch.sin(pa).view(*pa.shape, 1, 1)
        u, v = dx * c + dy * s, -dx * s + dy * c
        smaj = sig_maj.view(*sig_maj.shape, 1, 1)
        smin = sig_min.view(*sig_min.shape, 1, 1)

        # Only evaluate the kinds actually present in this chunk. Computing all
        # three and selecting with torch.where costs 3x the transcendentals for
        # a field that is 95% point sources.
        k = kind.view(*kind.shape, 1, 1)
        prof = torch.zeros_like(u)
        if bool((kind == 0).any()):     # anisotropic blob
            prof = torch.where(
                k == 0, torch.exp(-0.5 * ((u / smaj) ** 2 + (v / smin) ** 2)),
                prof)
        if bool((kind == 1).any()):     # limb-brightened shell
            r = (dx ** 2 + dy ** 2).sqrt()
            ssig = (thick.view(*thick.shape, 1, 1) / 2.0).clamp_min(1e-3)
            prof = torch.where(
                k == 1, torch.exp(-0.5 * ((r - smaj) / ssig) ** 2), prof)
        if bool((kind == 2).any()):     # elongated stand-in for render_filament
            prof = torch.where(
                k == 2,
                torch.exp(-0.5 * ((u / (smaj * 2.0)) ** 2 + (v / smin) ** 2)),
                prof)
        prof = self._unit_sum(prof)
        f4 = flux.view(*flux.shape, 1, 1)
        m4 = mask.view(*mask.shape, 1, 1).to(prof.dtype)
        return (prof * f4 * m4).sum(dim=1)

    def _centred_source(self, B: int, psfs: torch.Tensor):
        """The labelled source. Distributions match CutoutDataset exactly."""
        dev = self.device
        margin = self.cutout // 2 + 4
        cx = self._uniform(margin, self.H - margin, B)
        cy = self._uniform(margin, self.H - margin, B)
        # log-uniform flux, as CutoutDataset: exp(U(log 1e-4, log 1e-1))
        flux = torch.exp(self._uniform(math.log(_FLUX_MIN_JY),
                                       math.log(_FLUX_MAX_JY), B))

        # Kind uniform over morphologies. One draw for any multi-kind list, so
        # ("point", "blob") keeps its old random stream.
        n_m = len(self.morphologies)
        pick = (torch.randint(0, n_m, (B,), generator=self.gen, device=dev)
                if n_m > 1 else torch.zeros(B, dtype=torch.long, device=dev))
        def is_kind(name):
            if name not in self.morphologies:
                return torch.zeros(B, dtype=torch.bool, device=dev)
            return pick == self.morphologies.index(name)
        is_blob, is_amor = is_kind("blob"), is_kind("amorphous")
        is_edge = is_kind("edged")

        s_lo = self.blob_sigma_min_px
        sig_maj = self._uniform(s_lo, self.sigma_max_px, B)
        sig_min = s_lo + self._rand(B) * (sig_maj - s_lo)
        pa = self._uniform(0.0, math.pi, B)

        # Blob: unit-sum anisotropic Gaussian scaled by flux.
        prof = self._render_extended_flat(
            torch.zeros(B, dtype=torch.long, device=dev), cx, cy,
            sig_maj, sig_min, pa, torch.ones(B, device=dev),
        )                                        # (B, H, W), unit sum
        img = prof * (flux * is_blob.to(flux.dtype)).view(B, 1, 1)

        # Point: delta at the rounded pixel, as _render_kind does.
        pt = ~is_blob & ~is_amor & ~is_edge
        if bool(pt.any()):
            xi = cx.round().long().clamp_(0, self.W - 1)
            yi = cy.round().long().clamp_(0, self.H - 1)
            img.view(B, self.H * self.W).scatter_add_(
                1, (yi * self.W + xi).unsqueeze(1),
                (flux * pt.to(flux.dtype)).unsqueeze(1))

        # Point targets use the ROUNDED centre and beam-sigma shape.
        tx = torch.where(is_blob, cx, cx.round())
        ty = torch.where(is_blob, cy, cy.round())
        beam = torch.full_like(sig_maj, self.point_sigma_px)
        t_smaj = torch.where(is_blob, sig_maj, beam)
        t_smin = torch.where(is_blob, sig_min, beam)
        pt_pa = (self._uniform(-0.5 * math.pi, 0.5 * math.pi, B).to(pa.dtype)
                 if self.uniform_point_pa else torch.zeros_like(pa))
        t_pa = torch.where(is_blob, pa, pt_pa)
        cr, cc = ty.round().long(), tx.round().long()

        if bool(is_amor.any()):
            a = is_amor.nonzero(as_tuple=True)[0]
            a_img, a_flux, a_tx, a_ty, a_smaj, a_smin, a_pa, a_cr, a_cc = \
                self._amorphous(cx[a], cy[a], flux[a], psfs[a])
            img[a] = a_img
            flux, tx, ty = flux.clone(), tx.clone(), ty.clone()
            flux[a], tx[a], ty[a] = a_flux, a_tx, a_ty
            t_smaj[a], t_smin[a], t_pa[a] = a_smaj, a_smin, a_pa
            cr[a], cc[a] = a_cr, a_cc

        # Edged: a uniform-brightness ellipse (hard edge), semi-axes drawn as the
        # blob sizes. Label = the largest Gaussian that fits inside it at the
        # 3 sigma truncation (sigma = semi-axis / 3, same centre and PA) with
        # peak brightness equal to the ellipse's, so it never crosses the edge.
        if bool(is_edge.any()):
            e = is_edge.nonzero(as_tuple=True)[0]
            M = e.numel()
            ex, ey, ea, eb, ep = cx[e], cy[e], sig_maj[e], sig_min[e], pa[e]
            xg, yg = self._xg.view(1, 1, self.W), self._yg.view(1, self.H, 1)
            dx, dy = xg - ex.view(M, 1, 1), yg - ey.view(M, 1, 1)
            c_, s_ = torch.cos(ep).view(M, 1, 1), torch.sin(ep).view(M, 1, 1)
            u, v = dx * c_ + dy * s_, -dx * s_ + dy * c_
            q = (u / ea.view(M, 1, 1)) ** 2 + (v / eb.view(M, 1, 1)) ** 2
            top = (q <= 1.0).to(flux.dtype)
            empty = top.flatten(1).sum(dim=1) == 0      # ellipse smaller than a pixel
            if bool(empty.any()):
                k = empty.nonzero(as_tuple=True)[0]
                xi = ex[k].round().long().clamp(0, self.W - 1)
                yi = ey[k].round().long().clamp(0, self.H - 1)
                top[k, yi, xi] = 1.0
                q[k, yi, xi] = 0.0
            bright = flux[e] / top.flatten(1).sum(dim=1)
            img[e] = top * bright.view(M, 1, 1)
            # Label flux: Gaussian of peak `bright`, sigma = semi-axis/3, cut at
            # 3 sigma -- its footprint is exactly the ellipse, exponent 4.5 q.
            gsum = (torch.exp(-4.5 * q) * top).flatten(1).sum(dim=1)
            flux, tx, ty = flux.clone(), tx.clone(), ty.clone()
            flux[e] = bright * gsum
            tx[e], ty[e] = ex, ey
            t_smaj[e], t_smin[e], t_pa[e] = ea / 3.0, eb / 3.0, ep
            cr[e], cc[e] = ey.round().long(), ex.round().long()
        return img, flux, tx, ty, t_smaj, t_smin, t_pa, cr, cc

    def _amorphous(self, cx, cy, flux, psfs):
        """2-6 overlapping Gaussians sharing ``flux``. The label is the one
        component contributing most to the source's own dirty peak, and the
        cutout is centred on that peak, as minor_cycle centres on the residual
        peak. The loop then peels the source one component at a time."""
        M, dev = cx.shape[0], self.device
        K = _AMOR_K_MAX
        n = torch.randint(2, K + 1, (M,), generator=self.gen, device=dev)
        valid = torch.arange(K, device=dev).view(1, K) < n.view(M, 1)
        w = -torch.log(self._rand(M, K).clamp_min(1e-12)) * valid   # Dirichlet(1)
        w = w / w.sum(dim=1, keepdim=True)
        lim = 3.0 * _AMOR_SPREAD_PX
        ox = (torch.randn(M, K, generator=self.gen, device=dev)
              * _AMOR_SPREAD_PX).clamp(-lim, lim)
        oy = (torch.randn(M, K, generator=self.gen, device=dev)
              * _AMOR_SPREAD_PX).clamp(-lim, lim)
        s_lo = self.blob_sigma_min_px
        smaj = self._uniform(s_lo, self.sigma_max_px, M, K)
        smin = s_lo + self._rand(M, K) * (smaj - s_lo)
        pa = self._uniform(0.0, math.pi, M, K)

        img = torch.zeros(M, self.H, self.W, device=dev)
        dirty_k = []
        zero_kind = torch.zeros(M, dtype=torch.long, device=dev)
        for k in range(K):
            prof = self._render_extended_flat(
                zero_kind, cx + ox[:, k], cy + oy[:, k], smaj[:, k],
                smin[:, k], pa[:, k], torch.ones(M, device=dev),
            ) * (flux * w[:, k]).view(M, 1, 1)
            img += prof
            dirty_k.append(fft_convolve_same(prof, psfs))
        dk = torch.stack(dirty_k, dim=1)                     # (M, K, H, W)
        peak = dk.sum(dim=1).flatten(1).argmax(dim=1)        # (M,)
        contrib = dk.flatten(2).gather(
            2, peak.view(M, 1, 1).expand(M, K, 1)).squeeze(2)
        j = contrib.argmax(dim=1)
        pick = lambda t: t.gather(1, j.view(M, 1)).squeeze(1)
        half = self.cutout // 2
        cr = (peak // self.W).clamp(half, self.H - half)
        cc = (peak % self.W).clamp(half, self.W - half)
        return (img, flux * pick(w), cx + pick(ox), cy + pick(oy),
                pick(smaj), pick(smin), pick(pa), cr, cc)

    def _apply_snr_gates(self, img, flux, psfs):
        """The three linear-in-flux SNR gates; strictest rescale wins."""
        dirty_c = fft_convolve_same(img, psfs)
        sig = self.sigma_noise

        peak = dirty_c.abs().flatten(1).max(dim=1).values
        need = self.snr_min * sig
        scale_peak = torch.where(peak < need, need / peak.clamp_min(1e-30),
                                 torch.ones_like(peak))

        mf = ((dirty_c * img).flatten(1).sum(dim=1) /
              (sig * (img ** 2).flatten(1).sum(dim=1).sqrt()).clamp_min(1e-30))
        scale_mf = torch.where(mf < self.snr_min,
                               self.snr_min / mf.clamp_min(1e-30),
                               torch.ones_like(mf))

        fmax = dirty_c.flatten(1).max(dim=1).values.view(-1, 1, 1)
        foot = dirty_c > 0.01 * fmax
        big = dirty_c.masked_fill(~foot, float("inf"))
        min_sb = big.flatten(1).min(dim=1).values
        has = foot.flatten(1).any(dim=1)
        scale_sb = torch.where(
            has & (min_sb < 3.0 * sig),
            3.0 * sig / min_sb.clamp_min(1e-30), torch.ones_like(min_sb))

        scale = torch.maximum(torch.maximum(scale_peak, scale_mf), scale_sb)
        scale = scale.clamp_min(1.0)
        return img * scale.view(-1, 1, 1), flux * scale

    def _psf_shaped_noise(self, psf_cut: torch.Tensor) -> torch.Tensor:
        """Unit-variance noise with autocorrelation equal to the PSF.

        White noise filtered by sqrt(B^(k)) on a 2x padded grid, then
        centre-cropped. B^ is clamped at 0 (a truncated PSF has small negative
        spectral values). Pixel variance is mean(B^+), divided out so the
        result has unit variance and correlation B(tau)/B(0).
        """
        B, c, _ = psf_cut.shape
        n = 2 * c
        pad = torch.zeros(B, n, n, device=psf_cut.device, dtype=psf_cut.dtype)
        pad[:, c // 2:c // 2 + c, c // 2:c // 2 + c] = psf_cut
        spec = torch.fft.fft2(torch.fft.ifftshift(pad, dim=(-2, -1))).real
        spec = spec.clamp_min(0.0)
        w = torch.randn(B, n, n, generator=self.gen, device=psf_cut.device)
        x = torch.fft.ifft2(torch.fft.fft2(w) * spec.sqrt()).real
        x = x / spec.mean(dim=(-2, -1)).clamp_min(1e-30).sqrt().view(B, 1, 1)
        return x[:, c // 2:c // 2 + c, c // 2:c // 2 + c]

    # -- public ----------------------------------------------------------

    def sample(self, batch_size: int):
        B, dev = batch_size, self.device
        idx = torch.randint(0, len(self.psf_bank), (B,),
                            generator=self.gen, device=dev)
        psfs = self.psf_bank[idx]                            # (B, h, w)

        centred, flux, tx, ty, smaj, smin, pa, cr, cc = self._centred_source(B, psfs)
        centred, flux = self._apply_snr_gates(centred, flux, psfs)

        sky = self._distractor_sky(B) + centred
        dirty = fft_convolve_same(sky, psfs)

        # Crop `cutout` around the centred source. The margin guarantees the
        # window is fully inside the field, so no zero padding is needed.
        half = self.cutout // 2
        off = torch.arange(self.cutout, device=dev)
        rows = (cr.view(B, 1) - half + off.view(1, -1))
        cols = (cc.view(B, 1) - half + off.view(1, -1))
        bidx = torch.arange(B, device=dev).view(B, 1, 1)
        res = dirty[bidx, rows.unsqueeze(2), cols.unsqueeze(1)]   # (B, c, c)
        psf_cut = _centre_crop(psfs, self.cutout)

        # Noise is added AFTER the crop. White noise is i.i.d. per pixel, so
        # this is statistically identical to noising the 512 field and
        # cropping, at 1/16 the draws. Correlated noise is stationary, so the
        # same holds.
        if self.correlated_noise:
            res = res + self.sigma_noise * self._psf_shaped_noise(psf_cut)
        else:
            res = res + self.sigma_noise * torch.randn(
                res.shape, generator=self.gen, device=dev)

        # cond: sigma_local from the cutout (MAD), plus config one-hot.
        sigma_local = 1.4826 * res.flatten(1).abs().median(dim=1).values
        sigma_local = torch.where(sigma_local > 0, sigma_local,
                                  torch.full_like(sigma_local,
                                                  max(self.sigma_noise, 1e-12)))
        log_flux = torch.log(flux.clamp_min(1e-30))

        if self.scale_relative:
            # Normalisation rule: 1-homogeneous stat of the INPUT, output in the
            # same units, dimensionless ratio as conditioning.
            cscale = res.flatten(1).abs().max(dim=1).values
            cscale = torch.where(cscale > 0, cscale, torch.ones_like(cscale))
            res = res / cscale.view(B, 1, 1)
            sigma_local = sigma_local / cscale
            flux_term = log_flux - torch.log(cscale)
        else:
            flux_term = (log_flux - _LFO) / _LFS

        one_hot = torch.zeros(B, N_CONFIGS, device=dev)
        one_hot[:, self.config_idx] = 1.0
        cond = torch.cat([sigma_local.unsqueeze(1), one_hot], dim=1)

        img = torch.stack([res, psf_cut], dim=1)
        target = torch.stack([
            tx - cc.to(tx.dtype), ty - cr.to(ty.dtype),
            flux_term, torch.log(smaj), torch.log(smin), pa,
        ], dim=1)
        return img, cond, target


def _centre_crop(psfs: torch.Tensor, size: int) -> torch.Tensor:
    """Centre-crop (B, h, w) to (B, size, size); identity when already sized."""
    _, h, w = psfs.shape
    if h == size and w == size:
        return psfs
    r0, c0 = (h - size) // 2, (w - size) // 2
    return psfs[:, r0:r0 + size, c0:c0 + size]
