"""Exact-likelihood refit (mdn_asp/refit.py)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from mdn_asp.models.mdn_asp import MixParams

# conftest's proxy package maps mdn_asp.* to the flat root modules, so the
# packaged modules are loaded by path (same shim as tests/test_forward.py:134).
def _by_path(name: str):
    import importlib.util
    import sys
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "mdn_asp" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"mdn_asp.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod       # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


_mc = _by_path("minor_cycle")
AspenCommit, psf_convolve, render_aspen = (
    _mc.AspenCommit, _mc.psf_convolve, _mc.render_aspen)
_refit = _by_path("refit")
refit_commits, error_map_from_records = _refit.refit_commits, _refit.error_map_from_records

SIZE = 128


def gaussian_psf(size: int = SIZE, sigma: float = 2.1) -> np.ndarray:
    y = np.arange(size) - size // 2
    g = np.exp(-(y[:, None] ** 2 + y[None, :] ** 2) / (2 * sigma ** 2))
    return (g / g.max()).astype(np.float32)


def sidelobe_psf(size: int = SIZE, sigma: float = 2.1) -> np.ndarray:
    """A PSF with structure well outside the main lobe, like a real one.  A
    pure Gaussian has nothing outside the window, so windowed subtraction looks
    exact on it and hides the drift this guards."""
    y = np.arange(size) - size // 2
    r = np.hypot(y[:, None], y[None, :])
    main = np.exp(-r ** 2 / (2 * sigma ** 2))
    ring = 0.25 * np.cos(r / 3.0) * np.exp(-r / 40.0)
    psf = main + ring
    return (psf / psf.max()).astype(np.float32)


def empty_params(k: int = 5) -> MixParams:
    """A mixture that proposes nothing usable, so only the scale ladder runs."""
    return MixParams(logits=torch.zeros(1, k),
                     mu=torch.full((1, k, 7), float("nan")),
                     log_std=torch.zeros(1, k, 7))


def one_source(flux: float, sig: float, psf: np.ndarray):
    """Dirty image of a single centred Gaussian, and its true model."""
    truth = render_aspen(SIZE // 2, SIZE // 2, flux, sig, sig, 0.0, (SIZE, SIZE))
    return psf_convolve(truth, psf), truth


def commit_at(flux: float, sig: float) -> AspenCommit:
    return AspenCommit(cx=SIZE // 2, cy=SIZE // 2, flux=flux, sig_maj=sig,
                       sig_min=sig, pa=0.0, params=empty_params(), cut_scale=1.0)


def test_refit_recovers_size_and_flux_from_a_wrong_start():
    """A component committed at the wrong size and a tenth of the flux."""
    psf = gaussian_psf()
    true_flux, true_sig = 1.0, 4.0
    dirty, truth = one_source(true_flux, true_sig, psf)

    start = commit_at(flux=0.1 * true_flux, sig=1.4)
    model = render_aspen(SIZE // 2, SIZE // 2, start.flux, start.sig_maj,
                         start.sig_min, start.pa, (SIZE, SIZE))
    resid = dirty - psf_convolve(model, psf)

    for _ in range(6):        # coordinate descent, as the loop would call it
        resid, model, commits, stats = refit_commits(
            resid, model, psf, [start], sigma=1e-3, refit_gain=1.0,
            cutout=SIZE, sigma_min_px=0.5)
        start = commits[0]

    assert 1 / 1.5 < start.sig_maj / true_sig < 1.5, start.sig_maj
    assert 0.8 < start.flux / true_flux < 1.25, start.flux


def test_refit_never_increases_chi2():
    psf = gaussian_psf()
    dirty, _ = one_source(1.0, 3.0, psf)
    start = commit_at(flux=2.0, sig=8.0)          # too bright and too big
    model = render_aspen(SIZE // 2, SIZE // 2, start.flux, start.sig_maj,
                         start.sig_min, start.pa, (SIZE, SIZE))
    resid = dirty - psf_convolve(model, psf)

    for _ in range(4):
        resid, model, commits, stats = refit_commits(
            resid, model, psf, [start], sigma=1e-3, refit_gain=1.0,
            cutout=SIZE, sigma_min_px=0.5)
        assert stats["chi2_after"] <= stats["chi2_before"] * (1 + 1e-6), stats
        start = commits[0]


def test_refit_keeps_flux_non_negative_on_a_negative_bowl():
    """Fitting where the data is negative must not commit negative flux."""
    psf = gaussian_psf()
    dirty, _ = one_source(1.0, 3.0, psf)
    resid = -np.abs(dirty)                       # nothing positive to fit
    start = commit_at(flux=0.5, sig=3.0)
    model = render_aspen(SIZE // 2, SIZE // 2, start.flux, start.sig_maj,
                         start.sig_min, start.pa, (SIZE, SIZE))

    resid, model, commits, stats = refit_commits(
        resid, model, psf, [start], sigma=1e-3, refit_gain=1.0,
        cutout=SIZE, sigma_min_px=0.5)
    assert commits[0].flux >= 0.0
    assert model.min() >= -1e-12


def test_refit_respects_the_size_floor():
    psf = gaussian_psf()
    dirty, _ = one_source(1.0, 0.5, psf)         # a near-point source
    start = commit_at(flux=1.0, sig=3.0)
    model = render_aspen(SIZE // 2, SIZE // 2, start.flux, start.sig_maj,
                         start.sig_min, start.pa, (SIZE, SIZE))
    resid = dirty - psf_convolve(model, psf)

    for floor in (1.4, 0.5):
        _r, _m, commits, _s = refit_commits(
            resid, model, psf, [start], sigma=1e-3, refit_gain=1.0,
            cutout=SIZE, sigma_min_px=floor)
        assert commits[0].sig_maj >= floor - 1e-9


def test_refit_uses_the_mixture_shapes():
    """A mixture component at the true size must be reachable even when the
    scale ladder around the committed size does not contain it."""
    psf = gaussian_psf()
    true_sig = 5.0
    dirty, _ = one_source(1.0, true_sig, psf)
    start = commit_at(flux=1.0, sig=1.4)

    mu = torch.full((1, 5, 7), float("nan"))
    # 7D emitted layout: (x, y, log_flux, log_sig_maj, log_sig_min, sin2pa, cos2pa)
    mu[0, 0] = torch.tensor([0.0, 0.0, 0.0, math.log(true_sig),
                             math.log(true_sig), 0.0, 1.0])
    start = AspenCommit(**{**start.__dict__,
                           "params": MixParams(logits=torch.zeros(1, 5), mu=mu,
                                               log_std=torch.zeros(1, 5, 7))})
    model = render_aspen(SIZE // 2, SIZE // 2, start.flux, start.sig_maj,
                         start.sig_min, start.pa, (SIZE, SIZE))
    resid = dirty - psf_convolve(model, psf)

    _r, _m, commits, _s = refit_commits(
        resid, model, psf, [start], sigma=1e-3, refit_gain=1.0,
        cutout=SIZE, sigma_min_px=0.5)
    assert 1 / 1.2 < commits[0].sig_maj / true_sig < 1.2, commits[0].sig_maj


def test_size_bias_moves_the_choice_along_the_scale_ladder():
    """0 is the plain likelihood; a huge positive/negative bias picks the
    largest/smallest rung of the ladder around the committed size."""
    psf = gaussian_psf()
    sig = 3.0
    dirty, _ = one_source(1.0, sig, psf)
    start = commit_at(flux=1.0, sig=sig)
    model = render_aspen(SIZE // 2, SIZE // 2, start.flux, start.sig_maj,
                         start.sig_min, start.pa, (SIZE, SIZE))
    resid = dirty - psf_convolve(model, psf)

    def refit(**kw):
        return refit_commits(resid, model, psf, [start], sigma=1e-3,
                             refit_gain=1.0, cutout=SIZE, sigma_min_px=0.5, **kw)[2][0]

    plain, zero = refit(), refit(size_bias=0.0)
    assert (plain.sig_maj, plain.flux) == (zero.sig_maj, zero.flux)   # default unchanged
    assert 0.8 < plain.sig_maj / sig < 1.25, plain.sig_maj
    assert refit(size_bias=1e9).sig_maj == pytest.approx(1.6 * sig)
    assert refit(size_bias=-1e9).sig_maj == pytest.approx(0.6 * sig)


def test_error_map_follows_the_posterior_and_the_bias():
    """collect= returns the candidate posterior; with a huge size bias the shape
    posterior is a delta on one candidate, and the map is then s_c * g_c."""
    psf = gaussian_psf()
    sig = 3.0
    dirty, _ = one_source(1.0, sig, psf)
    start = AspenCommit(**{**commit_at(1.0, sig).__dict__, "cid": 7})
    model = render_aspen(SIZE // 2, SIZE // 2, start.flux, start.sig_maj,
                         start.sig_min, start.pa, (SIZE, SIZE))
    resid = dirty - psf_convolve(model, psf)

    recs = []
    refit_commits(resid, model, psf, [start], sigma=1e-3, refit_gain=1.0,
                  cutout=SIZE, sigma_min_px=0.5, collect=recs)
    assert len(recs) == 1 and recs[0]["cid"] == 7 and len(recs[0]["cands"]) >= 3

    shape = (SIZE, SIZE)
    plain = error_map_from_records(recs, shape)
    assert plain.shape == shape and np.isfinite(plain).all() and plain.min() >= 0
    big = error_map_from_records(recs, shape, size_bias=1e9)
    small = error_map_from_records(recs, shape, size_bias=-1e9)
    assert small.max() > big.max()           # narrow shape: higher, more compact peak

    smaj, smin_, pa, _J, den, _a, *_ = max(recs[0]["cands"], key=lambda c: c[0] * c[1])
    g = render_aspen(recs[0]["cx"], recs[0]["cy"], 1.0, smaj, smin_, pa, shape)
    expected = math.sqrt(recs[0]["sig2"] / den) * float(g.max())
    assert big.max() == pytest.approx(expected, rel=0.02)


def _shell(n=12, rad=28.0, flux=0.05, sig=4.0):
    """Overlapping components on a ring: the case where windowed subtraction
    leaves sidelobes behind and later Aspen fit them as their own flux."""
    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, n)
    out = []
    for t in th:
        out.append(commit_at(flux, sig))
        out[-1] = AspenCommit(**{**out[-1].__dict__,
                                 "cx": SIZE // 2 + rad * math.cos(t),
                                 "cy": SIZE // 2 + rad * math.sin(t)})
    return out


def test_returned_residual_matches_the_exact_operator():
    """The refit's own bookkeeping is windowed; the returned residual must
    still be r_start - B (model_new - model_old) under the FULL psf."""
    psf = sidelobe_psf()
    truth = _shell()
    sky = sum(render_aspen(c.cx, c.cy, c.flux, c.sig_maj, c.sig_min, c.pa,
                           (SIZE, SIZE)) for c in truth)
    dirty = psf_convolve(sky, psf)
    start = [AspenCommit(**{**c.__dict__, "flux": c.flux * 0.1}) for c in truth]
    model = sum(render_aspen(c.cx, c.cy, c.flux, c.sig_maj, c.sig_min, c.pa,
                             (SIZE, SIZE)) for c in start)
    resid = dirty - psf_convolve(model, psf)

    out_r, out_m, _c, stats = refit_commits(
        resid, model, psf, start, sigma=1e-4, refit_gain=0.5,
        cutout=64, sigma_min_px=0.5)
    exact = dirty - psf_convolve(out_m, psf)
    drift = float(np.abs(out_r - exact).max() / np.abs(exact).max())
    assert stats["exact_residual"] is True
    assert drift < 1e-5, drift


def test_windowed_bookkeeping_drifts_without_the_correction():
    """Guards the reason the correction exists: with it off, the returned
    residual is measurably not the exact one."""
    psf = sidelobe_psf()
    truth = _shell()
    sky = sum(render_aspen(c.cx, c.cy, c.flux, c.sig_maj, c.sig_min, c.pa,
                           (SIZE, SIZE)) for c in truth)
    dirty = psf_convolve(sky, psf)
    start = [AspenCommit(**{**c.__dict__, "flux": c.flux * 0.1}) for c in truth]
    model = sum(render_aspen(c.cx, c.cy, c.flux, c.sig_maj, c.sig_min, c.pa,
                             (SIZE, SIZE)) for c in start)
    resid = dirty - psf_convolve(model, psf)

    out_r, out_m, _c, _s = refit_commits(
        resid, model, psf, start, sigma=1e-4, refit_gain=0.5, cutout=64,
        sigma_min_px=0.5, exact_residual=False)
    exact = dirty - psf_convolve(out_m, psf)
    drift = float(np.abs(out_r - exact).max() / np.abs(exact).max())
    assert drift > 1e-3, drift


def test_zero_rejected_takes_flux_back_from_a_negative_bowl():
    """Over-subtracted: no positive amplitude fits, so the flux steps to 0."""
    psf = gaussian_psf()
    dirty, _ = one_source(1.0, 3.0, psf)
    resid = -np.abs(dirty)
    start = commit_at(flux=0.5, sig=3.0)
    model = render_aspen(SIZE // 2, SIZE // 2, start.flux, start.sig_maj,
                         start.sig_min, start.pa, (SIZE, SIZE))

    _, _, kept, _ = refit_commits(resid, model, psf, [start], sigma=1e-3,
                                  refit_gain=0.5, cutout=SIZE, sigma_min_px=0.5)
    assert kept[0].flux == start.flux

    r, m, commits, stats = refit_commits(resid, model, psf, [start], sigma=1e-3,
                                         refit_gain=0.5, cutout=SIZE,
                                         sigma_min_px=0.5, zero_rejected=True)
    assert stats["n_zeroed"] == 1
    assert commits[0].flux == pytest.approx(0.25)
    assert m.sum() == pytest.approx(0.25, rel=1e-4)
    assert stats["chi2_after"] < stats["chi2_before"]


def test_truncated_render_keeps_flux_inside_the_ellipse():
    full = render_aspen(64, 64, 2.0, 10.0, 5.0, 0.3, (128, 128))
    cut = render_aspen(64, 64, 2.0, 10.0, 5.0, 0.3, (128, 128), nsigma=3.0)
    assert np.array_equal(full, render_aspen(64, 64, 2.0, 10.0, 5.0, 0.3, (128, 128),
                                             nsigma=None))
    assert cut.sum() == pytest.approx(2.0, rel=1e-5)
    y, x = np.mgrid[:128, :128] - 64.0
    u = x * math.cos(0.3) + y * math.sin(0.3)
    v = -x * math.sin(0.3) + y * math.cos(0.3)
    assert np.all(cut[(u / 10.0) ** 2 + (v / 5.0) ** 2 > 9.0] == 0.0)


def test_refit_window_bookkeeping_is_exact_for_a_large_truncated_aspen():
    """sigma 42 at 3 sigma fits a 256 window, so the model the refit returns is
    exactly the full-frame render of the component it returns."""
    n, c = 512, 256
    psf = gaussian_psf(n)
    dirty = psf_convolve(render_aspen(256, 256, 1.0, 42.0, 30.0, 0.5, (n, n),
                                      nsigma=3.0), psf)
    start = AspenCommit(cx=256, cy=256, flux=0.3, sig_maj=42.0, sig_min=30.0, pa=0.5,
                        params=empty_params(), cut_scale=1.0)
    model = render_aspen(256, 256, 0.3, 42.0, 30.0, 0.5, (n, n), nsigma=3.0)
    resid = dirty - psf_convolve(model, psf)
    _, out_m, out, _ = refit_commits(resid, model, psf, [start], sigma=1e-3,
                                     refit_gain=1.0, cutout=c, sigma_min_px=0.5,
                                     sigma_max_px=42.0, render_nsigma=3.0)
    o = out[0]
    again = render_aspen(o.cx, o.cy, o.flux, o.sig_maj, o.sig_min, o.pa, (n, n), nsigma=3.0)
    assert np.abs(out_m - again).max() < 1e-6 * again.max()
    assert o.flux == pytest.approx(1.0, rel=0.05)


def test_psf_convolver_matches_psf_convolve():
    rng = np.random.default_rng(0)
    for n, h in [(64, 64), (64, 33), (50, 64)]:
        img = rng.standard_normal((n, n)).astype(np.float32)
        psf = rng.standard_normal((h, h)).astype(np.float32)
        ref = psf_convolve(img, psf)
        got = _mc.PSFConvolver(psf, (n, n))(img)
        assert np.abs(got - ref).max() < 1e-4 * np.abs(ref).max()


def test_truncated_render_box_matches_full_frame_evaluation():
    for cx, cy, smaj, smin, pa in [(64, 64, 10.0, 5.0, 0.3), (3.2, 120.7, 8.0, 2.0, 1.1),
                                   (64.0, 64.0, 0.1, 0.1, 0.0)]:
        cut = render_aspen(cx, cy, 2.0, smaj, smin, pa, (128, 128), nsigma=3.0)
        y, x = np.mgrid[:128, :128].astype(np.float64)
        u = (x - cx) * math.cos(pa) + (y - cy) * math.sin(pa)
        v = -(x - cx) * math.sin(pa) + (y - cy) * math.cos(pa)
        e = 0.5 * ((u / smaj) ** 2 + (v / smin) ** 2)
        g = np.where(e > 4.5, 0.0, np.exp(-e))
        ref = (g * 2.0 / g.sum()).astype(np.float32)
        assert np.abs(cut - ref).max() <= 1e-6 * ref.max()


class _StubMDN(torch.nn.Module):
    """Proposes one fixed shape and records the input it was given."""

    def __init__(self, sig: float, cutout: int = SIZE):
        super().__init__()
        self.cutout = cutout
        self._p = torch.nn.Parameter(torch.zeros(1))
        self.sig, self.seen = sig, []

    def forward(self, img, cond):
        self.seen.append(img[0, 0].detach().cpu().numpy())
        mu = torch.full((1, 5, 7), float("nan"))
        mu[0, 0] = torch.tensor([0.0, 0.0, 0.0, math.log(self.sig), math.log(self.sig), 0.0, 1.0])
        return MixParams(logits=torch.zeros(1, 5), mu=mu, log_std=torch.zeros(1, 5, 7))


def test_repropose_adds_the_fresh_network_shape():
    """The stored mixture is useless and the ladder cannot reach the true size
    in one step; a re-proposal at the true size must be picked, and the network
    must see the add-back residual."""
    psf = gaussian_psf()
    true_sig = 5.0
    dirty, _ = one_source(1.0, true_sig, psf)
    start = commit_at(flux=0.3, sig=1.4)
    model = render_aspen(SIZE // 2, SIZE // 2, start.flux, start.sig_maj,
                         start.sig_min, start.pa, (SIZE, SIZE))
    resid = dirty - psf_convolve(model, psf)

    _r, _m, off, _s = refit_commits(resid, model, psf, [start], sigma=1e-3,
                                    refit_gain=1.0, cutout=SIZE, sigma_min_px=0.5)
    assert off[0].sig_maj < 2.5, off[0].sig_maj

    stub = _StubMDN(true_sig)
    _r, _m, on, _s = refit_commits(resid, model, psf, [start], sigma=1e-3,
                                   refit_gain=1.0, cutout=SIZE, sigma_min_px=0.5,
                                   repropose=dict(model=stub, config_idx=3,
                                                  scale_relative=False, global_sigma=True))
    assert 1 / 1.2 < on[0].sig_maj / true_sig < 1.2, on[0].sig_maj
    # Single Aspen: its add-back residual is the dirty image.
    assert np.allclose(stub.seen[0], dirty, atol=1e-5)


def test_gradient_refit_moves_and_reshapes_a_wrong_aspen():
    """Wrong position, size, PA and flux: the joint fit must recover them,
    which the discrete refit cannot (its position is fixed)."""
    psf = gaussian_psf()
    true = dict(cx=62.3, cy=65.7, flux=1.0, sig_maj=4.0, sig_min=2.5, pa=0.4)
    truth = render_aspen(true["cx"], true["cy"], true["flux"], true["sig_maj"],
                         true["sig_min"], true["pa"], (SIZE, SIZE))
    dirty = psf_convolve(truth, psf)
    start = AspenCommit(cx=64.0, cy=64.0, flux=0.5, sig_maj=3.0, sig_min=3.0, pa=0.0,
                        params=empty_params(), cut_scale=1.0)
    model = render_aspen(start.cx, start.cy, start.flux, start.sig_maj, start.sig_min,
                         start.pa, (SIZE, SIZE))
    resid = dirty - psf_convolve(model, psf)

    r, m, out, st = _refit.refit_gradient(resid, model, psf, [start], sigma=1e-3,
                                          n_iter=300, sigma_min_px=0.5, max_window=SIZE)
    a = out[0]
    assert st["accepted"] and st["chi2_end"] < 0.01 * st["chi2_start"], st
    assert math.hypot(a.cx - true["cx"], a.cy - true["cy"]) < 0.3, (a.cx, a.cy)
    assert abs(a.flux - 1.0) < 0.1, a.flux
    assert abs(a.sig_maj - 4.0) < 0.4 and abs(a.sig_min - 2.5) < 0.4, (a.sig_maj, a.sig_min)
    # Bookkeeping exact: returned residual = dirty - B(returned model).
    assert np.allclose(r, dirty - psf_convolve(m, psf), atol=1e-5)
