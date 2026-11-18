"""MAP amplitudes in the refit and the joint solve.

The prior is the SBI proposal's own flux posterior; the ridge term is
sigma^2/tau^2 on the diagonal, identically in both solvers.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from mdn_asp.models.mdn_asp import MixParams


# conftest's proxy package maps mdn_asp.* to the flat root modules, so the
# packaged modules are loaded by path (same shim as tests/test_refit.py).
def _by_path(name: str):
    import importlib.util
    import sys
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "mdn_asp" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"mdn_asp.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_mc = _by_path("minor_cycle")
AspenCommit, psf_convolve, render_aspen = (
    _mc.AspenCommit, _mc.psf_convolve, _mc.render_aspen)
_refit = _by_path("refit")
flux_prior, refit_commits = _refit.flux_prior, _refit.refit_commits


def _commit(cx, cy, flux, prior_mean_jy, log_flux_sd, cut_scale=1.0, sig=2.0):
    """One commit whose proposal is a single mixture component with a prior
    mean of ``prior_mean_jy`` Jy.

    Dim 2 of the MDN is a STANDARDISED log flux in absolute mode, so the test
    must push the physical mean through ``standardise_log_flux`` -- writing
    ``log(mean)`` there instead makes the decoded prior exp(3.45*log(mean) -
    5.76), which for 1 mJy is 1.4e-13 Jy, and the test then passes for the
    wrong reason.
    """
    from mdn_asp.constants import standardise_log_flux
    mu = torch.zeros(1, 1, 7)
    if cut_scale != 1.0:
        mu[0, 0, 2] = math.log(prior_mean_jy / cut_scale)
    else:
        mu[0, 0, 2] = standardise_log_flux(math.log(prior_mean_jy))
    log_std = torch.full((1, 1, 7), -6.0)
    log_std[0, 0, 2] = math.log(log_flux_sd)
    return AspenCommit(cx=cx, cy=cy, flux=flux, sig_maj=sig, sig_min=sig, pa=0.0,
                       params=MixParams(logits=torch.zeros(1, 1), mu=mu,
                                        log_std=log_std),
                       cut_scale=cut_scale)


def _psf(n=64, s=1.6):
    y, x = np.mgrid[:n, :n]
    g = np.exp(-(((y - n // 2) ** 2 + (x - n // 2) ** 2) / (2 * s * s)))
    return (g / g.max()).astype(np.float32)


@pytest.mark.parametrize("cut_scale", [1.0, 4.0e-3])
def test_flux_prior_round_trips_the_mean_in_both_conventions(cut_scale):
    """The decoded prior mean must be the physical flux that went in, under
    BOTH the scale-relative and the legacy absolute standardisation."""
    from mdn_asp.constants import LOG_FLUX_SCALE
    want, sd = 2.5e-3, 0.25
    mean, tau = flux_prior(_commit(32, 32, 1e-3, want, sd, cut_scale=cut_scale))
    assert mean == pytest.approx(want, rel=1e-6)
    expect_sd = sd if cut_scale != 1.0 else sd * LOG_FLUX_SCALE
    assert tau == pytest.approx(mean * expect_sd, rel=1e-3)


def test_zero_weight_is_bit_identical_to_ml():
    psf = _psf()
    truth = render_aspen(32.0, 32.0, 5.0e-3, 2.0, 2.0, 0.0, (64, 64))
    resid = psf_convolve(truth, psf).astype(np.float32)
    commits = [_commit(32.0, 32.0, 5.0e-4, 3.0e-3, 0.3)]
    model = np.zeros((64, 64), np.float32)
    a = refit_commits(resid.copy(), model.copy(), psf, commits,
                      sigma=1e-5, flux_prior_weight=0.0)
    b = refit_commits(resid.copy(), model.copy(), psf, commits,
                      sigma=1e-5)
    assert a[3]["n_prior_applied"] == 0
    np.testing.assert_array_equal(a[0], b[0])
    assert a[2][0].flux == b[2][0].flux


def test_tight_prior_pulls_amplitude_toward_proposal_mean():
    """A tight prior must move the refit amplitude off the ML value, toward
    the proposal mean. This is the whole point of the MAP step."""
    psf = _psf()
    # Data holds 5 mJy; the proposal insists on 1 mJy with a tight posterior.
    truth = render_aspen(32.0, 32.0, 5.0e-3, 2.0, 2.0, 0.0, (64, 64))
    resid = psf_convolve(truth, psf).astype(np.float32)
    model = np.zeros((64, 64), np.float32)
    prior_mean, prior_sd = 1.0e-3, 0.02
    c = [_commit(32.0, 32.0, 5.0e-4, prior_mean, prior_sd)]

    ml = refit_commits(resid.copy(), model.copy(), psf, c, sigma=1e-5,
                       refit_gain=1.0, flux_prior_weight=0.0)[2][0].flux
    mp = refit_commits(resid.copy(), model.copy(), psf, c, sigma=1e-5,
                       refit_gain=1.0, flux_prior_weight=1e-3)[2][0].flux
    assert mp < ml, f"MAP {mp:.4e} did not fall below ML {ml:.4e}"
    assert abs(mp - prior_mean) < abs(ml - prior_mean)
    # and it must land BETWEEN the prior and the ML value, not outside them
    assert prior_mean <= mp <= ml, (prior_mean, mp, ml)

