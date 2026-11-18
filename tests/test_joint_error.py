import numpy as np
import torch

from mdn_asp.minor_cycle import PSFConvolver
from mdn_asp.refit import error_map_from_records, joint_amplitude_error


def _setup():
    H = W = 64
    yy, xx = np.mgrid[:H, :W]
    psf = np.exp(-0.5 * ((yy - 32) ** 2 + (xx - 32) ** 2) / 2.0 ** 2).astype(np.float32)
    wy, wx = np.mgrid[:16, :16]
    u = np.exp(-0.5 * ((wy - 8) ** 2 + (wx - 8) ** 2) / 3.0 ** 2).astype(np.float32)
    return H, W, psf, u / u.sum()


def _ubu(img, psf):
    c = PSFConvolver(psf, img.shape, torch.device("cpu"))
    return float((img * c.convolve_psd_torch(torch.from_numpy(img)).numpy()).sum())


def test_single_atom_matches_closed_form():
    H, W, psf, u = _setup()
    sd, st = joint_amplitude_error([(20, 36, 20, 36, u)], [1.0], (H, W), psf, 0.1)
    img = np.zeros((H, W), np.float32)
    img[20:36, 20:36] = u
    ref = 0.1 / np.sqrt(_ubu(img, psf)) * img
    assert st["n_active"] == 1 and st["n_floored"] == 0
    assert np.abs(sd - ref).max() <= 1e-6 * ref.max()


def test_overlapping_atoms_are_anticorrelated():
    H, W, psf, u = _setup()
    u2 = np.roll(u, 2, axis=1)
    sd, _ = joint_amplitude_error([(20, 36, 20, 36, u), (20, 36, 20, 36, u2)], [1.0, 1.0], (H, W), psf, 0.1)
    ind = np.zeros((H, W))
    for v in (u, u2):
        img = np.zeros((H, W), np.float32)
        img[20:36, 20:36] = v
        ind += (0.1 / np.sqrt(_ubu(img, psf)) * img) ** 2
    # Var(a1 u1 + a2 u2) with negative Cov(a1, a2) is below the independent sum where the atoms overlap
    assert sd.max() < np.sqrt(ind).max()


def test_inactive_atoms_are_ignored():
    H, W, psf, u = _setup()
    sd, st = joint_amplitude_error([(20, 36, 20, 36, u), (0, 16, 0, 16, u)], [1.0, 0.0], (H, W), psf, 0.1)
    assert st["n_active"] == 1
    assert sd[0:16, 0:16].max() == 0.0


def test_error_map_include_amp_false_drops_amplitude_term():
    rec = {"cx": 32.0, "cy": 32.0, "sig2": 0.01, "cands": [[3.0, 3.0, 0.0, 5.0, 2.0, 1.0]]}
    full = error_map_from_records([rec], (64, 64))
    shape_only = error_map_from_records([rec], (64, 64), include_amp=False)
    assert full.max() > 0.0
    assert shape_only.max() == 0.0   # one candidate: no shape spread


def test_mixture_prior_suppresses_zero_weight_candidate():
    # A point-like candidate with the best likelihood but ~zero mixture weight must not drive the error.
    rec = {"cx": 32.0, "cy": 32.0, "sig2": 0.01,
           "cands": [[6.0, 2.0, 0.0, 5.0, 2.0, 1.0, np.log(0.999)],
                     [0.55, 0.55, 0.0, 5.0, 2.0, 1.0, np.log(1e-6)]]}
    flat = [[*c[:6]] for c in rec["cands"]]
    with_prior = error_map_from_records([rec], (64, 64), include_amp=False)
    no_prior = error_map_from_records([{**rec, "cands": flat}], (64, 64), include_amp=False)
    assert with_prior.max() < 1e-2 * no_prior.max()


def test_candidate_log_prior_matches_mixture_and_inherits_nearest():
    from types import SimpleNamespace
    from mdn_asp.refit import _mixture_shapes, candidate_log_prior
    mu = torch.zeros(1, 2, 7)
    mu[0, 0, 3], mu[0, 0, 4] = np.log(6.0), np.log(2.0)      # 7d layout: x, y, flux, log smaj, log smin, sin2t, cos2t
    mu[0, 1, 3], mu[0, 1, 4] = np.log(0.6), np.log(0.6)
    mu[0, :, 6] = 1.0
    com = SimpleNamespace(params=SimpleNamespace(logits=torch.tensor([[2.0, -8.0]]), mu=mu))
    shapes = _mixture_shapes(com)
    lp = candidate_log_prior(com, shapes + [(shapes[0][0] * 1.25, shapes[0][1] * 1.25, shapes[0][2])],
                             cur=shapes[0][:2], sigma_min=0.5, sigma_max=20.0)
    lw = torch.log_softmax(torch.tensor([2.0, -8.0], dtype=torch.float64), 0).numpy()
    assert np.allclose(lp, [lw[0], lw[1], lw[0]])
