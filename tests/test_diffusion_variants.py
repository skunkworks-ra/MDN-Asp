"""T1: diffusion variants A and B see different information (scripts/train_tiled_asp.py).

A sees only the residual d - PSF * m, so two (sky, m) pairs with the same residual sky and the
same residual image must give A identical input and target. B sees d and m, so its input differs.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

_path = Path(__file__).resolve().parents[1] / "scripts" / "train_tiled_asp.py"
_spec = importlib.util.spec_from_file_location("train_tiled_asp", _path)
tta = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tta)

S = 64


def _args(variant):
    return argparse.Namespace(variant=variant, T=1000, sky_scale=10.0, patch=S, self_cond_p=0.0)


def _pair():
    """Two batches: (sky, m=0, d) and (sky + m2, m2, d + PSF * m2). Same residual sky, same residual."""
    g = torch.Generator().manual_seed(0)
    psf = torch.zeros(1, S, S)
    yy, xx = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
    psf[0] = torch.exp(-0.5 * ((yy - S // 2) ** 2 + (xx - S // 2) ** 2) / 2.0 ** 2)
    psf[0, S // 2, :] += 0.1                                     # a sidelobe spoke
    sky = torch.zeros(1, S, S)
    sky[0, 20, 30], sky[0, 40, 40] = 1.0, 0.5
    m2 = torch.rand(1, S, S, generator=g) * 0.3
    d = tta.fft_convolve_same(sky, psf)
    d2 = d + tta.fft_convolve_same(m2, psf)
    common = {"psf": psf, "psf_cut": psf, "sigma": torch.tensor([1e-3])}
    b1 = {**common, "img": d, "sky": sky}
    b2 = {**common, "img": d2, "sky": sky + m2}
    return b1, torch.zeros_like(m2), b2, m2


def _inputs(variant, batch, m):
    """(network input channels, target) as Diffusion.eps / Diffusion.loss build them."""
    a = _args(variant)
    model = tta.UNet(3 if variant == "A" else 4, base=32)
    diff = tta.Diffusion(model, a, "cpu")
    img, sig, u = diff.condition(batch, m)
    seen = {}
    model.forward = lambda x, t, log_sigma: (seen.update(x=x, ls=log_sigma), x[:, :1])[1]
    diff.eps(torch.zeros(1, S, S), torch.zeros(1, dtype=torch.long), img, sig, batch, m)
    return seen["x"], seen["ls"], diff.target(batch, m, u)


def test_a_sees_only_the_residual():
    b1, m1, b2, m2 = _pair()
    x1, ls1, t1 = _inputs("A", b1, m1)
    x2, ls2, t2 = _inputs("A", b2, m2)
    assert x1.shape[1] == 3                                      # x_t, residual, PSF: no m channel
    assert torch.allclose(x1, x2, atol=1e-5)
    assert torch.allclose(ls1, ls2)
    assert torch.allclose(t1, t2, atol=1e-5)


def test_b_sees_d_and_m():
    b1, m1, b2, m2 = _pair()
    x1, _, t1 = _inputs("B", b1, m1)
    x2, _, t2 = _inputs("B", b2, m2)
    assert x1.shape[1] == 4                                      # x_t, dirty, PSF, m
    assert not torch.allclose(x1[:, 1], x2[:, 1], atol=1e-3)     # dirty differs
    assert not torch.allclose(x1[:, 3], x2[:, 3], atol=1e-3)     # m differs
    assert not torch.allclose(t1, t2, atol=1e-3)                 # target = full sky


def test_val_loss_is_deterministic_with_own_passes():
    """Same weights, same generator seed and k: same loss (A and B with one own pass; B with
    one-step self-conditioning). Output layer perturbed so eps, m and the draws matter."""
    b1, _, _, _ = _pair()
    b1["sky"] = b1["sky"] / 10.0
    for variant, passes in (("A", 1), ("B", 1), ("B", 0)):
        a = _args(variant)
        a.a_passes, a.m_sample_steps, a.self_cond_p = passes, 3, 1.0
        torch.manual_seed(1)
        net = tta.UNet(3 if variant == "A" else 4, base=32)
        torch.nn.init.normal_(net.out[-1].weight, std=0.1)
        diff = tta.Diffusion(net, a, "cpu")
        k = torch.ones(1, dtype=torch.long) if passes else None
        l1 = diff.loss(b1, gen=torch.Generator().manual_seed(5), k=k)
        torch.rand(10)                                          # global RNG moves; loss must not
        l2 = diff.loss(b1, gen=torch.Generator().manual_seed(5), k=k)
        assert torch.isfinite(l1) and l1 == l2
