"""GPUCutoutGenerator: PSF-shaped noise and point-size label."""
import math

import torch

from mdn_asp.data.gpu_cutout_generator import GPUCutoutGenerator


def _gaussian_psf(size: int, sigma: float) -> torch.Tensor:
    ar = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-0.5 * (ar.view(-1, 1) ** 2 + ar.view(1, -1) ** 2) / sigma ** 2)
    return g / g.max()


def _gen(**kw) -> GPUCutoutGenerator:
    bank = _gaussian_psf(128, 2.0).unsqueeze(0)
    return GPUCutoutGenerator(bank, field_size=256, cutout_size=128,
                              n_sources=(1, 2), device="cpu", seed=0, **kw)


def test_psf_shaped_noise_variance_and_correlation():
    gen = _gen(correlated_noise=True)
    psf = gen.psf_bank[[0] * 16]
    x = gen._psf_shaped_noise(psf)
    var = float(x.var())
    lag1 = float((x[:, :, 1:] * x[:, :, :-1]).mean() / x.var())
    expected = math.exp(-0.5 / 2.0 ** 2)     # PSF(1 px) for sigma = 2 px
    assert abs(var - 1.0) < 0.05
    assert abs(lag1 - expected) < 0.05


def test_point_sigma_label():
    gen = _gen(point_sigma_px=0.25, morphologies=("point",))
    _, _, target = gen.sample(8)
    assert torch.allclose(target[:, 3], torch.full((8,), math.log(0.25)))
    assert torch.allclose(target[:, 4], torch.full((8,), math.log(0.25)))


def test_defaults_unchanged():
    gen = _gen(morphologies=("point",))
    _, _, target = gen.sample(4)
    assert torch.allclose(target[:, 3], torch.full((4,), math.log(1.4)))


def test_convolution_matches_deployment_operator():
    """Training dirty images must put a point's response where deployment does."""
    import numpy as np
    from mdn_asp.data.gpu_cutout_generator import fft_convolve_same
    from mdn_asp.minor_cycle import psf_convolve

    psf = _gaussian_psf(128, 2.0)
    sky = torch.zeros(1, 256, 256)
    sky[0, 100, 150] = 1.0
    ref = psf_convolve(sky[0].numpy(), psf.numpy())
    gpu = fft_convolve_same(sky, psf.unsqueeze(0))[0].numpy()
    for img in (ref, gpu):
        assert np.unravel_index(int(np.argmax(img)), img.shape) == (100, 150)
    assert np.allclose(gpu, ref, atol=1e-5)


def test_sigma_max_and_truncation_at_256():
    bank = _gaussian_psf(256, 2.0).unsqueeze(0)
    gen = GPUCutoutGenerator(bank, field_size=512, cutout_size=256, n_sources=(1, 2),
                             morphologies=("blob",), sigma_max_px=42.0,
                             render_nsigma=3.0, device="cpu", seed=0)
    img, _, target = gen.sample(8)
    assert img.shape == (8, 2, 256, 256)
    assert float(target[:, 3].exp().max()) <= 42.0
    one = gen._render_extended_flat(torch.zeros(1, dtype=torch.long),
                                    torch.tensor([256.0]), torch.tensor([256.0]),
                                    torch.tensor([20.0]), torch.tensor([20.0]),
                                    torch.tensor([0.0]), torch.ones(1))[0]
    assert abs(float(one.sum()) - 1.0) < 1e-5
    assert float(one[256, 256 + 61]) == 0.0 and float(one[256, 256 + 59]) > 0.0


def test_edged_label_fits_inside_the_top_hat():
    import math
    from mdn_asp.data.gpu_cutout_generator import GPUCutoutGenerator
    psf = torch.zeros(1, 64, 64); psf[0, 32, 32] = 1.0
    g = GPUCutoutGenerator(psf, field_size=128, cutout_size=64, morphologies=("edged",),
                           sigma_max_px=12.0, render_nsigma=3.0, device="cpu", seed=3)
    img, flux, tx, ty, smaj, smin, pa, cr, cc = g._centred_source(8, g.psf_bank[[0] * 8])
    for i in range(8):
        sky = img[i]
        inside = sky > 0
        bright = float(sky[inside].max())
        assert torch.allclose(sky[inside], torch.full_like(sky[inside], bright))  # uniform
        # label Gaussian (peak = bright, 3 sigma footprint) stays under the top-hat
        yy, xx = torch.meshgrid(torch.arange(128.), torch.arange(128.), indexing="ij")
        u = (xx - tx[i]) * math.cos(pa[i]) + (yy - ty[i]) * math.sin(pa[i])
        v = -(xx - tx[i]) * math.sin(pa[i]) + (yy - ty[i]) * math.cos(pa[i])
        e = 0.5 * ((u / smaj[i]) ** 2 + (v / smin[i]) ** 2)
        lab = torch.where(e <= 4.5, bright * torch.exp(-e), torch.zeros_like(e))
        assert float((lab - sky).max()) <= 1e-6 * bright
        assert abs(float(lab.sum()) - float(flux[i])) <= 1e-4 * float(flux[i])
