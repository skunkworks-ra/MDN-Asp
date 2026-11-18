"""Unit tests for publication math fixes, NNLS refit, and diffusion upgrades."""
from __future__ import annotations

import argparse
import importlib.util
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _load_script(stem: str):
    path = REPO_ROOT / "scripts" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[stem] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def repo_scratch():
    """Temporary test directory inside tests/ (never touches /tmp)."""
    d = Path(__file__).resolve().parent / "_scratch_pub_fixes"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _gaussian_psf(size: int = 128, sigma: float = 2.0) -> np.ndarray:
    y = np.arange(size) - size // 2
    g = np.exp(-(y[:, None] ** 2 + y[None, :] ** 2) / (2.0 * sigma ** 2))
    return (g / g.max()).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. MDN-Asp von Mises PA head, density mode, uniform point PA, HF round-trip
# ---------------------------------------------------------------------------

def test_von_mises_pa_density_integrates_to_one():
    from mdn_asp.models.mdn_asp import MDNAsp, MixParams, make_cond

    model = MDNAsp(base_channels=8, hidden=32, n_components=1, pa_head="von_mises")
    m_grid = 800
    step = math.pi / m_grid
    pg = torch.linspace(-0.5 * math.pi + 0.5 * step, 0.5 * math.pi - 0.5 * step, m_grid)
    log_norm_5d = -2.5 * math.log(2.0 * math.pi)

    for log_kappa in (-3.0, 0.0, 2.0, 5.0):
        for mu_pa in (-1.1, 0.0, 0.8):
            logits = torch.zeros(1, 1)
            mu = torch.zeros(1, 1, 7)
            mu[:, 0, 5] = math.sin(2.0 * mu_pa)
            mu[:, 0, 6] = math.cos(2.0 * mu_pa)
            log_std = torch.zeros(1, 1, 7)
            log_std[:, 0, 5] = log_kappa

            log_p_pa = []
            for i in range(m_grid):
                t_i = torch.zeros(1, 6)
                t_i[0, 5] = pg[i]
                nll_i = model.nll_loss(MixParams(logits, mu, log_std), t_i).item()
                log_p_pa.append(-nll_i - log_norm_5d)
            integral = float(np.sum(np.exp(np.array(log_p_pa))) * step)
            assert abs(integral - 1.0) < 5e-4, f"integral={integral} for log_kappa={log_kappa}, mu_pa={mu_pa}"

    # End-to-end forward, backward, sample, and density_mode check
    img = torch.randn(3, 2, 128, 128)
    cond = make_cond(torch.tensor([0.01, 0.02, 0.05]), torch.tensor([3, 3, 3]))
    params = model(img, cond)
    assert params.mu.shape == (3, 1, 7)
    tgt = torch.zeros(3, 6)
    tgt[:, 5] = torch.tensor([-0.4, 0.0, 1.2])
    loss = model.nll_loss(params, tgt)
    loss.backward()
    assert torch.isfinite(loss)
    samples = model.sample(params, n=16)
    assert samples.shape == (3, 16, 6)
    assert (samples[..., 5] >= -0.5 * math.pi - 1e-5).all()
    assert (samples[..., 5] <= 0.5 * math.pi + 1e-5).all()

    # Density mode picks sharp component over diffuse component of slightly higher weight
    m2 = MDNAsp(base_channels=8, hidden=32, n_components=2, pa_head="von_mises")
    logits2 = torch.tensor([[math.log(0.55), math.log(0.45)]])
    mu2 = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                         [0.0, 0.0, 0.0, 2.5, 2.5, 1.0, 0.0]]])
    log_std2 = torch.tensor([[[1.5, 1.5, 1.5, 1.5, 1.5, -2.0, 0.0],
                              [-1.5, -1.5, -1.5, -1.5, -1.5, 3.0, 0.0]]])
    p2 = MixParams(logits2, mu2, log_std2)
    w_mode = m2.mode(p2, density_mode=False)
    d_mode = m2.mode(p2, density_mode=True)
    assert abs(float(w_mode[0, 3]) - 1.0) < 1e-5
    assert abs(float(d_mode[0, 3]) - 2.5) < 1e-5


def test_uniform_point_pa_and_hf_roundtrip(repo_scratch):
    from mdn_asp.data.gpu_cutout_generator import GPUCutoutGenerator
    from mdn_asp.models.mdn_asp import (
        MDNAsp,
        checkpoint_is_scale_relative,
        checkpoint_sigma_min_px,
        load_mdn_checkpoint,
        make_cond,
    )

    psf = torch.from_numpy(_gaussian_psf(128, 2.0)).unsqueeze(0)
    gen_zero = GPUCutoutGenerator(psf, field_size=256, cutout_size=128, n_sources=(1, 2),
                                  morphologies=("point",), uniform_point_pa=False,
                                  device="cpu", seed=7)
    _, _, tgt_zero = gen_zero.sample(32)
    assert torch.allclose(tgt_zero[:, 5], torch.zeros_like(tgt_zero[:, 5]))

    gen_uni = GPUCutoutGenerator(psf, field_size=256, cutout_size=128, n_sources=(1, 2),
                                 morphologies=("point",), uniform_point_pa=True,
                                 device="cpu", seed=7)
    _, _, tgt_uni = gen_uni.sample(32)
    assert float(tgt_uni[:, 5].std()) > 0.4
    assert (tgt_uni[:, 5].abs() <= 0.5 * math.pi + 1e-5).all()

    # Save .pt checkpoint -> convert to HF -> reload and check metadata + outputs
    model = MDNAsp(
        base_channels=8, hidden=32, n_components=4, cutout=128,
        pa_head="von_mises", scale_relative=True, point_sigma_px=0.5,
    ).eval()
    pt_path = repo_scratch / "best.pt"
    torch.save({
        "model": model.state_dict(),
        "step": 100,
        "val_nll": -1.23,
        "cutout": 128,
        "pa_head": "von_mises",
        "args": {
            "base_channels": 8,
            "hidden": 32,
            "n_components": 4,
            "cutout": 128,
            "pa_head": "von_mises",
            "scale_relative": True,
            "point_sigma_px": 0.5,
            "sigma_max_px": 48.0,
        },
    }, pt_path)

    conv_mod = _load_script("convert_checkpoint_to_hf")
    hf_dir = repo_scratch / "hf_model"
    conv_mod.main(["--ckpt", str(pt_path), "--out_dir", str(hf_dir)])

    assert checkpoint_is_scale_relative(str(hf_dir)) is True
    assert abs(checkpoint_sigma_min_px(str(hf_dir)) - 0.5) < 1e-6
    reloaded = load_mdn_checkpoint(str(hf_dir), "cpu")
    assert reloaded.pa_head == "von_mises"

    x = torch.randn(2, 2, 128, 128)
    c = make_cond(torch.tensor([0.02, 0.05]), torch.tensor([3, 3]))
    with torch.no_grad():
        p_orig = model(x, c)
        p_new = reloaded(x, c)
    assert torch.allclose(p_orig.logits, p_new.logits, atol=1e-6)
    assert torch.allclose(p_orig.mu, p_new.mu, atol=1e-6)

    run_img = _load_script("run_imaging")
    digest = run_img._sha256(str(hf_dir))
    assert isinstance(digest, str) and len(digest) == 64


# ---------------------------------------------------------------------------
# 2. PSFConvolver exact adjoint, canonical axes, sidelobe detector, minor cycle
# ---------------------------------------------------------------------------

def test_psf_convolver_exact_adjoint_and_psd():
    from mdn_asp.minor_cycle import PSFConvolver

    rng = np.random.default_rng(42)
    psf = np.zeros((35, 41), dtype=np.float32)
    yy, xx = np.mgrid[:35, :41]
    psf += np.exp(-0.5 * ((yy - 17) ** 2 + (xx - 20) ** 2) / 2.2 ** 2).astype(np.float32)
    psf[10, 29] += 0.22
    psf[26, 8] -= 0.15
    psf /= psf.max()

    conv = PSFConvolver(psf, (48, 52), "cpu")
    x = rng.standard_normal((48, 52)).astype(np.float32)
    y = rng.standard_normal((48, 52)).astype(np.float32)

    bx = conv(x)
    bty = conv.adjoint(y)
    lhs = float(np.sum(bx.astype(np.float64) * y.astype(np.float64)))
    rhs = float(np.sum(x.astype(np.float64) * bty.astype(np.float64)))
    assert abs(lhs - rhs) / max(abs(lhs), 1.0) < 1e-5

    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    bpx = conv.convolve_psd_torch(xt)
    bpy = conv.convolve_psd_torch(yt)
    assert torch.allclose((bpx * yt).sum(), (xt * bpy).sum(), rtol=1e-5, atol=1e-5)
    assert float((xt * bpx).sum()) >= -1e-5


def test_canonical_axes_preserves_ellipse():
    from mdn_asp.minor_cycle import _canonical_axes, render_aspen

    smaj, smin, pa = _canonical_axes(2.0, 5.5, 0.35, 0.5, 96.0)
    assert abs(smaj - 5.5) < 1e-6
    assert abs(smin - 2.0) < 1e-6
    img_raw = render_aspen(32.0, 32.0, 1.0, 2.0, 5.5, 0.35, (64, 64))
    img_can = render_aspen(32.0, 32.0, 1.0, smaj, smin, pa, (64, 64))
    assert np.allclose(img_raw, img_can, atol=1e-6)


def test_measure_psf_sidelobe_positive_ring_and_pure_gaussian():
    from mdn_asp.minor_cycle import measure_psf_sidelobe

    pure = _gaussian_psf(128, 2.5)
    assert measure_psf_sidelobe(pure) == 0.0

    yy, xx = np.mgrid[:128, :128] - 64
    r = np.hypot(yy, xx)
    ring_psf = np.exp(-0.5 * (r / 2.2) ** 2) - 0.04 * np.exp(-0.5 * ((r - 8.0) / 1.5) ** 2)
    ring_psf += 0.14 * np.exp(-0.5 * ((r - 16.0) / 1.8) ** 2)
    ring_psf = (ring_psf / ring_psf.max()).astype(np.float32)
    sl = measure_psf_sidelobe(ring_psf)
    assert 0.12 <= sl <= 0.16, f"expected outer positive ring ~0.14, got {sl}"


def test_minor_cycle_negative_residual_flux_clip_and_uv_guard():
    from mdn_asp.constants import standardise_log_flux
    from mdn_asp.minor_cycle import minor_cycle, psf_convolve, render_aspen
    from mdn_asp.models.mdn_asp import MDNAsp, MixParams

    psf = _gaussian_psf(128, 2.0)
    tiny = MDNAsp(base_channels=8, hidden=16, n_components=1, cutout=128)

    # 1. All-negative residual exits cleanly with 0 commits
    neg_res = -np.ones((128, 128), dtype=np.float32)
    upd, commits, reason = minor_cycle(neg_res, psf, 0.01, 3, tiny, cutout=128)
    assert commits == []
    assert reason == "noise_floor"
    assert float(np.abs(upd).max()) == 0.0

    # 2. Dimensional flux_clip: wide component with flux 4.0 Jy > dirty_peak 1.0 Jy/beam
    #    has dirty peak < 1.0 Jy/beam, so flux_clip=True must NOT clip its flux to 1.0 Jy.
    class _FixedWideModel(torch.nn.Module):
        cutout = 128

        def forward(self, img, cond):
            b = img.shape[0]
            mu = torch.zeros(b, 1, 7)
            mu[:, 0, 2] = float(standardise_log_flux(np.float32(np.log(4.0))))
            mu[:, 0, 3] = math.log(8.0)
            mu[:, 0, 4] = math.log(8.0)
            mu[:, 0, 5] = 1.0
            return MixParams(torch.zeros(b, 1), mu, torch.zeros(b, 1, 7))

        def mode(self, params, density_mode=False):
            b = params.logits.shape[0]
            out = torch.zeros(b, 6)
            out[:, 2] = params.mu[:, 0, 2]
            out[:, 3] = math.log(8.0)
            out[:, 4] = math.log(8.0)
            return out

    wide_sky = render_aspen(64, 64, 4.0, 8.0, 8.0, 0.0, (128, 128))
    dirty_wide = psf_convolve(wide_sky, psf)
    dirty_wide = dirty_wide / dirty_wide.max()  # dirty_peak == 1.0 Jy/beam, integrated > 4 Jy
    _, commits_clip, _ = minor_cycle(
        dirty_wide, psf, 0.01, 3, _FixedWideModel(),
        loop_gain=1.0, max_components=1, cutout=128,
        scale_relative=False, flux_clip=True,
    )
    assert len(commits_clip) == 1
    assert commits_clip[0].flux > 2.5, f"dimensional flux_clip wrongly clipped extended flux: {commits_clip[0].flux}"

    # 3. Missing-short-spacing zero-mean PSF: uv_coupling_min + delta_fallback rejects resolved-out wide blob
    psf_wide = _gaussian_psf(128, 8.0)
    psf_zero_dc = psf - psf_wide * (psf.sum() / psf_wide.sum())
    psf_zero_dc = (psf_zero_dc / psf_zero_dc.max()).astype(np.float32)

    pt_dirty = psf_convolve(render_aspen(64, 64, 1.0, 0.5, 0.5, 0.0, (128, 128)), psf_zero_dc)
    _, commits_uv, _ = minor_cycle(
        pt_dirty, psf_zero_dc, 0.01, 3, _FixedWideModel(),
        loop_gain=1.0, max_components=1, sigma_min_px=0.5, cutout=128,
        profile_amp=True, delta_fallback=True, uv_coupling_min=0.2,
    )
    assert len(commits_uv) == 1
    assert abs(commits_uv[0].sig_maj - 0.5) < 1e-5, f"expected delta fallback, got sig_maj={commits_uv[0].sig_maj}"


# ---------------------------------------------------------------------------
# 3. Joint NNLS amplitude refit vs sequential CD and ker(B) damping
# ---------------------------------------------------------------------------

def test_nnls_single_atom_collinear_plateau_and_nullspace_damping():
    from mdn_asp.minor_cycle import psf_convolve, render_aspen
    from mdn_asp.refit import solve_nnls_amplitudes
    run_img = _load_script("run_imaging")

    psf = _gaussian_psf(64, 2.0)

    # 1. Single atom (M = 1): NNLS matches exact closed-form damped update to 1e-5
    for sig in (1.0, 8.0):
        u = render_aspen(32, 32, 1.0, sig, sig, 0.0, (64, 64))
        true_a, a0, gain = 2.4, 0.7, 0.6
        dirty = psf_convolve(true_a * u, psf)
        res = (dirty - psf_convolve(a0 * u, psf)).astype(np.float32)
        a_nnls, _, _, _ = solve_nnls_amplitudes(
            [(0, 64, 0, 64, u)], [a0], res, psf, refit_gain=gain, device="cpu"
        )
        expected = a0 + gain * (true_a - a0)
        assert abs(float(a_nnls[0]) - expected) < 5e-4, (a_nnls[0], expected)

    # 2. K = 10 identical overlapping tiles on a plateau: CD collapses onto tile 0, NNLS stays balanced
    plateau = render_aspen(32, 32, 5.0, 7.0, 7.0, 0.0, (64, 64))
    dirty_p = psf_convolve(plateau, psf)
    u_sub = plateau[16:48, 16:48].copy()
    tiles_cd = [{"y0": 16, "y1": 48, "x0": 16, "x1": 48, "u": u_sub.copy(), "a": 0.02} for _ in range(10)]
    tiles_nn = [{"y0": 16, "y1": 48, "x0": 16, "x1": 48, "u": u_sub.copy(), "a": 0.02} for _ in range(10)]
    m_init = 0.2 * plateau
    r_init = (dirty_p - psf_convolve(m_init, psf)).astype(np.float32)

    _, _, st_cd = run_img.refit_tiles(tiles_cd, r_init.copy(), psf, 1.0, torch.device("cpu"), method="cd")
    _, _, st_nn = run_img.refit_tiles(tiles_nn, r_init.copy(), psf, 1.0, torch.device("cpu"), method="nnls")

    a_cd = np.array([t["a"] for t in tiles_cd])
    a_nn = np.array([t["a"] for t in tiles_nn])
    assert a_cd[0] > 5.0 * np.median(a_cd[1:]), f"expected CD order bias, got {a_cd}"
    assert float(a_nn.max() - a_nn.min()) < 1e-4, f"expected symmetric NNLS amplitudes, got {a_nn}"
    assert abs(float(a_nn.sum()) - 1.0) < 1e-2
    assert st_nn["chi2_after"] <= st_nn["chi2_before"]

    # 3. Nullspace damping on zero-mean PSF: smooth wide Gaussian sitting inside UV hole is damped toward 0
    psf_wide = _gaussian_psf(64, 10.0)
    psf_zero_dc = (psf - psf_wide * (psf.sum() / psf_wide.sum())).astype(np.float32)
    psf_zero_dc /= psf_zero_dc.max()
    ghost_blob = render_aspen(32, 32, 1.0, 9.0, 9.0, 0.0, (64, 64))
    res_zero = np.zeros((64, 64), dtype=np.float32)

    a_undamped, _, _, _ = solve_nnls_amplitudes(
        [(0, 64, 0, 64, ghost_blob)], [3.0], res_zero, psf_zero_dc,
        refit_gain=1.0, nullspace_lam=0.0, device="cpu",
    )
    a_damped, _, _, _ = solve_nnls_amplitudes(
        [(0, 64, 0, 64, ghost_blob)], [3.0], res_zero, psf_zero_dc,
        refit_gain=1.0, nullspace_lam=2.5, device="cpu",
    )
    assert float(a_damped[0]) < 0.25 * float(a_undamped[0]), (a_damped[0], a_undamped[0])


# ---------------------------------------------------------------------------
# 4. Diffusion upgrades: conv stem, zero-pad shift, q-space DDS, AIC DOF pen
# ---------------------------------------------------------------------------

def test_diffusion_conv_stem_shift_zero_pad_q_space_and_dof_penalty():
    from mdn_asp.minor_cycle import psf_convolve
    from mdn_asp.models import diffusion as dmod
    run_img = _load_script("run_imaging")

    # 1. _shift_zero_pad does not wrap border flux around the torus
    x = torch.zeros(1, 32, 32)
    x[0, 16, 2] = 10.0
    shifted = dmod._shift_zero_pad(x, 0, -5)
    assert float(shifted[0, 16, -3:].abs().max()) == 0.0

    # 2. LDContextStem preserves outer-context gradient sensitivity on 128 -> 32
    a_ld = argparse.Namespace(
        variant="LD", T=20, sky_scale=10.0, patch=32, shape_size=32, ctx=128,
        base_channels=16, ld_ctx_encoder="conv", self_cond_p=0.0, a_passes=0,
        pred="eps", zero_weight=0.0, zero_floor=1e-3,
    )
    net_ld = dmod.build_diffusion_net(a_ld)
    torch.nn.init.normal_(net_ld.out[-1].weight, std=0.1)  # UNet.out[-1] is zero-init at step 0
    diff_ld = dmod.Diffusion(net_ld, a_ld, "cpu")
    img_ctx = torch.randn(1, 128, 128, requires_grad=True)
    psf_ctx = torch.from_numpy(_gaussian_psf(128, 2.0))[None]
    shape_t = torch.rand(1, 32, 32)
    batch_ld = {
        "img": img_ctx,
        "psf_cut": psf_ctx,
        "sigma": torch.tensor([0.05]),
        "shape": shape_t,
        "sky": shape_t,
    }
    loss_ld = diff_ld.loss(batch_ld, gen=torch.Generator().manual_seed(0))
    loss_ld.backward()
    assert float(img_ctx.grad[0, 8, 8].abs()) > 0.0, "outer context pixel must have non-zero gradient"

    # 3. Positive q-space + tapered DDS produces >= 0 atoms with zero borders
    a_p = argparse.Namespace(
        variant="P", T=10, sample_steps=4, sky_scale=10.0, patch=32, shape_size=32,
        base_channels=16, self_cond_p=0.0, a_passes=0, pred="eps", zero_weight=0.0,
    )
    net_p = dmod.build_diffusion_net(a_p)
    diff_p = dmod.Diffusion(net_p, a_p, "cpu")
    psf2 = torch.from_numpy(_gaussian_psf(64, 2.0))
    rw = torch.randn(32, 32) * 0.1
    rw[16, 16] = 1.0
    draws = diff_p.sample_shape_posterior(
        rw, psf2, 0.1, 4, gen=torch.Generator().manual_seed(3),
        cg_iters=2, q_space=True, taper=True,
    )
    assert draws.shape == (4, 32, 32)
    assert float(draws.min()) >= 0.0
    assert float(draws[:, 0, :].abs().max()) == 0.0
    assert float(draws[:, -1, :].abs().max()) == 0.0
    assert float(draws[:, :, 0].abs().max()) == 0.0
    assert float(draws[:, :, -1].abs().max()) == 0.0

    # 4. AIC effective_beam_dofs penalises diffuse noise draws on a pure point source
    psf_field = _gaussian_psf(64, 2.0)
    pt = np.zeros((64, 64), dtype=np.float32)
    pt[32, 32] = 1.0
    dirty_pt = psf_convolve(pt, psf_field)
    mp_args = argparse.Namespace(
        max_components=1, divergence_tol=0.2, diff_n_draws=4, diff_taper=True,
        diff_q_space=True, diff_dof_pen=4.0, diff_briggs_psd=False,
        diff_ld_cg_iters=0, refit_nullspace_lam=0.0, diff_scales="",
        diff_unmask_output=True, diff_save_draws=0, loop_gain=0.5,
    )
    _, _, steps, _, _, recs = run_img.diffusion_mp_cycle(
        diff_p, a_p, dirty_pt, psf_field, 0.05, mp_args, torch.device("cpu"),
        torch.Generator().manual_seed(11), np.ones((64, 64), dtype=bool),
        0.1, 0.05,
    )
    assert steps == 1
    assert recs[0]["winner"] == "delta"

    # 5. Unbiased _noise_term works on both full and boundary-truncated windows
    rng = np.random.default_rng(9)
    res_noise = rng.standard_normal((128, 128)).astype(np.float32) * 0.02
    mod_sparse = np.zeros((128, 128), dtype=np.float32)
    mod_sparse[64, 64] = 1.0
    rf = run_img._noise_term(res_noise, mod_sparse, _gaussian_psf(128, 2.0), 0.02, 4, 16, torch.device("cpu"))
    assert rf is not None
    w_full = torch.from_numpy(res_noise[:16, :16][None])
    w_crop = torch.zeros(1, 16, 16)
    w_crop[:, :10, :12] = torch.from_numpy(res_noise[:10, :12])
    r_full = rf(w_full)
    r_crop = rf(w_crop, valid_hw=(10, 12))
    assert torch.isfinite(r_full).all() and torch.isfinite(r_crop).all()


# ---------------------------------------------------------------------------
# 5. sim_recovery zero-shift check and CLI parser smoke suite for run_a100.sh
# ---------------------------------------------------------------------------

def test_sim_recovery_no_pixel_shift_and_cli_parsers():
    from mdn_asp.minor_cycle import psf_convolve

    sim_mod = _load_script("sim_recovery")
    run_img = _load_script("run_imaging")
    tr_mdn = _load_script("train_mdn_asp")
    tr_diff = _load_script("train_tiled_asp")
    ev_diff = _load_script("eval_diffusion")

    # Verify psf_convolve(..., gaussian_beam) has zero pixel shift on even-sized grids
    beam = sim_mod.gaussian_beam(64, 2.5)
    sky = np.zeros((64, 64), dtype=np.float32)
    sky[21, 37] = 1.0
    conv_img = psf_convolve(sky, beam)
    py, px = np.unravel_index(int(np.argmax(conv_img)), conv_img.shape)
    assert (py, px) == (21, 37), f"expected exact peak at (21, 37), got ({py}, {px})"

    # Smoke-parse exact flag combinations from scripts/run_a100.sh
    a_img_mdn = run_img.parse_args([
        "--vis", "dummy.ms", "--imagename", "jet", "--out_dir", "out",
        "--ckpt", "models/mdn-asp", "--sigma_min_px", "0.5", "--sigma_max_px", "96.0",
        "--fit_unmasked", "--mdn_density_mode", "--mdn_profile_amp",
        "--mdn_delta_fallback", "--uv_coupling_min", "0.15",
        "--refit_exact", "--major_before_refit", "--refit_all",
        "--refit_method", "nnls", "--refit_gain", "0.5", "--refit_nullspace_lam", "1e-4",
        "--rebuild_metric", "vis_chi2", "--grad_objective", "vis",
    ])
    assert a_img_mdn.refit_method == "nnls"
    assert a_img_mdn.mdn_density_mode is True

    a_img_diff = run_img.parse_args([
        "--vis", "dummy.ms", "--imagename", "cygA", "--out_dir", "out",
        "--diffusion_ckpt", "B2/best.pt", "--diff_mp", "--diff_n_draws", "10",
        "--diff_ld_ckpt", "A3/best.pt", "--diff_ld_draws", "10", "--diff_ld_cg_iters", "3",
        "--diff_scales", "16,32,64", "--fit_unmasked", "--diff_unmask_output",
        "--diff_taper", "--diff_q_space", "--diff_dof_pen", "2.0", "--diff_briggs_psd",
        "--refit_exact", "--major_before_refit", "--refit_method", "nnls",
        "--refit_nullspace_lam", "1e-5", "--diff_rebuild_drop", "0.2",
    ])
    assert a_img_diff.diff_q_space is True
    assert a_img_diff.diff_scales == "16,32,64"

    a_sim = sim_mod.parse_args([
        "--psf", "dummy.fits", "--out_dir", "out",
        "--diffusion_ckpt", "B2/best.pt", "--diff_n_draws", "10",
        "--diff_q_space", "--diff_dof_pen", "2.0", "--diff_briggs_psd",
        "--refit_exact", "--refit_method", "nnls", "--refit_nullspace_lam", "1e-5",
    ])
    assert a_sim.refit_exact is True and a_sim.refit_method == "nnls"

    a_tr_mdn = tr_mdn.parse_args([
        "--out_dir", "out", "--pa_head", "von_mises", "--uniform_point_pa",
        "--scale_relative", "--correlated_noise", "--cutout", "256", "--resume", "best.pt",
    ])
    assert a_tr_mdn.pa_head == "von_mises" and a_tr_mdn.uniform_point_pa is True

    a_tr_diff = tr_diff.parse_args([
        "--variant", "LD", "--out_dir", "out", "--ctx", "256", "--shape_size", "64",
        "--ld_ctx_encoder", "conv", "--ld_clean_prob", "0.5", "--resume", "ckpt_0010000.pt",
    ])
    assert a_tr_diff.ld_ctx_encoder == "conv" and a_tr_diff.ld_clean_prob == 0.5

    a_ev = ev_diff.parse_args([
        "--ckpt", "models/B2_prior", "--out_dir", "out", "--n_patches", "8", "--n_draws", "4"
    ])
    assert a_ev.n_patches == 8 and a_ev.n_draws == 4
