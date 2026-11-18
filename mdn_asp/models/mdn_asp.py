"""Mixture Density Network for 6D Aspen posteriors.

Input:   2-channel 128x128 image (residual cutout, PSF cutout).
Conditioning: (sigma_local, config_one_hot[4]) — length 5.
Output:  K=5 diagonal-Gaussian mixture over 6D target:
           (x, y, log_flux, log_sigma_maj, log_sigma_minor, PA)

PA is emitted internally as (sin 2*theta, cos 2*theta) — 7 scalars per
component — then decoded back to a scalar PA at sampling/mode.  This
collapses the PA <-> PA + pi ambiguity.

Architecture is a direct port of radiosharp/radiosharp/models/mdn.py with:
  - Output dimension raised from 4D to 6D (plus PA encoding -> 7 emitted).
  - Conditioning vector swapped: (sigma_local, config_one_hot) length 5,
    NOT the radiosharp (HA, length, dec) vector.
  - LOG_STD_MIN_PER_DIM extended to 7 dims; PA dims use the full -5 floor
    (PA uncertainty is inherently wide for symmetric sources).
  - Everything else (CNN encoder, FiLM blocks, logsumexp NLL) is unchanged.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K_COMPONENTS = 5

# 7 emitted scalars per component:
#   x, y, log_flux, log_sigma_maj, log_sigma_minor, sin2pa, cos2pa
N_EMITTED = 7

# 6D target space (PA decoded to scalar at output time)
N_DIM = 6

LOG_STD_MAX = 3.0

# Per-dim floors for the 7 emitted dims.
# log_sigma_maj and log_sigma_minor get the tighter -1.5 floor used in
# radiosharp for the scale dim (same reasoning: stop shattering into
# near-delta peaks).  PA encoding dims (sin2pa, cos2pa) use -5 — their
# natural range is [-1, 1] so the network has to be able to go narrow there.
LOG_STD_MIN_PER_EMITTED = (-5.0, -5.0, -5.0, -1.5, -1.5, -5.0, -5.0)

# Conditioning vector dimension: (sigma_local [1], config_one_hot [4])
COND_DIM = 5

# Number of VLA configurations encoded in the one-hot part
N_CONFIGS = 4  # A, B, C, D


# ---------------------------------------------------------------------------
# PA encoding / decoding
# ---------------------------------------------------------------------------

def encode_pa(pa_rad: torch.Tensor) -> torch.Tensor:
    """Encode scalar PA (radians) to (sin 2*theta, cos 2*theta).

    Input shape:  (...,)
    Output shape: (..., 2)
    """
    two_pa = 2.0 * pa_rad
    return torch.stack([torch.sin(two_pa), torch.cos(two_pa)], dim=-1)


def decode_pa(sin2pa: torch.Tensor, cos2pa: torch.Tensor) -> torch.Tensor:
    """Decode (sin 2*theta, cos 2*theta) back to scalar PA in (-pi/2, pi/2].

    Uses atan2 then halves the result.  Output is in (-pi/2, pi/2].

    Input shapes: (...,) each.
    Output shape: (...,)
    """
    return torch.atan2(sin2pa, cos2pa) * 0.5


# ---------------------------------------------------------------------------
# FiLM block (identical to radiosharp)
# ---------------------------------------------------------------------------

class FiLMBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.film = nn.Linear(cond_dim, 2 * dim)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.fc(h)
        gb = self.film(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        h = (1.0 + gamma) * h + beta
        return F.gelu(h)


# ---------------------------------------------------------------------------
# Mixture parameters container
# ---------------------------------------------------------------------------

class MixParams(NamedTuple):
    """Raw MDN outputs before PA decoding.

    logits:  (B, K)         — unnormalised mixture log-weights
    mu:      (B, K, 7)      — component means over 7 emitted dims
    log_std: (B, K, 7)      — component log-stds over 7 emitted dims
    """
    logits:  torch.Tensor
    mu:      torch.Tensor
    log_std: torch.Tensor


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MDNAsp(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/pjaganna/mdn-asp",
    license="gpl-3.0",
    tags=["radio-astronomy", "deconvolution", "mixture-density-network"],
):
    """MDN for 6D Aspen posteriors conditioned on (sigma_local, config_one_hot).

    Parameters
    ----------
    base_channels:
        Width multiplier for the CNN encoder.  Default 32 matches radiosharp.
        Use a smaller value (e.g. 8) for unit tests on CPU.
    hidden:
        MLP hidden width.  Default 256.  Use 64 for tests.
    n_components:
        Number of mixture components K.  Default 5 per plan.
    cond_dim:
        Conditioning vector length.  Default 5 (1 + 4).
    cutout:
        Input size (power of 2).  Each doubling over 128 adds one stride-2
        block, so the flattened features stay 4x4 and a 128 model is unchanged.
    """

    def __init__(
        self,
        base_channels: int = 32,
        hidden: int = 256,
        n_components: int = K_COMPONENTS,
        cond_dim: int = COND_DIM,
        cutout: int = 128,
        pa_head: str = "7d",
        scale_relative: bool | None = None,
        point_sigma_px: float | None = None,
    ):
        super().__init__()
        if pa_head not in ("7d", "von_mises"):
            raise ValueError(f"pa_head must be '7d' or 'von_mises', got {pa_head!r}")
        self.K = n_components
        self.cond_dim = cond_dim
        self.pa_head = pa_head
        self.scale_relative = scale_relative
        self.point_sigma_px = point_sigma_px
        n_blocks = int(math.log2(cutout // 4))
        if cutout < 128 or 4 * 2 ** n_blocks != cutout:
            raise ValueError(f"cutout must be a power of 2 >= 128, got {cutout}")
        self.cutout = cutout

        c = base_channels

        def block(ic: int, oc: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, stride=2, padding=1),
                nn.GroupNorm(1, oc),
                nn.GELU(),
                nn.Conv2d(oc, oc, 3, stride=1, padding=1),
                nn.GroupNorm(1, oc),
                nn.GELU(),
            )

        # 2-channel input: (residual, PSF); 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.enc = nn.Sequential(
            block(2, c),          # 64
            block(c, 2 * c),      # 32
            block(2 * c, 4 * c),  # 16
            block(4 * c, 4 * c),  # 8
            block(4 * c, 4 * c),  # 4
            *[block(4 * c, 4 * c) for _ in range(n_blocks - 5)],
        )

        # Flatten (not global avg pool) — preserves sub-pixel localisation
        # information (see radiosharp rationale).
        feat_dim = 4 * c * 4 * 4

        self.proj = nn.Linear(feat_dim, hidden)
        self.film1 = FiLMBlock(hidden, cond_dim)
        self.film2 = FiLMBlock(hidden, cond_dim)

        # K logits + K * N_EMITTED means + K * N_EMITTED log_stds
        self.head = nn.Linear(hidden, self.K * (1 + 2 * N_EMITTED))

        # Per-dim log_std floor tensor — registered as buffer so it moves
        # with .to(device) automatically.
        self.register_buffer(
            "_log_std_min",
            torch.tensor(LOG_STD_MIN_PER_EMITTED, dtype=torch.float32),
        )

    def forward(
        self,
        image: torch.Tensor,   # (B, 2, 128, 128)
        cond:  torch.Tensor,   # (B, cond_dim)
    ) -> MixParams:
        """Return mixture parameters over the 7-emitted-dim space."""
        B = image.shape[0]
        h = self.enc(image)
        h = h.flatten(1)
        h = F.gelu(self.proj(h))
        h = self.film1(h, cond)
        h = self.film2(h, cond)
        out = self.head(h)

        K, E = self.K, N_EMITTED
        logits  = out[:, :K]
        mu      = out[:, K : K + K * E].view(B, K, E)
        log_std = out[:, K + K * E :].view(B, K, E)

        log_std = torch.maximum(log_std, self._log_std_min)
        log_std = log_std.clamp_max(LOG_STD_MAX)
        if self.pa_head == "von_mises":
            # log_std[..., 5] holds log_kappa in [-5, 5]; slot 6 is unused (0).
            lk = out[:, K + K * E :].view(B, K, E)[..., 5].clamp(-5.0, 5.0)
            z = torch.zeros_like(lk)
            log_std = torch.cat([log_std[..., :5], lk.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)

        return MixParams(logits=logits, mu=mu, log_std=log_std)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def nll_loss(
        self,
        params:  MixParams,
        targets: torch.Tensor,  # (B, 6): x, y, log_flux, log_smaj, log_smin, PA
    ) -> torch.Tensor:
        """NLL of targets under the predicted mixture.

        For ``pa_head == '7d'``, PA (dim 5) is encoded to (sin2pa, cos2pa) and
        evaluated under a 7D diagonal Gaussian. For ``pa_head == 'von_mises'``,
        dims 0..4 use a 5D Gaussian and PA uses an exact normalized period-pi
        von Mises density on (-pi/2, pi/2]. Returns a scalar (mean over batch).
        """
        pa = targets[:, 5]                          # (B,)
        logits, mu, log_std = params
        log_w = F.log_softmax(logits, dim=-1)       # (B, K)

        if self.pa_head == "von_mises":
            y5 = targets[:, :5].unsqueeze(1)        # (B, 1, 5)
            mu5, ls5 = mu[..., :5], log_std[..., :5]
            var5 = torch.exp(2.0 * ls5)
            log_g5 = -0.5 * (
                ((y5 - mu5) ** 2) / var5 + 2.0 * ls5 + math.log(2.0 * math.pi)
            ).sum(dim=-1)                           # (B, K)
            mu_2pa = torch.atan2(mu[..., 5], mu[..., 6])  # (B, K)
            kappa = torch.exp(log_std[..., 5])            # (B, K)
            # i0e(k) = exp(-k) * I0(k), so log I0(k) = k + log(i0e(k)):
            # k * cos(d) - log(pi * I0(k)) = k * (cos(d) - 1) - log(pi) - log(i0e(k))
            log_vm = (
                kappa * (torch.cos(2.0 * pa.unsqueeze(1) - mu_2pa) - 1.0)
                - math.log(math.pi)
                - torch.log(torch.special.i0e(kappa).clamp_min(1e-30))
            )
            log_p = torch.logsumexp(log_w + log_g5 + log_vm, dim=-1)
            return -log_p.mean()

        pa_enc = encode_pa(pa)                      # (B, 2)
        # Rebuild target in 7D emitted space
        y7 = torch.cat([targets[:, :5], pa_enc], dim=-1)  # (B, 7)

        y7_exp = y7.unsqueeze(1)                    # (B, 1, 7)
        var = torch.exp(2.0 * log_std)              # (B, K, 7)
        log_comp = -0.5 * (
            ((y7_exp - mu) ** 2) / var
            + 2.0 * log_std
            + math.log(2.0 * math.pi)
        )
        log_comp = log_comp.sum(dim=-1)             # (B, K)
        log_p = torch.logsumexp(log_w + log_comp, dim=-1)  # (B,)
        return -log_p.mean()

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def mode(self, params: MixParams, density_mode: bool = False) -> torch.Tensor:
        """Argmax-component mean decoded to 6D.  Returns (B, 6)."""
        logits, mu, log_std = params
        if not density_mode:
            k = logits.argmax(dim=-1)                   # (B,)
        elif self.pa_head == "von_mises":
            log_w = F.log_softmax(logits, dim=-1)
            kappa = torch.exp(log_std[..., 5])
            score = (
                log_w
                - log_std[..., :5].sum(dim=-1)
                - torch.log(torch.special.i0e(kappa).clamp_min(1e-30))
            )
            k = score.argmax(dim=-1)
        else:
            score = F.log_softmax(logits, dim=-1) - log_std.sum(dim=-1)
            k = score.argmax(dim=-1)
        idx = k.view(-1, 1, 1).expand(-1, 1, N_EMITTED)
        mu_mode = mu.gather(1, idx).squeeze(1)          # (B, 7)
        return self._decode_7d_to_6d(mu_mode)

    @torch.no_grad()
    def sample(self, params: MixParams, n: int = 1) -> torch.Tensor:
        """Draw samples from the mixture.  Returns (B, n, 6)."""
        logits, mu, log_std = params
        B, K, E = mu.shape
        log_w = F.log_softmax(logits, dim=-1)
        probs = log_w.exp()
        comp = torch.multinomial(probs, n, replacement=True)  # (B, n)
        idx  = comp.unsqueeze(-1).expand(-1, -1, E)           # (B, n, E)
        mu_sel  = mu.gather(1, idx)                           # (B, n, E)
        if self.pa_head == "von_mises":
            std5 = log_std[..., :5].exp().gather(1, idx[..., :5])
            s5 = mu_sel[..., :5] + std5 * torch.randn_like(std5)
            mu_2pa = torch.atan2(mu_sel[..., 5], mu_sel[..., 6])
            kappa = log_std[..., 5].exp().gather(1, comp).clamp_min(1e-4)
            two_pa = torch.distributions.VonMises(mu_2pa, kappa).sample()
            pa = decode_pa(torch.sin(two_pa), torch.cos(two_pa))
            return torch.cat([s5, pa.unsqueeze(-1)], dim=-1)
        std_sel = log_std.exp().gather(1, idx)
        eps = torch.randn_like(mu_sel)
        raw = mu_sel + std_sel * eps                          # (B, n, 7)
        # Decode each sample
        out = torch.stack(
            [self._decode_7d_to_6d(raw[:, i, :]) for i in range(n)],
            dim=1,
        )
        return out  # (B, n, 6)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_7d_to_6d(v: torch.Tensor) -> torch.Tensor:
        """Convert (B, 7) emitted to (B, 6) with PA decoded to scalar."""
        sin2pa = v[:, 5]
        cos2pa = v[:, 6]
        pa = decode_pa(sin2pa, cos2pa)
        return torch.cat([v[:, :5], pa.unsqueeze(-1)], dim=-1)


# ---------------------------------------------------------------------------
# Convenience: build conditioning vector from raw inputs
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Checkpoint loading (single safe path for both inference and eval scripts)
# ---------------------------------------------------------------------------

# Architecture keys the constructor accepts; used to filter a run_config.
_ARCH_KEYS = (
    "base_channels", "hidden", "n_components", "cond_dim", "cutout",
    "pa_head", "scale_relative", "point_sigma_px",
)


def _read_checkpoint_meta(ckpt: str | Path) -> dict:
    """Gather training metadata from run_config.json, .pt['args'], or HF config.json."""
    path = Path(ckpt)
    meta: dict = {}
    if path.is_dir():
        hf_cfg = path / "config.json"
        if hf_cfg.is_file():
            meta.update({k: v for k, v in json.loads(hf_cfg.read_text()).items() if v is not None})
        rc = path / "run_config.json"
        if rc.is_file():
            meta.update(json.loads(rc.read_text()).get("args", {}))
    elif path.is_file():
        rc = path.parent / "run_config.json"
        if rc.is_file():
            meta.update(json.loads(rc.read_text()).get("args", {}))
        try:
            raw = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(raw, dict):
                if isinstance(raw.get("args"), dict):
                    for k, v in raw["args"].items():
                        meta.setdefault(k, v)
                if "scale_relative" in raw and raw["scale_relative"] is not None:
                    meta["scale_relative"] = bool(raw["scale_relative"])
        except Exception:
            pass
    return meta


def load_mdn_checkpoint(
    ckpt: str | Path,
    device: torch.device | str = "cpu",
) -> "MDNAsp":
    """Load an MDNAsp from a HF safetensors dir/repo or a .pt file."""
    path = Path(ckpt)
    if path.is_dir() or (path / "config.json").exists():
        model = MDNAsp.from_pretrained(str(path))
    elif path.is_file():
        raw = torch.load(path, map_location="cpu", weights_only=True)
        state = raw.get("model", raw.get("model_state_dict", raw)) if isinstance(raw, dict) else raw
        meta = _read_checkpoint_meta(path)
        arch = {k: meta[k] for k in _ARCH_KEYS if k in meta and meta[k] is not None}
        model = MDNAsp(**arch)
        model.load_state_dict(state)
    else:
        # Remote HF Hub repo ID
        model = MDNAsp.from_pretrained(str(ckpt))
    return model.to(device).eval()


def checkpoint_is_scale_relative(ckpt: str | Path) -> bool | None:
    """Report whether a checkpoint's flux head is scale-relative."""
    meta = _read_checkpoint_meta(ckpt)
    if "scale_relative" in meta and meta["scale_relative"] is not None:
        return bool(meta["scale_relative"])
    return None


def checkpoint_sigma_min_px(ckpt: str | Path) -> float | None:
    """Smallest source width (px) the checkpoint was trained on."""
    from mdn_asp.constants import BEAM_SIGMA_PX
    meta = _read_checkpoint_meta(ckpt)
    vals = [meta.get("point_sigma_px"), meta.get("blob_sigma_min_px")]
    if all(v is None for v in vals):
        return None
    present = [float(v) for v in vals if v is not None]
    if len(present) == 2:
        return min(present)
    return min(present[0], BEAM_SIGMA_PX)


def checkpoint_train_arg(ckpt: str | Path, key: str):
    """One training argument from run_config.json / .pt['args'] / config.json, or None."""
    return _read_checkpoint_meta(ckpt).get(key)


def make_cond(
    sigma_local: torch.Tensor,   # (B,)
    config_idx:  torch.Tensor,   # (B,) int in {0, 1, 2, 3}
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build the (B, 5) conditioning vector.

    sigma_local is not normalised here — caller is responsible.
    config_idx 0=A, 1=B, 2=C, 3=D.
    """
    B = sigma_local.shape[0]
    one_hot = F.one_hot(config_idx.long(), num_classes=N_CONFIGS).float()
    cond = torch.cat([sigma_local.unsqueeze(-1), one_hot], dim=-1)
    if device is not None:
        cond = cond.to(device)
    return cond
