"""Shared constants, moved verbatim from MAD-Clean data/extended_sky.py and
data/cutout_dataset.py."""

# G55 D-config L-band beam sigma (px).  Used for the lower bound on
# component scale; upper bound driven by 8 * BEAM_SIGMA_PX.
BEAM_SIGMA_PX: float = 1.4
SIGMA_MAX_PX: float = 8.0 * BEAM_SIGMA_PX  # ≈ 11.2 px

LOG_FLUX_OFFSET = -5.76
LOG_FLUX_SCALE  = 3.45


def standardise_log_flux(log_flux: float) -> float:
    return (log_flux - LOG_FLUX_OFFSET) / LOG_FLUX_SCALE


def unstandardise_log_flux(z: float) -> float:
    return z * LOG_FLUX_SCALE + LOG_FLUX_OFFSET
