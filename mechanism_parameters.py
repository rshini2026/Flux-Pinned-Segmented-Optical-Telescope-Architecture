"""Field-cooling-fixed physical parameter definitions.

This module centralizes the quantities used by the Field-cooling-fixed model.

Literature-derived / literature-fit quantities:
    - K_parallel(d): uses the published/digitized K11 separation trend.
    - K_perp(d): uses the published/digitized K33 separation trend.
    - c_T(T): normalized temperature scaling from the selected literature fit.
    - zeta: nominal damping ratio chosen from the published dynamic range.

Model assumptions / derived quantities:
    - One representative mirror segment has mass = 1.0 kg.
    - Segment inertia is derived from the modeled uniform regular-hex prism.
    - A pair of identical segments has effective relative mass/inertia of m/2,
      I/2 for the damping-ratio-to-coefficient conversion.
    - Each physical mechanism receives its full per-pair stiffness; there is
      no division by mechanism count in Field-cooling-fixed.
    - Individual mechanism rotational stiffness and translation-rotation
      coupling are not imposed; segment-level rotation emerges from the
      spatial distribution of translational mechanisms.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np

from empirical_mechanism_data import (
    fit_separation_models,
    fit_temperature_candidates,
    choose_temperature_model,
    CHIANG_T_K,
    CHIANG_FORCE_N,
    JIANG_T_K,
    JIANG_FORCE_N,
    jiang_pchip_temperature_ratio,
    temperature_pchip_ratio,
)


@dataclass(frozen=True)
class SegmentProperties:
    mass_kg: float = 1.0
    hex_circumradius_m: float = 0.50
    thickness_m: float = 0.05

    @property
    def inertia_x_kgm2(self) -> float:
        # Centroidal Ix for a regular hex lamina plus prism thickness term.
        return (5.0 / 24.0) * self.mass_kg * self.hex_circumradius_m**2 + (
            1.0 / 12.0
        ) * self.mass_kg * self.thickness_m**2

    @property
    def inertia_y_kgm2(self) -> float:
        return self.inertia_x_kgm2

    @property
    def inertia_z_kgm2(self) -> float:
        # Polar centroidal inertia for the regular-hex lamina/prism.
        return (5.0 / 12.0) * self.mass_kg * self.hex_circumradius_m**2

    @property
    def inertia_diag_kgm2(self) -> np.ndarray:
        return np.array(
            [self.inertia_x_kgm2, self.inertia_y_kgm2, self.inertia_z_kgm2],
            dtype=float,
        )


@dataclass(frozen=True)
class PhysicalConfig:
    segment: SegmentProperties = SegmentProperties()
    reference_temperature_k: float = 77.0
    nominal_gap_m: float = 0.010
    zeta_nominal: float = 0.03
    zeta_low: float = 0.02
    zeta_high: float = 0.04
    temperature_model: str = "chiang"


CONFIG = PhysicalConfig()
SEPARATION_FITS = fit_separation_models()
TEMPERATURE_CANDIDATES = {
    "chiang": fit_temperature_candidates(CHIANG_T_K, CHIANG_FORCE_N, 77.0),
    "jiang": fit_temperature_candidates(JIANG_T_K, JIANG_FORCE_N, 77.0),
}
TEMPERATURE_FITS = {
    name: choose_temperature_model(candidates)
    for name, candidates in TEMPERATURE_CANDIDATES.items()
}


# ---------------------------------------------------------------------------
# Literature-parameterized stiffness
# ---------------------------------------------------------------------------


def separation_domain_m() -> Tuple[float, float]:
    # K11/K33 share the same empirical data domain in the empirical parameterization.
    fit_lo = max(
        min(SEPARATION_FITS["K11"].d_ref_m, 0.005),
        0.005,
    )
    fit_hi = 0.030
    return fit_lo, fit_hi


def K_parallel(d_m: float) -> float:
    """Parallel/lateral stiffness from the K11 literature fit, N/m."""
    return float(SEPARATION_FITS["K11"].stiffness(float(d_m)))


def K_perp(d_m: float) -> float:
    """Perpendicular/axial stiffness from the K33 literature fit, N/m."""
    return float(SEPARATION_FITS["K33"].stiffness(float(d_m)))


def temperature_ratio(temp_k: float, model_name: str = "chiang") -> float:
    name = str(model_name).lower()
    if name not in ("chiang", "jiang"):
        raise ValueError("temperature model must be 'chiang' or 'jiang'")
    t = float(temp_k)
    if name == "chiang":
        if t < float(CHIANG_T_K.min()) or t > float(CHIANG_T_K.max()):
            raise ValueError(
                f"Temperature {t:.3f} K outside Chiang-PCHIP range "
                f"{CHIANG_T_K.min():.1f}-{CHIANG_T_K.max():.1f} K."
            )
        return float(temperature_pchip_ratio(
            t, 77.0, CHIANG_T_K, CHIANG_FORCE_N
        ))
    if name == "jiang":
        if t < float(JIANG_T_K.min()) or t > 88.0:
            raise ValueError(
                f"Temperature {t:.3f} K outside Jiang-PCHIP range "
                f"{JIANG_T_K.min():.1f}-88.0 K."
            )
        return float(jiang_pchip_temperature_ratio(t, 77.0))
    raise AssertionError("unreachable temperature-model branch")


def translational_stiffnesses(
    d_m: float,
    temp_k: float,
    model_name: str = "chiang",
) -> Tuple[float, float]:
    cT = temperature_ratio(temp_k, model_name)
    return K_parallel(d_m) * cT, K_perp(d_m) * cT


# ---------------------------------------------------------------------------
# Literature-calibrated damping
# ---------------------------------------------------------------------------


def effective_relative_mass_kg() -> float:
    """Reduced mass for two identical 1 kg segments in relative translation."""
    m = CONFIG.segment.mass_kg
    return 0.5 * m


def effective_relative_inertia_kgm2(axis: int) -> float:
    """Reduced inertia for two identical segments in relative rotation."""
    return 0.5 * CONFIG.segment.inertia_diag_kgm2[int(axis)]


def translational_damping_coefficients(
    k_parallel_npm: float,
    k_perp_npm: float,
    zeta: float | None = None,
) -> Tuple[float, float]:
    """Return pair-level c_parallel and c_perp in N s/m.

    The calculation uses c = 2*zeta*sqrt(k*m_eff), with m_eff = m/2 for
    two identical segments.
    """
    z = CONFIG.zeta_nominal if zeta is None else float(zeta)
    if z < 0.0:
        raise ValueError("zeta must be nonnegative")
    m_eff = effective_relative_mass_kg()
    c_parallel = 2.0 * z * math.sqrt(max(k_parallel_npm, 0.0) * m_eff)
    c_perp = 2.0 * z * math.sqrt(max(k_perp_npm, 0.0) * m_eff)
    return c_parallel, c_perp


# ---------------------------------------------------------------------------
# Mechanism-level 6x6 matrices
# ---------------------------------------------------------------------------


def mechanism_K6(
    d_m: float,
    temp_k: float,
    model_name: str = "chiang",
) -> np.ndarray:
    """Pure-translational local 6x6 stiffness matrix.

    Local ordering: [t, n, b, r_t, r_n, r_b].
    """
    k_parallel, k_perp = translational_stiffnesses(d_m, temp_k, model_name)
    return np.diag(
        [k_parallel, k_perp, k_parallel, 0.0, 0.0, 0.0]
    ).astype(float)


def mechanism_C6(
    d_m: float,
    temp_k: float,
    model_name: str = "chiang",
    zeta: float | None = None,
) -> np.ndarray:
    """Pure-translational local 6x6 damping matrix.

    Damping is derived from the published damping-ratio range. No direct
    mechanism rotational damping is imposed; distributed translational
    damping generates segment-level rotational damping through lever arms.
    """
    k_parallel, k_perp = translational_stiffnesses(d_m, temp_k, model_name)
    c_parallel, c_perp = translational_damping_coefficients(
        k_parallel, k_perp, zeta=zeta
    )
    return np.diag(
        [c_parallel, c_perp, c_parallel, 0.0, 0.0, 0.0]
    ).astype(float)


def parameter_summary(
    d_m: float = CONFIG.nominal_gap_m,
    temp_k: float = CONFIG.reference_temperature_k,
    model_name: str = CONFIG.temperature_model,
    zeta: float | None = None,
) -> dict:
    cT = temperature_ratio(temp_k, model_name)
    kp, kn = translational_stiffnesses(d_m, temp_k, model_name)
    cp, cn = translational_damping_coefficients(kp, kn, zeta=zeta)
    K = mechanism_K6(d_m, temp_k, model_name)
    C = mechanism_C6(d_m, temp_k, model_name, zeta=zeta)
    return {
        "temperature_model": model_name,
        "temperature_K": float(temp_k),
        "temperature_ratio": cT,
        "gap_m": float(d_m),
        "K_parallel_N_per_m": kp,
        "K_perp_N_per_m": kn,
        "zeta": CONFIG.zeta_nominal if zeta is None else float(zeta),
        "c_parallel_Ns_per_m": cp,
        "c_perp_Ns_per_m": cn,
        "segment_mass_kg": CONFIG.segment.mass_kg,
        "segment_inertia_x_kgm2": CONFIG.segment.inertia_x_kgm2,
        "segment_inertia_y_kgm2": CONFIG.segment.inertia_y_kgm2,
        "segment_inertia_z_kgm2": CONFIG.segment.inertia_z_kgm2,
        "K6": K.tolist(),
        "C6": C.tolist(),
    }


__all__ = [
    "CONFIG",
    "SEPARATION_FITS",
    "TEMPERATURE_FITS",
    "K_parallel",
    "K_perp",
    "temperature_ratio",
    "translational_stiffnesses",
    "effective_relative_mass_kg",
    "effective_relative_inertia_kgm2",
    "translational_damping_coefficients",
    "mechanism_K6",
    "mechanism_C6",
    "parameter_summary",
]
