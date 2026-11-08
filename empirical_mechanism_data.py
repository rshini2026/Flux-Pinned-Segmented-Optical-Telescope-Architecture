"""Empirical experimental parameterization layer.

It provides empirical, uncertainty-aware models
for translational stiffness versus separation and normalized temperature
scaling. The built-in data are digitized approximations from the cited figures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass(frozen=True)
class SeparationFit:
    d_ref_m: float
    k_ref: float
    lambda_m: float
    lambda_sigma_m: float
    r2: float

    def stiffness(self, d_m: np.ndarray | float) -> np.ndarray:
        d = np.asarray(d_m, dtype=float)
        return self.k_ref * np.exp(-(d - self.d_ref_m) / self.lambda_m)


@dataclass(frozen=True)
class TemperatureFit:
    model_name: str
    t_ref_k: float
    coeff: Tuple[float, ...]
    t_min_k: float
    t_max_k: float
    aicc: float
    rmse: float

    def ratio(self, temp_k: np.ndarray | float) -> np.ndarray:
        t = np.asarray(temp_k, dtype=float)
        if self.model_name == "linear":
            a, b = self.coeff
            y = a + b * t
        elif self.model_name == "quadratic":
            a, b, c = self.coeff
            y = a + b * t + c * t * t
        elif self.model_name == "logistic":
            rmin, rmax, t50, s = self.coeff
            y = rmin + (rmax - rmin) / (1.0 + np.exp((t - t50) / s))
        elif self.model_name == "exponential":
            rinf, A, tau = self.coeff
            y = rinf + A * np.exp(-(t - self.t_min_k) / tau)
        else:
            raise ValueError(f"Unknown temperature model: {self.model_name}")
        return y / self._raw_at_ref()

    def _raw_at_ref(self) -> float:
        t = np.array([self.t_ref_k], dtype=float)
        if self.model_name == "linear":
            a, b = self.coeff
            return float(a + b * t[0])
        if self.model_name == "quadratic":
            a, b, c = self.coeff
            return float(a + b * t[0] + c * t[0] * t[0])
        if self.model_name == "logistic":
            rmin, rmax, t50, s = self.coeff
            return float(rmin + (rmax - rmin) / (1.0 + np.exp((t[0] - t50) / s)))
        rinf, A, tau = self.coeff
        return float(rinf + A * np.exp(-(t[0] - self.t_min_k) / tau))

    def bounded_ratio(self, temp_k: np.ndarray | float) -> np.ndarray:
        t = np.asarray(temp_k, dtype=float)
        r = self.ratio(t)
        return np.clip(r, 0.0, None)


# ---------------------------------------------------------------------------
# Digitized/approximate literature temperature data.
# Source: Parameter Identification in Application, Fig. 2.
# The paper explicitly warns that the Chiang and Jiang temperature trends
# disagree and recommends application-specific measurement.
# ---------------------------------------------------------------------------
CHIANG_T_K = np.array([40, 50, 60, 70, 77, 85], dtype=float)
CHIANG_FORCE_N = np.array([0.170, 0.135, 0.110, 0.075, 0.045, 0.020], dtype=float)

# Exact tabulated maximum levitation-force data from Jiang et al., Physica C
# 378-381 (2002) 869-872.  The paper's plotted axis titles are transposed;
# these arrays follow the numerical table, not the mislabeled plot axes.
JIANG_T_K = np.array(
    [38.4, 48.5, 50.0, 54.3, 59.0, 64.5, 67.6,
     70.5, 75.5, 82.0, 88.0, 93.0, 100.0],
    dtype=float,
)
JIANG_FORCE_N = np.array(
    [32.5, 33.5, 32.5, 32.5, 30.6, 29.7, 29.6,
     28.7, 25.8, 21.1, 9.6, 0.0, 0.0],
    dtype=float,
)


def jiang_pchip_force_n(temp_k: np.ndarray | float) -> np.ndarray:
    """Shape-preserving interpolation of Jiang's tabulated force data."""
    t = np.asarray(temp_k, dtype=float)
    if np.any(t < JIANG_T_K.min()) or np.any(t > 88.0):
        raise ValueError(
            f"Jiang PCHIP temperature must be within "
            f"{JIANG_T_K.min():.1f}-88.0 K for this study."
        )
    interpolator = PchipInterpolator(JIANG_T_K, JIANG_FORCE_N, extrapolate=False)
    return np.asarray(interpolator(t), dtype=float)


def jiang_pchip_temperature_ratio(
    temp_k: np.ndarray | float,
    reference_temperature_k: float = 77.0,
) -> np.ndarray:
    """Return F_lev(T)/F_lev(77 K) from Jiang's tabulated data."""
    numerator = jiang_pchip_force_n(temp_k)
    denominator = float(jiang_pchip_force_n(reference_temperature_k))
    return numerator / denominator

# ---------------------------------------------------------------------------
# Digitized/approximate 0-deg separation data from the published K11-K33
# plots in measure_stiffn (2).pdf. These are intentionally isolated in this
# module so they can be replaced with higher-fidelity digitization.
# ---------------------------------------------------------------------------
SEP_D_MM = np.array([5, 6, 8, 10, 12, 15, 18, 20, 22, 25, 30], dtype=float)
SEP_K11 = np.array([100, 82, 62, 40, 27, 15, 9, 7, 5.5, 1.2, 0.15], dtype=float)
SEP_K22 = np.array([60, 48, 35, 25, 17, 11, 7, 5.0, 4.0, 2.5, 0.6], dtype=float)
SEP_K33 = np.array([160, 130, 105, 80, 55, 30, 17, 12, 9, 5.2, 3.4], dtype=float)

# Reference matrix from the static experiment at d=5 mm, beta=2 deg, N=1.
# Raw measured matrix, retained for provenance. Do not silently symmetrize it.
K_REFERENCE_RAW = np.array([
    [101.0, 18.1, 6.35, -0.72, -3.30, 0.0],
    [8.96, 75.4, -8.61, -4.31, -1.10, 0.0],
    [-34.9, 46.3, 196.0, -1.68, -0.09, 0.0],
    [0.56, 0.86, 0.06, -0.07, 0.0, 0.0],
    [-0.38, 0.16, -0.23, -0.02, -0.01, 0.0],
    [-0.06, 0.01, 0.01, 0.0, 0.01, 0.0],
], dtype=float)


def exp_decay_fit(d_mm: np.ndarray, k: np.ndarray, d_ref_mm: float = 5.0) -> SeparationFit:
    """Fit K(d)=Kref*exp(-(d-dref)/lambda) by linear regression in log(K)."""
    d = np.asarray(d_mm, dtype=float)
    y = np.asarray(k, dtype=float)
    mask = np.isfinite(d) & np.isfinite(y) & (y > 0.0)
    if mask.sum() < 3:
        raise ValueError("Need at least 3 positive stiffness points.")
    x = d[mask]
    z = np.log(y[mask])
    A = np.column_stack([np.ones_like(x), -(x - d_ref_mm)])
    coeff, *_ = np.linalg.lstsq(A, z, rcond=None)
    intercept, slope = coeff
    if slope <= 0:
        raise ValueError("Fitted decay length is non-positive; check data.")
    lambda_mm = 1.0 / slope
    kref = float(np.exp(intercept))
    zhat = A @ coeff
    ss_res = float(np.sum((z - zhat) ** 2))
    ss_tot = float(np.sum((z - np.mean(z)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    dof = max(1, len(x) - 2)
    s2 = ss_res / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    lambda_sigma_mm = float(np.sqrt(cov[1, 1]) / slope**2)
    return SeparationFit(
        d_ref_m=d_ref_mm * 1e-3,
        k_ref=kref,
        lambda_m=lambda_mm * 1e-3,
        lambda_sigma_m=lambda_sigma_mm * 1e-3,
        r2=r2,
    )


def fit_separation_models() -> Dict[str, SeparationFit]:
    return {
        "K11": exp_decay_fit(SEP_D_MM, SEP_K11),
        "K22": exp_decay_fit(SEP_D_MM, SEP_K22),
        "K33": exp_decay_fit(SEP_D_MM, SEP_K33),
    }


def _aicc(n: int, rss: float, p: int) -> float:
    if n <= p + 1 or rss <= 0:
        return float("inf")
    aic = n * np.log(rss / n) + 2 * p
    return aic + 2 * p * (p + 1) / (n - p - 1)


def fit_temperature_candidates(temp_k: np.ndarray, force_n: np.ndarray, t_ref_k: float) -> Dict[str, TemperatureFit]:
    """Fit several simple empirical forms and return them all for model comparison."""
    t = np.asarray(temp_k, dtype=float)
    y = np.asarray(force_n, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if len(t) < 5:
        raise ValueError("Need at least 5 temperature data points.")
    out: Dict[str, TemperatureFit] = {}

    # Linear and quadratic polynomial fits.
    for name, deg in (("linear", 1), ("quadratic", 2)):
        coeff = np.polyfit(t, y, deg)[::-1]
        pred = sum(coeff[i] * t ** i for i in range(len(coeff)))
        rss = float(np.sum((y - pred) ** 2))
        out[name] = TemperatureFit(name, t_ref_k, tuple(map(float, coeff)), float(t.min()), float(t.max()),
                                    _aicc(len(t), rss, len(coeff)), float(np.sqrt(rss / len(t))))

    # Logistic, least-squares using curve_fit.
    from scipy.optimize import curve_fit
    def logistic(t, rmin, rmax, t50, s):
        return rmin + (rmax - rmin) / (1.0 + np.exp((t - t50) / s))
    p0 = [float(y.min()), float(y.max()), float(np.median(t)), 5.0]
    bounds = ([0.0, 0.0, t.min() - 20.0, 0.2], [np.inf, np.inf, t.max() + 20.0, 100.0])
    try:
        popt, _ = curve_fit(logistic, t, y, p0=p0, bounds=bounds, maxfev=20000)
        pred = logistic(t, *popt)
        rss = float(np.sum((y - pred) ** 2))
        out["logistic"] = TemperatureFit("logistic", t_ref_k, tuple(map(float, popt)), float(t.min()), float(t.max()),
                                         _aicc(len(t), rss, 4), float(np.sqrt(rss / len(t))))
    except Exception:
        pass

    # Exponential decay toward an asymptote.
    def expo(t, rinf, A, tau):
        return rinf + A * np.exp(-(t - t.min()) / tau)
    p0 = [float(y.min()), float(y.max() - y.min()), max(1.0, 0.3 * (t.max() - t.min()))]
    bounds = ([0.0, 0.0, 0.2], [np.inf, np.inf, 200.0])
    try:
        popt, _ = curve_fit(expo, t, y, p0=p0, bounds=bounds, maxfev=20000)
        pred = expo(t, *popt)
        rss = float(np.sum((y - pred) ** 2))
        out["exponential"] = TemperatureFit("exponential", t_ref_k, tuple(map(float, popt)), float(t.min()), float(t.max()),
                                             _aicc(len(t), rss, 3), float(np.sqrt(rss / len(t))))
    except Exception:
        pass
    return out


def choose_temperature_model(candidates: Dict[str, TemperatureFit]) -> TemperatureFit:
    return min(candidates.values(), key=lambda m: m.aicc)


def temperature_pchip_ratio(temp_k: np.ndarray | float, t_ref_k: float, table_t: np.ndarray, table_force: np.ndarray) -> np.ndarray:
    """Monotone empirical interpolation; normalized to 1 at Tref.

    This is often safer than forcing a single global analytic form onto sparse
    and contradictory literature data.
    """
    interp = PchipInterpolator(table_t, table_force, extrapolate=False)
    f_ref = float(interp(t_ref_k))
    return np.asarray(interp(temp_k), dtype=float) / f_ref

# Runtime empirical-model helpers -------------------------------------------------------
def empirical_translational_stiffness(d_m: float) -> np.ndarray:
    fits = fit_separation_models()
    return np.array([
        float(fits["K11"].stiffness(d_m)),
        float(fits["K22"].stiffness(d_m)),
        float(fits["K33"].stiffness(d_m)),
    ], dtype=float)


def temperature_ratio(temp_k: float, model_name: str = "chiang") -> float:
    name = str(model_name).lower()
    if name == "chiang":
        fit = choose_temperature_model(
            fit_temperature_candidates(CHIANG_T_K, CHIANG_FORCE_N, 77.0)
        )
    elif name == "jiang":
        fit = choose_temperature_model(
            fit_temperature_candidates(JIANG_T_K, JIANG_FORCE_N, 77.0)
        )
    else:
        raise ValueError("temperature_model must be 'chiang' or 'jiang'")
    t = float(temp_k)
    if t < fit.t_min_k or t > min(fit.t_max_k, 85.0):
        raise ValueError(
            f"Temperature {t:.3f} K outside the empirical-model primary range "
            f"{fit.t_min_k:.1f}-{min(fit.t_max_k, 85.0):.1f} K."
        )
    return float(fit.bounded_ratio(t))
