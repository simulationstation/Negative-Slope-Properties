from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from .constants import PhysicalConstants
from .inversion import ForwardMuPosterior


@dataclass(frozen=True)
class SlopeStatistic:
    slope_draws: np.ndarray
    slope_mean: float
    slope_std: float
    p_slope_lt_0: float
    logA_grid: np.ndarray
    logmu_logA_samples: np.ndarray


def logA0_from_params(H0: np.ndarray, omega_k0: np.ndarray, *, constants: PhysicalConstants) -> np.ndarray:
    H0 = np.asarray(H0, dtype=float)
    omega_k0 = np.asarray(omega_k0, dtype=float)
    if H0.shape != omega_k0.shape:
        raise ValueError("H0/omega_k0 shape mismatch.")
    const_A = 4.0 * np.pi * (constants.c_km_s**2)
    denom0 = (H0**2) * (1.0 - omega_k0)
    if np.any(denom0 <= 0.0) or (not np.all(np.isfinite(denom0))):
        raise ValueError("Invalid A0 mapping: require finite positive H0^2*(1-omega_k0).")
    return np.log(const_A / denom0)


def area_from_H_samples(
    H_samples: np.ndarray,
    *,
    z_grid: np.ndarray,
    H0: np.ndarray,
    omega_k0: np.ndarray,
    constants: PhysicalConstants,
) -> np.ndarray:
    H_samples = np.asarray(H_samples, dtype=float)
    z_grid = np.asarray(z_grid, dtype=float)
    H0 = np.asarray(H0, dtype=float)
    omega_k0 = np.asarray(omega_k0, dtype=float)
    if H_samples.ndim != 2:
        raise ValueError("H_samples must be 2D (n_draws, n_z).")
    if z_grid.ndim != 1 or H_samples.shape[1] != z_grid.size:
        raise ValueError("z_grid/H_samples shape mismatch.")
    if H0.shape != (H_samples.shape[0],) or omega_k0.shape != (H_samples.shape[0],):
        raise ValueError("H0/omega_k0 shape mismatch with H_samples.")
    const_A = 4.0 * np.pi * (constants.c_km_s**2)
    u = H_samples**2
    denom = u - (H0[:, None] ** 2) * omega_k0[:, None] * (1.0 + z_grid[None, :]) ** 2
    if np.any(denom <= 0.0) or (not np.all(np.isfinite(denom))):
        raise ValueError("Invalid apparent-horizon denominator in area mapping.")
    return const_A / denom


def stable_logA_domain(logA_draws: np.ndarray) -> tuple[float, float]:
    logA_draws = np.asarray(logA_draws, dtype=float)
    if logA_draws.ndim != 2:
        raise ValueError("logA_draws must be 2D.")
    finite_rows = np.all(np.isfinite(logA_draws), axis=1)
    ok = logA_draws[finite_rows]
    if ok.size == 0:
        raise ValueError("No finite logA draws.")
    lo = np.percentile(ok, 2.0, axis=1)
    hi = np.percentile(ok, 98.0, axis=1)
    logA_min = float(np.max(lo))
    logA_max = float(np.min(hi))
    if np.isfinite(logA_min) and np.isfinite(logA_max) and (logA_max > logA_min):
        return logA_min, logA_max
    vals = ok[np.isfinite(ok)]
    if vals.size == 0:
        raise ValueError("No finite logA values available for fallback domain.")
    logA_min = float(np.percentile(vals, 2.0))
    logA_max = float(np.percentile(vals, 98.0))
    if not np.isfinite(logA_min) or not np.isfinite(logA_max) or logA_max <= logA_min:
        raise ValueError("Failed to determine a stable logA domain.")
    return logA_min, logA_max


def posterior_logmu_on_logA(
    post: ForwardMuPosterior,
    *,
    constants: PhysicalConstants,
    n_logA: int = 140,
) -> tuple[np.ndarray, np.ndarray]:
    H0_s = np.asarray(post.params["H0"], dtype=float)
    ok_s = np.asarray(post.params.get("omega_k0", np.zeros_like(H0_s)), dtype=float)
    logA_draws = np.log(
        area_from_H_samples(
            post.H_samples,
            z_grid=post.z_grid,
            H0=H0_s,
            omega_k0=ok_s,
            constants=constants,
        )
    )
    logA_min, logA_max = stable_logA_domain(logA_draws)
    logA_grid = np.linspace(logA_min, logA_max, int(n_logA))
    logA0 = logA0_from_params(H0_s, ok_s, constants=constants)
    x_grid = np.asarray(post.x_grid, dtype=float)
    out = np.empty((post.logmu_x_samples.shape[0], logA_grid.size), dtype=float)
    for j in range(out.shape[0]):
        xj = np.clip(logA_grid - float(logA0[j]), float(x_grid[0]), float(x_grid[-1]))
        out[j] = np.interp(xj, x_grid, np.asarray(post.logmu_x_samples[j], dtype=float))
    return logA_grid, out


def slope_draws_from_logmu_samples(
    *,
    logA_grid: np.ndarray,
    logmu_samples: np.ndarray,
    weight_mode: str = "variance",
) -> np.ndarray:
    logA_grid = np.asarray(logA_grid, dtype=float)
    logmu_samples = np.asarray(logmu_samples, dtype=float)
    if logA_grid.ndim != 1 or logmu_samples.ndim != 2 or logmu_samples.shape[1] != logA_grid.size:
        raise ValueError("Shape mismatch for slope computation.")
    if weight_mode == "variance":
        var = np.var(logmu_samples, axis=0, ddof=1)
        w = 1.0 / np.clip(var, 1e-12, np.inf)
    elif weight_mode == "uniform":
        w = np.ones(logA_grid.size, dtype=float)
    else:
        raise ValueError(f"Unsupported weight_mode '{weight_mode}'.")
    w = w / np.trapezoid(w, x=logA_grid)
    x0 = float(np.average(logA_grid, weights=w))
    x = logA_grid - x0
    X = np.column_stack([np.ones_like(x), x])
    XtW = X.T * w[None, :]
    beta = np.linalg.solve(XtW @ X, XtW @ logmu_samples.T)  # (2, n_draws)
    return np.asarray(beta[1], dtype=float)


def slope_stat_from_posterior(
    post: ForwardMuPosterior,
    *,
    constants: PhysicalConstants,
    n_logA: int = 140,
    weight_mode: str = "variance",
) -> SlopeStatistic:
    logA_grid, logmu_logA_samples = posterior_logmu_on_logA(post, constants=constants, n_logA=n_logA)
    s = slope_draws_from_logmu_samples(logA_grid=logA_grid, logmu_samples=logmu_logA_samples, weight_mode=weight_mode)
    s_mean = float(np.mean(s))
    s_std = float(np.std(s, ddof=1)) if s.size > 1 else 0.0
    return SlopeStatistic(
        slope_draws=s,
        slope_mean=s_mean,
        slope_std=s_std,
        p_slope_lt_0=float(np.mean(s < 0.0)),
        logA_grid=logA_grid,
        logmu_logA_samples=logmu_logA_samples,
    )


def wilson_interval(*, k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive.")
    if k < 0 or k > n:
        raise ValueError("k must satisfy 0 <= k <= n.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0,1).")
    # Good enough for 95% (and nearby) decision reporting without extra deps.
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-12 else 2.5758293035489004
    phat = float(k) / float(n)
    denom = 1.0 + (z * z) / float(n)
    center = (phat + (z * z) / (2.0 * float(n))) / denom
    half = (z / denom) * sqrt((phat * (1.0 - phat) / float(n)) + (z * z) / (4.0 * float(n * n)))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return float(lo), float(hi)

