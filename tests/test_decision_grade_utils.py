import numpy as np

from entropy_horizon_recon.decision_grade import slope_draws_from_logmu_samples, wilson_interval


def test_slope_draws_from_logmu_samples_recovers_linear_slope():
    logA = np.linspace(0.0, 1.0, 50)
    slopes = np.array([-0.3, -0.1, 0.2], dtype=float)
    draws = np.vstack([s * (logA - np.mean(logA)) for s in slopes])
    out = slope_draws_from_logmu_samples(logA_grid=logA, logmu_samples=draws, weight_mode="uniform")
    assert np.allclose(out, slopes, rtol=1e-12, atol=1e-12)


def test_wilson_interval_bounds_and_order():
    lo, hi = wilson_interval(k=5, n=20, confidence=0.95)
    assert 0.0 <= lo <= hi <= 1.0

