import numpy as np
import pytest

hp = pytest.importorskip("healpy")

from entropy_horizon_recon.optical_bias.estimators import weighted_linear_fit


def test_regression_random_positions_near_zero():
    rng = np.random.default_rng(1)
    n = 300
    kappa = rng.normal(scale=0.01, size=n)
    residuals = rng.normal(scale=0.1, size=n)
    weights = np.ones(n) / n
    reg = weighted_linear_fit(kappa, residuals, weights)
    # With x~O(1e-2) and y~O(1e-1), the slope b has O(1) sampling scatter even under the null.
    # A robust null check is that the standardized slope is not an extreme outlier.
    assert abs(reg["z"]) < 4.0
