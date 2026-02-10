import numpy as np
import pytest

from entropy_horizon_recon.optical_bias.injection import estimate_delta_h0_over_h0, inject_mu
from entropy_horizon_recon.optical_bias.weights import h0_estimator_weights
from entropy_horizon_recon.optical_bias.injection import delta_dl_over_dl


def test_injection_recovers_h0_bias():
    rng = np.random.default_rng(0)
    n = 200
    z = rng.uniform(0.02, 0.1, size=n)
    mu_model = 5.0 * np.log10(3000.0 * z) + 25.0
    sigma = np.full(n, 0.1)
    weights = h0_estimator_weights(z, sigma)
    kappa = rng.normal(scale=0.01, size=n)
    mu_inj = inject_mu(mu_model, kappa)
    dlnH0 = estimate_delta_h0_over_h0(mu_inj, mu_model, weights)
    # Expected mean kappa drives deltaH0/H0
    expected = float(np.sum(weights * kappa))
    assert np.isfinite(dlnH0)
    assert abs(dlnH0 - expected) < 5e-3
