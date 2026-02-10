import numpy as np

from entropy_horizon_recon.likelihoods_lensing import PlanckLensingProxyLogLike


def test_lensing_proxy_likelihood_smoke():
    like = PlanckLensingProxyLogLike(mean=0.589, sigma=0.020, meta={"source": "test"})
    model = like.predict(omega_m0=0.3, sigma8_0=0.8)
    assert np.isfinite(model)
    ll0 = like.loglike(model)
    ll1 = like.loglike(model + 0.02)
    assert np.isfinite(ll0)
    assert np.isfinite(ll1)
    assert ll0 > ll1  # Gaussian peak at the mean

