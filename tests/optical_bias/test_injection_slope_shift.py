import numpy as np

from entropy_horizon_recon.optical_bias.estimators import weighted_linear_fit


def test_injection_slope_shift_is_exact():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y0 = rng.normal(size=n)
    w = rng.uniform(0.1, 2.0, size=n)
    c = -0.37

    reg0 = weighted_linear_fit(x, y0, w)
    reg1 = weighted_linear_fit(x, y0 + c * x, w)
    assert np.isfinite(reg0["b"])
    assert np.isfinite(reg1["b"])
    assert abs((reg1["b"] - reg0["b"]) - c) < 1e-10
