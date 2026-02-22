import numpy as np

from entropy_horizon_recon.sbc import sample_sbc_prior_truth


def test_sample_sbc_prior_truth_linear_slope_mode():
    rng = np.random.default_rng(0)
    x_knots = np.linspace(-1.2, 0.0, 7)
    slope = -0.25
    truth = sample_sbc_prior_truth(
        rng,
        x_knots=x_knots,
        omega_m0_prior=(0.2, 0.4),
        H0_prior=(60.0, 80.0),
        r_d_prior=(130.0, 160.0),
        sigma_cc_jit_scale=1.0,
        sigma_sn_jit_scale=0.1,
        logmu_knot_scale=1.0,
        log_sigma_d2_prior=(-12.0, 3.0),
        sigma_d2_scale=0.2,
        mu_truth_mode="linear_slope",
        mu_truth_slope=slope,
    )
    assert np.allclose(truth.logmu_knots, slope * x_knots)

