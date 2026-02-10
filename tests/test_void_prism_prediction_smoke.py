import numpy as np

from entropy_horizon_recon.sirens import MuForwardPosterior
from entropy_horizon_recon.void_prism import eg_gr_baseline_from_background, predict_EG_void_from_mu


def _make_lcdm_posterior(*, n_draws: int = 3) -> MuForwardPosterior:
    H0 = 70.0
    om0 = 0.3
    z_grid = np.linspace(0.0, 1.0, 200)
    H = H0 * np.sqrt(om0 * (1.0 + z_grid) ** 3 + (1.0 - om0))
    H_samples = np.repeat(H.reshape((1, -1)), n_draws, axis=0)
    # Ensure the inferred x-domain covers the horizon-mapping x(z) over z in [0,1].
    # For this LCDM grid, x(z=1) ~ -1.13, so we choose a slightly wider domain.
    x_grid = np.array([-1.5, -0.9, -0.3, 0.0])
    logmu_x_samples = np.zeros((n_draws, x_grid.size))
    return MuForwardPosterior(
        x_grid=x_grid,
        logmu_x_samples=logmu_x_samples,
        z_grid=z_grid,
        H_samples=H_samples,
        H0=np.full(n_draws, H0),
        omega_m0=np.full(n_draws, om0),
        omega_k0=np.zeros(n_draws),
        sigma8_0=None,
    )


def test_void_prism_minimal_matches_gr_when_mu_is_one():
    post = _make_lcdm_posterior()
    z_eff = 0.5
    eg_min = predict_EG_void_from_mu(post, z_eff=z_eff, embedding="minimal", max_draws=None)
    eg_gr = eg_gr_baseline_from_background(post, z_eff=z_eff, max_draws=None)
    assert eg_min.shape == eg_gr.shape
    assert np.allclose(eg_min, eg_gr, rtol=1e-10, atol=0.0)


def test_void_prism_slip_scales_sigma_only():
    post = _make_lcdm_posterior()
    z_eff = 0.5
    eg_gr = eg_gr_baseline_from_background(post, z_eff=z_eff, max_draws=None)
    eg_slip = predict_EG_void_from_mu(post, z_eff=z_eff, embedding="slip_allowed", eta0=1.5, eta1=0.0, max_draws=None)
    # With mu=1 and muP=1, only Sigma gets scaled by eta0, so EG scales the same way.
    assert np.allclose(eg_slip, 1.5 * eg_gr, rtol=1e-10, atol=0.0)
