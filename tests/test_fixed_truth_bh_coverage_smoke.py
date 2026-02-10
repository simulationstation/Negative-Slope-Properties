import numpy as np

from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.cosmology import build_background_from_H_grid
from entropy_horizon_recon.inversion import infer_logmu_forward


def test_fixed_truth_bh_coverage_smoke():
    """Smoke-level fixed-truth coverage check for BH truth (μ=1 => logμ=0).

    This is intentionally small-N and low-cost: it should catch severe under-coverage
    regressions (e.g. derivative-noise inversion) without being flaky.
    """
    constants = PhysicalConstants()
    H0_true = 70.0
    omega_m0_true = 0.3
    z_max = 0.8
    z_grid = np.linspace(0.0, z_max, 80)
    H_true = H0_true * np.sqrt(omega_m0_true * (1.0 + z_grid) ** 3 + (1.0 - omega_m0_true))
    bg = build_background_from_H_grid(z_grid, H_true, constants=constants)

    # Small synthetic datasets
    z_sn = np.linspace(0.02, z_max, 30)
    M_true = -3.0
    m_true = 5.0 * np.log10(bg.Dl(z_sn)) + M_true
    sigma_m = 0.08
    cov_sn = (sigma_m**2) * np.eye(z_sn.size)

    z_cc = np.linspace(0.1, z_max, 6)
    H_cc_true = bg.H(z_cc)
    sigma_H = np.full_like(z_cc, 8.0)

    # x domain for μ(A): x = log(A/A0) ≈ 2 log(H0/H(z))
    H_zmax = float(H_true[-1])
    x_min = float(2.0 * np.log(H0_true / H_zmax))
    x_knots = np.linspace(1.25 * x_min, 0.0, 6)
    x_grid = np.linspace(x_min, 0.0, 40)

    rng = np.random.default_rng(0)
    n_rep = 3
    cover68_mu = []
    cover95_mu = []
    cover68_H = []
    cover95_H = []

    for i in range(n_rep):
        m_obs = m_true + sigma_m * rng.normal(size=m_true.shape)
        H_cc = H_cc_true + sigma_H * rng.normal(size=H_cc_true.shape)

        post = infer_logmu_forward(
            z_grid=z_grid,
            x_knots=x_knots,
            x_grid=x_grid,
            sn_z=z_sn,
            sn_m=m_obs,
            sn_cov=cov_sn,
            cc_z=z_cc,
            cc_H=H_cc,
            cc_sigma_H=sigma_H,
            bao_likes=[],
            constants=constants,
            n_walkers=32,
            n_steps=320,
            n_burn=110,
            seed=100 + i,
            n_processes=1,
            n_draws=200,
            progress=False,
        )

        lo68 = np.percentile(post.logmu_x_samples, 16.0, axis=0)
        hi68 = np.percentile(post.logmu_x_samples, 84.0, axis=0)
        lo95 = np.percentile(post.logmu_x_samples, 2.5, axis=0)
        hi95 = np.percentile(post.logmu_x_samples, 97.5, axis=0)
        cover68_mu.append(float(np.mean((0.0 >= lo68) & (0.0 <= hi68))))
        cover95_mu.append(float(np.mean((0.0 >= lo95) & (0.0 <= hi95))))

        lo68H = np.percentile(post.H_samples, 16.0, axis=0)
        hi68H = np.percentile(post.H_samples, 84.0, axis=0)
        lo95H = np.percentile(post.H_samples, 2.5, axis=0)
        hi95H = np.percentile(post.H_samples, 97.5, axis=0)
        cover68_H.append(float(np.mean((H_true >= lo68H) & (H_true <= hi68H))))
        cover95_H.append(float(np.mean((H_true >= lo95H) & (H_true <= hi95H))))

    # Avoid catastrophic under-coverage; allow some slack for small-N MCMC noise.
    assert float(np.mean(cover68_mu)) >= 0.50
    assert float(np.mean(cover95_mu)) >= 0.80
    assert float(np.mean(cover68_H)) >= 0.35
    assert float(np.mean(cover95_H)) >= 0.75
