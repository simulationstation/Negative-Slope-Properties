import numpy as np

from entropy_horizon_recon.growth import solve_growth_ode, solve_growth_ode_muP


def test_growth_solver_muP_matches_gr_when_muP_is_unity():
    H0 = 70.0
    omega_m0 = 0.3
    z_grid = np.linspace(0.0, 3.0, 600)
    H_grid = H0 * np.sqrt(omega_m0 * (1.0 + z_grid) ** 3 + (1.0 - omega_m0))
    muP_grid = np.ones_like(z_grid)

    sol_gr = solve_growth_ode(z_grid=z_grid, H_grid=H_grid, H0=H0, omega_m0=omega_m0, z_start=z_grid[-1], n_ext=1)
    sol_mu = solve_growth_ode_muP(
        z_grid=z_grid,
        H_grid=H_grid,
        H0=H0,
        omega_m0=omega_m0,
        muP_grid=muP_grid,
        z_start=z_grid[-1],
        n_ext=1,
    )

    # Exact equality is not guaranteed due to slightly different code paths; they should agree
    # numerically to tight tolerance for muP=1.
    assert np.allclose(sol_mu.D, sol_gr.D, rtol=1e-10, atol=0.0)
    assert np.allclose(sol_mu.f, sol_gr.f, rtol=1e-10, atol=0.0)


def test_growth_solver_muP_strengthens_growth_when_muP_gt_one():
    H0 = 70.0
    omega_m0 = 0.3
    z_grid = np.linspace(0.0, 3.0, 600)
    H_grid = H0 * np.sqrt(omega_m0 * (1.0 + z_grid) ** 3 + (1.0 - omega_m0))
    muP_grid = np.full_like(z_grid, 1.2)

    sol_gr = solve_growth_ode(z_grid=z_grid, H_grid=H_grid, H0=H0, omega_m0=omega_m0, z_start=z_grid[-1], n_ext=1)
    sol_mu = solve_growth_ode_muP(
        z_grid=z_grid,
        H_grid=H_grid,
        H0=H0,
        omega_m0=omega_m0,
        muP_grid=muP_grid,
        z_start=z_grid[-1],
        n_ext=1,
    )

    # With a stronger source term, growth is faster. After normalizing D(z=0)=1, the earlier-time
    # D(z) should be *smaller* (it has more growth left to reach 1 by z=0).
    z_check = 0.5
    x_check = -np.log1p(z_check)
    D_gr = float(np.interp(x_check, sol_gr.x_grid, sol_gr.D))
    D_mu = float(np.interp(x_check, sol_mu.x_grid, sol_mu.D))
    assert D_mu < D_gr
