import numpy as np

from entropy_horizon_recon.growth import predict_fsigma8, solve_growth_ode


def _lcdm_growth_reference(*, H0: float, omega_m0: float, sigma8_0: float, z_eval: np.ndarray) -> np.ndarray:
    """Reference fσ8(z) for flat ΛCDM via the exact integral growth-factor expression."""
    z_eval = np.asarray(z_eval, dtype=float)
    a_eval = 1.0 / (1.0 + z_eval)

    # Dense grid in a for accurate quadrature of D(a).
    a = np.logspace(-4, 0, 20000)
    E = np.sqrt(omega_m0 / a**3 + (1.0 - omega_m0))
    integrand = 1.0 / (a**3 * E**3)
    # cumulative trapezoid (manual; avoids requiring scipy)
    da = np.diff(a)
    I = np.empty_like(a)
    I[0] = 0.0
    I[1:] = np.cumsum(0.5 * da * (integrand[:-1] + integrand[1:]))
    D_unnorm = 2.5 * omega_m0 * E * I
    D = D_unnorm / D_unnorm[-1]
    # f = d ln D / d ln a
    f = np.gradient(np.log(np.clip(D, 1e-300, np.inf)), np.log(a))

    D_ev = np.interp(a_eval, a, D)
    f_ev = np.interp(a_eval, a, f)
    return f_ev * sigma8_0 * D_ev


def test_growth_solver_recovers_lcdm():
    H0 = 70.0
    omega_m0 = 0.3
    sigma8_0 = 0.8
    z_grid = np.linspace(0.0, 3.0, 600)
    H_grid = H0 * np.sqrt(omega_m0 * (1.0 + z_grid) ** 3 + (1.0 - omega_m0))

    # Smoke: growth solution is physical.
    sol = solve_growth_ode(z_grid=z_grid, H_grid=H_grid, H0=H0, omega_m0=omega_m0, z_start=z_grid[-1], n_ext=1)
    assert np.isfinite(sol.D).all()
    assert np.isfinite(sol.f).all()
    assert np.all(sol.D > 0)

    z_eval = np.array([0.0, 0.2, 0.5, 1.0, 2.0])
    fs8_model = predict_fsigma8(
        z_eval=z_eval,
        z_grid=z_grid,
        H_grid=H_grid,
        H0=H0,
        omega_m0=omega_m0,
        sigma8_0=sigma8_0,
    )
    fs8_ref = _lcdm_growth_reference(H0=H0, omega_m0=omega_m0, sigma8_0=sigma8_0, z_eval=z_eval)

    # Allow a small error due to finite z-grid resolution / RK4 stepping.
    assert np.allclose(fs8_model, fs8_ref, rtol=1e-2, atol=2e-3)
