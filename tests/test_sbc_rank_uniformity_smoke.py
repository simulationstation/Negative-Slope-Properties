import numpy as np

from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.likelihoods import BaoLogLike
from entropy_horizon_recon.sbc import run_sbc_prior_truth


def test_sbc_rank_uniformity_smoke():
    """Smoke test for prior-truth SBC machinery (small N, cheap chains).

    This test is not meant to certify uniformity (too small-N); it ensures:
      - the SBC loop runs end-to-end,
      - ranks are within bounds,
      - uniformity p-values are finite.
    """
    constants = PhysicalConstants()

    z_grid = np.linspace(0.0, 0.8, 60)
    x_knots = np.linspace(-1.4, 0.0, 6)
    x_grid = np.linspace(-1.2, 0.0, 40)

    # Tiny synthetic "template" datasets.
    sn_z = np.linspace(0.02, 0.8, 20)
    sn_cov = (0.08**2) * np.eye(sn_z.size)

    cc_z = np.linspace(0.1, 0.8, 5)
    cc_sigma_H = np.full_like(cc_z, 8.0)

    bao_template = BaoLogLike.from_arrays(
        dataset="desi_2024_bao_all",
        z=np.array([0.35]),
        y=np.array([0.0]),
        obs=np.array(["DM_over_rs"]),
        cov=np.array([[0.06**2]]),
        constants=constants,
    )

    res = run_sbc_prior_truth(
        seed=0,
        N=3,
        z_grid=z_grid,
        x_knots=x_knots,
        x_grid=x_grid,
        sn_z=sn_z,
        sn_cov=sn_cov,
        cc_z=cc_z,
        cc_sigma_H=cc_sigma_H,
        bao_templates=[bao_template],
        constants=constants,
        n_walkers=28,
        n_steps=240,
        n_burn=80,
        n_draws=120,
        n_processes=1,
        max_rss_mb=1024.0,
        sigma_cc_jit_scale=0.5,
        sigma_sn_jit_scale=0.02,
        sigma_d2_scale=0.185,
    )

    assert res["N"] == 3
    assert res["n_draws"] == 120
    for key, ranks in res["ranks"].items():
        r = np.asarray(ranks, dtype=int)
        assert r.shape == (3,)
        assert np.all((r >= 0) & (r <= 120)), f"rank out of bounds for {key}"
    for key, pv in res["pvalues"].items():
        assert 0.0 <= float(pv["chi2_p"]) <= 1.0
        assert 0.0 <= float(pv["ks_p"]) <= 1.0

