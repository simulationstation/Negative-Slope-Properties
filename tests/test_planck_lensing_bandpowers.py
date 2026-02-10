import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.ingest_planck_lensing_bandpowers import load_planck_lensing_bandpowers
from entropy_horizon_recon.likelihoods_planck_lensing_bandpowers import PlanckLensingBandpowerLogLike


def test_planck_lensing_bandpowers_shapes():
    paths = DataPaths(repo_root='.')
    bp = load_planck_lensing_bandpowers(paths=paths, dataset='consext8')
    assert bp.clpp.size == bp.cov.shape[0]
    assert bp.cov.shape[0] == bp.cov.shape[1]
    assert np.allclose(bp.cov, bp.cov.T, atol=1e-10)


def test_planck_lensing_bandpowers_loglike_finite():
    paths = DataPaths(repo_root='.')
    bp = load_planck_lensing_bandpowers(paths=paths, dataset='consext8')
    like = PlanckLensingBandpowerLogLike.from_data(
        clpp=bp.clpp,
        cov=bp.cov,
        template_clpp=bp.template_clpp,
        alpha=0.25,
        s8om_fid=0.589,
        meta=bp.meta,
    )
    model = like.predict(omega_m0=0.3, sigma8_0=0.8)
    ll = like.loglike(model)
    assert np.isfinite(ll)
