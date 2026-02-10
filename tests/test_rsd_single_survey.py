import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.ingest_rsd_single_survey import load_rsd_single_survey
from entropy_horizon_recon.likelihoods_rsd_single_survey import RsdFs8CovLogLike


def test_rsd_single_survey_covariance():
    paths = DataPaths(repo_root=".")
    rsd = load_rsd_single_survey(paths=paths, dataset="sdss_dr16_lrg_fsbao_dmdhfs8")
    assert rsd.z.size == rsd.fs8.size
    assert rsd.cov.shape == (rsd.z.size, rsd.z.size)
    assert np.allclose(rsd.cov, rsd.cov.T, atol=1e-10)


def test_rsd_single_survey_loglike_finite():
    paths = DataPaths(repo_root=".")
    rsd = load_rsd_single_survey(paths=paths, dataset="sdss_dr16_lrg_fsbao_dmdhfs8")
    like = RsdFs8CovLogLike.from_data(z=rsd.z, fs8=rsd.fs8, cov=rsd.cov, meta=rsd.meta)
    # Use exact data as model to yield finite likelihood.
    ll = like.loglike(rsd.fs8)
    assert np.isfinite(ll)
