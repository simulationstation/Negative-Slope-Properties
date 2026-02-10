import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.ingest_fullshape_pk import load_fullshape_pk
from entropy_horizon_recon.likelihoods_fullshape_pk import FullShapePkLogLike


def test_fullshape_pk_shapes():
    paths = DataPaths(repo_root='.')
    pk = load_fullshape_pk(paths=paths, dataset='shapefit_lrgz1_ngc_mono')
    assert pk.k.size == pk.pk.size
    assert pk.cov.shape == (pk.k.size, pk.k.size)
    assert np.allclose(pk.cov, pk.cov.T, atol=1e-10)


def test_fullshape_pk_loglike_finite():
    paths = DataPaths(repo_root='.')
    pk = load_fullshape_pk(paths=paths, dataset='shapefit_lrgz1_ngc_mono')
    like = FullShapePkLogLike.from_data(k=pk.k, pk=pk.pk, cov=pk.cov, z_eff=pk.z_eff, meta=pk.meta)
    model = like.predict(H0=70.0, omega_m0=0.3, omega_k0=0.0, sigma8_0=0.8, b1=2.0, pshot=1000.0)
    ll = like.loglike(model)
    assert np.isfinite(ll)
