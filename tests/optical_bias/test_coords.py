import numpy as np
import pytest

hp = pytest.importorskip("healpy")
astropy = pytest.importorskip("astropy")

from entropy_horizon_recon.optical_bias.maps import healpix_to_radec, radec_to_healpix


def test_radec_roundtrip_icrs():
    nside = 32
    ra = np.array([10.0, 120.0, 250.0])
    dec = np.array([-10.0, 5.0, 45.0])
    pix = radec_to_healpix(ra, dec, nside=nside, frame="icrs")
    ra2, dec2 = healpix_to_radec(pix, nside=nside, frame="icrs")
    # Round-trip within a pixel size (~1 deg at nside=32)
    assert np.all(np.abs(ra - ra2) < 2.0)
    assert np.all(np.abs(dec - dec2) < 2.0)


def test_radec_roundtrip_galactic():
    nside = 32
    ra = np.array([10.0, 120.0, 250.0])
    dec = np.array([-10.0, 5.0, 45.0])
    pix = radec_to_healpix(ra, dec, nside=nside, frame="galactic")
    ra2, dec2 = healpix_to_radec(pix, nside=nside, frame="galactic")
    assert np.all(np.abs(ra - ra2) < 2.0)
    assert np.all(np.abs(dec - dec2) < 2.0)
