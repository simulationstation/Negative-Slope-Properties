#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure this repo's `src/` is importable even if another checkout/version of
# `entropy_horizon_recon` is installed in the environment.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir():
    sys.path.insert(0, str(_SRC_DIR))

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pooch
from scipy import stats

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.repro import git_head_sha, git_is_dirty


def _utc_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%SUTC", time.gmtime())


def _log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except BrokenPipeError:
        return


def _json_default(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False, default=_json_default), encoding="ascii")


def _sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass(frozen=True)
class PlanckLensingProducts:
    component: str
    release: str
    lmax_tag: int
    tarball_url: str
    tarball_path: Path
    tarball_sha256: str | None
    dat_klm_path: Path
    mask_path: Path


def _ensure_planck_lensing_products(
    *,
    paths: DataPaths,
    component: str,
    release: str = "R3.00",
    lmax_tag: int = 4096,
    compute_sha256: bool = False,
) -> PlanckLensingProducts:
    component = component.upper()
    if component not in {"MV", "PP", "TT"}:
        raise ValueError("component must be one of: MV, PP, TT")

    base_url = "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/lensing/"
    fname = f"COM_Lensing_{lmax_tag}_{release}.tgz"
    tarball_url = base_url + fname

    cache_dir = paths.pooch_cache_dir / "planck2018_lensing_map"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # We intentionally do not pin known_hash here because the IRSA mirror is a
    # distribution endpoint; instead we optionally compute sha256 after fetch
    # and record it in the output meta for reproducibility.
    tarball_path_str = pooch.retrieve(url=tarball_url, known_hash=None, path=str(cache_dir), fname=fname, progressbar=True)
    tarball_path = Path(tarball_path_str)

    processed_dir = paths.processed_dir / "planck2018_lensing_map" / f"{release}_lmax{lmax_tag}"
    processed_dir.mkdir(parents=True, exist_ok=True)
    dat_klm_path = processed_dir / f"{component}_dat_klm.fits"
    mask_path = processed_dir / "mask.fits.gz"

    need_extract = (not dat_klm_path.exists()) or (not mask_path.exists())
    if need_extract:
        _log(f"[planck] extracting products from {tarball_path.name} (component={component})")
        with tarfile.open(tarball_path, mode="r:gz") as tf:
            member_dat = f"COM_Lensing_{lmax_tag}_{release}/{component}/dat_klm.fits"
            member_mask = f"COM_Lensing_{lmax_tag}_{release}/mask.fits.gz"
            with contextlib.ExitStack() as stack:
                f_dat = tf.extractfile(member_dat)
                if f_dat is None:
                    raise RuntimeError(f"Failed to extract {member_dat} from {tarball_path}")
                f_mask = tf.extractfile(member_mask)
                if f_mask is None:
                    raise RuntimeError(f"Failed to extract {member_mask} from {tarball_path}")
                dat_klm_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                # Stream extraction to disk (avoid holding the full FITS in RAM).
                out_dat = stack.enter_context(dat_klm_path.open("wb"))
                out_mask = stack.enter_context(mask_path.open("wb"))
                stack.callback(f_dat.close)
                stack.callback(f_mask.close)
                while True:
                    b = f_dat.read(1 << 20)
                    if not b:
                        break
                    out_dat.write(b)
                while True:
                    b = f_mask.read(1 << 20)
                    if not b:
                        break
                    out_mask.write(b)

    tarball_sha256 = None
    if compute_sha256:
        _log("[planck] computing tarball sha256 (one-time cost)")
        tarball_sha256 = _sha256_file(tarball_path)

    return PlanckLensingProducts(
        component=component,
        release=release,
        lmax_tag=int(lmax_tag),
        tarball_url=tarball_url,
        tarball_path=tarball_path,
        tarball_sha256=tarball_sha256,
        dat_klm_path=dat_klm_path,
        mask_path=mask_path,
    )


def _axis_from_args(args: argparse.Namespace) -> tuple[float, float]:
    if args.axis_results_json:
        p = Path(args.axis_results_json)
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, list) or not obj:
            raise ValueError("--axis-results-json must point to a non-empty JSON list.")
        # Use the max-|z_score| direction by default (most "anomalous" axis in the scan).
        best = max(obj, key=lambda d: abs(float(d.get("z_score", 0.0))))
        l_deg = float(best["axis_l_deg"])
        b_deg = float(best["axis_b_deg"])
        return l_deg, b_deg

    if args.axis_frame == "galactic":
        if args.axis_l_deg is None or args.axis_b_deg is None:
            raise ValueError("Provide --axis-l-deg/--axis-b-deg (or --axis-results-json).")
        return float(args.axis_l_deg), float(args.axis_b_deg)

    if args.axis_frame == "icrs":
        if args.axis_ra_deg is None or args.axis_dec_deg is None:
            raise ValueError("Provide --axis-ra-deg/--axis-dec-deg (or --axis-results-json).")
        try:
            import astropy.units as u
            from astropy.coordinates import SkyCoord
        except Exception as e:
            raise RuntimeError("axis_frame='icrs' requires astropy.") from e
        c = SkyCoord(ra=float(args.axis_ra_deg) * u.deg, dec=float(args.axis_dec_deg) * u.deg, frame="icrs").galactic
        return float(c.l.deg), float(c.b.deg)

    raise ValueError("Unsupported axis_frame.")


def _truncate_alm(*, alm_full: np.ndarray, lmax_full: int, lmax_out: int) -> np.ndarray:
    """Truncate a healpy alm array to l<=lmax_out (keeping m<=l)."""
    lmax_full = int(lmax_full)
    lmax_out = int(lmax_out)
    if lmax_out > lmax_full:
        raise ValueError("lmax_out must be <= lmax_full")
    out = np.zeros(hp.Alm.getsize(lmax_out, mmax=lmax_out), dtype=alm_full.dtype)
    for m in range(lmax_out + 1):
        for ell in range(m, lmax_out + 1):
            i_out = hp.Alm.getidx(lmax_out, ell, m)
            i_in = hp.Alm.getidx(lmax_full, ell, m)
            out[i_out] = alm_full[i_in]
    return out


def _weighted_mean_and_var(*, x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    wsum = float(np.sum(w))
    if not np.isfinite(wsum) or wsum <= 0.0:
        raise ValueError("Non-positive weight sum.")
    mean = float(np.sum(w * x) / wsum)
    var = float(np.sum(w * (x - mean) ** 2) / wsum)
    return mean, var


def _hemisphere_delta_var(
    *,
    kappa: np.ndarray,
    mask: np.ndarray,
    pix_vec: tuple[np.ndarray, np.ndarray, np.ndarray],
    axis_vec: np.ndarray,
    mask_min: float,
) -> dict[str, float]:
    vx, vy, vz = pix_vec
    axis_vec = np.asarray(axis_vec, dtype=float)
    if axis_vec.shape != (3,):
        raise ValueError("axis_vec must be shape (3,)")
    denom = float(np.linalg.norm(axis_vec))
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("Invalid axis vector.")
    ax, ay, az = (axis_vec / denom).tolist()

    sign = vx * ax + vy * ay + vz * az
    valid = mask > float(mask_min)

    north = valid & (sign >= 0.0)
    south = valid & ~north
    if not np.any(north) or not np.any(south):
        raise ValueError("Degenerate hemisphere split: empty north or south set.")

    wN = mask[north]
    xN = kappa[north]
    wS = mask[south]
    xS = kappa[south]
    meanN, varN = _weighted_mean_and_var(x=xN, w=wN)
    meanS, varS = _weighted_mean_and_var(x=xS, w=wS)

    if varN <= 0.0 or varS <= 0.0:
        raise ValueError("Non-positive variance encountered; check inputs.")

    delta = float(math.log(varN / varS))
    fsky_N = float(np.sum(wN) / np.sum(mask[valid]))
    fsky_S = float(np.sum(wS) / np.sum(mask[valid]))
    return {
        "delta_log_var": delta,
        "var_north": float(varN),
        "var_south": float(varS),
        "mean_north": float(meanN),
        "mean_south": float(meanS),
        "fsky_north": fsky_N,
        "fsky_south": fsky_S,
        "npix_valid": int(np.count_nonzero(valid)),
        "npix_north": int(np.count_nonzero(north)),
        "npix_south": int(np.count_nonzero(south)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Hemispherical Planck lensing (kappa) amplitude test along a fixed axis.")
    ap.add_argument("--component", default="MV", choices=["MV", "PP", "TT"], help="Planck lensing reconstruction variant.")
    ap.add_argument("--release", default="R3.00", help="Planck release tag (default: R3.00).")
    ap.add_argument("--lmax-tag", type=int, default=4096, help="Planck lensing product L_max tag in filename.")
    ap.add_argument("--nside", type=int, default=256, help="Output map nside for the statistic.")
    ap.add_argument("--lmin", type=int, default=8, help="Low-L cut (remove monopole/dipole + ultra-low-L).")
    ap.add_argument("--lmax", type=int, default=400, help="High-L cut (avoid noise-dominated tail).")
    ap.add_argument("--mask-min", type=float, default=0.5, help="Minimum mask weight to treat a pixel as usable.")
    ap.add_argument("--n-random", type=int, default=2000, help="Number of random-axis rotations for the null.")
    ap.add_argument("--seed", type=int, default=1234, help="RNG seed for random axes.")
    ap.add_argument("--compute-sha256", action="store_true", help="Compute sha256 of the downloaded tarball.")

    ap.add_argument("--axis-results-json", default=None, help="Optional path to axis_results.json from the anisotropy scan.")
    ap.add_argument("--axis-frame", default="galactic", choices=["galactic", "icrs"], help="Axis coordinate frame.")
    ap.add_argument("--axis-l-deg", type=float, default=None, help="Axis Galactic longitude l (deg).")
    ap.add_argument("--axis-b-deg", type=float, default=None, help="Axis Galactic latitude b (deg).")
    ap.add_argument("--axis-ra-deg", type=float, default=None, help="Axis RA (deg), if axis_frame=icrs.")
    ap.add_argument("--axis-dec-deg", type=float, default=None, help="Axis DEC (deg), if axis_frame=icrs.")

    ap.add_argument("--outdir", default=None, help="Output directory (default: outputs/planck_lensing_hemi_<stamp>).")
    args = ap.parse_args()

    paths = DataPaths(repo_root=_REPO_ROOT)
    outdir = Path(args.outdir) if args.outdir else (_REPO_ROOT / "outputs" / f"planck_lensing_hemi_{_utc_stamp()}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Axis: accept a fixed hypothesis direction (preferred) or pull from an axis scan.
    axis_l_deg, axis_b_deg = _axis_from_args(args)
    theta = np.deg2rad(90.0 - axis_b_deg)
    phi = np.deg2rad(axis_l_deg % 360.0)
    axis_vec = hp.ang2vec(theta, phi)

    _log(f"[axis] l={axis_l_deg:.3f} deg, b={axis_b_deg:.3f} deg (Galactic)")
    _log(f"[cfg] nside={args.nside} lmin={args.lmin} lmax={args.lmax} n_random={args.n_random} mask_min={args.mask_min}")

    prod = _ensure_planck_lensing_products(
        paths=paths,
        component=args.component,
        release=args.release,
        lmax_tag=args.lmax_tag,
        compute_sha256=bool(args.compute_sha256),
    )

    _log(f"[planck] dat_klm={prod.dat_klm_path}")
    _log(f"[planck] mask={prod.mask_path}")

    # Read the Planck convergence harmonics and truncate to the requested lmax.
    alm_full = hp.read_alm(str(prod.dat_klm_path))
    alm_full = np.asarray(alm_full, dtype=np.complex64)
    lmax_full = int(hp.Alm.getlmax(len(alm_full)))
    if args.lmax > lmax_full:
        raise ValueError(f"Requested lmax={args.lmax} exceeds file lmax={lmax_full}.")
    alm = _truncate_alm(alm_full=alm_full, lmax_full=lmax_full, lmax_out=int(args.lmax))
    fl = np.ones(int(args.lmax) + 1, dtype=float)
    fl[: int(args.lmin)] = 0.0
    alm = hp.almxfl(alm, fl)

    # Build a band-limited convergence map on a modest nside.
    kappa = hp.alm2map(alm, nside=int(args.nside), lmax=int(args.lmax), verbose=False)
    kappa = np.asarray(kappa, dtype=np.float32)

    # Read and downgrade the mask to the same nside. Keep weights in [0,1].
    mask_hi = hp.read_map(str(prod.mask_path), field=0, dtype=np.float32, verbose=False)
    mask = hp.ud_grade(mask_hi, nside_out=int(args.nside), power=0)
    mask = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)

    npix = hp.nside2npix(int(args.nside))
    pix = np.arange(npix, dtype=np.int64)
    pix_vec = hp.pix2vec(int(args.nside), pix)

    obs = _hemisphere_delta_var(
        kappa=kappa,
        mask=mask,
        pix_vec=pix_vec,
        axis_vec=axis_vec,
        mask_min=float(args.mask_min),
    )
    delta_obs = float(obs["delta_log_var"])
    _log(f"[obs] delta_log_var={delta_obs:+.6f} (log(var_N/var_S))")

    rng = np.random.default_rng(int(args.seed))
    n_random = int(args.n_random)
    deltas = np.empty(n_random, dtype=np.float64)
    for i in range(n_random):
        # Random axis from isotropic Gaussian direction.
        v = rng.normal(size=3)
        v /= float(np.linalg.norm(v))
        d = _hemisphere_delta_var(
            kappa=kappa,
            mask=mask,
            pix_vec=pix_vec,
            axis_vec=v,
            mask_min=float(args.mask_min),
        )
        deltas[i] = float(d["delta_log_var"])
        if (i + 1) % 250 == 0:
            _log(f"[null] {i+1}/{n_random} axes done")

    # Two-sided p-value for a fixed (a priori) axis.
    p_two = float((np.count_nonzero(np.abs(deltas) >= abs(delta_obs)) + 1) / (n_random + 1))
    z_two = float(stats.norm.isf(p_two / 2.0)) if p_two > 0 else float("inf")
    _log(f"[null] p(two-sided)={p_two:.6g} => z~{z_two:.3f}")

    out = {
        "axis_galactic": {"l_deg": axis_l_deg, "b_deg": axis_b_deg},
        "statistic": "delta_log_var = log(var_N/var_S) on band-limited kappa map",
        "delta_log_var_obs": delta_obs,
        "obs": obs,
        "null": {
            "n_random": n_random,
            "seed": int(args.seed),
            "delta_log_var_null_mean": float(np.mean(deltas)),
            "delta_log_var_null_std": float(np.std(deltas, ddof=1)),
            "p_two_sided": p_two,
            "z_two_sided_gauss_equiv": z_two,
        },
        "map_cfg": {
            "nside": int(args.nside),
            "lmin": int(args.lmin),
            "lmax": int(args.lmax),
            "mask_min": float(args.mask_min),
        },
        "planck_products": {
            "component": prod.component,
            "release": prod.release,
            "lmax_tag": prod.lmax_tag,
            "tarball_url": prod.tarball_url,
            "tarball_path": prod.tarball_path,
            "tarball_sha256": prod.tarball_sha256,
            "dat_klm_path": prod.dat_klm_path,
            "mask_path": prod.mask_path,
        },
        "repro": {"git_sha": git_head_sha(repo_root=_REPO_ROOT), "git_dirty": git_is_dirty(repo_root=_REPO_ROOT)},
    }
    _write_json(outdir / "planck_lensing_hemisphere_test.json", out)
    np.save(outdir / "delta_log_var_null.npy", deltas)

    # Plot: histogram of null with the observed line.
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.hist(deltas, bins=40, density=True, color="#5b8bb2", alpha=0.85, label="random-axis null")
    ax.axvline(delta_obs, color="#d1495b", lw=2.0, label=f"observed ({delta_obs:+.3g})")
    ax.set_xlabel(r"$\Delta \log \mathrm{Var}(\kappa)$ (north-south)")
    ax.set_ylabel("density")
    ax.set_title(f"Planck lensing hemisphere test (fixed axis)\n" f"p(two-sided)={p_two:.3g}, z~{z_two:.2f}")
    ax.legend(loc="best", frameon=False)
    try:
        fig.tight_layout()
    except Exception:
        # Avoid a hard failure if backend mathtext/layout has issues.
        pass
    fig.savefig(outdir / "planck_lensing_hemisphere_test.png", dpi=160)
    plt.close(fig)

    _log(f"[out] {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
