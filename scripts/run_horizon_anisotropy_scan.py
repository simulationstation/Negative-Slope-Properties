#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import healpy as hp
import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.departure import compute_departure_stats
from entropy_horizon_recon.ingest import load_pantheon_plus_sky
from entropy_horizon_recon.inversion import infer_logmu_forward
from entropy_horizon_recon.likelihoods import SNLogLike, bin_sn_loglike
from entropy_horizon_recon.repro import git_head_sha, git_is_dirty


def _utc_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%SUTC", time.gmtime())


def _log(msg: str) -> None:
    # Keep logs visible in nohup+run.log (when PYTHONUNBUFFERED=1).
    print(msg, flush=True)


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


def _write_json_atomic(path: Path, obj: Any) -> None:
    """Atomic JSON write (tmp + rename) to support concurrent shard processes safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, indent=2, sort_keys=False, default=_json_default)
    # tmp in the same directory so os.replace() is atomic on POSIX filesystems.
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _radec_to_galactic_unitvec(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Convert ICRS-ish RA/DEC (deg) to Galactic unit vectors using healpy's built-in rotator."""
    ra_deg = np.asarray(ra_deg, dtype=float)
    dec_deg = np.asarray(dec_deg, dtype=float)
    theta_c = np.deg2rad(90.0 - dec_deg)
    phi_c = np.deg2rad(ra_deg)
    rot = hp.Rotator(coord=["C", "G"])
    theta_g, phi_g = rot(theta_c, phi_c)
    return hp.ang2vec(theta_g, phi_g)


def _galactic_unitvec_to_radec(vec: np.ndarray) -> tuple[float, float]:
    """Convert a Galactic unit vector to (ra_deg, dec_deg) in equatorial coordinates."""
    vec = np.asarray(vec, dtype=float)
    if vec.shape != (3,):
        raise ValueError("vec must be shape (3,).")
    # vec -> (theta_g, phi_g)
    theta_g, phi_g = hp.vec2ang(vec)
    rot = hp.Rotator(coord=["G", "C"])
    theta_c, phi_c = rot(theta_g, phi_g)
    ra_deg = float(np.rad2deg(phi_c) % 360.0)
    dec_deg = float(90.0 - np.rad2deg(theta_c))
    return ra_deg, dec_deg


def _axis_grid_galactic(*, nside: int, nest: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pix, vec[pix], (l_deg,b_deg)[pix]) for pixel centers in Galactic coords."""
    nside = int(nside)
    if nside <= 0:
        raise ValueError("nside must be positive.")
    npix = hp.nside2npix(nside)
    pix = np.arange(npix, dtype=np.int64)
    theta, phi = hp.pix2ang(nside, pix, nest=bool(nest))
    vec = hp.ang2vec(theta, phi)  # Galactic basis by construction.
    l_deg = (np.rad2deg(phi) % 360.0).astype(float)
    b_deg = (90.0 - np.rad2deg(theta)).astype(float)
    lb = np.column_stack([l_deg, b_deg])
    return pix, vec, lb


def _antipode_pix(*, nside: int, vec: np.ndarray, nest: bool) -> int:
    """Return the pixel index containing the antipode direction (-vec)."""
    vec = np.asarray(vec, dtype=float)
    if vec.shape != (3,):
        raise ValueError("vec must be shape (3,).")
    return int(hp.vec2pix(int(nside), -float(vec[0]), -float(vec[1]), -float(vec[2]), nest=bool(nest)))


def _match_z_binwise(
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    z: np.ndarray,
    *,
    z_edges: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Downsample two index sets so they have matched z histograms.

    This is a pragmatic control for depth differences. It keeps min(count_a, count_b) per z-bin
    in both sets by random selection (seeded).
    """
    z = np.asarray(z, dtype=float)
    z_edges = np.asarray(z_edges, dtype=float)
    if z_edges.ndim != 1 or z_edges.size < 3 or np.any(np.diff(z_edges) <= 0):
        raise ValueError("z_edges must be strictly increasing and have >= 3 edges.")
    idx_a = np.asarray(idx_a, dtype=np.int64)
    idx_b = np.asarray(idx_b, dtype=np.int64)

    keep_a: list[np.ndarray] = []
    keep_b: list[np.ndarray] = []
    used = []
    for i in range(z_edges.size - 1):
        lo = float(z_edges[i])
        hi = float(z_edges[i + 1])
        if i < z_edges.size - 2:
            ma = idx_a[(z[idx_a] >= lo) & (z[idx_a] < hi)]
            mb = idx_b[(z[idx_b] >= lo) & (z[idx_b] < hi)]
        else:
            ma = idx_a[(z[idx_a] >= lo) & (z[idx_a] <= hi)]
            mb = idx_b[(z[idx_b] >= lo) & (z[idx_b] <= hi)]
        na = int(ma.size)
        nb = int(mb.size)
        nk = min(na, nb)
        if nk <= 0:
            used.append({"bin": [lo, hi], "na": na, "nb": nb, "keep": 0})
            continue
        sel_a = rng.choice(ma, size=nk, replace=False)
        sel_b = rng.choice(mb, size=nk, replace=False)
        keep_a.append(np.asarray(sel_a, dtype=np.int64))
        keep_b.append(np.asarray(sel_b, dtype=np.int64))
        used.append({"bin": [lo, hi], "na": na, "nb": nb, "keep": nk})
    if keep_a:
        out_a = np.sort(np.concatenate(keep_a))
        out_b = np.sort(np.concatenate(keep_b))
    else:
        out_a = np.zeros((0,), dtype=np.int64)
        out_b = np.zeros((0,), dtype=np.int64)
    meta = {"mode": "bin_downsample", "z_edges": z_edges.tolist(), "bins": used}
    return out_a, out_b, meta


def _permute_positions_within_survey(idsurvey: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a permutation index that shuffles within each survey group."""
    idsurvey = np.asarray(idsurvey, dtype=np.int64)
    perm = np.arange(idsurvey.size, dtype=np.int64)
    for sid in np.unique(idsurvey):
        idx = np.where(idsurvey == sid)[0]
        if idx.size <= 1:
            continue
        perm[idx] = rng.permutation(idx)
    return perm


def _make_stratified_folds(
    *,
    idsurvey: np.ndarray,
    z: np.ndarray,
    k_folds: int,
    z_bin_width: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign each SN to a fold, stratified by (survey_id, z-bin).

    This keeps survey footprint and depth distributions similar across folds.
    """
    idsurvey = np.asarray(idsurvey, dtype=np.int64)
    z = np.asarray(z, dtype=float)
    k_folds = int(k_folds)
    z_bin_width = float(z_bin_width)
    if k_folds < 2:
        raise ValueError("k_folds must be >= 2.")
    if not np.isfinite(z_bin_width) or z_bin_width <= 0:
        raise ValueError("z_bin_width must be finite and positive.")

    z_min = float(np.nanmin(z))
    z_max = float(np.nanmax(z))
    z_edges = np.arange(z_min, z_max + z_bin_width, z_bin_width)
    if z_edges.size < 3:
        raise ValueError("Too few z bins for stratification; increase z_bin_width.")
    z_bin = np.digitize(z, z_edges, right=False) - 1
    z_bin = np.clip(z_bin, 0, z_edges.size - 2).astype(np.int64)

    # Map each (survey, zbin) to a group id.
    # Use a simple hash into a 64-bit key.
    key = (idsurvey.astype(np.int64) << 32) ^ z_bin.astype(np.int64)
    uniq = np.unique(key)
    fold = np.full((z.size,), -1, dtype=np.int64)

    groups = []
    for g, k in enumerate(uniq.tolist()):
        idx = np.where(key == int(k))[0].astype(np.int64)
        if idx.size == 0:
            continue
        idx = rng.permutation(idx)
        # Round-robin assignment across folds within group.
        fold[idx] = np.arange(idx.size, dtype=np.int64) % k_folds
        sid = int(idsurvey[idx[0]])
        zb = int(z_bin[idx[0]])
        groups.append({"group": int(g), "survey_id": sid, "z_bin": zb, "n": int(idx.size)})
    if np.any(fold < 0):
        raise RuntimeError("Fold assignment failed for some entries.")
    meta = {"k_folds": k_folds, "z_bin_width": z_bin_width, "z_edges": z_edges.tolist(), "n_groups": int(len(groups)), "groups": groups}
    return fold, meta


def _match_fore_aft_by_group(
    idx_fore: np.ndarray,
    idx_aft: np.ndarray,
    *,
    group_key: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Downsample fore/aft so they match the same group histogram.

    group_key is an integer key per SN encoding (survey_id, z_bin), or any other
    grouping you want to enforce symmetry over.
    """
    idx_fore = np.asarray(idx_fore, dtype=np.int64)
    idx_aft = np.asarray(idx_aft, dtype=np.int64)
    group_key = np.asarray(group_key, dtype=np.int64)

    keep_f: list[np.ndarray] = []
    keep_a: list[np.ndarray] = []
    bins: list[dict[str, Any]] = []
    for g in np.unique(group_key[np.concatenate([idx_fore, idx_aft])]).tolist():
        g = int(g)
        mf = idx_fore[group_key[idx_fore] == g]
        ma = idx_aft[group_key[idx_aft] == g]
        nf = int(mf.size)
        na = int(ma.size)
        nk = min(nf, na)
        if nk <= 0:
            bins.append({"group": g, "n_fore": nf, "n_aft": na, "keep": 0})
            continue
        keep_f.append(rng.choice(mf, size=nk, replace=False).astype(np.int64))
        keep_a.append(rng.choice(ma, size=nk, replace=False).astype(np.int64))
        bins.append({"group": g, "n_fore": nf, "n_aft": na, "keep": nk})

    if keep_f:
        out_f = np.sort(np.concatenate(keep_f))
        out_a = np.sort(np.concatenate(keep_a))
    else:
        out_f = np.zeros((0,), dtype=np.int64)
        out_a = np.zeros((0,), dtype=np.int64)
    meta = {"mode": "group_downsample", "bins": bins}
    return out_f, out_a, meta


def _survey_z_group_key(*, idsurvey: np.ndarray, z: np.ndarray, z_edges: np.ndarray) -> np.ndarray:
    """Integer key encoding (survey_id, z_bin) for survey-aware matching."""
    idsurvey = np.asarray(idsurvey, dtype=np.int64)
    z = np.asarray(z, dtype=float)
    z_edges = np.asarray(z_edges, dtype=float)
    z_bin = np.digitize(z, z_edges, right=False) - 1
    z_bin = np.clip(z_bin, 0, z_edges.size - 2).astype(np.int64)
    return (idsurvey.astype(np.int64) << 32) ^ z_bin.astype(np.int64)


def _axisresult_from_dict(d: dict[str, Any]) -> AxisResult:
    """Parse an AxisResult dict (as written by this script) into the dataclass."""
    return AxisResult(
        axis_pix=int(d["axis_pix"]),
        axis_l_deg=float(d["axis_l_deg"]),
        axis_b_deg=float(d["axis_b_deg"]),
        n_fore=int(d["n_fore"]),
        n_aft=int(d["n_aft"]),
        s_fore_mean=float(d["s_fore_mean"]),
        s_fore_std=float(d["s_fore_std"]),
        s_aft_mean=float(d["s_aft_mean"]),
        s_aft_std=float(d["s_aft_std"]),
        delta_s=float(d["delta_s"]),
        sigma_delta_s=float(d["sigma_delta_s"]),
        z_score=float(d["z_score"]),
    )


def _seed_for_match(*, base_seed: int, fold: int, axis_pix: int, stage: str, mode: str, min_req: int) -> int:
    """Derive a deterministic RNG seed for (fold, axis, stage, match-mode, threshold).

    We avoid depending on global RNG state so runs can be resumed without changing selections.
    """
    stage_id = 1 if stage == "train" else 2
    mode_id = {"survey_z": 1, "z_bin": 2, "none": 3}.get(mode, 9)
    return int(base_seed) + 10_000_000 * int(stage_id) + 100_000 * int(fold) + 1000 * int(axis_pix) + 10 * int(mode_id) + int(min_req)


def _growth_gamma_prior_from_args(args: argparse.Namespace) -> tuple[float, float] | None:
    """Return growth-gamma prior tuple when gamma is sampled, else None."""
    if not bool(getattr(args, "growth_gamma_free", False)):
        return None
    lo, hi = [float(x) for x in getattr(args, "growth_gamma_prior")]
    return (lo, hi)


def _next_pow2(n: int) -> int:
    n = int(max(1, n))
    p = 1
    while p < n:
        p <<= 1
    return p


def _auto_nside_for_jobs(*, jobs: int, skip_antipodes: bool) -> int:
    """Choose a power-of-two nside with at least `jobs` axis tasks."""
    jobs = int(max(1, jobs))
    # Number of unique axes on a hemisphere scan:
    # - full grid: 12*nside^2
    # - with antipode skipping: about half => 6*nside^2
    denom = 6 if bool(skip_antipodes) else 12
    need = int(math.ceil(math.sqrt(float(jobs) / float(denom))))
    return _next_pow2(max(1, need))


def _apply_production_preset(args: argparse.Namespace) -> None:
    """Enforce production defaults for robust directional scans."""
    if str(getattr(args, "preset", "none")) != "production":
        return
    args.match_z = "bin_downsample"
    args.match_mode = "survey_z"
    args.null_mode = "shuffle_within_survey"
    args.null_stat = "max_abs_z"
    args.skip_antipodes = True
    args.train_skip_antipodes = True
    args.resume_scan = True
    args.test_parallel_fore_aft = True
    args.store_axis_posteriors = False
    args.store_train_posteriors = False
    # Treat gamma as a sampled nuisance instead of fixed GR-calibrated value.
    args.growth_mode = "gamma"
    args.growth_gamma_free = True
    # Ensure enough axis tasks to keep axis-level workers busy.
    hemi_jobs = max(int(getattr(args, "axis_jobs", 0)), int(getattr(args, "null_axis_jobs", 0)))
    if hemi_jobs > 0:
        args.axes_nside = max(int(getattr(args, "axes_nside", 1)), _auto_nside_for_jobs(jobs=hemi_jobs, skip_antipodes=True))
    train_jobs = int(getattr(args, "train_axis_jobs", 0))
    if train_jobs > 0:
        args.train_axes_nside = max(int(getattr(args, "train_axes_nside", 1)), _auto_nside_for_jobs(jobs=train_jobs, skip_antipodes=True))


def _match_test_split(
    idx_fore: np.ndarray,
    idx_aft: np.ndarray,
    *,
    z: np.ndarray,
    z_edges: np.ndarray,
    group_key: np.ndarray,
    base_seed: int,
    fold: int,
    axis_pix: int,
    match_z: str,
    match_mode: str,
    min_req: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Match fore/aft with fallbacks if the strict match yields too few points.

    Returns matched (idx_fore, idx_aft, meta). If all fallbacks fail, idx arrays may be empty and
    meta will include an 'attempts' record for debugging.
    """
    idx_fore = np.asarray(idx_fore, dtype=np.int64)
    idx_aft = np.asarray(idx_aft, dtype=np.int64)
    attempts: list[dict[str, Any]] = []

    if str(match_z) != "bin_downsample":
        meta = {"strategy": "none", "min_req_used": int(min_req), "attempts": [{"strategy": "none", "min_req": int(min_req), "n_fore": int(idx_fore.size), "n_aft": int(idx_aft.size)}]}
        return idx_fore, idx_aft, meta

    # Strategies from strict -> relaxed.
    if str(match_mode) == "survey_z":
        strategies = ["survey_z", "z_bin", "none"]
    else:
        strategies = ["z_bin", "none"]

    # Try to honor min_req, but allow controlled relaxation (still recorded).
    cand = [int(min_req), 60, 50, 40, 30]
    min_reqs: list[int] = []
    for x in cand:
        if x <= int(min_req) and x > 0 and x not in min_reqs:
            min_reqs.append(x)
    if not min_reqs:
        min_reqs = [max(int(min_req), 1)]

    for mr in min_reqs:
        for strat in strategies:
            if strat == "survey_z":
                rng = np.random.default_rng(_seed_for_match(base_seed=base_seed, fold=fold, axis_pix=axis_pix, stage="test", mode="survey_z", min_req=mr))
                f, a, meta = _match_fore_aft_by_group(idx_fore, idx_aft, group_key=group_key, rng=rng)
            elif strat == "z_bin":
                rng = np.random.default_rng(_seed_for_match(base_seed=base_seed, fold=fold, axis_pix=axis_pix, stage="test", mode="z_bin", min_req=mr))
                f, a, meta = _match_z_binwise(idx_fore, idx_aft, z, z_edges=z_edges, rng=rng)
            else:
                f, a, meta = idx_fore, idx_aft, {"mode": "none"}
            attempts.append({"strategy": strat, "min_req": int(mr), "n_fore": int(f.size), "n_aft": int(a.size)})
            if int(f.size) >= int(mr) and int(a.size) >= int(mr):
                return f, a, {"strategy": strat, "min_req_used": int(mr), "match_meta": meta, "attempts": attempts}

    return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), {"failed": True, "attempts": attempts}


def _bh_x_domain(*, z_max: float, mu_knots: int, mu_grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Choose x=log(A/A0) domain from a BH baseline guess (same logic as run_realdata_recon.py)."""
    z_max = float(z_max)
    if not np.isfinite(z_max) or z_max <= 0:
        raise ValueError("z_max must be finite and positive.")
    H0_guess = 70.0
    omega_m0_guess = 0.3
    H_zmax_guess = H0_guess * math.sqrt(omega_m0_guess * (1.0 + z_max) ** 3 + (1.0 - omega_m0_guess))
    x_min_guess = float(2.0 * math.log(H0_guess / H_zmax_guess))  # negative
    x_min = float(2.0 * x_min_guess)  # extra margin for safety
    x_knots = np.linspace(1.25 * x_min, 0.0, int(mu_knots))
    x_grid = np.linspace(x_min, 0.0, int(mu_grid))
    return x_knots, x_grid


def _run_subset_recon(
    *,
    out_dir: Path,
    sn_z: np.ndarray,
    sn_m: np.ndarray,
    sn_cov: np.ndarray,
    z_min: float,
    z_max: float,
    sn_like_bin_width: float,
    sn_like_min_per_bin: int,
    mu_knots: int,
    mu_grid: int,
    n_grid: int,
    mu_sampler: str,
    pt_ntemps: int,
    pt_tmax: float | None,
    mu_walkers: int,
    mu_steps: int,
    mu_burn: int,
    mu_draws: int,
    mu_procs: int,
    seed: int,
    omega_m0_prior: tuple[float, float],
    H0_prior: tuple[float, float],
    r_d_fixed: float,
    growth_mode: str = "ode",
    growth_gamma: float = 0.55,
    growth_gamma_prior: tuple[float, float] | None = None,
    sigma_sn_jit_scale: float,
    sigma_d2_scale: float,
    logmu_knot_scale: float | None,
    m_weight_mode: str,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run a single masked SN-only reconstruction and write outputs into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".done"
    if marker.exists() and (out_dir / "samples" / "mu_forward_posterior.npz").exists() and (out_dir / "summary.json").exists():
        return json.loads((out_dir / "summary.json").read_text(encoding="ascii"))

    sn_like = SNLogLike.from_arrays(z=sn_z, m=sn_m, cov=sn_cov)

    # Robust SN binning: when hemispheres are aggressively matched (especially survey x z),
    # some subsets can become "narrow" in z and/or sparse in certain bins. The linear compression
    # is only used as a speed-up, so we adapt binning parameters to avoid hard failures.
    used_binning = None
    last_err: Exception | None = None
    bw0 = float(sn_like_bin_width)
    mpb0 = int(sn_like_min_per_bin)
    bw_grid = [bw0, bw0 / 2.0, bw0 / 4.0]
    mpb_grid = [mpb0, max(mpb0 // 2, 3), 3]
    for bw in bw_grid:
        if not np.isfinite(bw) or bw <= 0:
            continue
        z_edges = np.arange(float(z_min), float(z_max) + float(bw), float(bw))
        if z_edges.size < 3:
            continue
        for mpb in mpb_grid:
            try:
                sn_like_bin = bin_sn_loglike(sn_like, z_edges=z_edges, min_per_bin=int(mpb))
                used_binning = {"bin_width": float(bw), "min_per_bin": int(mpb), "n_bins": int(sn_like_bin.z.size)}
                last_err = None
                break
            except ValueError as e:
                last_err = e
                msg = str(e)
                # Only downgrade binning if the failure is about bin occupancy.
                if "Too few populated SN bins" not in msg:
                    raise
                continue
        if used_binning is not None:
            break
    if used_binning is None:
        raise ValueError(f"SN binning failed after fallbacks: {last_err}")

    z_grid = np.linspace(0.0, float(z_max), int(n_grid))
    x_knots, x_grid = _bh_x_domain(z_max=float(z_max), mu_knots=int(mu_knots), mu_grid=int(mu_grid))

    # Massive axis-parallel scans can produce huge, interleaved stdout from inference;
    # optionally silence per-axis inference logs (run-level logs still exist).
    def _do_infer():
        return infer_logmu_forward(
            z_grid=z_grid,
            x_knots=x_knots,
            x_grid=x_grid,
            sn_z=sn_like_bin.z,
            sn_m=sn_like_bin.m,
            sn_cov=sn_like_bin.cov,
            sn_marg="M",
            cc_z=np.zeros((0,), dtype=float),
            cc_H=np.zeros((0,), dtype=float),
            cc_sigma_H=np.zeros((0,), dtype=float),
            bao_likes=[],
            fsbao_likes=[],
            rsd_like=None,
            lensing_like=None,
            pk_like=None,
            constants=PhysicalConstants(),
            sampler_kind=str(mu_sampler),
            pt_ntemps=int(pt_ntemps),
            pt_tmax=float(pt_tmax) if pt_tmax is not None else None,
            n_walkers=int(mu_walkers),
            n_steps=int(mu_steps),
            n_burn=int(mu_burn),
            seed=int(seed),
            n_processes=int(mu_procs),
            n_draws=int(mu_draws),
            omega_m0_prior=(float(omega_m0_prior[0]), float(omega_m0_prior[1])),
            H0_prior=(float(H0_prior[0]), float(H0_prior[1])),
            r_d_fixed=float(r_d_fixed),
            growth_mode=str(growth_mode),
            growth_gamma=float(growth_gamma),
            growth_gamma_prior=(
                (float(growth_gamma_prior[0]), float(growth_gamma_prior[1]))
                if growth_gamma_prior is not None
                else None
            ),
            sigma_sn_jit_scale=float(sigma_sn_jit_scale),
            sigma_d2_scale=float(sigma_d2_scale),
            logmu_knot_scale=float(logmu_knot_scale) if logmu_knot_scale is not None else 1.0,
            progress=True,
        )

    if quiet:
        with open(os.devnull, "w", encoding="ascii") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
            mu_post = _do_infer()
    else:
        mu_post = _do_infer()

    dep = compute_departure_stats(
        logA_grid=mu_post.x_grid,  # shift-invariant for (m,s) in M0
        logmu_samples=mu_post.logmu_x_samples,
        weight_mode=str(m_weight_mode),
    )

    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "samples" / "mu_forward_posterior.npz",
        x_grid=mu_post.x_grid,
        logmu_x_samples=mu_post.logmu_x_samples,
        z_grid=mu_post.z_grid,
        H_samples=mu_post.H_samples,
        H0=mu_post.params["H0"],
        omega_m0=mu_post.params["omega_m0"],
        omega_k0=mu_post.params.get("omega_k0"),
        r_d_Mpc=mu_post.params["r_d_Mpc"],
        growth_gamma=mu_post.params.get("growth_gamma"),
        sigma_cc_jit=mu_post.params["sigma_cc_jit"],
        sigma_sn_jit=mu_post.params["sigma_sn_jit"],
        sigma_d2=mu_post.params["sigma_d2"],
    )
    _write_json(out_dir / "samples" / "mu_forward_meta.json", {"meta": mu_post.meta})

    # Store the raw scar draws (m_draw, s_draw) for downstream comparisons (delta_s posteriors).
    # This is cheap relative to inference and avoids needing to rerun MCMC for simple delta tests.
    logA_grid = np.asarray(mu_post.x_grid, dtype=float)
    draws = np.asarray(mu_post.logmu_x_samples, dtype=float)
    var = np.var(draws, axis=0, ddof=1)
    if str(m_weight_mode) == "variance":
        w = 1.0 / np.clip(var, 1e-12, np.inf)
    else:
        w = np.ones_like(var)
    w = w / np.trapezoid(w, x=logA_grid)
    x0 = float(np.average(logA_grid, weights=w))
    x = logA_grid - x0
    m_draw = np.trapezoid(draws * w[None, :], x=logA_grid, axis=1)
    # Weighted slope per draw.
    X = np.column_stack([np.ones_like(x), x])
    XtW = (X.T * w)
    beta0 = np.linalg.solve(XtW @ X, XtW)  # (2,n)
    s_draw = beta0[1] @ draws.T  # (n_draws,)
    np.savez_compressed(out_dir / "samples" / "scar_draws.npz", m=m_draw, s=s_draw, w=w, logA_grid=logA_grid)

    summary = {
        "n_sn_in": int(sn_z.size),
        "n_sn_binned": int(sn_like_bin.z.size),
        "z_min": float(z_min),
        "z_max": float(z_max),
        "sn_binning": used_binning,
        "mu_post_meta": mu_post.meta,
        "departure": dep,
    }
    _write_json(out_dir / "summary.json", summary)
    marker.write_text("ok\n", encoding="ascii")
    return summary


def _infer_subset_s_stats(
    *,
    sn_z: np.ndarray,
    sn_m: np.ndarray,
    sn_cov: np.ndarray,
    z_min: float,
    z_max: float,
    sn_like_bin_width: float,
    sn_like_min_per_bin: int,
    mu_knots: int,
    mu_grid: int,
    n_grid: int,
    mu_sampler: str,
    pt_ntemps: int,
    pt_tmax: float | None,
    mu_walkers: int,
    mu_steps: int,
    mu_burn: int,
    mu_draws: int,
    mu_procs: int,
    seed: int,
    omega_m0_prior: tuple[float, float],
    H0_prior: tuple[float, float],
    r_d_fixed: float,
    growth_mode: str = "ode",
    growth_gamma: float = 0.55,
    growth_gamma_prior: tuple[float, float] | None = None,
    sigma_sn_jit_scale: float,
    sigma_d2_scale: float,
    logmu_knot_scale: float | None,
    m_weight_mode: str,
    quiet: bool = False,
) -> tuple[float, float]:
    """Ephemeral SN-only inference returning only (s_mean, s_std).

    This avoids writing per-axis sample files (critical for large null batteries).
    """
    sn_like = SNLogLike.from_arrays(z=sn_z, m=sn_m, cov=sn_cov)

    # Robust SN binning (same fallback logic as _run_subset_recon).
    used_binning = None
    last_err: Exception | None = None
    bw0 = float(sn_like_bin_width)
    mpb0 = int(sn_like_min_per_bin)
    bw_grid = [bw0, bw0 / 2.0, bw0 / 4.0]
    mpb_grid = [mpb0, max(mpb0 // 2, 3), 3]
    for bw in bw_grid:
        if not np.isfinite(bw) or bw <= 0:
            continue
        z_edges = np.arange(float(z_min), float(z_max) + float(bw), float(bw))
        if z_edges.size < 3:
            continue
        for mpb in mpb_grid:
            try:
                sn_like_bin = bin_sn_loglike(sn_like, z_edges=z_edges, min_per_bin=int(mpb))
                used_binning = {"bin_width": float(bw), "min_per_bin": int(mpb), "n_bins": int(sn_like_bin.z.size)}
                last_err = None
                break
            except ValueError as e:
                last_err = e
                msg = str(e)
                if "Too few populated SN bins" not in msg:
                    raise
                continue
        if used_binning is not None:
            break
    if used_binning is None:
        raise ValueError(f"SN binning failed after fallbacks: {last_err}")

    z_grid = np.linspace(0.0, float(z_max), int(n_grid))
    x_knots, x_grid = _bh_x_domain(z_max=float(z_max), mu_knots=int(mu_knots), mu_grid=int(mu_grid))

    def _do_infer():
        return infer_logmu_forward(
            z_grid=z_grid,
            x_knots=x_knots,
            x_grid=x_grid,
            sn_z=sn_like_bin.z,
            sn_m=sn_like_bin.m,
            sn_cov=sn_like_bin.cov,
            sn_marg="M",
            cc_z=np.zeros((0,), dtype=float),
            cc_H=np.zeros((0,), dtype=float),
            cc_sigma_H=np.zeros((0,), dtype=float),
            bao_likes=[],
            fsbao_likes=[],
            rsd_like=None,
            lensing_like=None,
            pk_like=None,
            constants=PhysicalConstants(),
            sampler_kind=str(mu_sampler),
            pt_ntemps=int(pt_ntemps),
            pt_tmax=float(pt_tmax) if pt_tmax is not None else None,
            n_walkers=int(mu_walkers),
            n_steps=int(mu_steps),
            n_burn=int(mu_burn),
            seed=int(seed),
            n_processes=int(mu_procs),
            n_draws=int(mu_draws),
            omega_m0_prior=(float(omega_m0_prior[0]), float(omega_m0_prior[1])),
            H0_prior=(float(H0_prior[0]), float(H0_prior[1])),
            r_d_fixed=float(r_d_fixed),
            growth_mode=str(growth_mode),
            growth_gamma=float(growth_gamma),
            growth_gamma_prior=(
                (float(growth_gamma_prior[0]), float(growth_gamma_prior[1]))
                if growth_gamma_prior is not None
                else None
            ),
            sigma_sn_jit_scale=float(sigma_sn_jit_scale),
            sigma_d2_scale=float(sigma_d2_scale),
            logmu_knot_scale=float(logmu_knot_scale) if logmu_knot_scale is not None else 1.0,
            progress=False,
        )

    if quiet:
        with open(os.devnull, "w", encoding="ascii") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
            mu_post = _do_infer()
    else:
        mu_post = _do_infer()

    dep = compute_departure_stats(
        logA_grid=mu_post.x_grid,  # shift-invariant for (m,s) in M0
        logmu_samples=mu_post.logmu_x_samples,
        weight_mode=str(m_weight_mode),
    )
    return float(dep["slope"]["mean"]), float(dep["slope"]["std"])


@dataclass(frozen=True)
class AxisResult:
    axis_pix: int
    axis_l_deg: float
    axis_b_deg: float
    n_fore: int
    n_aft: int
    s_fore_mean: float
    s_fore_std: float
    s_aft_mean: float
    s_aft_std: float
    delta_s: float
    sigma_delta_s: float
    z_score: float


@dataclass(frozen=True)
class _TrainAxisTask:
    fold: int
    axis_pix: int
    axis_l_deg: float
    axis_b_deg: float
    axis_vec: tuple[float, float, float]


_TRAIN_AXIS_GLOBAL: dict[str, Any] | None = None


@dataclass(frozen=True)
class _NullAxisTask:
    rep: int
    axis_ord: int
    axis_pix: int
    axis_l_deg: float
    axis_b_deg: float
    axis_vec: tuple[float, float, float]


_NULL_AXIS_GLOBAL: dict[str, Any] | None = None


def _null_axis_worker(task: _NullAxisTask) -> tuple[int, int, float, float, float] | None:
    """Compute one null axis z-score (no disk writes).

    Designed for mp.get_context("fork") so arrays are inherited by copy-on-write.
    Returns (rep, axis_ord, delta_s, sigma_delta_s, z_score) or None (axis skipped).
    """
    g = _NULL_AXIS_GLOBAL
    if g is None:
        raise RuntimeError("null axis worker missing global context")

    rep = int(task.rep)
    axis_ord = int(task.axis_ord)
    p = int(task.axis_pix)
    vec = np.asarray(task.axis_vec, dtype=float)

    if "v_null_stack" in g:
        rep_to_stack: np.ndarray = g["rep_to_stack"]
        if rep < 0 or rep >= int(rep_to_stack.size):
            raise RuntimeError(f"null axis worker rep index out of bounds: rep={rep}")
        stack_idx = int(rep_to_stack[rep])
        if stack_idx < 0:
            return None
        v_null = np.asarray(g["v_null_stack"][stack_idx], dtype=float)
    else:
        v_null = np.asarray(g["v_null"], dtype=float)
    z: np.ndarray = g["z"]
    m: np.ndarray = g["m"]
    cov: np.ndarray = g["cov"]
    z_edges: np.ndarray | None = g["z_edges"]
    group_key: np.ndarray | None = g["group_key"]
    args = g["args"]
    base_seed = int(g["base_seed"])

    proj = v_null @ vec
    idx_fore = np.where(proj > 0.0)[0].astype(np.int64)
    idx_aft = np.where(proj < 0.0)[0].astype(np.int64)

    if str(args.match_z) == "bin_downsample":
        mr = int(args.min_sn_per_side)
        if str(args.match_mode) == "survey_z":
            if group_key is None:
                raise RuntimeError("group_key missing for survey_z matching")
            rng_match = np.random.default_rng(
                _seed_for_match(base_seed=base_seed, fold=rep, axis_pix=p, stage="train", mode="survey_z", min_req=mr)
            )
            idx_fore, idx_aft, _ = _match_fore_aft_by_group(idx_fore, idx_aft, group_key=group_key, rng=rng_match)
        else:
            if z_edges is None:
                raise RuntimeError("z_edges missing for z_only matching")
            rng_match = np.random.default_rng(_seed_for_match(base_seed=base_seed, fold=rep, axis_pix=p, stage="train", mode="z_bin", min_req=mr))
            idx_fore, idx_aft, _ = _match_z_binwise(idx_fore, idx_aft, z, z_edges=z_edges, rng=rng_match)

    if int(idx_fore.size) < int(args.min_sn_per_side) or int(idx_aft.size) < int(args.min_sn_per_side):
        return None

    s_f, ss_f = _infer_subset_s_stats(
        sn_z=z[idx_fore],
        sn_m=m[idx_fore],
        sn_cov=cov[np.ix_(idx_fore, idx_fore)],
        z_min=float(g["z_min"]),
        z_max=float(g["z_max"]),
        sn_like_bin_width=float(args.sn_like_bin_width),
        sn_like_min_per_bin=int(args.sn_like_min_per_bin),
        mu_knots=int(args.mu_knots),
        mu_grid=int(args.mu_grid),
        n_grid=int(args.n_grid),
        mu_sampler=str(args.mu_sampler),
        pt_ntemps=int(args.pt_ntemps),
        pt_tmax=float(args.pt_tmax) if args.pt_tmax is not None else None,
        mu_walkers=int(args.mu_walkers),
        mu_steps=int(args.mu_steps),
        mu_burn=int(args.mu_burn),
        mu_draws=int(args.mu_draws),
        mu_procs=int(g["mu_procs"]),
        seed=base_seed + 1_000_000 + 10_000 * rep + int(p) + 111,
        omega_m0_prior=(float(args.omega_m0_prior[0]), float(args.omega_m0_prior[1])),
        H0_prior=(float(args.H0_prior[0]), float(args.H0_prior[1])),
        r_d_fixed=float(args.r_d_fixed),
        growth_mode=str(args.growth_mode),
        growth_gamma=float(args.growth_gamma),
        growth_gamma_prior=_growth_gamma_prior_from_args(args),
        sigma_sn_jit_scale=float(args.sigma_sn_jit_scale),
        sigma_d2_scale=float(args.sigma_d2_scale),
        logmu_knot_scale=float(args.logmu_knot_scale) if args.logmu_knot_scale is not None else None,
        m_weight_mode=str(args.m_weight_mode),
        quiet=bool(g["quiet"]),
    )
    s_a, ss_a = _infer_subset_s_stats(
        sn_z=z[idx_aft],
        sn_m=m[idx_aft],
        sn_cov=cov[np.ix_(idx_aft, idx_aft)],
        z_min=float(g["z_min"]),
        z_max=float(g["z_max"]),
        sn_like_bin_width=float(args.sn_like_bin_width),
        sn_like_min_per_bin=int(args.sn_like_min_per_bin),
        mu_knots=int(args.mu_knots),
        mu_grid=int(args.mu_grid),
        n_grid=int(args.n_grid),
        mu_sampler=str(args.mu_sampler),
        pt_ntemps=int(args.pt_ntemps),
        pt_tmax=float(args.pt_tmax) if args.pt_tmax is not None else None,
        mu_walkers=int(args.mu_walkers),
        mu_steps=int(args.mu_steps),
        mu_burn=int(args.mu_burn),
        mu_draws=int(args.mu_draws),
        mu_procs=int(g["mu_procs"]),
        seed=base_seed + 2_000_000 + 10_000 * rep + int(p) + 222,
        omega_m0_prior=(float(args.omega_m0_prior[0]), float(args.omega_m0_prior[1])),
        H0_prior=(float(args.H0_prior[0]), float(args.H0_prior[1])),
        r_d_fixed=float(args.r_d_fixed),
        growth_mode=str(args.growth_mode),
        growth_gamma=float(args.growth_gamma),
        growth_gamma_prior=_growth_gamma_prior_from_args(args),
        sigma_sn_jit_scale=float(args.sigma_sn_jit_scale),
        sigma_d2_scale=float(args.sigma_d2_scale),
        logmu_knot_scale=float(args.logmu_knot_scale) if args.logmu_knot_scale is not None else None,
        m_weight_mode=str(args.m_weight_mode),
        quiet=bool(g["quiet"]),
    )

    delta_s = float(s_f - s_a)
    sig = float(math.sqrt(max(ss_f**2 + ss_a**2, 1e-12)))
    z_sc = float(delta_s / sig) if sig > 0 else float("nan")
    return int(rep), int(axis_ord), float(delta_s), float(sig), float(z_sc)


@dataclass(frozen=True)
class _HemiAxisTask:
    axis_pix: int
    axis_l_deg: float
    axis_b_deg: float
    axis_vec: tuple[float, float, float]


_HEMI_AXIS_GLOBAL: dict[str, Any] | None = None


def _hemi_axis_worker(task: _HemiAxisTask) -> dict[str, Any] | None:
    """Compute one hemisphere-scan axis result (fore/aft), writing outputs to disk."""
    g = _HEMI_AXIS_GLOBAL
    if g is None:
        raise RuntimeError("hemi axis worker missing global context")

    out_dir: Path = g["out_dir"]
    z: np.ndarray = g["z"]
    m: np.ndarray = g["m"]
    cov: np.ndarray = g["cov"]
    v: np.ndarray = g["v"]
    z_edges: np.ndarray | None = g["z_edges"]
    group_key: np.ndarray | None = g["group_key"]
    args = g["args"]
    quiet = bool(g["quiet"])

    p = int(task.axis_pix)
    vec = np.asarray(task.axis_vec, dtype=float)
    l_deg = float(task.axis_l_deg)
    b_deg = float(task.axis_b_deg)

    axis_dir = out_dir / f"axis_nside{int(args.axes_nside)}_{'nest' if bool(args.axes_nest) else 'ring'}_pix{int(p):05d}"
    cached_res = axis_dir / "result.json"
    if cached_res.exists():
        try:
            return json.loads(cached_res.read_text(encoding="utf-8"))
        except Exception:
            pass

    proj = v @ vec
    idx_fore = np.where(proj > 0.0)[0].astype(np.int64)
    idx_aft = np.where(proj < 0.0)[0].astype(np.int64)
    zmatch_meta: dict[str, Any] | None = None

    if str(args.match_z) == "bin_downsample":
        mr = int(args.min_sn_per_side)
        if str(args.match_mode) == "survey_z":
            if group_key is None:
                raise RuntimeError("group_key missing for survey_z matching")
            rng_match = np.random.default_rng(_seed_for_match(base_seed=int(args.seed), fold=0, axis_pix=p, stage="train", mode="survey_z", min_req=mr))
            idx_fore, idx_aft, zmatch_meta = _match_fore_aft_by_group(idx_fore, idx_aft, group_key=group_key, rng=rng_match)
        else:
            if z_edges is None:
                raise RuntimeError("z_edges missing for z_only matching")
            rng_match = np.random.default_rng(_seed_for_match(base_seed=int(args.seed), fold=0, axis_pix=p, stage="train", mode="z_bin", min_req=mr))
            idx_fore, idx_aft, zmatch_meta = _match_z_binwise(idx_fore, idx_aft, z, z_edges=z_edges, rng=rng_match)

    if int(idx_fore.size) < int(args.min_sn_per_side) or int(idx_aft.size) < int(args.min_sn_per_side):
        return None

    axis_dir.mkdir(parents=True, exist_ok=True)
    _write_json(axis_dir / "axis.json", {"axis_pix": int(p), "l_deg": float(l_deg), "b_deg": float(b_deg), "match": zmatch_meta})

    if bool(getattr(args, "store_axis_posteriors", False)):
        fore = _run_subset_recon(
            out_dir=axis_dir / "fore",
            sn_z=z[idx_fore],
            sn_m=m[idx_fore],
            sn_cov=cov[np.ix_(idx_fore, idx_fore)],
            z_min=float(g["z_min"]),
            z_max=float(g["z_max"]),
            sn_like_bin_width=float(args.sn_like_bin_width),
            sn_like_min_per_bin=int(args.sn_like_min_per_bin),
            mu_knots=int(args.mu_knots),
            mu_grid=int(args.mu_grid),
            n_grid=int(args.n_grid),
            mu_sampler=str(args.mu_sampler),
            pt_ntemps=int(args.pt_ntemps),
            pt_tmax=float(args.pt_tmax) if args.pt_tmax is not None else None,
            mu_walkers=int(args.mu_walkers),
            mu_steps=int(args.mu_steps),
            mu_burn=int(args.mu_burn),
            mu_draws=int(args.mu_draws),
            mu_procs=int(g["mu_procs"]),
            seed=int(args.seed) + 1000 + int(p),
            omega_m0_prior=(float(args.omega_m0_prior[0]), float(args.omega_m0_prior[1])),
            H0_prior=(float(args.H0_prior[0]), float(args.H0_prior[1])),
            r_d_fixed=float(args.r_d_fixed),
            growth_mode=str(args.growth_mode),
            growth_gamma=float(args.growth_gamma),
            growth_gamma_prior=_growth_gamma_prior_from_args(args),
            sigma_sn_jit_scale=float(args.sigma_sn_jit_scale),
            sigma_d2_scale=float(args.sigma_d2_scale),
            logmu_knot_scale=float(args.logmu_knot_scale) if args.logmu_knot_scale is not None else None,
            m_weight_mode=str(args.m_weight_mode),
            quiet=quiet,
        )
        aft = _run_subset_recon(
            out_dir=axis_dir / "aft",
            sn_z=z[idx_aft],
            sn_m=m[idx_aft],
            sn_cov=cov[np.ix_(idx_aft, idx_aft)],
            z_min=float(g["z_min"]),
            z_max=float(g["z_max"]),
            sn_like_bin_width=float(args.sn_like_bin_width),
            sn_like_min_per_bin=int(args.sn_like_min_per_bin),
            mu_knots=int(args.mu_knots),
            mu_grid=int(args.mu_grid),
            n_grid=int(args.n_grid),
            mu_sampler=str(args.mu_sampler),
            pt_ntemps=int(args.pt_ntemps),
            pt_tmax=float(args.pt_tmax) if args.pt_tmax is not None else None,
            mu_walkers=int(args.mu_walkers),
            mu_steps=int(args.mu_steps),
            mu_burn=int(args.mu_burn),
            mu_draws=int(args.mu_draws),
            mu_procs=int(g["mu_procs"]),
            seed=int(args.seed) + 2000 + int(p),
            omega_m0_prior=(float(args.omega_m0_prior[0]), float(args.omega_m0_prior[1])),
            H0_prior=(float(args.H0_prior[0]), float(args.H0_prior[1])),
            r_d_fixed=float(args.r_d_fixed),
            growth_mode=str(args.growth_mode),
            growth_gamma=float(args.growth_gamma),
            growth_gamma_prior=_growth_gamma_prior_from_args(args),
            sigma_sn_jit_scale=float(args.sigma_sn_jit_scale),
            sigma_d2_scale=float(args.sigma_d2_scale),
            logmu_knot_scale=float(args.logmu_knot_scale) if args.logmu_knot_scale is not None else None,
            m_weight_mode=str(args.m_weight_mode),
            quiet=quiet,
        )

        s_f = float(fore["departure"]["slope"]["mean"])
        ss_f = float(fore["departure"]["slope"]["std"])
        s_a0 = float(aft["departure"]["slope"]["mean"])
        ss_a = float(aft["departure"]["slope"]["std"])
    else:
        s_f, ss_f = _infer_subset_s_stats(
            sn_z=z[idx_fore],
            sn_m=m[idx_fore],
            sn_cov=cov[np.ix_(idx_fore, idx_fore)],
            z_min=float(g["z_min"]),
            z_max=float(g["z_max"]),
            sn_like_bin_width=float(args.sn_like_bin_width),
            sn_like_min_per_bin=int(args.sn_like_min_per_bin),
            mu_knots=int(args.mu_knots),
            mu_grid=int(args.mu_grid),
            n_grid=int(args.n_grid),
            mu_sampler=str(args.mu_sampler),
            pt_ntemps=int(args.pt_ntemps),
            pt_tmax=float(args.pt_tmax) if args.pt_tmax is not None else None,
            mu_walkers=int(args.mu_walkers),
            mu_steps=int(args.mu_steps),
            mu_burn=int(args.mu_burn),
            mu_draws=int(args.mu_draws),
            mu_procs=int(g["mu_procs"]),
            seed=int(args.seed) + 1000 + int(p),
            omega_m0_prior=(float(args.omega_m0_prior[0]), float(args.omega_m0_prior[1])),
            H0_prior=(float(args.H0_prior[0]), float(args.H0_prior[1])),
            r_d_fixed=float(args.r_d_fixed),
            growth_mode=str(args.growth_mode),
            growth_gamma=float(args.growth_gamma),
            growth_gamma_prior=_growth_gamma_prior_from_args(args),
            sigma_sn_jit_scale=float(args.sigma_sn_jit_scale),
            sigma_d2_scale=float(args.sigma_d2_scale),
            logmu_knot_scale=float(args.logmu_knot_scale) if args.logmu_knot_scale is not None else None,
            m_weight_mode=str(args.m_weight_mode),
            quiet=quiet,
        )
        s_a0, ss_a = _infer_subset_s_stats(
            sn_z=z[idx_aft],
            sn_m=m[idx_aft],
            sn_cov=cov[np.ix_(idx_aft, idx_aft)],
            z_min=float(g["z_min"]),
            z_max=float(g["z_max"]),
            sn_like_bin_width=float(args.sn_like_bin_width),
            sn_like_min_per_bin=int(args.sn_like_min_per_bin),
            mu_knots=int(args.mu_knots),
            mu_grid=int(args.mu_grid),
            n_grid=int(args.n_grid),
            mu_sampler=str(args.mu_sampler),
            pt_ntemps=int(args.pt_ntemps),
            pt_tmax=float(args.pt_tmax) if args.pt_tmax is not None else None,
            mu_walkers=int(args.mu_walkers),
            mu_steps=int(args.mu_steps),
            mu_burn=int(args.mu_burn),
            mu_draws=int(args.mu_draws),
            mu_procs=int(g["mu_procs"]),
            seed=int(args.seed) + 2000 + int(p),
            omega_m0_prior=(float(args.omega_m0_prior[0]), float(args.omega_m0_prior[1])),
            H0_prior=(float(args.H0_prior[0]), float(args.H0_prior[1])),
            r_d_fixed=float(args.r_d_fixed),
            growth_mode=str(args.growth_mode),
            growth_gamma=float(args.growth_gamma),
            growth_gamma_prior=_growth_gamma_prior_from_args(args),
            sigma_sn_jit_scale=float(args.sigma_sn_jit_scale),
            sigma_d2_scale=float(args.sigma_d2_scale),
            logmu_knot_scale=float(args.logmu_knot_scale) if args.logmu_knot_scale is not None else None,
            m_weight_mode=str(args.m_weight_mode),
            quiet=quiet,
        )
    delta_s = float(s_f - s_a0)
    sigma_delta = float(math.sqrt(max(ss_f**2 + ss_a**2, 1e-30)))
    zscore = float(delta_s / sigma_delta)

    row = AxisResult(
        axis_pix=int(p),
        axis_l_deg=float(l_deg),
        axis_b_deg=float(b_deg),
        n_fore=int(idx_fore.size),
        n_aft=int(idx_aft.size),
        s_fore_mean=float(s_f),
        s_fore_std=float(ss_f),
        s_aft_mean=float(s_a0),
        s_aft_std=float(ss_a),
        delta_s=float(delta_s),
        sigma_delta_s=float(sigma_delta),
        z_score=float(zscore),
    )
    out = asdict(row)
    _write_json(axis_dir / "result.json", out)
    return out


def _train_axis_worker(task: _TrainAxisTask) -> dict[str, Any] | None:
    """Compute one TRAIN axis result (fore/aft) in crossfit mode.

    Designed to be used with mp.get_context("fork") so large arrays are inherited by copy-on-write.
    Returns a result dict (AxisResult asdict) or None (skipped axis).
    """
    g = _TRAIN_AXIS_GLOBAL
    if g is None:
        raise RuntimeError("train axis worker missing global context")

    train_scan_dir: Path = g["train_scan_dir"]
    z: np.ndarray = g["z"]
    m: np.ndarray = g["m"]
    cov: np.ndarray = g["cov"]
    v: np.ndarray = g["v"]
    group_key: np.ndarray = g["group_key"]
    train_idx: np.ndarray = g["train_idx"]
    z_edges: np.ndarray | None = g["z_edges"]
    args = g["args"]
    base_seed = int(g["base_seed"])

    p = int(task.axis_pix)
    vec = np.asarray(task.axis_vec, dtype=float)
    l_deg = float(task.axis_l_deg)
    b_deg = float(task.axis_b_deg)

    axis_dir = train_scan_dir / f"axis_nside{int(args.train_axes_nside)}_{'nest' if bool(args.train_axes_nest) else 'ring'}_pix{int(p):05d}"
    cached_res = axis_dir / "result.json"
    if cached_res.exists():
        try:
            return json.loads(cached_res.read_text(encoding="utf-8"))
        except Exception:
            pass

    proj = v[train_idx] @ vec
    idx_fore = train_idx[np.where(proj > 0.0)[0]].astype(np.int64)
    idx_aft = train_idx[np.where(proj < 0.0)[0]].astype(np.int64)
    zmatch_meta: dict[str, Any] | None = None

    if str(args.match_z) == "bin_downsample":
        mr = int(args.min_sn_per_side)
        if str(args.match_mode) == "survey_z":
            rng_match = np.random.default_rng(
                _seed_for_match(base_seed=base_seed, fold=int(task.fold), axis_pix=int(p), stage="train", mode="survey_z", min_req=mr)
            )
            idx_fore, idx_aft, zmatch_meta = _match_fore_aft_by_group(idx_fore, idx_aft, group_key=group_key, rng=rng_match)
        else:
            if z_edges is None:
                raise RuntimeError("z_edges missing for z_only matching")
            rng_match = np.random.default_rng(_seed_for_match(base_seed=base_seed, fold=int(task.fold), axis_pix=int(p), stage="train", mode="z_bin", min_req=mr))
            idx_fore, idx_aft, zmatch_meta = _match_z_binwise(idx_fore, idx_aft, z, z_edges=z_edges, rng=rng_match)

    if int(idx_fore.size) < int(args.min_sn_per_side) or int(idx_aft.size) < int(args.min_sn_per_side):
        return None

    axis_dir.mkdir(parents=True, exist_ok=True)
    _write_json(axis_dir / "axis.json", {"axis_pix": int(p), "l_deg": float(l_deg), "b_deg": float(b_deg), "match": zmatch_meta})

    if bool(getattr(args, "store_train_posteriors", False)):
        fore = _run_subset_recon(
            out_dir=axis_dir / "fore",
            sn_z=z[idx_fore],
            sn_m=m[idx_fore],
            sn_cov=cov[np.ix_(idx_fore, idx_fore)],
            z_min=float(g["z_min"]),
            z_max=float(g["z_max"]),
            sn_like_bin_width=float(args.sn_like_bin_width),
            sn_like_min_per_bin=int(args.sn_like_min_per_bin),
            mu_knots=int(args.train_mu_knots),
            mu_grid=int(args.mu_grid),
            n_grid=int(args.n_grid),
            mu_sampler=str(g["train_mu_sampler"]),
            pt_ntemps=int(g["train_pt_ntemps"]),
            pt_tmax=float(g["train_pt_tmax"]) if g["train_pt_tmax"] is not None else None,
            mu_walkers=int(g["train_mu_walkers"]),
            mu_steps=int(args.train_mu_steps),
            mu_burn=int(args.train_mu_burn),
            mu_draws=int(args.train_mu_draws),
            mu_procs=int(g["train_mu_procs"]),
            seed=base_seed + 10_000 * int(task.fold) + int(p) + 111,
            omega_m0_prior=(float(args.omega_m0_prior[0]), float(args.omega_m0_prior[1])),
            H0_prior=(float(args.H0_prior[0]), float(args.H0_prior[1])),
            r_d_fixed=float(args.r_d_fixed),
            growth_mode=str(args.growth_mode),
            growth_gamma=float(args.growth_gamma),
            growth_gamma_prior=_growth_gamma_prior_from_args(args),
            sigma_sn_jit_scale=float(args.sigma_sn_jit_scale),
            sigma_d2_scale=float(args.sigma_d2_scale),
            logmu_knot_scale=float(args.logmu_knot_scale) if args.logmu_knot_scale is not None else None,
            m_weight_mode=str(args.m_weight_mode),
            quiet=bool(g["train_quiet"]),
        )
        aft = _run_subset_recon(
            out_dir=axis_dir / "aft",
            sn_z=z[idx_aft],
            sn_m=m[idx_aft],
            sn_cov=cov[np.ix_(idx_aft, idx_aft)],
            z_min=float(g["z_min"]),
            z_max=float(g["z_max"]),
            sn_like_bin_width=float(args.sn_like_bin_width),
            sn_like_min_per_bin=int(args.sn_like_min_per_bin),
            mu_knots=int(args.train_mu_knots),
            mu_grid=int(args.mu_grid),
            n_grid=int(args.n_grid),
            mu_sampler=str(g["train_mu_sampler"]),
            pt_ntemps=int(g["train_pt_ntemps"]),
            pt_tmax=float(g["train_pt_tmax"]) if g["train_pt_tmax"] is not None else None,
            mu_walkers=int(g["train_mu_walkers"]),
            mu_steps=int(args.train_mu_steps),
            mu_burn=int(args.train_mu_burn),
            mu_draws=int(args.train_mu_draws),
            mu_procs=int(g["train_mu_procs"]),
            seed=base_seed + 10_000 * int(task.fold) + int(p) + 222,
            omega_m0_prior=(float(args.omega_m0_prior[0]), float(args.omega_m0_prior[1])),
            H0_prior=(float(args.H0_prior[0]), float(args.H0_prior[1])),
            r_d_fixed=float(args.r_d_fixed),
            growth_mode=str(args.growth_mode),
            growth_gamma=float(args.growth_gamma),
            growth_gamma_prior=_growth_gamma_prior_from_args(args),
            sigma_sn_jit_scale=float(args.sigma_sn_jit_scale),
            sigma_d2_scale=float(args.sigma_d2_scale),
            logmu_knot_scale=float(args.logmu_knot_scale) if args.logmu_knot_scale is not None else None,
            m_weight_mode=str(args.m_weight_mode),
            quiet=bool(g["train_quiet"]),
        )

        s_f_mean = float(fore["departure"]["slope"]["mean"])
        s_f_std = float(fore["departure"]["slope"]["std"])
        s_a_mean = float(aft["departure"]["slope"]["mean"])
        s_a_std = float(aft["departure"]["slope"]["std"])
    else:
        s_f_mean, s_f_std = _infer_subset_s_stats(
            sn_z=z[idx_fore],
            sn_m=m[idx_fore],
            sn_cov=cov[np.ix_(idx_fore, idx_fore)],
            z_min=float(g["z_min"]),
            z_max=float(g["z_max"]),
            sn_like_bin_width=float(args.sn_like_bin_width),
            sn_like_min_per_bin=int(args.sn_like_min_per_bin),
            mu_knots=int(args.train_mu_knots),
            mu_grid=int(args.mu_grid),
            n_grid=int(args.n_grid),
            mu_sampler=str(g["train_mu_sampler"]),
            pt_ntemps=int(g["train_pt_ntemps"]),
            pt_tmax=float(g["train_pt_tmax"]) if g["train_pt_tmax"] is not None else None,
            mu_walkers=int(g["train_mu_walkers"]),
            mu_steps=int(args.train_mu_steps),
            mu_burn=int(args.train_mu_burn),
            mu_draws=int(args.train_mu_draws),
            mu_procs=int(g["train_mu_procs"]),
            seed=base_seed + 10_000 * int(task.fold) + int(p) + 111,
            omega_m0_prior=(float(args.omega_m0_prior[0]), float(args.omega_m0_prior[1])),
            H0_prior=(float(args.H0_prior[0]), float(args.H0_prior[1])),
            r_d_fixed=float(args.r_d_fixed),
            growth_mode=str(args.growth_mode),
            growth_gamma=float(args.growth_gamma),
            growth_gamma_prior=_growth_gamma_prior_from_args(args),
            sigma_sn_jit_scale=float(args.sigma_sn_jit_scale),
            sigma_d2_scale=float(args.sigma_d2_scale),
            logmu_knot_scale=float(args.logmu_knot_scale) if args.logmu_knot_scale is not None else None,
            m_weight_mode=str(args.m_weight_mode),
            quiet=bool(g["train_quiet"]),
        )
        s_a_mean, s_a_std = _infer_subset_s_stats(
            sn_z=z[idx_aft],
            sn_m=m[idx_aft],
            sn_cov=cov[np.ix_(idx_aft, idx_aft)],
            z_min=float(g["z_min"]),
            z_max=float(g["z_max"]),
            sn_like_bin_width=float(args.sn_like_bin_width),
            sn_like_min_per_bin=int(args.sn_like_min_per_bin),
            mu_knots=int(args.train_mu_knots),
            mu_grid=int(args.mu_grid),
            n_grid=int(args.n_grid),
            mu_sampler=str(g["train_mu_sampler"]),
            pt_ntemps=int(g["train_pt_ntemps"]),
            pt_tmax=float(g["train_pt_tmax"]) if g["train_pt_tmax"] is not None else None,
            mu_walkers=int(g["train_mu_walkers"]),
            mu_steps=int(args.train_mu_steps),
            mu_burn=int(args.train_mu_burn),
            mu_draws=int(args.train_mu_draws),
            mu_procs=int(g["train_mu_procs"]),
            seed=base_seed + 10_000 * int(task.fold) + int(p) + 222,
            omega_m0_prior=(float(args.omega_m0_prior[0]), float(args.omega_m0_prior[1])),
            H0_prior=(float(args.H0_prior[0]), float(args.H0_prior[1])),
            r_d_fixed=float(args.r_d_fixed),
            growth_mode=str(args.growth_mode),
            growth_gamma=float(args.growth_gamma),
            growth_gamma_prior=_growth_gamma_prior_from_args(args),
            sigma_sn_jit_scale=float(args.sigma_sn_jit_scale),
            sigma_d2_scale=float(args.sigma_d2_scale),
            logmu_knot_scale=float(args.logmu_knot_scale) if args.logmu_knot_scale is not None else None,
            m_weight_mode=str(args.m_weight_mode),
            quiet=bool(g["train_quiet"]),
        )
    delta_s = s_f_mean - s_a_mean
    sig = math.sqrt(max(s_f_std**2 + s_a_std**2, 1e-12))
    z_sc = float(delta_s / sig)
    ar = AxisResult(
        axis_pix=int(p),
        axis_l_deg=float(l_deg),
        axis_b_deg=float(b_deg),
        n_fore=int(idx_fore.size),
        n_aft=int(idx_aft.size),
        s_fore_mean=s_f_mean,
        s_fore_std=s_f_std,
        s_aft_mean=s_a_mean,
        s_aft_std=s_a_std,
        delta_s=float(delta_s),
        sigma_delta_s=float(sig),
        z_score=z_sc,
    )
    out = asdict(ar)
    _write_json(axis_dir / "result.json", out)
    return out


def _fit_dipole_from_deltas(vecs: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> dict[str, Any]:
    """Fit y_i = D . n_i with weighted least squares (returns D and covariance)."""
    X = np.asarray(vecs, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("vecs must be (N,3).")
    if y.shape != (X.shape[0],) or sigma.shape != (X.shape[0],):
        raise ValueError("y/sigma shape mismatch.")
    w = 1.0 / np.clip(sigma**2, 1e-30, np.inf)
    XtW = X.T * w[None, :]
    cov = np.linalg.inv(XtW @ X)
    D = cov @ (XtW @ y)
    D = np.asarray(D, dtype=float)
    cov = np.asarray(cov, dtype=float)
    amp = float(np.linalg.norm(D))
    if amp > 0:
        var_amp = float((D.T @ cov @ D) / (amp**2))
        sig_amp = math.sqrt(max(var_amp, 0.0))
    else:
        sig_amp = float("inf")
    return {
        "D_vec": D.tolist(),
        "D_cov": cov.tolist(),
        "D_amp": amp,
        "D_amp_sigma": sig_amp,
        "T_amp": (amp / sig_amp) if (sig_amp > 0 and np.isfinite(sig_amp)) else None,
    }


def _fit_monopole_plus_dipole(vecs: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> dict[str, Any]:
    """Fit y_i = a + D . n_i with weighted least squares (returns a, D and covariance)."""
    X3 = np.asarray(vecs, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if X3.ndim != 2 or X3.shape[1] != 3:
        raise ValueError("vecs must be (N,3).")
    if y.shape != (X3.shape[0],) or sigma.shape != (X3.shape[0],):
        raise ValueError("y/sigma shape mismatch.")
    X = np.column_stack([np.ones((X3.shape[0],), dtype=float), X3])
    w = 1.0 / np.clip(sigma**2, 1e-30, np.inf)
    XtW = X.T * w[None, :]
    cov = np.linalg.inv(XtW @ X)
    beta = cov @ (XtW @ y)
    a = float(beta[0])
    D = np.asarray(beta[1:], dtype=float)
    amp = float(np.linalg.norm(D))
    covD = cov[1:, 1:]
    if amp > 0:
        var_amp = float((D.T @ covD @ D) / (amp**2))
        sig_amp = math.sqrt(max(var_amp, 0.0))
    else:
        sig_amp = float("inf")
    return {
        "s_mono": a,
        "beta": beta.tolist(),
        "cov": cov.tolist(),
        "D_vec": D.tolist(),
        "D_cov": covD.tolist(),
        "D_amp": amp,
        "D_amp_sigma": sig_amp,
        "T_amp": (amp / sig_amp) if (sig_amp > 0 and np.isfinite(sig_amp)) else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Horizon anisotropy test: hemisphere scan over Pantheon+ sky.")
    ap.add_argument("--mode", type=str, default="hemisphere_scan", choices=["hemisphere_scan", "patch_map", "crossfit"])
    ap.add_argument(
        "--preset",
        type=str,
        default="none",
        choices=["none", "production"],
        help="Optional preset bundle. 'production' enforces crossfit-safe matching, null defaults, and free-gamma settings.",
    )
    ap.add_argument("--out", type=Path, default=None, help="Output directory (default: outputs/horizon_anisotropy/<timestamp>).")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed for z-matching downsampling.")
    ap.add_argument("--axes-nside", type=int, default=2, help="HEALPix nside for hemisphere scan axes (Galactic).")
    ap.add_argument("--axes-nest", action="store_true", help="Use NEST ordering for axis pixels (default: RING).")
    ap.add_argument(
        "--skip-antipodes",
        action="store_true",
        help=(
            "Skip antipodal duplicate axes in hemisphere_scan mode. "
            "Axis n and -n define the same split (swapping fore/aft), so this halves work safely."
        ),
    )
    ap.add_argument(
        "--axis-jobs",
        type=int,
        default=0,
        help="If >1, parallelize hemisphere_scan axis computations across this many worker processes (sets --mu-procs=1).",
    )
    ap.add_argument("--axis-pix", type=int, default=None, help="If set, run only this axis pixel index (in the chosen nside/order).")
    ap.add_argument("--max-axes", type=int, default=0, help="If >0, run at most this many axes (after filtering).")
    ap.add_argument("--patch-nside", type=int, default=2, help="HEALPix nside for patch-map mode (Galactic).")
    ap.add_argument("--patch-nest", action="store_true", help="Use NEST ordering for patch pixels (default: RING).")
    ap.add_argument("--subset", type=str, default="cosmology", choices=["cosmology", "all", "shoes_hubble_flow"])
    ap.add_argument("--cov-kind", type=str, default="stat+sys", choices=["stat+sys", "statonly"])
    ap.add_argument("--z-column", type=str, default="zHD", choices=["zHD", "zCMB", "zHEL"])
    ap.add_argument("--z-min", type=float, default=0.02)
    ap.add_argument("--z-max", type=float, default=0.62)
    ap.add_argument("--match-z", type=str, default="none", choices=["none", "bin_downsample"])
    ap.add_argument("--match-z-bin-width", type=float, default=0.05, help="Bin width for match-z=bin_downsample.")
    ap.add_argument(
        "--match-mode",
        type=str,
        default="z_only",
        choices=["z_only", "survey_z"],
        help=(
            "How to symmetrize fore/aft hemispheres. z_only matches N(z) only; "
            "survey_z matches (survey_id x z_bin) to suppress footprint-driven dipoles."
        ),
    )
    ap.add_argument("--min-sn-per-side", type=int, default=200, help="Skip axes with fewer SNe than this per side after matching.")
    ap.add_argument("--min-sn-per-patch", type=int, default=250, help="Skip patches with fewer SNe than this in patch_map mode.")
    ap.add_argument(
        "--null-reps",
        type=int,
        default=0,
        help=(
            "If >0, run a null battery by shuffling SN sky positions (survey-aware) and recomputing the scan "
            "statistic. NOTE: this can be extremely expensive unless you use small mu_steps/draws."
        ),
    )
    ap.add_argument(
        "--null-mode",
        type=str,
        default="shuffle_within_survey",
        choices=["shuffle_within_survey", "shuffle_all"],
        help="Null generator for sky positions (tries to preserve survey footprint).",
    )
    ap.add_argument(
        "--null-stat",
        type=str,
        default="max_abs_z",
        choices=["max_abs_z", "dipole_T"],
        help="Null test statistic to record.",
    )
    ap.add_argument(
        "--null-axis-jobs",
        type=int,
        default=0,
        help="If >1, parallelize per-replicate null axis computations across this many worker processes (sets --mu-procs=1).",
    )
    ap.add_argument(
        "--null-rep-start",
        type=int,
        default=0,
        help=(
            "For sharded null batteries: first null replicate index to compute (inclusive). "
            "Existing rep files are always reused/skipped."
        ),
    )
    ap.add_argument(
        "--null-rep-end",
        type=int,
        default=None,
        help=(
            "For sharded null batteries: stop replicate index (exclusive). "
            "Default computes through --null-reps."
        ),
    )
    ap.add_argument(
        "--null-finalize-only",
        action="store_true",
        help=(
            "Do not compute new null reps. Instead, read existing null_reps/rep*.json files and "
            "write/refresh null_summary.json. Useful after sharded runs complete."
        ),
    )
    ap.add_argument(
        "--resume-scan",
        action="store_true",
        help=(
            "If axis_results.json exists in --out, reuse it and skip recomputing the hemisphere scan. "
            "This is required when running multiple null shards against the same out dir."
        ),
    )
    ap.add_argument(
        "--store-axis-posteriors",
        action="store_true",
        help=(
            "Write per-axis fore/aft posterior files during hemisphere_scan. "
            "Default keeps only compact per-axis summaries to reduce I/O."
        ),
    )
    ap.add_argument(
        "--store-train-posteriors",
        action="store_true",
        help=(
            "Write per-axis fore/aft posterior files during crossfit TRAIN scan. "
            "Default keeps only compact per-axis summaries to reduce I/O."
        ),
    )

    # Forward-mu inference knobs (SN-only).
    ap.add_argument("--sn-like-bin-width", type=float, default=0.05)
    ap.add_argument("--sn-like-min-per-bin", type=int, default=20)
    ap.add_argument("--mu-knots", type=int, default=6)
    ap.add_argument("--mu-grid", type=int, default=120)
    ap.add_argument("--n-grid", type=int, default=200)
    ap.add_argument("--mu-sampler", type=str, default="ptemcee", choices=["emcee", "ptemcee"])
    ap.add_argument("--pt-ntemps", type=int, default=6)
    ap.add_argument("--pt-tmax", type=float, default=20.0)
    ap.add_argument("--mu-walkers", type=int, default=48)
    ap.add_argument("--mu-steps", type=int, default=800)
    ap.add_argument("--mu-burn", type=int, default=250)
    ap.add_argument("--mu-draws", type=int, default=400)
    ap.add_argument("--mu-procs", type=int, default=0, help="Worker processes (0=auto-ish, default uses 1).")
    ap.add_argument("--omega-m0-prior", type=float, nargs=2, default=[0.2, 0.4])
    ap.add_argument("--H0-prior", type=float, nargs=2, default=[60.0, 80.0])
    ap.add_argument("--r-d-fixed", type=float, default=147.78, help="Fix r_d to remove an unconstrained dimension in SN-only runs.")
    ap.add_argument(
        "--growth-mode",
        type=str,
        default="ode",
        choices=["ode", "gamma"],
        help="Growth solver mode forwarded to inference (gamma mode supports free-gamma sampling).",
    )
    ap.add_argument("--growth-gamma", type=float, default=0.55, help="Fixed growth index gamma when not sampling it.")
    ap.add_argument(
        "--growth-gamma-free",
        action="store_true",
        help="Sample growth gamma from --growth-gamma-prior instead of fixing it.",
    )
    ap.add_argument(
        "--growth-gamma-prior",
        type=float,
        nargs=2,
        default=[0.2, 1.2],
        help="Uniform prior bounds for sampled growth gamma.",
    )
    ap.add_argument("--sigma-sn-jit-scale", type=float, default=0.05)
    ap.add_argument("--sigma-d2-scale", type=float, default=0.185)
    ap.add_argument("--logmu-knot-scale", type=float, default=None)
    ap.add_argument("--m-weight-mode", type=str, default="variance", choices=["variance", "uniform"])

    # Cross-fit mode knobs.
    ap.add_argument("--kfold", type=int, default=5, help="Number of folds for crossfit mode (train/test axis selection).")
    ap.add_argument("--split-z-bin-width", type=float, default=0.05, help="z-bin width for (survey x z) stratified folds.")
    ap.add_argument("--train-axes-nside", type=int, default=4, help="Axis grid nside for train axis scan (crossfit mode).")
    ap.add_argument("--train-axes-nest", action="store_true", help="Use NEST ordering for train axes (crossfit mode).")
    ap.add_argument("--train-max-axes", type=int, default=0, help="If >0, cap number of train axes (crossfit mode).")
    ap.add_argument(
        "--train-skip-antipodes",
        action="store_true",
        help=(
            "Skip antipodal duplicate axes in the TRAIN scan (crossfit mode). "
            "This is safe because axis n and -n define the same hemisphere split (just swapping fore/aft)."
        ),
    )
    ap.add_argument(
        "--train-axis-jobs",
        type=int,
        default=0,
        help=(
            "If >1, parallelize TRAIN axis computations across this many worker processes. "
            "Recommended to keep --train-mu-procs small (often 1) to avoid nested multiprocessing overhead."
        ),
    )
    ap.add_argument("--train-mu-knots", type=int, default=2, help="Number of mu knots in train scan (crossfit mode).")
    ap.add_argument("--train-mu-steps", type=int, default=400, help="MCMC steps for train scan (crossfit mode).")
    ap.add_argument("--train-mu-burn", type=int, default=150, help="Burn-in steps for train scan (crossfit mode).")
    ap.add_argument("--train-mu-draws", type=int, default=200, help="Posterior draws for train scan (crossfit mode).")
    ap.add_argument(
        "--train-mu-sampler",
        type=str,
        default="inherit",
        choices=["inherit", "emcee", "ptemcee"],
        help="Sampler for train scan (crossfit mode). 'inherit' uses --mu-sampler.",
    )
    ap.add_argument("--train-mu-walkers", type=int, default=0, help="Walkers for train scan (0=use --mu-walkers).")
    ap.add_argument(
        "--train-mu-procs",
        type=int,
        default=0,
        help="Worker processes for train scan (0=auto: min(64, --mu-procs); if --mu-procs<=0 then 1).",
    )
    ap.add_argument("--train-pt-ntemps", type=int, default=0, help="ptemcee temps for train scan (0=use --pt-ntemps).")
    ap.add_argument("--train-pt-tmax", type=float, default=None, help="ptemcee Tmax for train scan (default uses --pt-tmax).")
    ap.add_argument("--test-mu-knots", type=int, default=6, help="Number of mu knots for held-out confirmation (crossfit mode).")
    ap.add_argument("--test-mu-steps", type=int, default=2500, help="MCMC steps for held-out confirmation (crossfit mode).")
    ap.add_argument("--test-mu-burn", type=int, default=800, help="Burn-in steps for held-out confirmation (crossfit mode).")
    ap.add_argument("--test-mu-draws", type=int, default=800, help="Posterior draws for held-out confirmation (crossfit mode).")
    ap.add_argument(
        "--test-mu-procs",
        type=int,
        default=0,
        help="Worker processes for held-out confirmation (crossfit mode). 0=use --mu-procs.",
    )
    ap.add_argument(
        "--test-parallel-fore-aft",
        action="store_true",
        help="Run held-out confirmation fore/aft reconstructions concurrently (two subprocesses).",
    )
    ap.add_argument(
        "--test-min-sn-per-side",
        type=int,
        default=80,
        help="Minimum SNe per hemisphere in held-out confirmation (crossfit mode).",
    )

    args = ap.parse_args()
    _apply_production_preset(args)

    if bool(args.growth_gamma_free) and str(args.growth_mode) != "gamma":
        raise ValueError("--growth-gamma-free requires --growth-mode gamma.")
    if float(args.growth_gamma) <= 0 or not np.isfinite(float(args.growth_gamma)):
        raise ValueError("--growth-gamma must be finite and > 0.")
    g_lo, g_hi = [float(x) for x in args.growth_gamma_prior]
    if not (np.isfinite(g_lo) and np.isfinite(g_hi) and g_hi > g_lo and g_lo > 0):
        raise ValueError("--growth-gamma-prior must satisfy 0 < lo < hi and both finite.")

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out) if args.out is not None else (repo_root / "outputs" / "horizon_anisotropy" / _utc_stamp())
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp_utc": _utc_stamp(),
        "git": {"sha": git_head_sha(repo_root=repo_root), "dirty": git_is_dirty(repo_root=repo_root)},
        "args": vars(args),
    }
    # Avoid shards clobbering the canonical manifest for a run directory.
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        _write_json(out_dir / f"manifest_{manifest['timestamp_utc']}.json", manifest)
    else:
        _write_json(manifest_path, manifest)

    # Resolve mu_procs.
    mu_procs = int(args.mu_procs)
    if mu_procs <= 0:
        # SN-only loglike is not too expensive; keep conservative default.
        mu_procs = 1
    if int(getattr(args, "axis_jobs", 0)) > 1 or int(getattr(args, "null_axis_jobs", 0)) > 1:
        # Avoid nested multiprocessing overhead.
        mu_procs = 1

    # Crossfit TRAIN override knobs.
    train_mu_sampler = str(args.mu_sampler) if str(getattr(args, "train_mu_sampler", "inherit")) == "inherit" else str(args.train_mu_sampler)
    train_mu_walkers = int(args.mu_walkers) if int(getattr(args, "train_mu_walkers", 0)) <= 0 else int(args.train_mu_walkers)
    train_pt_ntemps = int(args.pt_ntemps) if int(getattr(args, "train_pt_ntemps", 0)) <= 0 else int(args.train_pt_ntemps)
    train_pt_tmax = float(args.pt_tmax) if getattr(args, "train_pt_tmax", None) is None else float(args.train_pt_tmax)
    train_mu_procs = int(getattr(args, "train_mu_procs", 0))
    if train_mu_procs <= 0:
        # For TRAIN scans the log-prob is cheap; too many workers wastes time in IPC overhead.
        train_mu_procs = int(min(64, mu_procs)) if int(mu_procs) > 1 else 1
    train_axis_jobs = int(getattr(args, "train_axis_jobs", 0))
    if train_axis_jobs > 1 and train_mu_procs > 1:
        # Nested pools are usually slower + can oversubscribe. Prefer axis-level parallelism.
        train_mu_procs = 1

    test_mu_procs = int(getattr(args, "test_mu_procs", 0))
    if test_mu_procs <= 0:
        test_mu_procs = int(mu_procs)
    if bool(getattr(args, "test_parallel_fore_aft", False)) and test_mu_procs > 1 and test_mu_procs == int(mu_procs):
        # If user didn't set test-mu-procs explicitly, default to splitting the pool.
        test_mu_procs = max(1, int(mu_procs) // 2)

    paths = DataPaths(repo_root=repo_root)
    sn = load_pantheon_plus_sky(paths=paths, cov_kind=str(args.cov_kind), subset=str(args.subset), z_column=str(args.z_column))

    # Global z cuts.
    z = np.asarray(sn.z, dtype=float)
    m = np.asarray(sn.m_b_corr, dtype=float)
    cov = np.asarray(sn.cov, dtype=float)
    v = _radec_to_galactic_unitvec(sn.ra_deg, sn.dec_deg)

    z_min = float(args.z_min)
    z_max = float(args.z_max)
    m_z = (z >= z_min) & (z <= z_max) & np.isfinite(z) & np.isfinite(m)
    if not np.any(m_z):
        raise RuntimeError("No Pantheon+ entries pass z cuts; check z_min/z_max.")

    keep = np.where(m_z)[0].astype(np.int64)
    z = z[keep]
    m = m[keep]
    cov = cov[np.ix_(keep, keep)]
    v = v[keep]
    idsurvey = np.asarray(sn.idsurvey, dtype=np.int64)[keep]

    mode = str(args.mode)

    # --- Cross-fit mode: out-of-sample axis selection + held-out confirmation ---
    if mode == "crossfit":
        base_seed = int(args.seed)
        rng = np.random.default_rng(base_seed)
        fold_assign_path = out_dir / "fold_assignments.npz"
        if fold_assign_path.exists():
            fold_id = np.asarray(np.load(fold_assign_path)["fold_id"], dtype=np.int64)
            if fold_id.shape != (z.size,):
                raise RuntimeError(f"Existing fold_assignments.npz has wrong shape: {fold_id.shape}, expected ({z.size},)")
            _log(f"[crossfit] loaded fold assignments from {fold_assign_path}")
        else:
            fold_id, fold_meta = _make_stratified_folds(
                idsurvey=idsurvey,
                z=z,
                k_folds=int(args.kfold),
                z_bin_width=float(args.split_z_bin_width),
                rng=rng,
            )
            np.savez_compressed(fold_assign_path, fold_id=fold_id)
            _write_json(out_dir / "folds.json", {"meta": fold_meta})
            _log(f"[crossfit] wrote fold assignments to {fold_assign_path}")

        # Group key for (survey x zbin) matching in hemispheres.
        # Use the same z-bin definition as the match-z bins.
        bw = float(args.match_z_bin_width)
        z_edges = np.arange(z_min, z_max + bw, bw)
        z_bin = np.digitize(z, z_edges, right=False) - 1
        z_bin = np.clip(z_bin, 0, z_edges.size - 2).astype(np.int64)
        group_key = (idsurvey.astype(np.int64) << 32) ^ z_bin.astype(np.int64)

        # Axis grid for train scan.
        pix, axis_vecs, lb = _axis_grid_galactic(nside=int(args.train_axes_nside), nest=bool(args.train_axes_nest))

        fold_rows: list[dict[str, Any]] = []
        test_z_scores: list[float] = []

        for k in range(int(args.kfold)):
            fold_dir = out_dir / f"fold{int(k):02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_idx = np.where(fold_id != int(k))[0].astype(np.int64)
            test_idx = np.where(fold_id == int(k))[0].astype(np.int64)
            _write_json(
                fold_dir / "split.json",
                {"kfold": int(args.kfold), "fold": int(k), "n_train": int(train_idx.size), "n_test": int(test_idx.size)},
            )

            _log(f"[crossfit] fold {k+1}/{int(args.kfold)}: n_train={train_idx.size} n_test={test_idx.size}")

            # --- TRAIN: scan axes and select best ---
            train_scan_dir = fold_dir / "train_scan"
            train_scan_dir.mkdir(parents=True, exist_ok=True)
            cached_axis_results = train_scan_dir / "axis_results.json"
            cached_best_axis = train_scan_dir / "train_best_axis.json"
            train_axis_rows: list[AxisResult] = []
            train_rows_json: list[dict[str, Any]] = []

            # If a completed train scan exists, load it (enables resume without recomputing 100s of axes).
            if cached_axis_results.exists():
                try:
                    cached = json.loads(cached_axis_results.read_text(encoding="utf-8"))
                    if not isinstance(cached, list):
                        raise TypeError("axis_results.json must be a list")
                    train_rows_json = [dict(x) for x in cached]
                    train_axis_rows = [_axisresult_from_dict(d) for d in train_rows_json]
                    _log(f"[crossfit][train] fold={k} loaded cached axis_results.json (axes={len(train_axis_rows)})")
                except Exception as e:
                    _log(f"[crossfit][train] fold={k} failed to load cached axis_results.json: {type(e).__name__}: {e} (will recompute)")
                    train_rows_json = []
                    train_axis_rows = []

            if not train_axis_rows:
                # Build task list (honor max_axes + antipode skipping).
                tasks: list[_TrainAxisTask] = []
                for i, (p, vec, (l_deg, b_deg)) in enumerate(zip(pix.tolist(), axis_vecs, lb.tolist(), strict=True)):
                    if int(args.train_max_axes) > 0 and int(i) >= int(args.train_max_axes):
                        break
                    if bool(getattr(args, "train_skip_antipodes", False)):
                        p_anti = _antipode_pix(nside=int(args.train_axes_nside), vec=np.asarray(vec, dtype=float), nest=bool(args.train_axes_nest))
                        if int(p) > int(p_anti):
                            continue
                    tasks.append(
                        _TrainAxisTask(
                            fold=int(k),
                            axis_pix=int(p),
                            axis_l_deg=float(l_deg),
                            axis_b_deg=float(b_deg),
                            axis_vec=(float(vec[0]), float(vec[1]), float(vec[2])),
                        )
                    )

                global _TRAIN_AXIS_GLOBAL
                _TRAIN_AXIS_GLOBAL = {
                    "train_scan_dir": train_scan_dir,
                    "z": z,
                    "m": m,
                    "cov": cov,
                    "v": v,
                    "group_key": group_key,
                    "train_idx": train_idx,
                    "z_edges": z_edges,
                    "z_min": z_min,
                    "z_max": z_max,
                    "args": args,
                    "base_seed": base_seed,
                    "train_mu_sampler": train_mu_sampler,
                    "train_pt_ntemps": train_pt_ntemps,
                    "train_pt_tmax": train_pt_tmax,
                    "train_mu_walkers": train_mu_walkers,
                    "train_mu_procs": train_mu_procs,
                    "train_quiet": True if train_axis_jobs > 1 else False,
                }

                t0 = time.time()
                if train_axis_jobs > 1:
                    import multiprocessing as mp

                    ctx = mp.get_context("fork")
                    _log(f"[crossfit][train] fold={k} parallel axis scan: jobs={train_axis_jobs} tasks={len(tasks)} train_mu_procs={train_mu_procs}")
                    with ctx.Pool(processes=int(train_axis_jobs)) as pool:
                        for j, out in enumerate(pool.imap_unordered(_train_axis_worker, tasks, chunksize=1), start=1):
                            if out is None:
                                continue
                            try:
                                train_axis_rows.append(_axisresult_from_dict(out))
                                train_rows_json.append(dict(out))
                            except Exception:
                                continue
                            if j % 10 == 0:
                                dt = time.time() - t0
                                _log(f"[crossfit][train] fold={k} collected={len(train_axis_rows)} elapsed={dt/60:.1f} min")
                else:
                    for j, tsk in enumerate(tasks, start=1):
                        out = _train_axis_worker(tsk)
                        if out is None:
                            continue
                        train_axis_rows.append(_axisresult_from_dict(out))
                        train_rows_json.append(dict(out))
                        if j % 10 == 0:
                            dt = time.time() - t0
                            _log(f"[crossfit][train] fold={k} collected={len(train_axis_rows)} elapsed={dt/60:.1f} min")

                _TRAIN_AXIS_GLOBAL = None

                if not train_axis_rows:
                    raise RuntimeError(f"Crossfit fold {k}: no train axes produced results; loosen cuts.")

                _write_json(train_scan_dir / "axis_results.json", train_rows_json)

            # Best axis (prefer cached if present).
            if cached_best_axis.exists():
                best_train = _axisresult_from_dict(json.loads(cached_best_axis.read_text(encoding="utf-8")))
            else:
                best_train = max(train_axis_rows, key=lambda r: abs(r.z_score))
                _write_json(train_scan_dir / "train_best_axis.json", asdict(best_train))
            _log(f"[crossfit][train] fold={k} best axis pix={best_train.axis_pix} (l,b)=({best_train.axis_l_deg:.2f},{best_train.axis_b_deg:.2f}) z={best_train.z_score:+.3f}")

            # --- TEST: confirm only at the chosen axis ---
            test_dir = fold_dir / "test_confirm"
            test_dir.mkdir(parents=True, exist_ok=True)
            test_result_path = test_dir / "test_result.json"
            if test_result_path.exists():
                try:
                    fold_rows.append(json.loads(test_result_path.read_text(encoding="utf-8")))
                    test_z_scores.append(float(fold_rows[-1]["test"]["z_score"]))
                    _log(f"[crossfit][test] fold={k} loaded cached test_result.json z_test={test_z_scores[-1]:+.3f}")
                    continue
                except Exception:
                    pass

            vec_best = axis_vecs[pix.tolist().index(best_train.axis_pix)]  # axis_vecs aligned with pix list
            proj_test = v[test_idx] @ vec_best
            idx_fore_t = test_idx[np.where(proj_test > 0.0)[0]].astype(np.int64)
            idx_aft_t = test_idx[np.where(proj_test < 0.0)[0]].astype(np.int64)
            idx_fore_t, idx_aft_t, tmatch_meta = _match_test_split(
                idx_fore_t,
                idx_aft_t,
                z=z,
                z_edges=z_edges,
                group_key=group_key,
                base_seed=base_seed,
                fold=k,
                axis_pix=int(best_train.axis_pix),
                match_z=str(args.match_z),
                match_mode=str(args.match_mode),
                min_req=int(args.test_min_sn_per_side),
            )
            if int(idx_fore_t.size) == 0 or int(idx_aft_t.size) == 0:
                raise RuntimeError(
                    f"Crossfit fold {k}: test split too small after all matching fallbacks for best axis (pix={best_train.axis_pix}). "
                    f"Consider lowering --test-min-sn-per-side and/or --min-sn-per-side, or using --match-mode z_only."
                )

            _write_json(test_dir / "axis.json", {"axis_pix": int(best_train.axis_pix), "l_deg": float(best_train.axis_l_deg), "b_deg": float(best_train.axis_b_deg), "match": tmatch_meta})
            _log(f"[crossfit][test] fold={k} n_fore={idx_fore_t.size} n_aft={idx_aft_t.size} (best axis)")

            def _run_test_side(*, side: str, idx: np.ndarray, seed0: int) -> dict[str, Any]:
                return _run_subset_recon(
                    out_dir=test_dir / side,
                    sn_z=z[idx],
                    sn_m=m[idx],
                    sn_cov=cov[np.ix_(idx, idx)],
                    z_min=z_min,
                    z_max=z_max,
                    sn_like_bin_width=float(args.sn_like_bin_width),
                    sn_like_min_per_bin=int(args.sn_like_min_per_bin),
                    mu_knots=int(args.test_mu_knots),
                    mu_grid=int(args.mu_grid),
                    n_grid=int(args.n_grid),
                    mu_sampler=str(args.mu_sampler),
                    pt_ntemps=int(args.pt_ntemps),
                    pt_tmax=float(args.pt_tmax) if args.pt_tmax is not None else None,
                    mu_walkers=int(args.mu_walkers),
                    mu_steps=int(args.test_mu_steps),
                    mu_burn=int(args.test_mu_burn),
                    mu_draws=int(args.test_mu_draws),
                    mu_procs=int(test_mu_procs),
                    seed=int(seed0),
                    omega_m0_prior=(float(args.omega_m0_prior[0]), float(args.omega_m0_prior[1])),
                    H0_prior=(float(args.H0_prior[0]), float(args.H0_prior[1])),
                    r_d_fixed=float(args.r_d_fixed),
                    growth_mode=str(args.growth_mode),
                    growth_gamma=float(args.growth_gamma),
                    growth_gamma_prior=_growth_gamma_prior_from_args(args),
                    sigma_sn_jit_scale=float(args.sigma_sn_jit_scale),
                    sigma_d2_scale=float(args.sigma_d2_scale),
                    logmu_knot_scale=float(args.logmu_knot_scale) if args.logmu_knot_scale is not None else None,
                    m_weight_mode=str(args.m_weight_mode),
                    quiet=False,
                )

            seed_fore = int(args.seed) + 20_000 * int(k) + int(best_train.axis_pix) + 333
            seed_aft = int(args.seed) + 20_000 * int(k) + int(best_train.axis_pix) + 444

            if bool(getattr(args, "test_parallel_fore_aft", False)):
                import multiprocessing as mp

                ctx = mp.get_context("fork")
                q: mp.Queue = ctx.Queue()

                procs: list[mp.Process] = []
                results: dict[str, dict[str, Any]] = {}
                errors: dict[str, str] = {}

                def _target(side: str, idx: np.ndarray, seed0: int):
                    try:
                        q.put((side, _run_test_side(side=side, idx=idx, seed0=seed0), None))
                    except Exception as e:
                        q.put((side, None, f"{type(e).__name__}: {e}"))

                # Only spawn processes for missing sides (resume-friendly).
                for side, idx, seed0 in [
                    ("fore", idx_fore_t, seed_fore),
                    ("aft", idx_aft_t, seed_aft),
                ]:
                    done_marker = (test_dir / side / ".done")
                    if done_marker.exists() and (test_dir / side / "summary.json").exists():
                        results[side] = json.loads((test_dir / side / "summary.json").read_text(encoding="ascii"))
                        continue
                    p0 = ctx.Process(target=_target, args=(side, idx, seed0))
                    p0.daemon = False
                    p0.start()
                    procs.append(p0)

                # Collect spawned results.
                needed = len(procs)
                while needed > 0:
                    side, summ, err = q.get()
                    needed -= 1
                    if err is not None:
                        errors[str(side)] = str(err)
                    else:
                        results[str(side)] = dict(summ)

                for p0 in procs:
                    p0.join()

                if errors:
                    raise RuntimeError(f"held-out test parallel fore/aft failed: {errors}")

                fore_t = results["fore"]
                aft_t = results["aft"]
            else:
                fore_t = _run_test_side(side="fore", idx=idx_fore_t, seed0=seed_fore)
                aft_t = _run_test_side(side="aft", idx=idx_aft_t, seed0=seed_aft)

            # Use the scalar summaries as the headline, plus a delta_s posterior sign probability.
            s_f_mean = float(fore_t["departure"]["slope"]["mean"])
            s_f_std = float(fore_t["departure"]["slope"]["std"])
            s_a_mean = float(aft_t["departure"]["slope"]["mean"])
            s_a_std = float(aft_t["departure"]["slope"]["std"])
            delta_s = float(s_f_mean - s_a_mean)
            sig = float(math.sqrt(max(s_f_std**2 + s_a_std**2, 1e-12)))
            z_sc = float(delta_s / sig)

            # Load scar draws for a crude P(delta_s>0) diagnostic (independent-draw approximation).
            try:
                sf = np.load(test_dir / "fore" / "samples" / "scar_draws.npz")
                sa = np.load(test_dir / "aft" / "samples" / "scar_draws.npz")
                ds = np.asarray(sf["s"], dtype=float) - np.asarray(sa["s"], dtype=float)
                p_pos = float(np.mean(ds > 0.0))
            except Exception:
                p_pos = float("nan")

            test_res = {
                "fold": int(k),
                "best_axis_train": asdict(best_train),
                "test": {"delta_s": delta_s, "sigma_delta_s": sig, "z_score": z_sc, "P(delta_s>0)": p_pos},
            }
            _write_json(test_dir / "test_result.json", test_res)
            fold_rows.append(test_res)
            test_z_scores.append(z_sc)
            _log(f"[crossfit][test] fold={k} z_test={z_sc:+.3f} delta_s={delta_s:+.3f} +/- {sig:.3f} P(ds>0)={p_pos:.3f}")

        # Aggregate across folds (Stouffer).
        z_arr = np.asarray(test_z_scores, dtype=float)
        z_comb = float(np.sum(z_arr) / math.sqrt(max(z_arr.size, 1)))
        summary = {
            "mode": "crossfit",
            "kfold": int(args.kfold),
            "match_z": str(args.match_z),
            "match_mode": str(args.match_mode),
            "train_axes_nside": int(args.train_axes_nside),
            "train_axes_nest": bool(args.train_axes_nest),
            "train_mu_knots": int(args.train_mu_knots),
            "test_mu_knots": int(args.test_mu_knots),
            "fold_results": fold_rows,
            "z_scores_test": test_z_scores,
            "z_stouffer": z_comb,
            "notes": [
                "This is an out-of-sample axis-selection test: axis chosen on train, evaluated on held-out test fold.",
                "The fold-level z-scores are not independent, but Stouffer is a convenient aggregate diagnostic.",
                "For publication, add a null battery that reruns the full crossfit procedure under survey-aware shuffles.",
            ],
        }
        _write_json(out_dir / "crossfit_summary.json", summary)
        return 0

    # --- Patch map mode (texture map / multipole-style visualization) ---
    if mode == "patch_map":
        patch_nside = int(args.patch_nside)
        patch_nest = bool(args.patch_nest)
        patch_pix, patch_vecs, patch_lb = _axis_grid_galactic(nside=patch_nside, nest=patch_nest)
        pix_sn = hp.vec2pix(patch_nside, v[:, 0], v[:, 1], v[:, 2], nest=patch_nest).astype(np.int64, copy=False)

        rows: list[dict[str, Any]] = []
        for p, vec, (l_deg, b_deg) in zip(patch_pix.tolist(), patch_vecs, patch_lb.tolist()):
            idx = np.where(pix_sn == int(p))[0].astype(np.int64)
            if int(idx.size) < int(args.min_sn_per_patch):
                continue
            patch_dir = out_dir / f"patch_nside{patch_nside}_{'nest' if patch_nest else 'ring'}_pix{int(p):05d}"
            _write_json(
                patch_dir / "patch.json",
                {"patch_pix": int(p), "l_deg": float(l_deg), "b_deg": float(b_deg), "n_sn": int(idx.size)},
            )

            summ = _run_subset_recon(
                out_dir=patch_dir,
                sn_z=z[idx],
                sn_m=m[idx],
                sn_cov=cov[np.ix_(idx, idx)],
                z_min=z_min,
                z_max=z_max,
                sn_like_bin_width=float(args.sn_like_bin_width),
                sn_like_min_per_bin=int(args.sn_like_min_per_bin),
                mu_knots=int(args.mu_knots),
                mu_grid=int(args.mu_grid),
                n_grid=int(args.n_grid),
                mu_sampler=str(args.mu_sampler),
                pt_ntemps=int(args.pt_ntemps),
                pt_tmax=float(args.pt_tmax) if args.pt_tmax is not None else None,
                mu_walkers=int(args.mu_walkers),
                mu_steps=int(args.mu_steps),
                mu_burn=int(args.mu_burn),
                mu_draws=int(args.mu_draws),
                mu_procs=int(mu_procs),
                seed=int(args.seed) + 3000 + int(p),
                omega_m0_prior=(float(args.omega_m0_prior[0]), float(args.omega_m0_prior[1])),
                H0_prior=(float(args.H0_prior[0]), float(args.H0_prior[1])),
                r_d_fixed=float(args.r_d_fixed),
                growth_mode=str(args.growth_mode),
                growth_gamma=float(args.growth_gamma),
                growth_gamma_prior=_growth_gamma_prior_from_args(args),
                sigma_sn_jit_scale=float(args.sigma_sn_jit_scale),
                sigma_d2_scale=float(args.sigma_d2_scale),
                logmu_knot_scale=float(args.logmu_knot_scale) if args.logmu_knot_scale is not None else None,
                m_weight_mode=str(args.m_weight_mode),
            )

            s_mean = float(summ["departure"]["slope"]["mean"])
            s_std = float(summ["departure"]["slope"]["std"])
            rows.append(
                {
                    "patch_pix": int(p),
                    "l_deg": float(l_deg),
                    "b_deg": float(b_deg),
                    "n_sn": int(idx.size),
                    "s_mean": s_mean,
                    "s_std": s_std,
                }
            )

        if not rows:
            raise RuntimeError("No patches produced results; loosen min-sn-per-patch or check z cuts.")

        _write_json(out_dir / "patch_results.json", rows)

        # Fit s(n) = s_mono + D.n from patch centers.
        dip = None
        l_hat = float("nan")
        b_hat = float("nan")
        if len(rows) >= 4:
            vec_by_pix = {int(p): patch_vecs[i] for i, p in enumerate(patch_pix.tolist())}
            X = np.stack([vec_by_pix[int(r["patch_pix"])] for r in rows], axis=0)
            y = np.array([float(r["s_mean"]) for r in rows], dtype=float)
            sig = np.array([max(float(r["s_std"]), 1e-6) for r in rows], dtype=float)
            try:
                dip = _fit_monopole_plus_dipole(X, y, sig)
                D = np.array(dip["D_vec"], dtype=float)
                amp = float(dip["D_amp"])
                if amp > 0:
                    theta = math.acos(max(min(D[2] / amp, 1.0), -1.0))
                    phi = math.atan2(D[1], D[0])
                    l_hat = (math.degrees(phi) % 360.0)
                    b_hat = 90.0 - math.degrees(theta)
            except np.linalg.LinAlgError:
                dip = None

        # Attempt a Mollweide visualization.
        fig_note = None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            npix_patch = hp.nside2npix(patch_nside)
            m_s = np.full((npix_patch,), hp.UNSEEN, dtype=float)
            for r in rows:
                m_s[int(r["patch_pix"])] = float(r["s_mean"])
            (out_dir / "figures").mkdir(parents=True, exist_ok=True)
            hp.mollview(
                m_s,
                nest=patch_nest,
                coord=["G"],
                title="Pantheon+ patch-wise slope scar s (mean)",
                unit="s",
                cmap="coolwarm",
            )
            plt.savefig(out_dir / "figures" / "s_map_mollview.png", dpi=200)
            plt.close()
        except Exception as e:
            fig_note = f"plot_failed: {type(e).__name__}: {e}"

        summary = {
            "mode": "patch_map",
            "n_patches_done": int(len(rows)),
            "patch_nside": int(patch_nside),
            "patch_nest": bool(patch_nest),
            "dipole_fit": ({**dip, "D_l_deg": float(l_hat), "D_b_deg": float(b_hat)} if dip is not None else None),
            "figure_note": fig_note,
            "notes": [
                "Patch-wise reconstructions use Pantheon+ only (SN-only logmu inference).",
                "Dipole fit is performed on patch s_mean values: s(n)=s_mono + D.n.",
                "Interpretation requires null tests (survey footprint, z distribution, and look-elsewhere controls).",
            ],
        }
        _write_json(out_dir / "scan_summary.json", summary)
        return 0

    # --- Hemisphere scan mode (dipole search) ---
    pix, axis_vecs, lb = _axis_grid_galactic(nside=int(args.axes_nside), nest=bool(args.axes_nest))

    axis_rows: list[AxisResult] = []
    rows_json: list[dict[str, Any]] = []

    # Precompute z-bins for optional matching.
    z_edges = None
    if str(args.match_z) == "bin_downsample":
        bw = float(args.match_z_bin_width)
        if bw <= 0:
            raise ValueError("match-z-bin-width must be positive.")
        z_edges = np.arange(z_min, z_max + bw, bw)
        if z_edges.size < 3:
            raise ValueError("Too few z bins; adjust match-z-bin-width.")

    # Survey-aware group key for matching (optional).
    group_key = None
    if str(args.match_z) == "bin_downsample" and str(args.match_mode) == "survey_z":
        assert z_edges is not None
        group_key = _survey_z_group_key(idsurvey=idsurvey, z=z, z_edges=z_edges)

    # Axis task list (optionally skip antipodes).
    tasks: list[_HemiAxisTask] = []
    for i, (p, vec, (l_deg, b_deg)) in enumerate(zip(pix.tolist(), axis_vecs, lb.tolist(), strict=True)):
        if args.axis_pix is not None and int(p) != int(args.axis_pix):
            continue
        if int(args.max_axes) > 0 and len(tasks) >= int(args.max_axes):
            break
        if bool(getattr(args, "skip_antipodes", False)):
            p_anti = _antipode_pix(nside=int(args.axes_nside), vec=np.asarray(vec, dtype=float), nest=bool(args.axes_nest))
            if int(p) > int(p_anti):
                continue
        tasks.append(
            _HemiAxisTask(
                axis_pix=int(p),
                axis_l_deg=float(l_deg),
                axis_b_deg=float(b_deg),
                axis_vec=(float(vec[0]), float(vec[1]), float(vec[2])),
            )
        )

    axis_results_path = out_dir / "axis_results.json"
    scan_summary_path = out_dir / "scan_summary.json"
    resume_scan = bool(getattr(args, "resume_scan", False))

    if resume_scan and axis_results_path.exists():
        rows_json = json.loads(axis_results_path.read_text(encoding="ascii"))
        axis_rows = [_axisresult_from_dict(d) for d in rows_json]
        if not axis_rows:
            raise RuntimeError(f"{axis_results_path} exists but contains no axis rows.")
        _log(f"[hemi] resumed axis scan from {axis_results_path} (axes={len(axis_rows)})")
    else:
        global _HEMI_AXIS_GLOBAL
        _HEMI_AXIS_GLOBAL = {
            "out_dir": out_dir,
            "z": z,
            "m": m,
            "cov": cov,
            "v": v,
            "z_edges": z_edges,
            "group_key": group_key,
            "args": args,
            "z_min": z_min,
            "z_max": z_max,
            "mu_procs": int(mu_procs),
            "quiet": True if int(getattr(args, "axis_jobs", 0)) > 1 else False,
        }

        axis_jobs = int(getattr(args, "axis_jobs", 0))
        if axis_jobs > 1:
            import multiprocessing as mp

            ctx = mp.get_context("fork")
            _log(f"[hemi] axis scan parallel: jobs={axis_jobs} axes={len(tasks)} mu_procs={mu_procs}")
            with ctx.Pool(processes=int(axis_jobs)) as pool:
                res = pool.map(_hemi_axis_worker, tasks)
        else:
            res = [_hemi_axis_worker(t) for t in tasks]

        for r in res:
            if r is None:
                continue
            row = _axisresult_from_dict(r)
            axis_rows.append(row)
            rows_json.append(asdict(row))

        if not axis_rows:
            raise RuntimeError("No axes produced results; loosen min-sn-per-side or check data cuts.")

        _write_json(axis_results_path, rows_json)

    # Find max-|Z| axis.
    zs = np.array([r.z_score for r in axis_rows], dtype=float)
    j = int(np.argmax(np.abs(zs)))
    best = axis_rows[j]

    # Fit dipole vector from delta_s measurements.
    # Use axis vectors (Galactic unit vecs) corresponding to computed results.
    pix_to_vec = {int(p): axis_vecs[i] for i, p in enumerate(pix.tolist())}
    X = np.stack([pix_to_vec[int(r.axis_pix)] for r in axis_rows], axis=0)
    y = np.array([r.delta_s for r in axis_rows], dtype=float)
    sig = np.array([r.sigma_delta_s for r in axis_rows], dtype=float)
    dip: dict[str, Any] | None = None
    l_hat = float("nan")
    b_hat = float("nan")
    if int(len(axis_rows)) >= 3:
        try:
            dip = _fit_dipole_from_deltas(X, y, sig)
            D = np.array(dip["D_vec"], dtype=float)
            amp = float(dip["D_amp"])
            if amp > 0:
                # Convert dipole direction back to (l,b).
                theta = math.acos(max(min(D[2] / amp, 1.0), -1.0))
                phi = math.atan2(D[1], D[0])
                l_hat = (math.degrees(phi) % 360.0)
                b_hat = 90.0 - math.degrees(theta)
        except np.linalg.LinAlgError:
            dip = None

    # Convert best axis center and (if available) dipole direction to RA/DEC.
    best_vec = pix_to_vec[int(best.axis_pix)]
    best_ra, best_dec = _galactic_unitvec_to_radec(best_vec)
    dip_ra = float("nan")
    dip_dec = float("nan")
    if dip is not None:
        D = np.array(dip["D_vec"], dtype=float)
        amp = float(dip["D_amp"])
        if amp > 0:
            dip_ra, dip_dec = _galactic_unitvec_to_radec(D / amp)

    # Optional quick diagnostic plot: delta_s vs cos(angle to dipole axis).
    fig_note = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not resume_scan:
            (out_dir / "figures").mkdir(parents=True, exist_ok=True)
        # Use best axis direction if dipole fit unavailable.
        ref = best_vec
        if dip is not None:
            D = np.array(dip["D_vec"], dtype=float)
            amp = float(dip["D_amp"])
            if amp > 0:
                ref = D / amp
        cosang = X @ ref
        if not resume_scan:
            plt.figure(figsize=(5.6, 3.8))
            plt.axhline(0.0, color="k", linewidth=1, alpha=0.4)
            plt.errorbar(cosang, y, yerr=sig, fmt="o", ms=4, alpha=0.85)
            plt.xlabel("cos(angle to reference axis)")
            plt.ylabel("delta_s (fore - aft)")
            plt.title("Hemisphere scan: delta_s vs axis angle")
            plt.tight_layout()
            plt.savefig(out_dir / "figures" / "delta_s_vs_cosangle.png", dpi=200)
            plt.close()
    except Exception as e:
        fig_note = f"plot_failed: {type(e).__name__}: {e}"

    summary = {
        "mode": "hemisphere_scan",
        "n_axes_done": int(len(axis_rows)),
        "best_axis": asdict(best),
        "best_axis_radec_deg": {"ra": float(best_ra), "dec": float(best_dec)},
        "dipole_fit": (
            {**dip, "D_l_deg": float(l_hat), "D_b_deg": float(b_hat), "D_ra_deg": float(dip_ra), "D_dec_deg": float(dip_dec)}
            if dip is not None
            else None
        ),
        "figure_note": fig_note,
        "notes": [
            "This is a hemisphere-split reconstruction scan using Pantheon+ only (SN-only logmu inference).",
            "Dipole fit uses the hemisphere delta_s measurements: delta_s(axis) ~ D . n_axis (approx).",
            "Interpretation requires a null battery / look-elsewhere correction and robustness checks.",
        ],
    }
    if not resume_scan:
        _write_json(scan_summary_path, summary)

    # Null battery: compute a look-elsewhere calibrated p-value for the chosen statistic.
    n_null = int(args.null_reps)
    if n_null > 0:
        # Record both key statistics for convenience (only one is used as the primary "t" distribution).
        t_data_max_abs_z = float(np.max(np.abs([r.z_score for r in axis_rows]))) if axis_rows else float("nan")
        t_data_dipole_T = float(dip["T_amp"]) if (dip is not None and dip.get("T_amp") is not None) else float("nan")

        t_data = t_data_max_abs_z if str(args.null_stat) == "max_abs_z" else t_data_dipole_T
        # Lightweight, resumable null battery: store only per-replicate stats to avoid exploding file counts.
        null_dir = out_dir / "null_reps"
        null_dir.mkdir(parents=True, exist_ok=True)

        # Write the axis grid once so per-rep arrays have an unambiguous index convention.
        axis_grid_path = out_dir / "axis_grid.json"
        if not axis_grid_path.exists():
            try:
                axis_grid = {
                    "axes_nside": int(args.axes_nside),
                    "axes_nest": bool(args.axes_nest),
                    "skip_antipodes": bool(getattr(args, "skip_antipodes", False)),
                    "axis_pix": [int(t.axis_pix) for t in tasks],
                    "axis_l_deg": [float(t.axis_l_deg) for t in tasks],
                    "axis_b_deg": [float(t.axis_b_deg) for t in tasks],
                    "axis_vec": [[float(t.axis_vec[0]), float(t.axis_vec[1]), float(t.axis_vec[2])] for t in tasks],
                }
                _write_json_atomic(axis_grid_path, axis_grid)
            except Exception:
                # If multiple shards race, or if we can't write for some reason, proceed anyway.
                pass

        null_axis_jobs = int(getattr(args, "null_axis_jobs", 0))
        if null_axis_jobs > 1:
            import multiprocessing as mp

            ctx = mp.get_context("fork")

        rep_start = int(getattr(args, "null_rep_start", 0))
        rep_end_arg = getattr(args, "null_rep_end", None)
        rep_end = int(n_null) if rep_end_arg is None else int(rep_end_arg)
        rep_start = max(0, min(rep_start, int(n_null)))
        rep_end = max(0, min(rep_end, int(n_null)))
        if rep_end < rep_start:
            raise ValueError("null-rep-end must be >= null-rep-start.")
        full_range = (rep_start == 0) and (rep_end == int(n_null))

        def _finalize_null_summary() -> None:
            # Each rep file stores: t (primary stat), and also max_abs_z/dipole_T when available.
            ts: list[float] = []
            ts_max_abs_z: list[float] = []
            ts_dipole_T: list[float] = []
            n_present = 0
            for k in range(int(n_null)):
                rp = null_dir / f"rep{int(k):04d}.json"
                if not rp.exists():
                    continue
                try:
                    d = json.loads(rp.read_text(encoding="utf-8"))
                    ts.append(float(d["t"]))
                    if "max_abs_z" in d and d["max_abs_z"] is not None:
                        ts_max_abs_z.append(float(d["max_abs_z"]))
                    if "dipole_T" in d and d["dipole_T"] is not None:
                        ts_dipole_T.append(float(d["dipole_T"]))
                    n_present += 1
                except Exception:
                    continue
            arr = np.asarray(ts, dtype=float)
            ok = np.isfinite(arr)
            p_value = float(np.mean(arr[ok] >= float(t_data))) if (np.isfinite(t_data) and np.any(ok)) else None

            # Secondary p-values for convenience.
            def _p_one_sided(samples: list[float], t_obs: float) -> float | None:
                a = np.asarray(samples, dtype=float)
                ok2 = np.isfinite(a)
                if not (np.isfinite(t_obs) and np.any(ok2)):
                    return None
                return float(np.mean(a[ok2] >= float(t_obs)))

            p_max_abs_z = _p_one_sided(ts_max_abs_z, t_data_max_abs_z)
            p_dipole_T = _p_one_sided(ts_dipole_T, t_data_dipole_T)
            _write_json_atomic(
                out_dir / "null_summary.json",
                {
                    "null_reps": int(n_null),
                    "null_mode": str(args.null_mode),
                    "null_stat": str(args.null_stat),
                    "t_data": float(t_data) if np.isfinite(t_data) else None,
                    "t_data_by_stat": {
                        "max_abs_z": float(t_data_max_abs_z) if np.isfinite(t_data_max_abs_z) else None,
                        "dipole_T": float(t_data_dipole_T) if np.isfinite(t_data_dipole_T) else None,
                    },
                    "t_null_mean": float(np.mean(arr[ok])) if np.any(ok) else None,
                    "t_null_std": float(np.std(arr[ok], ddof=1)) if np.sum(ok) > 1 else None,
                    "p_value_one_sided": p_value,
                    "p_value_one_sided_by_stat": {
                        "max_abs_z": p_max_abs_z,
                        "dipole_T": p_dipole_T,
                    },
                    "n_present": int(n_present),
                    "n_missing": int(int(n_null) - int(n_present)),
                    "note": (
                        "Look-elsewhere calibration under the same scan procedure. "
                        "p_value is meaningful only when n_present == null_reps."
                    ),
                },
            )

        if bool(getattr(args, "null_finalize_only", False)):
            _log(f"[null] finalize-only: reading {null_dir} and writing null_summary.json")
            _finalize_null_summary()
            return 0

        if rep_start != 0 or rep_end != int(n_null):
            _log(f"[null] shard reps {rep_start}:{rep_end} of {n_null} (resume-friendly)")
        else:
            _log(f"[null] running full null battery reps 0:{n_null}")

        pending_reps = [int(i) for i in range(rep_start, rep_end) if not (null_dir / f"rep{int(i):04d}.json").exists()]
        if not pending_reps:
            _log("[null] no pending reps in requested shard; all rep files already present.")
        else:
            n_axes = int(len(tasks))
            axis_vec_by_ord = [[float(t.axis_vec[0]), float(t.axis_vec[1]), float(t.axis_vec[2])] for t in tasks]
            rep_rows: dict[int, dict[str, Any]] = {
                int(i): {
                    "delta_s": [None] * n_axes,
                    "sigma_delta_s": [None] * n_axes,
                    "z_scores": [None] * n_axes,
                    "vecs_ok": [],
                    "deltas_ok": [],
                    "sigs_ok": [],
                    "zs_ok": [],
                }
                for i in pending_reps
            }

            global _NULL_AXIS_GLOBAL
            if null_axis_jobs > 1:
                # Build all null sky permutations once, then fan out all (rep,axis) tasks
                # through a single long-lived worker pool to avoid per-rep pool churn.
                rep_to_stack = np.full((int(n_null),), -1, dtype=np.int64)
                v_null_stack = np.empty((int(len(pending_reps)), int(v.shape[0]), 3), dtype=float)
                for stack_idx, i in enumerate(pending_reps):
                    rng_i = np.random.default_rng(int(args.seed) + 12345 + int(i))
                    if str(args.null_mode) == "shuffle_all":
                        perm = rng_i.permutation(v.shape[0])
                    else:
                        perm = _permute_positions_within_survey(idsurvey, rng_i)
                    v_null_stack[int(stack_idx)] = v[perm]
                    rep_to_stack[int(i)] = int(stack_idx)

                _NULL_AXIS_GLOBAL = {
                    "v_null_stack": v_null_stack,
                    "rep_to_stack": rep_to_stack,
                    "z": z,
                    "m": m,
                    "cov": cov,
                    "z_edges": z_edges,
                    "group_key": group_key,
                    "args": args,
                    "base_seed": int(args.seed),
                    "z_min": z_min,
                    "z_max": z_max,
                    "mu_procs": int(mu_procs),
                    "quiet": True,
                }
                tasks_all: list[_NullAxisTask] = []
                for i in pending_reps:
                    for j, t in enumerate(tasks):
                        tasks_all.append(
                            _NullAxisTask(
                                rep=int(i),
                                axis_ord=int(j),
                                axis_pix=int(t.axis_pix),
                                axis_l_deg=float(t.axis_l_deg),
                                axis_b_deg=float(t.axis_b_deg),
                                axis_vec=t.axis_vec,
                            )
                        )
                _log(f"[null] pooled axis-worker run: reps={len(pending_reps)} axes={n_axes} tasks={len(tasks_all)} jobs={null_axis_jobs}")
                with ctx.Pool(processes=int(null_axis_jobs)) as pool:
                    for out in pool.imap_unordered(_null_axis_worker, tasks_all, chunksize=1):
                        if out is None:
                            continue
                        rep_i, axis_ord, d_s, s_s, z_s = out
                        if rep_i not in rep_rows:
                            continue
                        if not (np.isfinite(float(d_s)) and np.isfinite(float(s_s)) and np.isfinite(float(z_s))):
                            continue
                        row_i = rep_rows[int(rep_i)]
                        j = int(axis_ord)
                        row_i["delta_s"][j] = float(d_s)
                        row_i["sigma_delta_s"][j] = float(s_s)
                        row_i["z_scores"][j] = float(z_s)
                        row_i["vecs_ok"].append(axis_vec_by_ord[j])
                        row_i["deltas_ok"].append(float(d_s))
                        row_i["sigs_ok"].append(float(s_s))
                        row_i["zs_ok"].append(float(z_s))
            else:
                # Serial fallback (low-core / debug mode): compute each rep independently.
                for i in pending_reps:
                    rng_i = np.random.default_rng(int(args.seed) + 12345 + int(i))
                    if str(args.null_mode) == "shuffle_all":
                        perm = rng_i.permutation(v.shape[0])
                    else:
                        perm = _permute_positions_within_survey(idsurvey, rng_i)
                    v_null = v[perm]
                    _NULL_AXIS_GLOBAL = {
                        "v_null": v_null,
                        "z": z,
                        "m": m,
                        "cov": cov,
                        "z_edges": z_edges,
                        "group_key": group_key,
                        "args": args,
                        "base_seed": int(args.seed),
                        "z_min": z_min,
                        "z_max": z_max,
                        "mu_procs": int(mu_procs),
                        "quiet": False,
                    }
                    for j, t in enumerate(tasks):
                        out = _null_axis_worker(
                            _NullAxisTask(
                                rep=int(i),
                                axis_ord=int(j),
                                axis_pix=int(t.axis_pix),
                                axis_l_deg=float(t.axis_l_deg),
                                axis_b_deg=float(t.axis_b_deg),
                                axis_vec=t.axis_vec,
                            )
                        )
                        if out is None:
                            continue
                        rep_i, axis_ord, d_s, s_s, z_s = out
                        if not (np.isfinite(float(d_s)) and np.isfinite(float(s_s)) and np.isfinite(float(z_s))):
                            continue
                        row_i = rep_rows[int(rep_i)]
                        jj = int(axis_ord)
                        row_i["delta_s"][jj] = float(d_s)
                        row_i["sigma_delta_s"][jj] = float(s_s)
                        row_i["z_scores"][jj] = float(z_s)
                        row_i["vecs_ok"].append(axis_vec_by_ord[jj])
                        row_i["deltas_ok"].append(float(d_s))
                        row_i["sigs_ok"].append(float(s_s))
                        row_i["zs_ok"].append(float(z_s))

            _NULL_AXIS_GLOBAL = None

            for done, i in enumerate(pending_reps, start=1):
                row_i = rep_rows[int(i)]
                zs_ok = [float(x) for x in row_i["zs_ok"]]
                deltas_ok = [float(x) for x in row_i["deltas_ok"]]
                sigs_ok = [float(x) for x in row_i["sigs_ok"]]
                vecs_ok = [list(vv) for vv in row_i["vecs_ok"]]

                t_max_abs_z = float(np.max(np.abs(zs_ok))) if zs_ok else float("nan")

                dip_i = None
                t_dipole_T: float | None = None
                if len(vecs_ok) >= 3:
                    try:
                        dip_i = _fit_dipole_from_deltas(
                            np.asarray(vecs_ok, dtype=float),
                            np.asarray(deltas_ok, dtype=float),
                            np.asarray(sigs_ok, dtype=float),
                        )
                        t0 = dip_i.get("T_amp")
                        t_dipole_T = float(t0) if t0 is not None and np.isfinite(float(t0)) else None
                    except Exception:
                        dip_i = None
                        t_dipole_T = None

                if str(args.null_stat) == "dipole_T":
                    t_i = float(t_dipole_T) if t_dipole_T is not None else float("nan")
                else:
                    t_i = float(t_max_abs_z)

                _write_json_atomic(
                    null_dir / f"rep{int(i):04d}.json",
                    {
                        "rep": int(i),
                        "t": float(t_i),
                        "t_stat": str(args.null_stat),
                        "n_axes_ok": int(len(zs_ok)),
                        "max_abs_z": float(t_max_abs_z) if np.isfinite(t_max_abs_z) else None,
                        "dipole_T": float(t_dipole_T) if t_dipole_T is not None else None,
                        "dipole_fit": dip_i,
                        "z_scores": row_i["z_scores"],
                        "delta_s": row_i["delta_s"],
                        "sigma_delta_s": row_i["sigma_delta_s"],
                    },
                )

                if done % 10 == 0 or done == int(len(pending_reps)):
                    _log(f"[null] rep {int(i)+1}/{n_null} t={t_i:.3f} (ok_axes={len(zs_ok)}/{len(tasks)})")

        if full_range:
            _finalize_null_summary()
        else:
            _write_json_atomic(
                out_dir / f"null_shard_{rep_start:04d}_{rep_end:04d}.json",
                {
                    "null_reps_total": int(n_null),
                    "rep_start": int(rep_start),
                    "rep_end_exclusive": int(rep_end),
                    "note": "This is a shard partial completion marker; run --null-finalize-only to write null_summary.json once all shards finish.",
                },
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
