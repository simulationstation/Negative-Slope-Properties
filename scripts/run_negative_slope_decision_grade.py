#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.cosmology import build_background_from_H_grid
from entropy_horizon_recon.decision_grade import slope_stat_from_posterior, wilson_interval
from entropy_horizon_recon.ingest import load_bao, load_chronometers, load_pantheon_plus
from entropy_horizon_recon.inversion import infer_logmu_forward
from entropy_horizon_recon.likelihoods import BaoLogLike, ChronometerLogLike, SNLogLike, bin_sn_loglike
from entropy_horizon_recon.sbc import run_sbc_prior_truth, uniformity_pvalues


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")
        f.flush()


def _parse_csv_floats(text: str) -> list[float]:
    vals = [float(t.strip()) for t in str(text).split(",") if t.strip()]
    if not vals:
        raise ValueError("Expected at least one float value.")
    return vals


def _dense_domain_zmax(
    z: np.ndarray,
    *,
    z_min: float,
    z_max_cap: float,
    bin_width: float,
    min_per_bin: int,
) -> float:
    z = np.asarray(z, dtype=float)
    z = z[(z >= z_min) & (z <= z_max_cap)]
    if z.size == 0:
        raise ValueError("No SN redshifts in selected z-range.")
    edges = np.arange(z_min, z_max_cap + bin_width, bin_width)
    counts, _ = np.histogram(z, bins=edges)
    ok = counts >= int(min_per_bin)
    if not np.any(ok):
        return float(z_min + bin_width)
    last_good = int(np.where(ok)[0].max())
    return float(edges[last_good + 1])


def _logmeanexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    m = float(np.max(x))
    return float(m + np.log(np.mean(np.exp(x - m))))


def _threshold_for_alpha(null_stats: np.ndarray, alpha: float) -> tuple[float, int]:
    x = np.sort(np.asarray(null_stats, dtype=float))
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite null statistics.")
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")
    cands = np.unique(x)
    thr = float(np.nextafter(cands[0], -np.inf))
    k = 0
    for t in cands:
        kk = int(np.sum(x <= float(t)))
        if (kk / x.size) <= alpha + 1e-15:
            thr = float(t)
            k = kk
        else:
            break
    return thr, k


def _extract_scalar_stats(res: dict[str, Any], *, key: str = "scar_s_post_mean") -> np.ndarray:
    reps = res.get("replicates", [])
    vals: list[float] = []
    for row in reps:
        if key not in row:
            continue
        v = float(row[key])
        if np.isfinite(v):
            vals.append(v)
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        raise RuntimeError(f"No finite replicate values for key '{key}'.")
    return arr


def _stats(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "p16": float(np.percentile(arr, 16.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p84": float(np.percentile(arr, 84.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _mean_ci95(arr: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(x))
    if x.size == 1:
        return m, m, m
    se = float(np.std(x, ddof=1) / np.sqrt(float(x.size)))
    half = 1.959963984540054 * se
    return m, float(m - half), float(m + half)


def _to_float(x: Any, *, default: float = float("nan")) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(y):
        return float(default)
    return float(y)


def _log_stage(runtime_log: Path | None, *, stage: str, event: str, **extra: Any) -> None:
    if runtime_log is None:
        return
    rec = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "stage": str(stage),
        "event": str(event),
    }
    rec.update(extra)
    _append_jsonl(runtime_log, rec)


@dataclass(frozen=True)
class DataStack:
    constants: PhysicalConstants
    z_min: float
    z_max: float
    z_grid: np.ndarray
    x_knots: np.ndarray
    x_grid: np.ndarray
    sn_like_bin: SNLogLike
    cc_like: ChronometerLogLike
    bao_likes: list[BaoLogLike]
    holdout_bao: BaoLogLike | None


def _load_data_stack(args: argparse.Namespace) -> DataStack:
    constants = PhysicalConstants()
    paths = DataPaths(repo_root=Path.cwd())
    sn = load_pantheon_plus(
        paths=paths,
        cov_kind=str(args.sn_cov_kind),
        subset="cosmology",
        z_column=str(args.sn_z_column),
    )
    z_max = _dense_domain_zmax(
        sn.z,
        z_min=float(args.z_min),
        z_max_cap=float(args.z_max_cap),
        bin_width=float(args.sn_bin_width),
        min_per_bin=int(args.sn_min_per_bin),
    )
    sn_like = SNLogLike.from_pantheon(sn, z_min=float(args.z_min), z_max=float(z_max))
    z_edges = np.arange(float(args.z_min), float(z_max) + float(args.sn_like_bin_width), float(args.sn_like_bin_width))
    sn_like_bin = bin_sn_loglike(sn_like, z_edges=z_edges, min_per_bin=int(args.sn_like_min_per_bin))

    cc = load_chronometers(paths=paths, variant="BC03_all")
    cc_like = ChronometerLogLike.from_data(cc, z_min=float(args.z_min), z_max=float(z_max))

    bao_likes: list[BaoLogLike] = []
    for dataset in ("sdss_dr12_consensus_bao", "sdss_dr16_lrg_bao_dmdh", "desi_2024_bao_all"):
        bao = load_bao(paths=paths, dataset=dataset)
        try:
            bl = BaoLogLike.from_data(
                bao,
                dataset=dataset,
                constants=constants,
                z_min=float(args.z_min),
                z_max=float(z_max),
            )
        except ValueError:
            continue
        bao_likes.append(bl)
    if not bao_likes:
        raise RuntimeError("No BAO datasets remain in the selected z-domain.")

    holdout_name = str(args.holdout_probe).strip()
    holdout_mode = holdout_name.lower()
    holdout = None
    if holdout_name and holdout_mode not in {"all", "none", "off"}:
        for bl in bao_likes:
            if bl.dataset == holdout_name:
                holdout = bl
                break
        if holdout is None:
            raise ValueError(f"Requested holdout probe '{holdout_name}' not found among BAO datasets.")

    z_grid = np.linspace(0.0, float(z_max), int(args.n_grid))
    h0_guess = 70.0
    om0_guess = 0.3
    h_zmax_guess = h0_guess * np.sqrt(om0_guess * (1.0 + float(z_max)) ** 3 + (1.0 - om0_guess))
    x_min_guess = float(2.0 * np.log(h0_guess / h_zmax_guess))
    x_min = float(2.0 * x_min_guess)
    x_knots = np.linspace(1.25 * x_min, 0.0, int(args.mu_knots))
    x_grid = np.linspace(x_min, 0.0, int(args.mu_grid))
    return DataStack(
        constants=constants,
        z_min=float(args.z_min),
        z_max=float(z_max),
        z_grid=z_grid,
        x_knots=x_knots,
        x_grid=x_grid,
        sn_like_bin=sn_like_bin,
        cc_like=cc_like,
        bao_likes=bao_likes,
        holdout_bao=holdout,
    )


def _base_infer_kwargs(stack: DataStack, args: argparse.Namespace, *, seed: int, bao_likes: list[BaoLogLike] | None = None) -> dict[str, Any]:
    use_bao = list(stack.bao_likes if bao_likes is None else bao_likes)
    # Keep walkers valid for all mandatory real-data variants (free/fixedOm/residual/curved).
    # Base ndim ~= n_mu_knots + 6; residual adds 5 knots; curved adds one Ω_k parameter.
    min_walkers = 2 * (int(args.mu_knots) + 12)
    return {
        "z_grid": stack.z_grid,
        "x_knots": stack.x_knots,
        "x_grid": stack.x_grid,
        "sn_z": stack.sn_like_bin.z,
        "sn_m": stack.sn_like_bin.m,
        "sn_cov": stack.sn_like_bin.cov,
        "sn_marg": str(args.sn_marg),
        "sn_mstep_z": float(args.sn_mstep_z),
        "cc_z": stack.cc_like.z,
        "cc_H": stack.cc_like.H,
        "cc_sigma_H": stack.cc_like.sigma_H,
        "bao_likes": use_bao,
        "fsbao_likes": [],
        "constants": stack.constants,
        "sampler_kind": str(args.mu_sampler),
        "pt_ntemps": int(args.pt_ntemps),
        "pt_tmax": float(args.pt_tmax) if args.pt_tmax is not None else None,
        "n_walkers": max(int(min_walkers), int(args.mu_walkers)),
        "n_steps": int(args.mu_steps),
        "n_burn": int(args.mu_burn),
        "seed": int(seed),
        "n_processes": int(args.mu_procs),
        "n_draws": int(args.mu_draws),
        "omega_m0_prior": (float(args.omega_m0_min), float(args.omega_m0_max)),
        "sigma_cc_jit_scale": float(args.sigma_cc_jit_scale),
        "sigma_sn_jit_scale": float(args.sigma_sn_jit_scale),
        "sigma_d2_scale": float(args.sigma_d2_scale),
        "logmu_knot_scale": float(args.logmu_knot_scale),
        "max_rss_mb": float(args.max_rss_mb) if args.max_rss_mb > 0 else None,
        "progress": bool(args.progress),
    }


def _holdout_predictive_loglikes(post, holdout_like: BaoLogLike, *, constants: PhysicalConstants) -> np.ndarray:
    r_d = np.asarray(post.params["r_d_Mpc"], dtype=float)
    out = np.empty(post.H_samples.shape[0], dtype=float)
    for j in range(out.size):
        bg = build_background_from_H_grid(post.z_grid, post.H_samples[j], constants=constants)
        y_model = holdout_like.predict(bg, r_d_Mpc=float(r_d[j]))
        out[j] = float(holdout_like.loglike(y_model))
    return out


def _run_holdout_check(
    stack: DataStack,
    args: argparse.Namespace,
    *,
    seed: int,
    runtime_log: Path | None = None,
) -> dict[str, Any]:
    probe = str(args.holdout_probe).strip()
    probe_norm = probe.lower()
    if probe_norm in {"", "none", "off"}:
        return {"enabled": False}

    if probe_norm == "all":
        holdouts = list(stack.bao_likes)
    else:
        holdouts = [bl for bl in stack.bao_likes if bl.dataset == probe]
        if not holdouts:
            raise ValueError(f"Requested holdout probe '{probe}' not found among BAO datasets.")

    if not holdouts:
        return {"enabled": False}

    folds: list[dict[str, Any]] = []
    for i, holdout in enumerate(holdouts):
        train_bao = [bl for bl in stack.bao_likes if bl.dataset != holdout.dataset]
        if not train_bao:
            continue
        fold_seed = int(seed + 7000 + 100 * i)
        _log_stage(runtime_log, stage="holdout", event="fold_start", fold=i, holdout=holdout.dataset, seed=fold_seed)
        t0 = time.perf_counter()

        free_kwargs = _base_infer_kwargs(stack, args, seed=fold_seed + 1, bao_likes=train_bao)
        free_post = infer_logmu_forward(**free_kwargs)
        bh_kwargs = dict(free_kwargs)
        bh_kwargs["seed"] = int(fold_seed + 2)
        bh_kwargs["fixed_logmu_knots"] = "bh"
        bh_post = infer_logmu_forward(**bh_kwargs)

        ll_free = _holdout_predictive_loglikes(free_post, holdout, constants=stack.constants)
        ll_bh = _holdout_predictive_loglikes(bh_post, holdout, constants=stack.constants)
        lpd_free = _logmeanexp(ll_free)
        lpd_bh = _logmeanexp(ll_bh)

        slope_free = slope_stat_from_posterior(
            free_post,
            constants=stack.constants,
            n_logA=int(args.n_logA),
            weight_mode=str(args.weight_mode),
        )
        neg = slope_free.slope_draws < 0.0
        nonneg = ~neg
        lpd_neg = _logmeanexp(ll_free[neg]) if np.any(neg) else float("nan")
        lpd_nonneg = _logmeanexp(ll_free[nonneg]) if np.any(nonneg) else float("nan")
        dt = float(time.perf_counter() - t0)
        fold = {
            "fold": int(i),
            "seed_free": int(fold_seed + 1),
            "seed_bh": int(fold_seed + 2),
            "holdout_dataset": holdout.dataset,
            "n_train_bao": int(len(train_bao)),
            "lpd_free": float(lpd_free),
            "lpd_bh_fixed": float(lpd_bh),
            "delta_lpd_free_minus_bh": float(lpd_free - lpd_bh),
            "lpd_free_slope_lt0": float(lpd_neg),
            "lpd_free_slope_ge0": float(lpd_nonneg),
            "delta_lpd_lt0_minus_ge0": float(lpd_neg - lpd_nonneg) if np.isfinite(lpd_neg) and np.isfinite(lpd_nonneg) else float("nan"),
            "p_slope_lt0_train": float(np.mean(slope_free.slope_draws < 0.0)),
            "free_acceptance": _to_float(free_post.meta.get("acceptance_fraction_mean", np.nan)),
            "bh_acceptance": _to_float(bh_post.meta.get("acceptance_fraction_mean", np.nan)),
            "free_ess_min": _to_float(free_post.meta.get("ess_min", np.nan)),
            "bh_ess_min": _to_float(bh_post.meta.get("ess_min", np.nan)),
            "runtime_s": dt,
        }
        folds.append(fold)
        _log_stage(
            runtime_log,
            stage="holdout",
            event="fold_done",
            fold=i,
            holdout=holdout.dataset,
            runtime_s=dt,
            free_acceptance=fold["free_acceptance"],
            bh_acceptance=fold["bh_acceptance"],
        )

    if not folds:
        return {"enabled": False}

    d1 = np.asarray([float(f["delta_lpd_free_minus_bh"]) for f in folds], dtype=float)
    d2 = np.asarray([float(f["delta_lpd_lt0_minus_ge0"]) for f in folds], dtype=float)
    mean1, lo1, hi1 = _mean_ci95(d1)
    mean2, lo2, hi2 = _mean_ci95(d2)
    return {
        "enabled": True,
        "mode": "all_folds" if probe_norm == "all" else "single_fold",
        "n_folds": int(len(folds)),
        "folds": folds,
        "delta_lpd_free_minus_bh_mean": float(mean1),
        "delta_lpd_free_minus_bh_ci95": [float(lo1), float(hi1)],
        "delta_lpd_lt0_minus_ge0_mean": float(mean2),
        "delta_lpd_lt0_minus_ge0_ci95": [float(lo2), float(hi2)],
    }


def _combine_sbc_chunks(
    *,
    chunks: list[dict[str, Any]],
    seed: int,
    mu_truth_mode: str,
    mu_truth_slope: float,
    rank_bins: int = 20,
) -> dict[str, Any]:
    if not chunks:
        raise RuntimeError("No SBC chunks to combine.")
    chunks = list(chunks)
    chunks.sort(key=lambda x: int(x.get("chunk_start", 0)))

    total_n = int(sum(int(c.get("N", 0)) for c in chunks))
    n_draws = int(chunks[0].get("n_draws", 0))
    rank_keys = sorted({str(k) for c in chunks for k in (c.get("ranks", {}) or {}).keys()})
    ranks: dict[str, list[int]] = {k: [] for k in rank_keys}
    for c in chunks:
        rr = c.get("ranks", {}) or {}
        for k in rank_keys:
            ranks[k].extend(int(x) for x in rr.get(k, []))
    pvals = {k: uniformity_pvalues(np.asarray(v, dtype=int), n_draws=int(n_draws), n_bins=int(rank_bins)) for k, v in ranks.items()}

    cov_keys = sorted({str(k) for c in chunks for k in (c.get("coverage", {}) or {}).keys()})
    coverage: dict[str, dict[str, float]] = {}
    for k in cov_keys:
        num68 = 0.0
        num95 = 0.0
        den = 0.0
        for c in chunks:
            nn = float(c.get("N", 0))
            cc = (c.get("coverage", {}) or {}).get(k, {})
            num68 += nn * float(cc.get("cover_68", 0.0))
            num95 += nn * float(cc.get("cover_95", 0.0))
            den += nn
        if den > 0:
            coverage[k] = {"cover_68": float(num68 / den), "cover_95": float(num95 / den)}

    replicates: list[dict[str, Any]] = []
    idx = 0
    for c in chunks:
        for rep in (c.get("replicates", []) or []):
            row = dict(rep)
            row["i"] = int(idx)
            replicates.append(row)
            idx += 1

    acc_vals = np.asarray([_to_float(r.get("acceptance_fraction_mean", np.nan)) for r in replicates], dtype=float)
    lp_total = 0
    lp_bad = 0
    reason_counts: dict[str, int] = {}
    first_meta = chunks[0].get("meta", {}) or {}
    for c in chunks:
        lp = ((c.get("meta", {}) or {}).get("logprob", {}) or {})
        lp_total += int(lp.get("total_calls") or 0)
        lp_bad += int(lp.get("invalid_calls") or 0)
        for kk, vv in (lp.get("invalid_reason_counts") or {}).items():
            reason_counts[str(kk)] = int(reason_counts.get(str(kk), 0)) + int(vv)

    return {
        "N": int(total_n),
        "n_draws": int(n_draws),
        "ranks": {k: [int(x) for x in v] for k, v in ranks.items()},
        "pvalues": pvals,
        "coverage": coverage,
        "replicates": replicates,
        "meta": {
            "seed": int(seed),
            "mu_truth_mode": str(mu_truth_mode),
            "mu_truth_slope": float(mu_truth_slope),
            "rank_bins": int(rank_bins),
            "sampler_kind": str(first_meta.get("sampler_kind", "")),
            "pt_ntemps": int(first_meta.get("pt_ntemps", 0)),
            "pt_tmax": first_meta.get("pt_tmax", None),
            "n_walkers": int(first_meta.get("n_walkers", 0)),
            "n_steps": int(first_meta.get("n_steps", 0)),
            "n_burn": int(first_meta.get("n_burn", 0)),
            "n_processes": int(first_meta.get("n_processes", 0)),
            "acceptance_fraction_mean_mean": float(np.nanmean(acc_vals)) if acc_vals.size > 0 else float("nan"),
            "logprob": {
                "total_calls": int(lp_total),
                "invalid_calls": int(lp_bad),
                "invalid_rate": float(lp_bad) / float(lp_total) if lp_total > 0 else None,
                "invalid_reason_counts": reason_counts,
            },
            "chunk_count": int(len(chunks)),
        },
    }


def _run_sbc_set(
    *,
    stack: DataStack,
    args: argparse.Namespace,
    seed: int,
    n_rep: int,
    mu_truth_mode: str,
    mu_truth_slope: float = 0.0,
    out_path: Path | None = None,
    label: str,
    runtime_log: Path | None = None,
) -> dict[str, Any]:
    if out_path is None:
        raise ValueError("out_path is required for resumable SBC runs.")

    if bool(args.resume) and out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if int(existing.get("N", -1)) == int(n_rep):
                meta = existing.get("meta", {}) or {}
                if str(meta.get("mu_truth_mode", mu_truth_mode)) == str(mu_truth_mode) and abs(float(meta.get("mu_truth_slope", mu_truth_slope)) - float(mu_truth_slope)) < 1e-12:
                    print(f"[resume] using cached SBC set: {out_path}", flush=True)
                    return existing
        except Exception:
            pass

    ckpt_dir = out_path.parent / "checkpoints" / str(label)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    chunk_n = max(1, int(args.sbc_chunk_size))
    if chunk_n > int(n_rep):
        chunk_n = int(n_rep)

    def _load_chunk_if_valid(path: Path, start_i: int, end_i: int) -> dict[str, Any] | None:
        this_n = int(end_i - start_i)
        try:
            chunk = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if int(chunk.get("N", -1)) != int(this_n):
            return None
        meta = chunk.get("meta", {}) or {}
        if str(meta.get("mu_truth_mode", mu_truth_mode)) != str(mu_truth_mode):
            return None
        try:
            slope_meta = float(meta.get("mu_truth_slope", mu_truth_slope))
        except Exception:
            slope_meta = float(mu_truth_slope)
        if abs(float(slope_meta) - float(mu_truth_slope)) >= 1e-12:
            return None
        chunk["chunk_start"] = int(start_i)
        chunk["chunk_end"] = int(end_i)
        return chunk

    def _chunk_bounds_from_name(path: Path) -> tuple[int, int] | None:
        parts = path.stem.split("_")
        if len(parts) != 3 or parts[0] != "chunk":
            return None
        try:
            start_i = int(parts[1])
            end_i = int(parts[2])
        except ValueError:
            return None
        if start_i < 0 or end_i <= start_i:
            return None
        return int(start_i), int(end_i)

    chunks: list[dict[str, Any]] = []
    covered = np.zeros(int(n_rep), dtype=bool)
    if bool(args.resume):
        for chunk_path in sorted(ckpt_dir.glob("chunk_*.json")):
            bounds = _chunk_bounds_from_name(chunk_path)
            if bounds is None:
                continue
            start_i, end_i = bounds
            if start_i >= int(n_rep) or end_i > int(n_rep):
                continue
            if np.any(covered[start_i:end_i]):
                continue
            chunk = _load_chunk_if_valid(chunk_path, start_i, end_i)
            if chunk is None:
                continue
            covered[start_i:end_i] = True
            chunks.append(chunk)
            print(f"[resume] {label} chunk loaded ({start_i}:{end_i})", flush=True)
            _log_stage(
                runtime_log,
                stage=f"sbc:{label}",
                event="chunk_resume",
                start=start_i,
                end=end_i,
                path=str(chunk_path),
            )

    pending: list[tuple[int, int]] = []
    for window_start in range(0, int(n_rep), int(chunk_n)):
        window_end = min(int(n_rep), int(window_start + chunk_n))
        i = int(window_start)
        while i < int(window_end):
            while i < int(window_end) and bool(covered[i]):
                i += 1
            if i >= int(window_end):
                break
            start_i = int(i)
            while i < int(window_end) and not bool(covered[i]):
                i += 1
            pending.append((int(start_i), int(i)))

    n_chunks = int(len(pending))
    for ci, (start, end) in enumerate(pending):
        this_n = int(end - start)
        chunk_path = ckpt_dir / f"chunk_{start:06d}_{end:06d}.json"
        if bool(args.resume) and chunk_path.exists():
            chunk = _load_chunk_if_valid(chunk_path, int(start), int(end))
            if chunk is not None and not np.any(covered[int(start) : int(end)]):
                covered[int(start) : int(end)] = True
                chunks.append(chunk)
                print(f"[resume] {label} chunk {ci+1}/{n_chunks} loaded ({start}:{end})", flush=True)
                _log_stage(runtime_log, stage=f"sbc:{label}", event="chunk_resume", chunk=ci, start=start, end=end, path=str(chunk_path))
                continue

        worker_event_path = ckpt_dir / f"worker_events_{start:06d}_{end:06d}.jsonl" if bool(args.sbc_worker_events) else None
        if worker_event_path is not None:
            worker_event_path.parent.mkdir(parents=True, exist_ok=True)
            worker_event_path.write_text("", encoding="utf-8")
        print(f"[run] {label} chunk {ci+1}/{n_chunks} start ({start}:{end})", flush=True)
        _log_stage(
            runtime_log,
            stage=f"sbc:{label}",
            event="chunk_start",
            chunk=ci,
            start=start,
            end=end,
            seed=int(seed + start),
            progress_path=str(ckpt_dir / f"progress_{start:06d}_{end:06d}.jsonl"),
            worker_event_path=str(worker_event_path) if worker_event_path is not None else None,
        )
        t0 = time.perf_counter()
        out = run_sbc_prior_truth(
            seed=int(seed + start),
            N=int(this_n),
            z_grid=stack.z_grid,
            x_knots=stack.x_knots,
            x_grid=stack.x_grid,
            sn_z=stack.sn_like_bin.z,
            sn_cov=stack.sn_like_bin.cov,
            cc_z=stack.cc_like.z,
            cc_sigma_H=stack.cc_like.sigma_H,
            bao_templates=stack.bao_likes,
            constants=stack.constants,
            sampler_kind=str(args.mu_sampler),
            pt_ntemps=int(args.pt_ntemps),
            pt_tmax=float(args.pt_tmax) if args.pt_tmax is not None else None,
            n_walkers=max(2 * (int(args.mu_knots) + 6), int(args.mu_walkers)),
            n_steps=int(args.mu_steps),
            n_burn=int(args.mu_burn),
            n_draws=int(args.mu_draws),
            n_processes=int(args.sbc_procs),
            max_rss_mb=float(args.max_rss_mb) if args.max_rss_mb > 0 else None,
            omega_m0_prior=(float(args.omega_m0_min), float(args.omega_m0_max)),
            sigma_cc_jit_scale=float(args.sigma_cc_jit_scale),
            sigma_sn_jit_scale=float(args.sigma_sn_jit_scale),
            sigma_d2_scale=float(args.sigma_d2_scale),
            logmu_knot_scale=float(args.logmu_knot_scale),
            mu_truth_mode=str(mu_truth_mode),
            mu_truth_slope=float(mu_truth_slope),
            progress=bool(args.progress),
            progress_path=ckpt_dir / f"progress_{start:06d}_{end:06d}.jsonl",
            progress_every=int(args.sbc_progress_every),
            worker_event_path=worker_event_path,
            inner_quiet=bool(args.sbc_inner_quiet),
            limit_blas_threads=bool(args.sbc_limit_blas_threads),
            worker_heartbeat_s=float(args.sbc_worker_heartbeat_s),
            worker_tasks_per_child=int(args.sbc_worker_max_tasks),
        )
        dt = float(time.perf_counter() - t0)
        out["chunk_start"] = int(start)
        out["chunk_end"] = int(end)
        _write_json_atomic(chunk_path, out)
        covered[int(start) : int(end)] = True
        chunks.append(out)
        _log_stage(
            runtime_log,
            stage=f"sbc:{label}",
            event="chunk_done",
            chunk=ci,
            start=start,
            end=end,
            runtime_s=dt,
            acceptance=_to_float((out.get("meta", {}) or {}).get("acceptance_fraction_mean_mean", np.nan)),
        )

    if not np.all(covered):
        missing = int(np.sum(~covered))
        raise RuntimeError(f"SBC chunk coverage incomplete for '{label}': {missing} replicate(s) missing.")

    combined = _combine_sbc_chunks(
        chunks=chunks,
        seed=int(seed),
        mu_truth_mode=str(mu_truth_mode),
        mu_truth_slope=float(mu_truth_slope),
        rank_bins=20,
    )
    _write_json_atomic(out_path, combined)
    return combined


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Decision-grade negative-slope calibration and BH-null exceedance report bundle."
    )
    ap.add_argument("--out", default=None, help="Output REPORTS bundle directory.")
    ap.add_argument("--seed", type=int, default=20260222)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Resume from chunk checkpoints and cached stage outputs.")
    ap.add_argument("--sbc-chunk-size", type=int, default=50, help="Chunk size for resumable SBC simulation batches.")
    ap.add_argument("--sbc-progress-every", type=int, default=10, help="Progress cadence (replicates) for SBC chunks.")
    ap.add_argument("--sbc-worker-events", action=argparse.BooleanOptionalAction, default=True, help="Write per-worker JSONL events for SBC chunks.")
    ap.add_argument("--sbc-worker-heartbeat-s", type=float, default=30.0, help="Heartbeat cadence in seconds while a worker inference is running (0 disables).")
    ap.add_argument("--sbc-inner-quiet", action=argparse.BooleanOptionalAction, default=True, help="Suppress inner infer_logmu_forward worker logs during SBC.")
    ap.add_argument("--sbc-limit-blas-threads", action=argparse.BooleanOptionalAction, default=True, help="Limit BLAS/OpenMP threadpools to 1 inside SBC workers.")
    ap.add_argument("--sbc-worker-max-tasks", type=int, default=8, help="Pool worker recycle period for SBC (0 disables recycling).")

    ap.add_argument("--z-min", type=float, default=0.02)
    ap.add_argument("--z-max-cap", type=float, default=1.2)
    ap.add_argument("--sn-bin-width", type=float, default=0.1)
    ap.add_argument("--sn-min-per-bin", type=int, default=60)
    ap.add_argument("--sn-like-bin-width", type=float, default=0.05)
    ap.add_argument("--sn-like-min-per-bin", type=int, default=20)
    ap.add_argument("--sn-cov-kind", type=str, default="stat+sys", choices=["stat+sys", "statonly"])
    ap.add_argument("--sn-z-column", type=str, default="zHD", choices=["zHD", "zCMB", "zHEL"])
    ap.add_argument("--sn-marg", type=str, default="M", choices=["M", "Mz", "Mstep"])
    ap.add_argument("--sn-mstep-z", type=float, default=0.15)

    ap.add_argument("--n-grid", type=int, default=200)
    ap.add_argument("--mu-knots", type=int, default=8)
    ap.add_argument("--mu-grid", type=int, default=120)
    ap.add_argument("--n-logA", type=int, default=140)
    ap.add_argument("--weight-mode", type=str, default="variance", choices=["variance", "uniform"])

    ap.add_argument("--mu-sampler", type=str, default="emcee", choices=["emcee", "ptemcee"])
    ap.add_argument("--pt-ntemps", type=int, default=8)
    ap.add_argument("--pt-tmax", type=float, default=None)
    ap.add_argument("--mu-walkers", type=int, default=64)
    ap.add_argument("--mu-steps", type=int, default=1500)
    ap.add_argument("--mu-burn", type=int, default=500)
    ap.add_argument("--mu-draws", type=int, default=800)
    ap.add_argument("--mu-procs", type=int, default=1)
    ap.add_argument("--sbc-procs", type=int, default=1)
    ap.add_argument("--max-rss-mb", type=float, default=1536.0)
    ap.add_argument("--omega-m0-min", type=float, default=0.2)
    ap.add_argument("--omega-m0-max", type=float, default=0.4)
    ap.add_argument("--sigma-d2-scale", type=float, default=0.185)
    ap.add_argument("--logmu-knot-scale", type=float, default=1.0)
    ap.add_argument("--sigma-cc-jit-scale", type=float, default=10.0)
    ap.add_argument("--sigma-sn-jit-scale", type=float, default=0.05)

    ap.add_argument("--n-bh-null", type=int, default=2000, help="BH-null full end-to-end simulation count.")
    ap.add_argument("--n-sbc", type=int, default=200, help="Prior-truth SBC replicate count.")
    ap.add_argument("--n-bh-coverage", type=int, default=200, help="BH fixed-truth coverage replicate count.")
    ap.add_argument("--n-tpr", type=int, default=500, help="Injected-slope replicate count per effect size.")
    ap.add_argument("--slope-effects", default="-0.10,-0.20,-0.30", help="Comma-separated injected slope values.")
    ap.add_argument("--alpha-targets", default="0.05,0.01", help="Comma-separated FPR targets.")

    ap.add_argument("--omega-k0-prior", type=float, nargs=2, default=[-0.02, 0.02], metavar=("LOW", "HIGH"))
    ap.add_argument("--holdout-probe", type=str, default="all", help="BAO holdout mode: 'all', specific dataset name, or 'none'.")
    ap.add_argument("--power-min-tpr", type=float, default=0.5, help="Minimum plausible-effect TPR for smoking-gun stop rule.")
    ap.add_argument("--p-small-cut", type=float, default=0.01, help="One-sided BH-null p-value cut for smoking-gun stop rule.")
    ap.add_argument("--min-ess", type=float, default=100.0, help="Minimum ESS per mapping variant for convergence gate.")
    ap.add_argument("--acc-min", type=float, default=0.1, help="Minimum acceptance mean for convergence gate.")
    ap.add_argument("--acc-max", type=float, default=0.8, help="Maximum acceptance mean for convergence gate.")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / "REPORTS" / f"negative_slope_decision_grade_{_utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    runtime_log = out_dir / "runtime_events.jsonl"

    slope_effects = _parse_csv_floats(args.slope_effects)
    alpha_targets = _parse_csv_floats(args.alpha_targets)

    _log_stage(runtime_log, stage="run", event="start", seed=int(args.seed), out_dir=str(out_dir))
    _log_stage(
        runtime_log,
        stage="config",
        event="resolved",
        mu_procs=int(args.mu_procs),
        sbc_procs=int(args.sbc_procs),
        n_bh_null=int(args.n_bh_null),
        n_tpr=int(args.n_tpr),
        slope_effects=[float(x) for x in slope_effects],
        alpha_targets=[float(a) for a in alpha_targets],
        resume=bool(args.resume),
        sbc_chunk_size=int(args.sbc_chunk_size),
        sbc_worker_events=bool(args.sbc_worker_events),
        sbc_worker_heartbeat_s=float(args.sbc_worker_heartbeat_s),
        sbc_inner_quiet=bool(args.sbc_inner_quiet),
        sbc_limit_blas_threads=bool(args.sbc_limit_blas_threads),
        sbc_worker_max_tasks=int(args.sbc_worker_max_tasks),
    )

    t_stack = time.perf_counter()
    stack = _load_data_stack(args)
    _log_stage(runtime_log, stage="data", event="loaded", runtime_s=float(time.perf_counter() - t_stack), z_max=float(stack.z_max), n_bao=int(len(stack.bao_likes)))

    # Real data: baseline + mapping variants (mandatory).
    variant_cache = data_dir / "real_data_variants.json"
    variant_slope: dict[str, dict[str, Any]] = {}
    obs_stat = float("nan")
    bh_obs_value = float("nan")
    bh_obs_p_lt0 = float("nan")
    if bool(args.resume) and variant_cache.exists():
        cached = json.loads(variant_cache.read_text(encoding="utf-8"))
        variant_summary = cached.get("variant_slope", {}) or {}
        for name, row in variant_summary.items():
            draw_path = data_dir / f"variant_slope_draws_{name}.npy"
            if not draw_path.exists():
                raise RuntimeError(f"Missing cached draw file for variant '{name}': {draw_path}")
            variant_slope[name] = dict(row)
            variant_slope[name]["draws"] = np.load(draw_path)
        obs_stat = _to_float(cached.get("obs_stat", np.nan))
        bh_obs_value = _to_float(cached.get("bh_obs_value", np.nan))
        bh_obs_p_lt0 = _to_float(cached.get("bh_obs_p_lt0", np.nan))
        print(f"[resume] using cached real-data variants: {variant_cache}", flush=True)
    else:
        _log_stage(runtime_log, stage="real_data", event="start")
        t_real = time.perf_counter()
        base_kwargs = _base_infer_kwargs(stack, args, seed=int(args.seed) + 101)
        post_free = infer_logmu_forward(**base_kwargs)

        v0_kwargs = dict(base_kwargs)
        v0_kwargs["seed"] = int(args.seed) + 102
        v0_kwargs["omega_m0_fixed"] = 0.3
        post_v0 = infer_logmu_forward(**v0_kwargs)

        v2_kwargs = dict(base_kwargs)
        v2_kwargs["seed"] = int(args.seed) + 103
        v2_kwargs["use_residual"] = True
        post_v2 = infer_logmu_forward(**v2_kwargs)

        vk_kwargs = dict(base_kwargs)
        vk_kwargs["seed"] = int(args.seed) + 104
        vk_kwargs["omega_k0_prior"] = (float(args.omega_k0_prior[0]), float(args.omega_k0_prior[1]))
        post_vk = infer_logmu_forward(**vk_kwargs)

        variants = {
            "V1_free": post_free,
            "V0_fixedOm": post_v0,
            "V2_residual": post_v2,
            "V1_curved": post_vk,
        }
        for name, post in variants.items():
            stat = slope_stat_from_posterior(
                post,
                constants=stack.constants,
                n_logA=int(args.n_logA),
                weight_mode=str(args.weight_mode),
            )
            ess_min = _to_float(post.meta.get("ess_min", np.nan))
            acc = _to_float(post.meta.get("acceptance_fraction_mean", np.nan))
            conv = bool(np.isfinite(ess_min) and ess_min >= float(args.min_ess) and np.isfinite(acc) and float(args.acc_min) <= acc <= float(args.acc_max))
            variant_slope[name] = {
                "mean": float(stat.slope_mean),
                "sd": float(stat.slope_std),
                "p_slope_lt0": float(stat.p_slope_lt_0),
                "draws": np.asarray(stat.slope_draws, dtype=float),
                "ess_min": ess_min,
                "acceptance_fraction_mean": acc,
                "converged": conv,
            }
            np.save(data_dir / f"variant_slope_draws_{name}.npy", np.asarray(stat.slope_draws, dtype=float))
            _log_stage(
                runtime_log,
                stage="real_data",
                event="variant_done",
                variant=name,
                slope_mean=float(stat.slope_mean),
                p_slope_lt0=float(stat.p_slope_lt_0),
                ess_min=ess_min,
                acceptance=acc,
                converged=conv,
            )

        obs_stat = float(variant_slope["V1_free"]["mean"])

        # BH fit for observed-data comparison.
        bh_kwargs = dict(base_kwargs)
        bh_kwargs["seed"] = int(args.seed) + 105
        bh_kwargs["fixed_logmu_knots"] = "bh"
        post_bh_obs = infer_logmu_forward(**bh_kwargs)
        bh_obs_stat = slope_stat_from_posterior(
            post_bh_obs,
            constants=stack.constants,
            n_logA=int(args.n_logA),
            weight_mode=str(args.weight_mode),
        )
        bh_obs_value = float(bh_obs_stat.slope_mean)
        bh_obs_p_lt0 = float(bh_obs_stat.p_slope_lt_0)
        _write_json_atomic(
            variant_cache,
            {
                "obs_stat": float(obs_stat),
                "bh_obs_value": float(bh_obs_value),
                "bh_obs_p_lt0": float(bh_obs_p_lt0),
                "variant_slope": {
                    k: {
                        "mean": float(v["mean"]),
                        "sd": float(v["sd"]),
                        "p_slope_lt0": float(v["p_slope_lt0"]),
                        "ess_min": _to_float(v["ess_min"]),
                        "acceptance_fraction_mean": _to_float(v["acceptance_fraction_mean"]),
                        "converged": bool(v["converged"]),
                    }
                    for k, v in variant_slope.items()
                },
            },
        )
        _log_stage(runtime_log, stage="real_data", event="done", runtime_s=float(time.perf_counter() - t_real))

    # Decision-grade simulation batteries.
    _log_stage(runtime_log, stage="sbc", event="start")
    t_sbc = time.perf_counter()
    null_res = _run_sbc_set(
        stack=stack,
        args=args,
        seed=int(args.seed) + 5000,
        n_rep=int(args.n_bh_null),
        mu_truth_mode="bh",
        out_path=data_dir / "bh_null_replicates.json",
        label="bh_null",
        runtime_log=runtime_log,
    )
    null_stats = _extract_scalar_stats(null_res)

    tpr_stats_by_slope: dict[str, np.ndarray] = {}
    tpr_res_meta: dict[str, Any] = {}
    for i, s in enumerate(slope_effects):
        key = f"{s:.6f}"
        res_s = _run_sbc_set(
            stack=stack,
            args=args,
            seed=int(args.seed) + 6000 + 100 * i,
            n_rep=int(args.n_tpr),
            mu_truth_mode="linear_slope",
            mu_truth_slope=float(s),
            out_path=data_dir / f"slope_replicates_{i:02d}.json",
            label=f"slope_{i:02d}",
            runtime_log=runtime_log,
        )
        vals = _extract_scalar_stats(res_s)
        tpr_stats_by_slope[key] = vals
        tpr_res_meta[key] = {"seed": int(args.seed) + 6000 + 100 * i, "n_rep": int(vals.size)}

    sbc_res = _run_sbc_set(
        stack=stack,
        args=args,
        seed=int(args.seed) + 7000,
        n_rep=int(args.n_sbc),
        mu_truth_mode="prior",
        out_path=data_dir / "sbc_prior_replicates.json",
        label="sbc_prior",
        runtime_log=runtime_log,
    )
    bh_cov_res = _run_sbc_set(
        stack=stack,
        args=args,
        seed=int(args.seed) + 8000,
        n_rep=int(args.n_bh_coverage),
        mu_truth_mode="bh",
        out_path=data_dir / "bh_coverage_replicates.json",
        label="bh_coverage",
        runtime_log=runtime_log,
    )
    _log_stage(runtime_log, stage="sbc", event="done", runtime_s=float(time.perf_counter() - t_sbc))

    holdout_cache = data_dir / "holdout_summary.json"
    if bool(args.resume) and holdout_cache.exists():
        holdout = json.loads(holdout_cache.read_text(encoding="utf-8"))
        print(f"[resume] using cached holdout summary: {holdout_cache}", flush=True)
    else:
        holdout = _run_holdout_check(stack, args, seed=int(args.seed), runtime_log=runtime_log)
        _write_json_atomic(holdout_cache, holdout)

    # (1) Low-FPR threshold calibration with CI.
    cal_rows: list[dict[str, Any]] = []
    for alpha in alpha_targets:
        thr, k_fpr = _threshold_for_alpha(null_stats, float(alpha))
        n_null = int(null_stats.size)
        fpr = float(k_fpr / max(1, n_null))
        fpr_ci = wilson_interval(k=int(k_fpr), n=n_null, confidence=0.95)

        for slope_key, vals in sorted(tpr_stats_by_slope.items(), key=lambda kv: float(kv[0])):
            k_tpr = int(np.sum(vals <= thr))
            n_tpr = int(vals.size)
            tpr = float(k_tpr / max(1, n_tpr))
            tpr_ci = wilson_interval(k=int(k_tpr), n=n_tpr, confidence=0.95)
            cal_rows.append(
                {
                    "alpha_target": float(alpha),
                    "threshold": float(thr),
                    "injected_slope": float(slope_key),
                    "fpr": fpr,
                    "fpr_ci95": [float(fpr_ci[0]), float(fpr_ci[1])],
                    "tpr": tpr,
                    "tpr_ci95": [float(tpr_ci[0]), float(tpr_ci[1])],
                    "n_null": n_null,
                    "n_tpr": n_tpr,
                    "seed_null": int(args.seed) + 5000,
                    "seed_tpr": int(tpr_res_meta[slope_key]["seed"]),
                }
            )

    # (2) BH-null exceedance p-value.
    k_one = int(np.sum(null_stats <= obs_stat))
    n_one = int(null_stats.size)
    p_one_sided = float(k_one / max(1, n_one))
    p_one_ci = wilson_interval(k=k_one, n=n_one, confidence=0.95)
    null_center = float(np.median(null_stats))
    k_two = int(np.sum(np.abs(null_stats - null_center) >= abs(obs_stat - null_center)))
    n_two = int(null_stats.size)
    p_two_sided = float(k_two / max(1, n_two))
    p_two_ci = wilson_interval(k=k_two, n=n_two, confidence=0.95)

    # Figure (i): FPR/TPR vs threshold.
    all_stats = [null_stats] + [v for v in tpr_stats_by_slope.values()]
    lo = float(min(np.min(v) for v in all_stats))
    hi = float(max(np.max(v) for v in all_stats))
    grid = np.linspace(lo, hi, 300)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    fpr_curve = np.array([np.mean(null_stats <= t) for t in grid], dtype=float)
    ax1.plot(grid, fpr_curve, label="FPR (BH null)", lw=2.2, color="C0")
    for j, (skey, vals) in enumerate(sorted(tpr_stats_by_slope.items(), key=lambda kv: float(kv[0]))):
        tpr_curve = np.array([np.mean(vals <= t) for t in grid], dtype=float)
        ax1.plot(grid, tpr_curve, label=f"TPR (slope={float(skey):.2f})", lw=1.8, color=f"C{j+1}")
    for alpha in alpha_targets:
        thr, _ = _threshold_for_alpha(null_stats, float(alpha))
        ax1.axvline(thr, color="0.35", ls="--", lw=1.2, alpha=0.8)
    ax1.set_xlabel("Decision threshold on slope statistic")
    ax1.set_ylabel("Rate")
    ax1.set_title("FPR/TPR vs threshold")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=False, fontsize=9)
    fig1.tight_layout()
    fig1.savefig(fig_dir / "fpr_tpr_vs_threshold.png", dpi=220)
    plt.close(fig1)

    # Figure (ii): BH-null histogram + observed statistic marker.
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(null_stats, bins=40, alpha=0.85, color="#4E79A7", edgecolor="white")
    ax2.axvline(obs_stat, color="#E15759", lw=2.2, label="Observed statistic")
    ax2.set_xlabel("Slope statistic")
    ax2.set_ylabel("Count")
    ax2.set_title("BH-null distribution with observed statistic")
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False)
    txt = (
        f"p_one={p_one_sided:.4g} [{p_one_ci[0]:.4g},{p_one_ci[1]:.4g}]\n"
        f"p_two={p_two_sided:.4g} [{p_two_ci[0]:.4g},{p_two_ci[1]:.4g}]"
    )
    ax2.text(0.98, 0.98, txt, transform=ax2.transAxes, ha="right", va="top")
    fig2.tight_layout()
    fig2.savefig(fig_dir / "bh_null_histogram_with_obs.png", dpi=220)
    plt.close(fig2)

    # Figure (iii): real-data slope posteriors across mandatory mapping variants.
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for j, (name, row) in enumerate(variant_slope.items()):
        draws = np.asarray(row["draws"], dtype=float)
        ax3.hist(draws, bins=40, density=True, histtype="step", lw=1.8, label=name, color=f"C{j}")
    ax3.axvline(0.0, color="0.25", ls="--", lw=1.2)
    ax3.set_xlabel("Slope statistic")
    ax3.set_ylabel("Density")
    ax3.set_title("Real-data slope posteriors across mapping variants")
    ax3.grid(alpha=0.25)
    ax3.legend(frameon=False, fontsize=9)
    fig3.tight_layout()
    fig3.savefig(fig_dir / "real_data_slope_posteriors_mapping_variants.png", dpi=220)
    plt.close(fig3)

    # Figure (iv): held-out predictive score deltas (with fold uncertainty).
    holdout_fig_rel = "heldout_predictive_scores.png"
    if holdout.get("enabled", False) and holdout.get("folds"):
        folds = list(holdout.get("folds", []))
        labels = [str(f.get("holdout_dataset", f"fold{i}")) for i, f in enumerate(folds)]
        d_free_bh = np.asarray([float(f.get("delta_lpd_free_minus_bh", np.nan)) for f in folds], dtype=float)
        d_sign = np.asarray([float(f.get("delta_lpd_lt0_minus_ge0", np.nan)) for f in folds], dtype=float)
        x = np.arange(len(labels))
        fig4, ax4 = plt.subplots(figsize=(9, 5))
        ax4.axhline(0.0, color="0.35", lw=1.1, ls="--")
        ax4.bar(x - 0.18, d_free_bh, width=0.34, label="ΔLPD (free - BH fixed)", color="#4E79A7", alpha=0.9)
        ax4.bar(x + 0.18, d_sign, width=0.34, label="ΔLPD (slope<0 - slope>=0)", color="#F28E2B", alpha=0.9)
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels, rotation=15, ha="right")
        ax4.set_ylabel("Predictive score difference")
        ax4.set_title("Held-out predictive score deltas by fold")
        ax4.grid(alpha=0.25, axis="y")
        ax4.legend(frameon=False, fontsize=9)
        fig4.tight_layout()
        fig4.savefig(fig_dir / holdout_fig_rel, dpi=220)
        plt.close(fig4)
    else:
        holdout_fig_rel = ""

    # Stop rule.
    alpha05_rows = [r for r in cal_rows if abs(float(r["alpha_target"]) - 0.05) < 1e-12]
    best_power = float(max((float(r["tpr"]) for r in alpha05_rows), default=float("nan")))
    has_low_fpr = any(float(r["fpr"]) <= 0.05 + 1e-12 for r in alpha05_rows)
    has_power = np.isfinite(best_power) and (best_power >= float(args.power_min_tpr))
    p_small = bool(p_one_sided <= float(args.p_small_cut))
    variant_names_sorted = sorted(variant_slope.keys())
    variant_means = [float(variant_slope[k]["mean"]) for k in variant_names_sorted]
    ref_sign = float(np.sign(float(variant_slope.get("V1_free", {}).get("mean", np.nan))))
    mapping_sign_agreement = bool(np.isfinite(ref_sign) and ref_sign != 0.0 and all(float(np.sign(m)) == ref_sign for m in variant_means))
    mapping_convergence_pass = True
    for k in variant_names_sorted:
        row = variant_slope[k]
        acc = _to_float(row.get("acceptance_fraction_mean", np.nan))
        ess = _to_float(row.get("ess_min", np.nan))
        conv = bool(row.get("converged", np.isfinite(acc) and np.isfinite(ess) and ess >= float(args.min_ess)))
        mapping_convergence_pass = bool(mapping_convergence_pass and conv)
        row["converged"] = bool(conv)
    smoking_gun = bool(has_low_fpr and has_power and p_small and mapping_sign_agreement and mapping_convergence_pass)
    framing = "smoking_gun" if smoking_gun else "suggestive_preference"

    # Summary JSON.
    summary = {
        "created_utc": _utc_stamp(),
        "out_dir": str(out_dir),
        "config": {
            "seed": int(args.seed),
            "z_min": float(args.z_min),
            "z_max": float(stack.z_max),
            "alpha_targets": [float(a) for a in alpha_targets],
            "slope_effects": [float(x) for x in slope_effects],
            "n_bh_null": int(args.n_bh_null),
            "n_tpr": int(args.n_tpr),
            "n_sbc": int(args.n_sbc),
            "n_bh_coverage": int(args.n_bh_coverage),
            "mu_procs": int(args.mu_procs),
            "sbc_procs": int(args.sbc_procs),
            "resume": bool(args.resume),
            "sbc_chunk_size": int(args.sbc_chunk_size),
            "sbc_progress_every": int(args.sbc_progress_every),
            "sbc_worker_events": bool(args.sbc_worker_events),
            "sbc_worker_heartbeat_s": float(args.sbc_worker_heartbeat_s),
            "sbc_inner_quiet": bool(args.sbc_inner_quiet),
            "sbc_limit_blas_threads": bool(args.sbc_limit_blas_threads),
            "sbc_worker_max_tasks": int(args.sbc_worker_max_tasks),
        },
        "observed_statistic": {
            "scalar_name": "scar_s_post_mean",
            "obs_value": float(obs_stat),
            "bh_fixed_obs_value": float(bh_obs_value),
            "bh_fixed_p_slope_lt0": float(bh_obs_p_lt0),
            "p_slope_lt0_obs": float(variant_slope["V1_free"]["p_slope_lt0"]),
        },
        "calibration_table": cal_rows,
        "bh_null_exceedance": {
            "k_one_sided": int(k_one),
            "n_one_sided": int(n_one),
            "p_one_sided": float(p_one_sided),
            "p_one_sided_ci95": [float(p_one_ci[0]), float(p_one_ci[1])],
            "k_two_sided": int(k_two),
            "n_two_sided": int(n_two),
            "p_two_sided": float(p_two_sided),
            "p_two_sided_ci95": [float(p_two_ci[0]), float(p_two_ci[1])],
            "null_stats": _stats(null_stats),
        },
        "mapping_variants": {
            k: {
                "mean": float(v["mean"]),
                "sd": float(v["sd"]),
                "p_slope_lt0": float(v["p_slope_lt0"]),
                "n_draws": int(np.asarray(v["draws"]).size),
                "ess_min": _to_float(v.get("ess_min", np.nan)),
                "acceptance_fraction_mean": _to_float(v.get("acceptance_fraction_mean", np.nan)),
                "converged": bool(v.get("converged", False)),
            }
            for k, v in variant_slope.items()
        },
        "sbc_rank_uniformity": {
            "N": int(sbc_res["N"]),
            "pvalues": sbc_res.get("pvalues", {}),
            "coverage": sbc_res.get("coverage", {}),
        },
        "bh_coverage_check": {
            "N": int(bh_cov_res["N"]),
            "coverage": bh_cov_res.get("coverage", {}),
        },
        "heldout_prediction": holdout,
        "stop_rule": {
            "has_low_fpr_alpha05": bool(has_low_fpr),
            "best_tpr_alpha05": float(best_power),
            "power_min_tpr": float(args.power_min_tpr),
            "p_one_sided_small_cut": float(args.p_small_cut),
            "p_one_sided_is_small": bool(p_small),
            "mapping_sign_agreement": bool(mapping_sign_agreement),
            "mapping_convergence_pass": bool(mapping_convergence_pass),
            "min_ess": float(args.min_ess),
            "acc_range": [float(args.acc_min), float(args.acc_max)],
            "classification": framing,
        },
        "figures": {
            "fpr_tpr_vs_threshold": str((fig_dir / "fpr_tpr_vs_threshold.png").resolve()),
            "bh_null_hist_with_obs": str((fig_dir / "bh_null_histogram_with_obs.png").resolve()),
            "mapping_variant_slope_posteriors": str((fig_dir / "real_data_slope_posteriors_mapping_variants.png").resolve()),
            "heldout_scores": str((fig_dir / holdout_fig_rel).resolve()) if holdout_fig_rel else None,
        },
        "runtime_log_jsonl": str(runtime_log.resolve()),
    }
    _write_json_atomic(out_dir / "summary.json", summary)

    # Master report markdown.
    lines: list[str] = []
    lines.append("# Decision-Grade Negative-Slope Report")
    lines.append("")
    lines.append("## Calibration Table")
    lines.append("")
    lines.append("| alpha target | threshold | injected slope | FPR (95% CI) | TPR (95% CI) | N_sims | seeds |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|")
    for row in cal_rows:
        fpr_ci = row["fpr_ci95"]
        tpr_ci = row["tpr_ci95"]
        seeds = f"null={row['seed_null']}, tpr={row['seed_tpr']}"
        n_sims = f"null={row['n_null']}, tpr={row['n_tpr']}"
        lines.append(
            f"| {row['alpha_target']:.2f} | {row['threshold']:.6f} | {row['injected_slope']:.3f} | "
            f"{row['fpr']:.4f} [{fpr_ci[0]:.4f},{fpr_ci[1]:.4f}] | "
            f"{row['tpr']:.4f} [{tpr_ci[0]:.4f},{tpr_ci[1]:.4f}] | {n_sims} | {seeds} |"
        )
    lines.append("")
    lines.append("## BH-Null Exceedance")
    lines.append("")
    lines.append(f"- scalar statistic (`scar_s_post_mean`) observed: `{obs_stat:.6f}`")
    lines.append(f"- one-sided p: `{p_one_sided:.6g}` (95% CI: `[{p_one_ci[0]:.6g}, {p_one_ci[1]:.6g}]`, k/n=`{k_one}/{n_one}`)")
    lines.append(f"- two-sided p: `{p_two_sided:.6g}` (95% CI: `[{p_two_ci[0]:.6g}, {p_two_ci[1]:.6g}]`, k/n=`{k_two}/{n_two}`)")
    lines.append("")
    lines.append("## Stop Rule")
    lines.append("")
    lines.append(f"- classification: `{framing}`")
    lines.append(f"- low-FPR+power pass (alpha=0.05): `{bool(has_low_fpr and has_power)}`")
    lines.append(f"- p-value-small pass: `{p_small}`")
    lines.append(f"- mapping-sign agreement pass: `{mapping_sign_agreement}`")
    lines.append(f"- mapping-convergence pass: `{mapping_convergence_pass}`")
    lines.append("")
    lines.append("## Held-Out Prediction")
    lines.append("")
    if holdout.get("enabled", False):
        lines.append(f"- mode: `{holdout.get('mode', 'unknown')}` with `{int(holdout.get('n_folds', 0))}` fold(s)")
        lines.append(
            f"- mean ΔLPD (free - BH fixed): `{float(holdout.get('delta_lpd_free_minus_bh_mean', np.nan)):.6f}` "
            f"(95% CI: `[{float((holdout.get('delta_lpd_free_minus_bh_ci95') or [np.nan, np.nan])[0]):.6f}, "
            f"{float((holdout.get('delta_lpd_free_minus_bh_ci95') or [np.nan, np.nan])[1]):.6f}]`)"
        )
        lines.append(
            f"- mean ΔLPD (slope<0 - slope>=0): `{float(holdout.get('delta_lpd_lt0_minus_ge0_mean', np.nan)):.6f}` "
            f"(95% CI: `[{float((holdout.get('delta_lpd_lt0_minus_ge0_ci95') or [np.nan, np.nan])[0]):.6f}, "
            f"{float((holdout.get('delta_lpd_lt0_minus_ge0_ci95') or [np.nan, np.nan])[1]):.6f}]`)"
        )
    else:
        lines.append("- disabled")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("- `figures/fpr_tpr_vs_threshold.png`")
    lines.append("- `figures/bh_null_histogram_with_obs.png`")
    lines.append("- `figures/real_data_slope_posteriors_mapping_variants.png`")
    if holdout_fig_rel:
        lines.append(f"- `figures/{holdout_fig_rel}`")
    (out_dir / "master_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Raw arrays for reproducibility.
    np.save(out_dir / "data" / "null_stats.npy", np.asarray(null_stats, dtype=float))
    for key, vals in tpr_stats_by_slope.items():
        tag = key.replace("-", "m").replace(".", "p")
        np.save(out_dir / "data" / f"tpr_stats_slope_{tag}.npy", np.asarray(vals, dtype=float))
    for name, row in variant_slope.items():
        np.save(out_dir / "data" / f"variant_slope_draws_{name}.npy", np.asarray(row["draws"], dtype=float))

    _log_stage(runtime_log, stage="run", event="done", classification=str(framing))
    print(f"Wrote REPORTS bundle: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
