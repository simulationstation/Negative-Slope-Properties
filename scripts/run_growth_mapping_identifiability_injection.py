#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.growth import solve_growth_ode_muP
from entropy_horizon_recon.ingest_rsd_single_survey import load_rsd_single_survey
from entropy_horizon_recon.sirens import load_mu_forward_posterior, x_of_z_from_H


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _parse_csv_str(text: str) -> list[str]:
    vals = [t.strip() for t in str(text).split(",") if t.strip()]
    if not vals:
        raise ValueError("Expected at least one value.")
    return vals


def _parse_csv_floats(text: str) -> list[float]:
    vals = [float(t.strip()) for t in str(text).split(",") if t.strip()]
    if not vals:
        raise ValueError("Expected at least one float.")
    return vals


def _discover_run_dirs(grid_dir: Path) -> list[Path]:
    csv_path = grid_dir / "grid_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing grid_results.csv at {csv_path}")
    out: list[Path] = []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rd = (row.get("run_dir") or "").strip()
            if rd:
                out.append(Path(rd).resolve())
    uniq = sorted({p for p in out})
    if not uniq:
        raise RuntimeError("No run_dir discovered from grid_results.csv")
    return uniq


def _stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "p16": float(np.percentile(x, 16.0)),
        "p50": float(np.percentile(x, 50.0)),
        "p84": float(np.percentile(x, 84.0)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _logmeanexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    m = float(np.max(x))
    return float(m + np.log(np.mean(np.exp(x - m))))


def _parse_truth_spec(text: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for token in _parse_csv_str(text):
        if ":" not in token:
            raise ValueError(f"Invalid truth spec token: {token} (expected mode:nu)")
        mode_raw, nu_raw = token.split(":", 1)
        mode = mode_raw.strip().lower()
        if mode not in {"relative", "absolute"}:
            raise ValueError(f"Unknown truth mode '{mode}' in token '{token}'")
        nu = float(nu_raw.strip())
        out.append((mode, nu))
    if not out:
        raise ValueError("Empty truth spec.")
    return out


@dataclass(frozen=True)
class Scenario:
    dataset: str
    z: np.ndarray
    fs8_obs: np.ndarray
    cov: np.ndarray
    inv_cov: np.ndarray
    cinv_one: np.ndarray
    one_cinv_one: float


def _build_scenarios(repo_root: Path, datasets: list[str]) -> list[Scenario]:
    paths = DataPaths(repo_root=repo_root)
    out: list[Scenario] = []
    for ds in datasets:
        rsd = load_rsd_single_survey(paths=paths, dataset=ds)
        z = np.asarray(rsd.z, dtype=float)
        fs8 = np.asarray(rsd.fs8, dtype=float)
        cov = np.asarray(rsd.cov, dtype=float)
        inv_cov = np.linalg.inv(cov)
        one = np.ones(z.size, dtype=float)
        cinv_one = inv_cov @ one
        one_cinv_one = float(one @ cinv_one)
        out.append(
            Scenario(
                dataset=ds,
                z=z,
                fs8_obs=fs8,
                cov=cov,
                inv_cov=inv_cov,
                cinv_one=cinv_one,
                one_cinv_one=one_cinv_one,
            )
        )
    return out


@dataclass(frozen=True)
class PrecomputeChunkTask:
    run_dir: str
    run_label: str
    draw_idx: list[int]
    mode: str
    nu_grid: list[float]
    z_by_dataset: list[list[float]]
    nmax: int


def _compute_chunk(task: PrecomputeChunkTask) -> dict[str, Any]:
    post = load_mu_forward_posterior(task.run_dir)
    n_draw = len(task.draw_idx)
    n_nu = len(task.nu_grid)
    n_ds = len(task.z_by_dataset)
    pred = np.full((n_draw, n_nu, n_ds, int(task.nmax)), np.nan, dtype=float)
    used_draw_idx: list[int] = []
    n_skipped = 0

    for out_i, idx in enumerate(task.draw_idx):
        try:
            h0 = float(post.H0[int(idx)])
            om0 = float(post.omega_m0[int(idx)])
            ok0 = float(post.omega_k0[int(idx)])
            s80 = float(post.sigma8_0[int(idx)])
            hz = np.asarray(post.H_samples[int(idx)], dtype=float)

            xz = x_of_z_from_H(post.z_grid, hz, H0=h0, omega_k0=ok0)
            xmin, xmax = float(post.x_grid[0]), float(post.x_grid[-1])
            xz = np.clip(xz, xmin, xmax)
            logmu = np.interp(xz, post.x_grid, np.asarray(post.logmu_x_samples[int(idx)], dtype=float))
            mu = np.exp(logmu)
            if task.mode == "relative":
                base = mu / max(float(mu[0]), 1e-12)
            elif task.mode == "absolute":
                base = mu
            else:
                raise ValueError(f"Unknown mode: {task.mode}")
            base = np.clip(base, 1e-12, 1e12)
            log_base = np.log(base)

            for nu_i, nu in enumerate(task.nu_grid):
                if float(nu) == 0.0:
                    muP = np.ones_like(base)
                else:
                    muP = np.exp(float(nu) * log_base)
                muP = np.clip(muP, 1e-8, 1e8)
                sol = solve_growth_ode_muP(
                    z_grid=post.z_grid,
                    H_grid=hz,
                    H0=h0,
                    omega_m0=om0,
                    omega_k0=ok0,
                    muP_grid=muP,
                    muP_highz=1.0,
                )
                for ds_i, z_list in enumerate(task.z_by_dataset):
                    z_obs = np.asarray(z_list, dtype=float)
                    x_obs = -np.log1p(z_obs)
                    d_obs = np.interp(x_obs, sol.x_grid, sol.D)
                    f_obs = np.interp(x_obs, sol.x_grid, sol.f)
                    fs8_pred = f_obs * s80 * d_obs
                    pred[out_i, nu_i, ds_i, : z_obs.size] = fs8_pred
            used_draw_idx.append(int(idx))
        except Exception:
            n_skipped += 1
            continue

    valid_rows = np.isfinite(pred).all(axis=(1, 2, 3))
    pred_valid = pred[valid_rows]
    used_idx_valid = [used_draw_idx[i] for i, ok in enumerate(valid_rows) if ok]
    return {
        "run_label": task.run_label,
        "mode": task.mode,
        "draw_idx": used_idx_valid,
        "pred": pred_valid,
        "n_attempted": int(len(task.draw_idx)),
        "n_used": int(pred_valid.shape[0]),
        "n_skipped": int(n_skipped + (len(task.draw_idx) - len(used_draw_idx))),
    }


def _profile_chi2_with_offset(
    residual: np.ndarray,
    inv_cov: np.ndarray,
    cinv_one: np.ndarray,
    one_cinv_one: float,
    sigma_delta: float,
) -> np.ndarray:
    # residual shape: [n_draw, n_nu, n_point]
    tmp = np.einsum("dnk,kl->dnl", residual, inv_cov, optimize=True)
    chi0 = np.einsum("dnl,dnl->dn", tmp, residual, optimize=True)
    if sigma_delta <= 0.0:
        return chi0
    b = np.einsum("dnk,k->dn", residual, cinv_one, optimize=True)
    a = float(one_cinv_one + 1.0 / (sigma_delta * sigma_delta))
    return chi0 - (b * b) / a


def _evaluate_mode_scores(
    pred_mode: np.ndarray,
    y_by_dataset: list[np.ndarray],
    scenarios: list[Scenario],
    sigma_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    # pred_mode shape: [n_draw, n_nu, n_ds, nmax]
    n_draw, n_nu, n_ds, _ = pred_mode.shape
    chi2_total = np.zeros((n_draw, n_nu), dtype=float)
    for ds_i in range(n_ds):
        scen = scenarios[ds_i]
        n_pt = int(scen.z.size)
        pred = pred_mode[:, :, ds_i, :n_pt]
        y = y_by_dataset[ds_i][None, None, :]
        residual = y - pred
        chi_ds = _profile_chi2_with_offset(
            residual=residual,
            inv_cov=scen.inv_cov,
            cinv_one=scen.cinv_one,
            one_cinv_one=scen.one_cinv_one,
            sigma_delta=float(sigma_delta),
        )
        chi2_total += chi_ds

    score_by_nu = np.array([_logmeanexp(-0.5 * chi2_total[:, i]) for i in range(n_nu)], dtype=float)
    mean_dchi_by_nu = np.array([float(np.mean(chi2_total[:, i] - chi2_total[:, 0])) for i in range(n_nu)], dtype=float)
    return score_by_nu, mean_dchi_by_nu


def _build_report_lines(summary: dict[str, Any]) -> list[str]:
    lines = [
        "# Growth mapping identifiability injections",
        "",
        f"- Output: `{summary['out_dir']}`",
        f"- Truth spec: `{summary['truth_spec']}`",
        f"- Reps per truth: `{summary['reps_per_truth']}`",
        f"- Completed reps: `{summary['n_reps_done']}/{summary['n_reps_total']}`",
        f"- Datasets: `{', '.join(summary['datasets'])}`",
        f"- Nu grid: `{summary['nu_grid']}`",
        f"- Offset prior sigma: `{summary['offset_prior_sigma']}`",
        "",
        "## Confusion matrix",
        "",
    ]
    for truth_mode, row in summary["confusion"].items():
        parts = [f"{k}:{v}" for k, v in sorted(row.items())]
        lines.append(f"- truth `{truth_mode}` -> " + ", ".join(parts))
    lines.extend(
        [
            "",
            "## Accuracy",
            "",
            f"- overall accuracy: `{summary['accuracy']['overall']:.4f}`",
        ]
    )
    for mode, acc in sorted(summary["accuracy"]["by_truth"].items()):
        lines.append(f"- truth `{mode}` accuracy: `{acc:.4f}`")
    lines.extend(["", "## Delta score stats (relative - absolute)", ""])
    for mode, stats in sorted(summary["delta_score_stats_by_truth"].items()):
        lines.append(
            f"- `{mode}`: mean `{stats.get('mean', float('nan')):.6f}`, "
            f"p16/p50/p84 `{stats.get('p16', float('nan')):.6f}/"
            f"{stats.get('p50', float('nan')):.6f}/{stats.get('p84', float('nan')):.6f}`"
        )
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Confusion-matrix identifiability injections for relative vs absolute growth coupling mapping."
    )
    ap.add_argument("--grid-dir", default="outputs/hubble_tension_mg_transfer_map_full_v1_20260208_045425UTC")
    ap.add_argument("--run-dirs", default="", help="Optional comma list. If empty, discover from grid-dir.")
    ap.add_argument(
        "--datasets",
        default="sdss_dr12_consensus_fs,sdss_dr16_lrg_fsbao_dmdhfs8",
        help="Comma-separated RSD datasets (full scenarios only).",
    )
    ap.add_argument("--truth-spec", default="relative:0.7,absolute:0.0")
    ap.add_argument("--nu-grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--reps-per-truth", type=int, default=64)
    ap.add_argument("--draws-per-run", type=int, default=24)
    ap.add_argument("--offset-prior-sigma", type=float, default=0.03)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=20260208)
    ap.add_argument("--heartbeat-sec", type=float, default=60.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))
    grid_dir = Path(args.grid_dir).resolve()
    if str(args.run_dirs).strip():
        run_dirs = [Path(x).resolve() for x in _parse_csv_str(args.run_dirs)]
    else:
        run_dirs = _discover_run_dirs(grid_dir)
    datasets = _parse_csv_str(args.datasets)
    nu_grid = _parse_csv_floats(args.nu_grid)
    truth_spec = _parse_truth_spec(args.truth_spec)

    out_dir = Path(args.out) if args.out else Path("outputs") / f"growth_mapping_identifiability_{_utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    reps_dir = out_dir / "reps"
    partials_dir = out_dir / "partials"
    reps_dir.mkdir(parents=True, exist_ok=True)
    partials_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _build_scenarios(Path.cwd(), datasets)
    z_by_dataset = [[float(z) for z in s.z.tolist()] for s in scenarios]
    n_ds = len(scenarios)
    nmax = int(max(s.z.size for s in scenarios))
    n_nu = len(nu_grid)

    _write_json_atomic(
        out_dir / "manifest.json",
        {
            "created_utc": _utc_stamp(),
            "grid_dir": str(grid_dir),
            "run_dirs": [str(x) for x in run_dirs],
            "datasets": datasets,
            "truth_spec": [[m, n] for (m, n) in truth_spec],
            "nu_grid": nu_grid,
            "reps_per_truth": int(args.reps_per_truth),
            "draws_per_run": int(args.draws_per_run),
            "offset_prior_sigma": float(args.offset_prior_sigma),
            "workers": int(args.workers),
            "seed": int(args.seed),
        },
    )

    # Stage 1: precompute model prediction cache for both mapping modes.
    modes = sorted({m for (m, _nu) in truth_spec} | {"relative", "absolute"})
    precompute_tasks: list[PrecomputeChunkTask] = []
    for mode in modes:
        for r_i, run_dir in enumerate(run_dirs):
            post = load_mu_forward_posterior(run_dir)
            n_all = int(post.H0.size)
            n_take = int(min(max(1, int(args.draws_per_run)), n_all))
            draw_idx = rng.choice(n_all, size=n_take, replace=False).astype(int).tolist()
            precompute_tasks.append(
                PrecomputeChunkTask(
                    run_dir=str(run_dir),
                    run_label=run_dir.name,
                    draw_idx=draw_idx,
                    mode=mode,
                    nu_grid=[float(x) for x in nu_grid],
                    z_by_dataset=z_by_dataset,
                    nmax=nmax,
                )
            )

    pre_total = len(precompute_tasks)
    pre_done = 0
    last_hb = 0.0
    _write_json_atomic(
        out_dir / "progress.json",
        {
            "updated_utc": _utc_stamp(),
            "stage": "precompute",
            "precompute_done": 0,
            "precompute_total": int(pre_total),
            "done_reps": 0,
            "total_reps": int(len(truth_spec) * int(args.reps_per_truth)),
            "pct": 0.0,
        },
    )

    pre_rows: list[dict[str, Any]] = []
    if pre_total > 0:
        workers = int(min(max(1, int(args.workers)), pre_total))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut_to_task = {ex.submit(_compute_chunk, t): t for t in precompute_tasks}
            while fut_to_task:
                try:
                    completed = list(as_completed(list(fut_to_task.keys()), timeout=1.0))
                except TimeoutError:
                    completed = []
                for fut in completed:
                    task = fut_to_task.pop(fut)
                    try:
                        row = fut.result()
                    except Exception as e:
                        _write_json_atomic(
                            partials_dir / f"precompute_{task.mode}_{task.run_label}_error.json",
                            {
                                "updated_utc": _utc_stamp(),
                                "stage": "precompute",
                                "mode": task.mode,
                                "run_label": task.run_label,
                                "error": str(e),
                            },
                        )
                        pre_done += 1
                        continue
                    pre_rows.append(row)
                    pre_done += 1

                now = time.time()
                if (now - last_hb) >= float(args.heartbeat_sec) or pre_done == pre_total:
                    _write_json_atomic(
                        out_dir / "progress.json",
                        {
                            "updated_utc": _utc_stamp(),
                            "stage": "precompute",
                            "precompute_done": int(pre_done),
                            "precompute_total": int(pre_total),
                            "done_reps": 0,
                            "total_reps": int(len(truth_spec) * int(args.reps_per_truth)),
                            "pct": float(100.0 * pre_done / max(1, pre_total)),
                        },
                    )
                    print(
                        f"[heartbeat] precompute={pre_done}/{pre_total} "
                        f"({100.0 * pre_done / max(1, pre_total):.1f}%)",
                        flush=True,
                    )
                    last_hb = now

    # Build dense cache arrays by mode.
    cache: dict[str, dict[str, Any]] = {}
    for mode in modes:
        rows = [r for r in pre_rows if str(r["mode"]) == mode and int(r["n_used"]) > 0]
        if not rows:
            raise RuntimeError(f"No usable precompute rows for mode '{mode}'")
        preds = [np.asarray(r["pred"], dtype=float) for r in rows]
        pred_cat = np.concatenate(preds, axis=0)
        run_labels: list[str] = []
        draw_ids: list[int] = []
        for r in rows:
            run_labels.extend([str(r["run_label"])] * int(r["n_used"]))
            draw_ids.extend([int(x) for x in r["draw_idx"]])
        cache[mode] = {
            "pred": pred_cat,  # [n_draw, n_nu, n_ds, nmax]
            "run_label": run_labels,
            "draw_idx": draw_ids,
        }

    # Stage 2: identifiability injections (serial for stability, vectorized scoring).
    rep_tasks: list[tuple[str, float, int]] = []
    for truth_mode, truth_nu in truth_spec:
        for rep_i in range(int(args.reps_per_truth)):
            rep_tasks.append((truth_mode, float(truth_nu), int(rep_i)))

    done_rows: list[dict[str, Any]] = []
    pending: list[tuple[str, float, int]] = []
    if bool(args.resume):
        for truth_mode, truth_nu, rep_i in rep_tasks:
            rep_name = f"{truth_mode}_nu{truth_nu:.3f}_rep{rep_i:04d}".replace(".", "p").replace("-", "m")
            rep_path = reps_dir / f"{rep_name}.json"
            if rep_path.exists():
                done_rows.append(json.loads(rep_path.read_text(encoding="utf-8")))
            else:
                pending.append((truth_mode, truth_nu, rep_i))
    else:
        pending = list(rep_tasks)

    total_reps = len(rep_tasks)
    done_reps = len(done_rows)
    last_hb = 0.0
    _write_json_atomic(
        out_dir / "progress.json",
        {
            "updated_utc": _utc_stamp(),
            "stage": "rep_loop",
            "precompute_done": int(pre_total),
            "precompute_total": int(pre_total),
            "done_reps": int(done_reps),
            "total_reps": int(total_reps),
            "pct": float(100.0 * done_reps / max(1, total_reps)),
        },
    )

    for truth_mode, truth_nu, rep_i in pending:
        rep_name = f"{truth_mode}_nu{truth_nu:.3f}_rep{rep_i:04d}".replace(".", "p").replace("-", "m")
        rep_path = reps_dir / f"{rep_name}.json"
        partial_path = partials_dir / f"{rep_name}_partial.json"

        truth_cache = cache[truth_mode]
        pred_truth = np.asarray(truth_cache["pred"], dtype=float)
        nu_idx_truth = int(np.argmin(np.abs(np.asarray(nu_grid, dtype=float) - float(truth_nu))))
        n_truth_draw = int(pred_truth.shape[0])
        d_truth = int(rng.integers(0, n_truth_draw))

        # Generate synthetic observed y from truth draw and truth nu.
        y_by_dataset: list[np.ndarray] = []
        for ds_i, scen in enumerate(scenarios):
            n_pt = int(scen.z.size)
            mu_true = pred_truth[d_truth, nu_idx_truth, ds_i, :n_pt]
            y = rng.multivariate_normal(mean=mu_true, cov=scen.cov)
            y_by_dataset.append(np.asarray(y, dtype=float))

        _write_json_atomic(
            partial_path,
            {
                "updated_utc": _utc_stamp(),
                "stage": "evaluate",
                "truth_mode": truth_mode,
                "truth_nu": float(truth_nu),
                "rep_index": int(rep_i),
                "truth_draw_local_index": int(d_truth),
            },
        )

        model_out: dict[str, Any] = {}
        for model_mode in modes:
            score_by_nu, mean_dchi_by_nu = _evaluate_mode_scores(
                pred_mode=np.asarray(cache[model_mode]["pred"], dtype=float),
                y_by_dataset=y_by_dataset,
                scenarios=scenarios,
                sigma_delta=float(args.offset_prior_sigma),
            )
            best_i = int(np.nanargmax(score_by_nu))
            model_out[model_mode] = {
                "best_nu": float(nu_grid[best_i]),
                "best_score": float(score_by_nu[best_i]),
                "score_by_nu": [float(x) for x in score_by_nu.tolist()],
                "mean_delta_chi2_by_nu_vs_nu0": [float(x) for x in mean_dchi_by_nu.tolist()],
            }

        delta_rel_abs = float(model_out["relative"]["best_score"] - model_out["absolute"]["best_score"])
        recovered = "relative" if delta_rel_abs >= 0.0 else "absolute"

        row = {
            "truth_mode": truth_mode,
            "truth_nu": float(truth_nu),
            "rep_index": int(rep_i),
            "truth_draw_local_index": int(d_truth),
            "truth_draw_run_label": str(truth_cache["run_label"][d_truth]),
            "truth_draw_idx": int(truth_cache["draw_idx"][d_truth]),
            "offset_prior_sigma": float(args.offset_prior_sigma),
            "models": model_out,
            "delta_score_relative_minus_absolute": delta_rel_abs,
            "recovered_mode": recovered,
            "is_correct_recovery": bool(recovered == truth_mode),
        }
        _write_json_atomic(rep_path, row)
        done_rows.append(row)
        done_reps += 1

        now = time.time()
        if (now - last_hb) >= float(args.heartbeat_sec) or done_reps == total_reps:
            _write_json_atomic(
                out_dir / "progress.json",
                {
                    "updated_utc": _utc_stamp(),
                    "stage": "rep_loop",
                    "precompute_done": int(pre_total),
                    "precompute_total": int(pre_total),
                    "done_reps": int(done_reps),
                    "total_reps": int(total_reps),
                    "pct": float(100.0 * done_reps / max(1, total_reps)),
                    "recent_partial": json.loads(partial_path.read_text(encoding="utf-8")),
                },
            )
            print(
                f"[heartbeat] reps={done_reps}/{total_reps} "
                f"({100.0 * done_reps / max(1, total_reps):.1f}%)",
                flush=True,
            )
            last_hb = now

    # Summary
    confusion: dict[str, dict[str, int]] = {}
    by_truth: dict[str, list[dict[str, Any]]] = {}
    for row in done_rows:
        t = str(row["truth_mode"])
        r = str(row["recovered_mode"])
        by_truth.setdefault(t, []).append(row)
        confusion.setdefault(t, {})
        confusion[t][r] = int(confusion[t].get(r, 0) + 1)

    acc_by_truth: dict[str, float] = {}
    for t, rows in by_truth.items():
        corr = np.mean([bool(x["is_correct_recovery"]) for x in rows]) if rows else float("nan")
        acc_by_truth[t] = float(corr)
    acc_overall = float(np.mean([bool(x["is_correct_recovery"]) for x in done_rows])) if done_rows else float("nan")

    delta_by_truth = {
        t: _stats(np.asarray([float(x["delta_score_relative_minus_absolute"]) for x in rows], dtype=float))
        for t, rows in by_truth.items()
    }
    delta_all = _stats(np.asarray([float(x["delta_score_relative_minus_absolute"]) for x in done_rows], dtype=float))

    summary = {
        "created_utc": _utc_stamp(),
        "out_dir": str(out_dir),
        "grid_dir": str(grid_dir),
        "datasets": datasets,
        "truth_spec": [[m, float(nu)] for (m, nu) in truth_spec],
        "nu_grid": [float(x) for x in nu_grid],
        "reps_per_truth": int(args.reps_per_truth),
        "n_reps_total": int(total_reps),
        "n_reps_done": int(len(done_rows)),
        "draws_per_run": int(args.draws_per_run),
        "offset_prior_sigma": float(args.offset_prior_sigma),
        "confusion": confusion,
        "accuracy": {"overall": acc_overall, "by_truth": acc_by_truth},
        "delta_score_stats_all": delta_all,
        "delta_score_stats_by_truth": delta_by_truth,
    }
    _write_json_atomic(out_dir / "summary.json", summary)
    _write_json_atomic(out_dir / "rows_all.json", done_rows)
    _write_json_atomic(
        out_dir / "progress.json",
        {
            "updated_utc": _utc_stamp(),
            "stage": "finished",
            "precompute_done": int(pre_total),
            "precompute_total": int(pre_total),
            "done_reps": int(len(done_rows)),
            "total_reps": int(total_reps),
            "pct": 100.0 if total_reps > 0 else 0.0,
        },
    )

    lines = _build_report_lines(summary)
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote identifiability outputs: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

