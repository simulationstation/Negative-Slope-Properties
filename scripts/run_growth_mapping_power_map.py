#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


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


def _parse_truth_tokens(text: str) -> list[tuple[str, float, str]]:
    out: list[tuple[str, float, str]] = []
    for token in _parse_csv_str(text):
        if ":" not in token:
            raise ValueError(f"Invalid truth token '{token}' (expected mode:nu)")
        mode_raw, nu_raw = token.split(":", 1)
        mode = mode_raw.strip().lower()
        if mode not in {"relative", "absolute"}:
            raise ValueError(f"Unknown truth mode '{mode}'")
        nu = float(nu_raw.strip())
        key = f"{mode}:nu{nu:.3f}".replace(".", "p").replace("-", "m")
        out.append((mode, nu, key))
    return out


def _load_rows(rows_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    delta = []
    labels = []
    for r in rows:
        if "delta_score_relative_minus_absolute" not in r or "truth_mode" not in r:
            continue
        delta.append(float(r["delta_score_relative_minus_absolute"]))
        labels.append(1 if str(r["truth_mode"]) == "relative" else 0)
    x = np.asarray(delta, dtype=float)
    y = np.asarray(labels, dtype=int)
    if x.size == 0:
        raise RuntimeError(f"No usable rows in {rows_path}")
    return x, y


def _candidate_thresholds(x: np.ndarray) -> np.ndarray:
    xs = np.unique(np.asarray(x, dtype=float))
    if xs.size == 1:
        return np.asarray([xs[0]], dtype=float)
    mids = (xs[:-1] + xs[1:]) * 0.5
    return np.concatenate(([xs[0] - 1e-9], mids, [xs[-1] + 1e-9]))


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tpr = float(tp / max(1, tp + fn))
    tnr = float(tn / max(1, tn + fp))
    return 0.5 * (tpr + tnr)


def _fit_threshold_from_rows(train_rows: Path) -> float:
    x, y = _load_rows(train_rows)
    best_thr = 0.0
    best_ba = -1.0
    for thr in _candidate_thresholds(x):
        pred = (x >= thr).astype(int)
        ba = _balanced_accuracy(y, pred)
        if ba > best_ba:
            best_ba = ba
            best_thr = float(thr)
    return float(best_thr)


def _stats(x: list[float]) -> dict[str, float]:
    arr = np.asarray(x, dtype=float)
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Run holdout power map for mapping identifiability.")
    ap.add_argument("--grid-dir", default="outputs/hubble_tension_mg_transfer_map_full_v1_20260208_045425UTC")
    ap.add_argument(
        "--datasets",
        default="sdss_dr12_consensus_fs,sdss_dr16_lrg_fsbao_dmdhfs8",
        help="Comma-separated RSD datasets.",
    )
    ap.add_argument(
        "--truth-grid",
        default="relative:0.5,relative:0.7,relative:0.9,absolute:0.0,absolute:0.2",
        help="Comma-separated truth tokens mode:nu.",
    )
    ap.add_argument("--holdouts", type=int, default=3)
    ap.add_argument("--reps-per-truth", type=int, default=128)
    ap.add_argument("--draws-per-run", type=int, default=48)
    ap.add_argument("--offset-prior-sigma", type=float, default=0.03)
    ap.add_argument("--nu-grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=20260208)
    ap.add_argument("--heartbeat-sec", type=float, default=60.0)
    ap.add_argument("--threshold", type=float, default=None, help="Fixed threshold for classify relative if delta>=thr.")
    ap.add_argument("--calibration-train-rows", default="", help="Optional rows_all.json used to fit threshold if --threshold is omitted.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"growth_mapping_power_map_{_utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    partials_dir = out_dir / "partials"
    runs_dir.mkdir(parents=True, exist_ok=True)
    partials_dir.mkdir(parents=True, exist_ok=True)

    truth_grid = _parse_truth_tokens(args.truth_grid)
    datasets = ",".join(_parse_csv_str(args.datasets))
    grid_dir = str(Path(args.grid_dir).resolve())
    nu_grid = str(args.nu_grid)
    heartbeat_sec = float(args.heartbeat_sec)
    workers = int(max(1, int(args.workers)))

    if args.threshold is not None:
        threshold = float(args.threshold)
        threshold_source = "cli"
    else:
        cal_path = Path(str(args.calibration_train_rows)).resolve() if str(args.calibration_train_rows).strip() else None
        if cal_path is None or (not cal_path.exists()):
            raise ValueError("Provide --threshold or valid --calibration-train-rows.")
        threshold = _fit_threshold_from_rows(cal_path)
        threshold_source = str(cal_path)

    manifest = {
        "created_utc": _utc_stamp(),
        "grid_dir": grid_dir,
        "datasets": datasets,
        "truth_grid": [[m, nu, key] for (m, nu, key) in truth_grid],
        "holdouts": int(args.holdouts),
        "reps_per_truth": int(args.reps_per_truth),
        "draws_per_run": int(args.draws_per_run),
        "offset_prior_sigma": float(args.offset_prior_sigma),
        "nu_grid": nu_grid,
        "workers": workers,
        "seed": int(args.seed),
        "threshold": float(threshold),
        "threshold_source": threshold_source,
    }
    _write_json_atomic(out_dir / "manifest.json", manifest)

    run_tasks: list[dict[str, Any]] = []
    for p_i, (mode, nu, key) in enumerate(truth_grid):
        for h in range(int(args.holdouts)):
            sub_seed = int(args.seed) + 100000 * p_i + 1000 * h + 17
            name = f"{key}_h{h:02d}"
            sub_out = runs_dir / name
            run_tasks.append(
                {
                    "name": name,
                    "mode": mode,
                    "nu": float(nu),
                    "truth_spec": f"{mode}:{nu}",
                    "seed": sub_seed,
                    "out": sub_out,
                }
            )

    rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    if bool(args.resume):
        for t in run_tasks:
            row_path = Path(t["out"]) / "rows_all.json"
            if row_path.exists():
                x, y = _load_rows(row_path)
                pred_rel = (x >= threshold).astype(int)
                rrate = float(np.mean(pred_rel))
                rows.append(
                    {
                        "name": t["name"],
                        "mode": t["mode"],
                        "nu": float(t["nu"]),
                        "seed": int(t["seed"]),
                        "out": str(t["out"]),
                        "n": int(x.size),
                        "pred_relative_rate": rrate,
                        "pred_absolute_rate": float(1.0 - rrate),
                        "mean_delta": float(np.mean(x)),
                        "sd_delta": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
                    }
                )
            else:
                pending.append(t)
    else:
        pending = list(run_tasks)

    total = len(run_tasks)
    done = len(rows)
    last_hb = 0.0
    _write_json_atomic(
        out_dir / "progress.json",
        {
            "updated_utc": _utc_stamp(),
            "stage": "running",
            "done_runs": int(done),
            "total_runs": int(total),
            "pct": float(100.0 * done / max(1, total)),
            "threshold": float(threshold),
        },
    )

    for task in pending:
        sub_out = Path(task["out"])
        sub_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            ".venv/bin/python",
            "scripts/run_growth_mapping_identifiability_injection.py",
            "--grid-dir",
            grid_dir,
            "--datasets",
            datasets,
            "--truth-spec",
            str(task["truth_spec"]),
            "--nu-grid",
            nu_grid,
            "--reps-per-truth",
            str(int(args.reps_per_truth)),
            "--draws-per-run",
            str(int(args.draws_per_run)),
            "--offset-prior-sigma",
            str(float(args.offset_prior_sigma)),
            "--workers",
            str(workers),
            "--seed",
            str(int(task["seed"])),
            "--heartbeat-sec",
            str(heartbeat_sec),
            "--resume",
            "--out",
            str(sub_out),
        ]
        proc = subprocess.run(cmd, cwd=str(Path.cwd()), capture_output=True, text=True)
        if proc.returncode != 0:
            _write_json_atomic(
                partials_dir / f"{task['name']}_error.json",
                {
                    "updated_utc": _utc_stamp(),
                    "name": task["name"],
                    "mode": task["mode"],
                    "nu": float(task["nu"]),
                    "seed": int(task["seed"]),
                    "returncode": int(proc.returncode),
                    "stderr_tail": proc.stderr[-2000:],
                    "stdout_tail": proc.stdout[-2000:],
                },
            )
            done += 1
            now = time.time()
            if (now - last_hb) >= heartbeat_sec or done == total:
                _write_json_atomic(
                    out_dir / "progress.json",
                    {
                        "updated_utc": _utc_stamp(),
                        "stage": "running",
                        "done_runs": int(done),
                        "total_runs": int(total),
                        "pct": float(100.0 * done / max(1, total)),
                        "threshold": float(threshold),
                    },
                )
                print(f"[heartbeat] done={done}/{total} ({100.0 * done / max(1, total):.1f}%)", flush=True)
                last_hb = now
            continue

        row_path = sub_out / "rows_all.json"
        x, y = _load_rows(row_path)
        pred_rel = (x >= threshold).astype(int)
        rrate = float(np.mean(pred_rel))
        row = {
            "name": task["name"],
            "mode": task["mode"],
            "nu": float(task["nu"]),
            "seed": int(task["seed"]),
            "out": str(sub_out),
            "n": int(x.size),
            "pred_relative_rate": rrate,
            "pred_absolute_rate": float(1.0 - rrate),
            "mean_delta": float(np.mean(x)),
            "sd_delta": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        }
        rows.append(row)
        _write_json_atomic(partials_dir / f"{task['name']}_partial.json", row)
        done += 1

        now = time.time()
        if (now - last_hb) >= heartbeat_sec or done == total:
            _write_json_atomic(
                out_dir / "progress.json",
                {
                    "updated_utc": _utc_stamp(),
                    "stage": "running",
                    "done_runs": int(done),
                    "total_runs": int(total),
                    "pct": float(100.0 * done / max(1, total)),
                    "threshold": float(threshold),
                    "recent_partial": row,
                },
            )
            print(f"[heartbeat] done={done}/{total} ({100.0 * done / max(1, total):.1f}%)", flush=True)
            last_hb = now

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"{row['mode']}:nu{float(row['nu']):.3f}"
        grouped.setdefault(key, []).append(row)

    point_summary: dict[str, Any] = {}
    for key, vals in sorted(grouped.items()):
        mode = key.split(":")[0]
        rr = [float(v["pred_relative_rate"]) for v in vals]
        md = [float(v["mean_delta"]) for v in vals]
        point_summary[key] = {
            "mode": mode,
            "n_holdouts_done": int(len(vals)),
            "pred_relative_rate": _stats(rr),
            "mean_delta": _stats(md),
            "metric_name": "TPR_relative" if mode == "relative" else "FPR_relative",
            "metric_value": float(np.mean(rr)) if rr else float("nan"),
        }

    rel_metrics = [float(v["metric_value"]) for v in point_summary.values() if v["mode"] == "relative"]
    abs_metrics = [float(v["metric_value"]) for v in point_summary.values() if v["mode"] == "absolute"]
    global_summary = {
        "mean_tpr_relative_points": float(np.mean(rel_metrics)) if rel_metrics else float("nan"),
        "mean_fpr_absolute_points": float(np.mean(abs_metrics)) if abs_metrics else float("nan"),
        "separation_tpr_minus_fpr": float(np.mean(rel_metrics) - np.mean(abs_metrics))
        if (rel_metrics and abs_metrics)
        else float("nan"),
    }

    summary = {
        "created_utc": _utc_stamp(),
        "out_dir": str(out_dir),
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "n_runs_total": int(total),
        "n_runs_done": int(done),
        "point_summary": point_summary,
        "global_summary": global_summary,
    }
    _write_json_atomic(out_dir / "summary.json", summary)
    _write_json_atomic(out_dir / "rows_all.json", rows)
    _write_json_atomic(
        out_dir / "progress.json",
        {
            "updated_utc": _utc_stamp(),
            "stage": "finished",
            "done_runs": int(done),
            "total_runs": int(total),
            "pct": 100.0 if total > 0 else 0.0,
            "threshold": float(threshold),
        },
    )

    lines = [
        "# Growth mapping power map",
        "",
        f"- Output: `{out_dir}`",
        f"- Threshold: `{threshold:.6f}` (source: `{threshold_source}`)",
        f"- Runs done: `{done}/{total}`",
        "",
        "## Point summary",
        "",
    ]
    for key, s in point_summary.items():
        lines.append(
            f"- `{key}` ({s['metric_name']}): mean `{s['metric_value']:.6f}`; "
            f"holdouts `{s['n_holdouts_done']}`"
        )
    lines.extend(
        [
            "",
            "## Global",
            "",
            f"- mean TPR(relative points): `{global_summary['mean_tpr_relative_points']:.6f}`",
            f"- mean FPR(absolute points): `{global_summary['mean_fpr_absolute_points']:.6f}`",
            f"- separation (TPR-FPR): `{global_summary['separation_tpr_minus_fpr']:.6f}`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote power-map outputs: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

