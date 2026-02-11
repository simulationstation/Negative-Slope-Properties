#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _utc_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%SUTC", time.gmtime())


def _log(msg: str) -> None:
    print(msg, flush=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="ascii")


def _parse_seeds(text: str) -> list[int]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        raise ValueError("No seeds provided.")
    out: list[int] = []
    for p in parts:
        out.append(int(p))
    if len(set(out)) != len(out):
        raise ValueError(f"Duplicate seeds are not allowed: {out}")
    return out


def _run(cmd: list[str], *, dry_run: bool) -> None:
    _log(f"[run] {shlex.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _lb_to_vec(l_deg: float, b_deg: float) -> np.ndarray:
    l = np.deg2rad(float(l_deg))
    b = np.deg2rad(float(b_deg))
    cb = np.cos(b)
    return np.array([cb * np.cos(l), cb * np.sin(l), np.sin(b)], dtype=float)


def _axis_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    # Hemisphere axes are sign-degenerate (n and -n are equivalent), so use |dot|.
    c = float(np.clip(np.abs(np.dot(v1, v2)), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(c)))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Production orchestrator for directional anisotropy runs: "
            "crossfit confirmation + look-elsewhere null battery + multi-seed stability summary."
        )
    )
    ap.add_argument("--out-base", type=Path, default=None, help="Base output directory.")
    ap.add_argument("--scan-script", type=Path, default=Path(__file__).with_name("run_horizon_anisotropy_scan.py"))
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--seeds", type=str, default="123,321,2024", help="Comma-separated RNG seeds.")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--match-z-bin-width", type=float, default=0.05)
    ap.add_argument("--min-sn-per-side", type=int, default=200)
    ap.add_argument("--test-min-sn-per-side", type=int, default=80)
    ap.add_argument("--r-d-fixed", type=float, default=147.78)
    ap.add_argument("--growth-gamma", type=float, default=0.55)
    ap.add_argument("--growth-gamma-prior", type=float, nargs=2, default=[0.2, 1.2])
    ap.add_argument("--mu-procs", type=int, default=1)
    ap.add_argument("--axis-jobs", type=int, default=0)
    ap.add_argument("--null-axis-jobs", type=int, default=0)
    ap.add_argument("--train-axis-jobs", type=int, default=0)
    ap.add_argument("--null-reps", type=int, default=400, help="Total null replicates per seed (hemisphere scan).")
    ap.add_argument("--null-shard-size", type=int, default=100, help="Null replicate shard size.")
    ap.add_argument(
        "--z-threshold",
        type=float,
        default=1.5,
        help="Pass criterion: crossfit Stouffer z must be >= this value.",
    )
    ap.add_argument(
        "--null-p-threshold",
        type=float,
        default=0.05,
        help="Pass criterion: look-elsewhere calibrated p-value must be <= this value.",
    )
    ap.add_argument(
        "--axis-consistency-max-angle-deg",
        type=float,
        default=45.0,
        help="Pass criterion: max pairwise angle between best-axis directions across seeds.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    if int(args.kfold) < 2:
        raise ValueError("--kfold must be >= 2.")
    if float(args.match_z_bin_width) <= 0:
        raise ValueError("--match-z-bin-width must be > 0.")
    if int(args.null_reps) <= 0:
        raise ValueError("--null-reps must be > 0 for production null calibration.")
    if int(args.null_shard_size) <= 0:
        raise ValueError("--null-shard-size must be > 0.")
    if float(args.growth_gamma) <= 0:
        raise ValueError("--growth-gamma must be > 0.")
    g_lo, g_hi = [float(x) for x in args.growth_gamma_prior]
    if not (g_lo > 0 and g_hi > g_lo):
        raise ValueError("--growth-gamma-prior must satisfy 0 < lo < hi.")

    repo_root = Path(__file__).resolve().parents[1]
    scan_script = Path(args.scan_script).resolve()
    if not scan_script.exists():
        raise FileNotFoundError(f"Scan script not found: {scan_script}")

    out_base = (
        Path(args.out_base)
        if args.out_base is not None
        else (repo_root / "outputs" / "horizon_anisotropy_production" / _utc_stamp())
    )
    out_base.mkdir(parents=True, exist_ok=True)

    criteria = {
        "timestamp_utc": _utc_stamp(),
        "seeds": seeds,
        "crossfit": {
            "z_stouffer_min": float(args.z_threshold),
            "kfold": int(args.kfold),
        },
        "null_calibration": {
            "look_elsewhere_p_max": float(args.null_p_threshold),
            "null_reps": int(args.null_reps),
            "null_shard_size": int(args.null_shard_size),
            "stat": "max_abs_z",
            "mode": "shuffle_within_survey",
        },
        "stability": {
            "max_pairwise_axis_angle_deg": float(args.axis_consistency_max_angle_deg),
            "axis_source": "hemisphere_scan.scan_summary.best_axis",
            "antipode_equivalent": True,
        },
        "notes": [
            "Criteria written before execution to avoid post-hoc threshold changes.",
            "Crossfit establishes out-of-sample directional significance.",
            "Look-elsewhere p-values are calibrated from hemisphere-scan null batteries.",
        ],
    }
    _write_json(out_base / "acceptance_criteria.json", criteria)

    common = [
        args.python,
        str(scan_script),
        "--preset",
        "production",
        "--match-z-bin-width",
        str(args.match_z_bin_width),
        "--min-sn-per-side",
        str(args.min_sn_per_side),
        "--test-min-sn-per-side",
        str(args.test_min_sn_per_side),
        "--r-d-fixed",
        str(args.r_d_fixed),
        "--growth-mode",
        "gamma",
        "--growth-gamma",
        str(args.growth_gamma),
        "--growth-gamma-free",
        "--growth-gamma-prior",
        str(g_lo),
        str(g_hi),
        "--mu-procs",
        str(args.mu_procs),
    ]

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        seed_dir = out_base / f"seed_{int(seed):04d}"
        crossfit_out = seed_dir / "crossfit"
        hemi_out = seed_dir / "hemisphere_null"
        crossfit_out.mkdir(parents=True, exist_ok=True)
        hemi_out.mkdir(parents=True, exist_ok=True)

        crossfit_cmd = common + [
            "--mode",
            "crossfit",
            "--out",
            str(crossfit_out),
            "--seed",
            str(seed),
            "--kfold",
            str(args.kfold),
            "--train-axis-jobs",
            str(args.train_axis_jobs),
        ]
        _run(crossfit_cmd, dry_run=bool(args.dry_run))

        shard_starts = list(range(0, int(args.null_reps), int(args.null_shard_size)))
        for rs in shard_starts:
            re = min(int(args.null_reps), int(rs + int(args.null_shard_size)))
            null_cmd = common + [
                "--mode",
                "hemisphere_scan",
                "--out",
                str(hemi_out),
                "--seed",
                str(seed),
                "--axis-jobs",
                str(args.axis_jobs),
                "--null-axis-jobs",
                str(args.null_axis_jobs),
                "--null-reps",
                str(args.null_reps),
                "--null-rep-start",
                str(rs),
                "--null-rep-end",
                str(re),
            ]
            _run(null_cmd, dry_run=bool(args.dry_run))

        finalize_cmd = common + [
            "--mode",
            "hemisphere_scan",
            "--out",
            str(hemi_out),
            "--seed",
            str(seed),
            "--axis-jobs",
            str(args.axis_jobs),
            "--null-axis-jobs",
            str(args.null_axis_jobs),
            "--null-reps",
            str(args.null_reps),
            "--null-finalize-only",
        ]
        _run(finalize_cmd, dry_run=bool(args.dry_run))

        if bool(args.dry_run):
            continue

        crossfit_summary = _load_json(crossfit_out / "crossfit_summary.json")
        scan_summary = _load_json(hemi_out / "scan_summary.json")
        null_summary = _load_json(hemi_out / "null_summary.json")
        z_stouffer = float(crossfit_summary["z_stouffer"])
        p_null = null_summary.get("p_value_one_sided")
        p_null = float(p_null) if p_null is not None else np.nan
        best_axis = dict(scan_summary["best_axis"])
        per_seed.append(
            {
                "seed": int(seed),
                "crossfit_out": str(crossfit_out),
                "hemisphere_out": str(hemi_out),
                "crossfit_z_stouffer": z_stouffer,
                "null_p_value_one_sided": p_null,
                "best_axis": best_axis,
                "pass_crossfit": bool(np.isfinite(z_stouffer) and z_stouffer >= float(args.z_threshold)),
                "pass_null": bool(np.isfinite(p_null) and p_null <= float(args.null_p_threshold)),
            }
        )

    if bool(args.dry_run):
        _log(f"[dry-run] wrote acceptance criteria to {out_base / 'acceptance_criteria.json'}")
        return 0

    vecs = []
    for row in per_seed:
        l_deg = float(row["best_axis"]["axis_l_deg"])
        b_deg = float(row["best_axis"]["axis_b_deg"])
        vecs.append(_lb_to_vec(l_deg, b_deg))

    pairwise = []
    for i in range(len(per_seed)):
        for j in range(i + 1, len(per_seed)):
            ang = _axis_angle_deg(vecs[i], vecs[j])
            pairwise.append({"seed_i": int(per_seed[i]["seed"]), "seed_j": int(per_seed[j]["seed"]), "angle_deg": float(ang)})
    max_ang = float(max((p["angle_deg"] for p in pairwise), default=0.0))

    summary = {
        "timestamp_utc": _utc_stamp(),
        "out_base": str(out_base),
        "criteria_path": str(out_base / "acceptance_criteria.json"),
        "per_seed": per_seed,
        "stability": {
            "pairwise_axis_angles_deg": pairwise,
            "max_pairwise_axis_angle_deg": max_ang,
            "pass": bool(max_ang <= float(args.axis_consistency_max_angle_deg)),
        },
    }
    summary["overall_pass"] = bool(
        summary["stability"]["pass"] and all(bool(r["pass_crossfit"]) and bool(r["pass_null"]) for r in per_seed)
    )
    _write_json(out_base / "production_summary.json", summary)
    _log(f"[done] wrote {out_base / 'production_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
