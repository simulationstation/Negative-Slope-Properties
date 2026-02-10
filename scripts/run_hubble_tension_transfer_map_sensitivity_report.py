#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _stats(x: pd.Series) -> dict[str, float]:
    arr = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "p16": float(np.percentile(arr, 16.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p84": float(np.percentile(arr, 84.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _group_effect(df: pd.DataFrame, factor: str, metric: str) -> dict[str, Any]:
    grouped = (
        df.groupby(factor, dropna=False)[metric]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    grouped = grouped.sort_values("mean")
    means = grouped["mean"].to_numpy(dtype=float)
    if means.size == 0:
        swing = float("nan")
    else:
        swing = float(np.max(means) - np.min(means))
    return {
        "factor": factor,
        "metric": metric,
        "swing": swing,
        "groups": grouped.to_dict(orient="records"),
    }


def _pair_effect(df: pd.DataFrame, factor_a: str, factor_b: str, metric: str) -> dict[str, Any]:
    grouped = (
        df.groupby([factor_a, factor_b], dropna=False)[metric]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    grouped = grouped.sort_values("mean")
    means = grouped["mean"].to_numpy(dtype=float)
    swing = float(np.max(means) - np.min(means)) if means.size else float("nan")
    return {
        "factors": [factor_a, factor_b],
        "metric": metric,
        "swing": swing,
        "groups": grouped.to_dict(orient="records"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sensitivity attribution report for Hubble tension transfer-map grid runs."
    )
    ap.add_argument("--grid-csv", required=True, help="Path to grid_results.csv")
    ap.add_argument(
        "--metrics",
        default="tension_relief_fraction_anchor_gr,anchor_gap_local_minus_gr_sigma,tension_relief_fraction_vs_planck_local_baseline",
        help="Comma-separated metric columns to analyze.",
    )
    ap.add_argument(
        "--factors",
        default="run_dir,sigma_highz_frac,h0_local_ref,local_mode,gr_omega_mode",
        help="Comma-separated factor columns for main-effect analysis.",
    )
    ap.add_argument(
        "--pairs",
        default="h0_local_ref:gr_omega_mode,sigma_highz_frac:gr_omega_mode,local_mode:gr_omega_mode",
        help="Comma-separated factor pairs as a:b for interaction summaries.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for reports (default: sibling of grid csv).",
    )
    args = ap.parse_args()

    grid_csv = Path(args.grid_csv).resolve()
    if not grid_csv.exists():
        raise FileNotFoundError(f"Missing grid CSV: {grid_csv}")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else grid_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    factors = [f.strip() for f in str(args.factors).split(",") if f.strip()]
    pair_specs = [p.strip() for p in str(args.pairs).split(",") if p.strip()]

    df = pd.read_csv(grid_csv)
    for col in metrics + factors:
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")

    for spec in pair_specs:
        a, b = [s.strip() for s in spec.split(":", 1)]
        if a not in df.columns or b not in df.columns:
            raise ValueError(f"Pair uses missing columns: {spec}")

    payload: dict[str, Any] = {
        "created_utc": _utc_now(),
        "grid_csv": str(grid_csv),
        "n_rows": int(df.shape[0]),
        "metrics": {},
        "factors": factors,
        "pairs": pair_specs,
    }

    md_lines: list[str] = []
    md_lines.append("# Hubble Transfer-Map Sensitivity Attribution\n")
    md_lines.append(f"- Generated: `{payload['created_utc']}`")
    md_lines.append(f"- Grid: `{grid_csv}`")
    md_lines.append(f"- Rows: `{int(df.shape[0])}`\n")

    for metric in metrics:
        metric_stats = _stats(df[metric])
        main_effects = [_group_effect(df, factor=f, metric=metric) for f in factors]
        main_effects = sorted(main_effects, key=lambda d: d["swing"], reverse=True)
        pair_effects = []
        for spec in pair_specs:
            a, b = [s.strip() for s in spec.split(":", 1)]
            pair_effects.append(_pair_effect(df, factor_a=a, factor_b=b, metric=metric))
        pair_effects = sorted(pair_effects, key=lambda d: d["swing"], reverse=True)

        payload["metrics"][metric] = {
            "overall": metric_stats,
            "main_effects_ranked": main_effects,
            "pair_effects_ranked": pair_effects,
        }

        md_lines.append(f"## Metric: `{metric}`\n")
        md_lines.append(
            f"- Overall: mean `{metric_stats['mean']:.6f}`, sd `{metric_stats['sd']:.6f}`, p16/p50/p84 = "
            f"`{metric_stats['p16']:.6f}` / `{metric_stats['p50']:.6f}` / `{metric_stats['p84']:.6f}`"
        )
        md_lines.append("- Main effects by mean swing (max group mean - min group mean):")
        for eff in main_effects:
            md_lines.append(f"  - `{eff['factor']}`: swing `{eff['swing']:.6f}`")
        md_lines.append("- Pair effects by mean swing:")
        for pe in pair_effects:
            f0, f1 = pe["factors"]
            md_lines.append(f"  - `{f0}:{f1}`: swing `{pe['swing']:.6f}`")
        md_lines.append("")

    json_path = out_dir / "sensitivity_attribution.json"
    md_path = out_dir / "sensitivity_attribution.md"
    _write_json_atomic(json_path, payload)
    _write_text_atomic(md_path, "\n".join(md_lines).strip() + "\n")
    print(f"[done] wrote {json_path}")
    print(f"[done] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
