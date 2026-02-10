#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _parse_float_list(text: str) -> list[float]:
    vals = [float(t.strip()) for t in str(text).split(",") if t.strip()]
    if not vals:
        raise ValueError("Expected at least one float.")
    return vals


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


def _interp_safe(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return float("nan")
    x_ok = x[mask]
    y_ok = y[mask]
    order = np.argsort(x_ok)
    return float(np.interp(float(x0), x_ok[order], y_ok[order]))


def _analyze_case(case_dir: Path, z_min: float, z_max: float, z_probe: list[float]) -> dict[str, Any]:
    tables = case_dir / "tables"
    prof = json.loads((tables / "expansion_profile_quantiles.json").read_text(encoding="utf-8"))
    summary = json.loads((tables / "summary.json").read_text(encoding="utf-8"))

    z = np.asarray(prof["z"], dtype=float)
    ratio = np.asarray(prof["h_ratio_planck_q50"], dtype=float)
    om0 = float(summary.get("references", {}).get("omega_m_planck", 0.315))

    e_planck = np.sqrt(om0 * (1.0 + z) ** 3 + (1.0 - om0))
    e_mg = (ratio / ratio[0]) * e_planck
    e2 = e_mg**2

    dlnh_dz = np.gradient(np.log(e_mg), z)
    q_z = -1.0 + (1.0 + z) * dlnh_dz
    omega_m_z = om0 * (1.0 + z) ** 3 / e2
    denom = 3.0 * (1.0 - omega_m_z)
    w_eff = (2.0 * q_z - 1.0) / denom
    w_eff[np.abs(denom) < 1e-8] = np.nan

    om_diag = np.full_like(z, np.nan, dtype=float)
    mask_om = np.abs((1.0 + z) ** 3 - 1.0) > 1e-8
    om_diag[mask_om] = (e2[mask_om] - 1.0) / ((1.0 + z[mask_om]) ** 3 - 1.0)

    in_win = (z >= float(z_min)) & (z <= float(z_max))
    w_win = w_eff[in_win]
    om_win = om_diag[in_win]

    phantom_frac = float(np.mean(w_win < -1.0)) if np.any(np.isfinite(w_win)) else float("nan")
    w_outside_sanity = float(np.mean((w_win < -2.0) | (w_win > 0.0))) if np.any(np.isfinite(w_win)) else float("nan")
    om_drift = float(np.nanmax(om_win) - np.nanmin(om_win)) if np.any(np.isfinite(om_win)) else float("nan")

    probes = {f"w_eff_z{zp:.3f}".replace(".", "p"): _interp_safe(z, w_eff, zp) for zp in z_probe}
    probes.update({f"Om_z{zp:.3f}".replace(".", "p"): _interp_safe(z, om_diag, zp) for zp in z_probe})

    row = {
        "label": case_dir.name,
        "run_dir": str(summary.get("run_dir", "")),
        "sigma_highz_frac": float(summary.get("references", {}).get("sigma_highz_frac", float("nan"))),
        "h0_local_ref": float(summary.get("references", {}).get("h0_local_ref", float("nan"))),
        "local_mode": str(summary.get("references", {}).get("local_mode", "")),
        "gr_omega_mode": str(summary.get("references", {}).get("gr_omega_mode", "")),
        "omega_m_planck": om0,
        "q0": float(_interp_safe(z, q_z, 0.0)),
        "w_eff_mean_window": float(np.nanmean(w_win)) if np.any(np.isfinite(w_win)) else float("nan"),
        "w_eff_min_window": float(np.nanmin(w_win)) if np.any(np.isfinite(w_win)) else float("nan"),
        "w_eff_max_window": float(np.nanmax(w_win)) if np.any(np.isfinite(w_win)) else float("nan"),
        "w_eff_phantom_frac_window": phantom_frac,
        "w_eff_outside_sanity_frac_window": w_outside_sanity,
        "Om_mean_window": float(np.nanmean(om_win)) if np.any(np.isfinite(om_win)) else float("nan"),
        "Om_sd_window": float(np.nanstd(om_win, ddof=1)) if np.sum(np.isfinite(om_win)) > 1 else 0.0,
        "Om_drift_window": om_drift,
        "Om_mean_minus_omega_m0": float(np.nanmean(om_win) - om0) if np.any(np.isfinite(om_win)) else float("nan"),
    }
    row.update(probes)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="Dark-energy consistency diagnostics from transfer-map outputs.")
    ap.add_argument(
        "--grid-dir",
        default="outputs/hubble_tension_mg_transfer_map_full_v1_20260208_045425UTC",
        help="Transfer-map grid directory containing `cases/`.",
    )
    ap.add_argument("--z-min", type=float, default=0.02, help="Window min z for summary metrics.")
    ap.add_argument("--z-max", type=float, default=0.62, help="Window max z for summary metrics.")
    ap.add_argument("--z-probe", default="0.2,0.35,0.5,0.62", help="Comma-separated probe redshifts.")
    ap.add_argument("--out", default=None, help="Output directory.")
    args = ap.parse_args()

    z_probe = _parse_float_list(args.z_probe)
    grid_dir = Path(args.grid_dir)
    case_dirs = sorted((grid_dir / "cases").glob("*"))
    if not case_dirs:
        raise FileNotFoundError(f"No case directories found under: {grid_dir / 'cases'}")

    out_dir = Path(args.out) if args.out else Path("outputs") / f"dark_energy_consistency_{_utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for case_dir in case_dirs:
        if not case_dir.is_dir():
            continue
        if not (case_dir / "tables" / "expansion_profile_quantiles.json").exists():
            continue
        rows.append(_analyze_case(case_dir, float(args.z_min), float(args.z_max), z_probe))

    if not rows:
        raise RuntimeError("No analyzable cases found.")

    metric_keys = [
        "q0",
        "w_eff_mean_window",
        "w_eff_min_window",
        "w_eff_max_window",
        "w_eff_phantom_frac_window",
        "w_eff_outside_sanity_frac_window",
        "Om_mean_window",
        "Om_sd_window",
        "Om_drift_window",
        "Om_mean_minus_omega_m0",
    ]
    metric_keys += [f"w_eff_z{zp:.3f}".replace(".", "p") for zp in z_probe]
    metric_keys += [f"Om_z{zp:.3f}".replace(".", "p") for zp in z_probe]

    summary = {
        "created_utc": _utc_stamp(),
        "grid_dir": str(grid_dir.resolve()),
        "n_cases": int(len(rows)),
        "window": {"z_min": float(args.z_min), "z_max": float(args.z_max)},
        "metrics": {k: _stats(np.asarray([r.get(k, float("nan")) for r in rows], dtype=float)) for k in metric_keys},
        "fractions": {
            "cases_with_window_phantom_gt_50pct": float(
                np.mean(np.asarray([r["w_eff_phantom_frac_window"] for r in rows], dtype=float) > 0.5)
            ),
            "cases_with_any_outside_sanity": float(
                np.mean(np.asarray([r["w_eff_outside_sanity_frac_window"] for r in rows], dtype=float) > 0.0)
            ),
            "cases_with_Om_drift_gt_0p05": float(
                np.mean(np.asarray([r["Om_drift_window"] for r in rows], dtype=float) > 0.05)
            ),
        },
    }

    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "rows.json", rows)

    csv_path = out_dir / "rows.csv"
    cols = sorted(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# Dark-energy consistency diagnostics",
        "",
        f"- Grid: `{grid_dir.resolve()}`",
        f"- Cases analyzed: `{len(rows)}`",
        f"- Redshift window: `[{float(args.z_min):.3f}, {float(args.z_max):.3f}]`",
        "",
        "## Key outputs",
        "",
        f"- `w_eff_mean_window` p16/p50/p84: "
        f"`{summary['metrics']['w_eff_mean_window'].get('p16', float('nan')):.4f}` / "
        f"`{summary['metrics']['w_eff_mean_window'].get('p50', float('nan')):.4f}` / "
        f"`{summary['metrics']['w_eff_mean_window'].get('p84', float('nan')):.4f}`",
        f"- `Om_drift_window` p16/p50/p84: "
        f"`{summary['metrics']['Om_drift_window'].get('p16', float('nan')):.4f}` / "
        f"`{summary['metrics']['Om_drift_window'].get('p50', float('nan')):.4f}` / "
        f"`{summary['metrics']['Om_drift_window'].get('p84', float('nan')):.4f}`",
        f"- Fraction with >50% phantom in window: `{summary['fractions']['cases_with_window_phantom_gt_50pct']:.3f}`",
        f"- Fraction with any `w_eff` outside sanity range [-2, 0]: "
        f"`{summary['fractions']['cases_with_any_outside_sanity']:.3f}`",
        "",
        "## Artifacts",
        "",
        "- `summary.json`",
        "- `rows.json`",
        "- `rows.csv`",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote dark-energy consistency outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
