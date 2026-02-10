from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _summ(arr: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(arr, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    p_gt0 = float(np.mean(arr > 0.0))
    return mean, std, p_gt0


def _weighted_slope_draws(logA: np.ndarray, draws: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least-squares slope per draw for y ~ a + b x where x is centered."""
    x0 = float(np.average(logA, weights=w))
    x = logA - x0
    # Solve (X^T W X) beta = (X^T W y) per draw.
    X = np.column_stack([np.ones_like(x), x])
    XtW = (X.T * w)
    XtWX = XtW @ X
    inv = np.linalg.inv(XtWX)
    beta = (inv @ XtW) @ draws.T  # (2,n_draws)
    return np.asarray(beta[1], dtype=float)


def _departure_draws(logA: np.ndarray, logmu_draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logA = np.asarray(logA, dtype=float)
    draws = np.asarray(logmu_draws, dtype=float)
    if draws.ndim != 2 or draws.shape[1] != logA.size:
        raise ValueError("Grid mismatch for logmu samples.")
    var = np.var(draws, axis=0, ddof=1)
    w = 1.0 / np.clip(var, 1e-12, np.inf)
    w = w / np.trapezoid(w, x=logA)
    m_draw = np.trapezoid(draws * w[None, :], x=logA, axis=1)
    s_draw = _weighted_slope_draws(logA, draws, w)
    return m_draw, s_draw


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize multi-start M0 runs for multimodality diagnostics.")
    ap.add_argument("--base", type=Path, default=Path("outputs/multistart"))
    ap.add_argument("--glob", type=str, default="M0_start*")
    ap.add_argument("--out-csv", type=Path, default=Path("outputs/multistart/summary.csv"))
    ap.add_argument("--out-md", type=Path, default=Path("outputs/multistart/summary.md"))
    args = ap.parse_args()

    runs = sorted([p for p in args.base.glob(args.glob) if p.is_dir()])
    if not runs:
        raise SystemExit(f"No run dirs found under {args.base} matching {args.glob!r}.")

    try:
        from scipy.stats import wasserstein_distance  # type: ignore
    except Exception:
        wasserstein_distance = None

    rows = []
    m_draws_by = {}
    for d in runs:
        npz = np.load(d / "samples" / "logmu_logA_samples.npz")
        logA = np.asarray(npz["logA"], dtype=float)
        logmu = np.asarray(npz["logmu_samples"], dtype=float)
        m_draw, s_draw = _departure_draws(logA, logmu)

        meta = _json_load(d / "samples" / "mu_forward_meta.json")["meta"]
        accept = meta.get("acceptance_fraction_mean", None)
        ess_min = meta.get("ess_min", None)
        tau_by = meta.get("tau_by", None) or {}
        tau_est = float(np.max(list(tau_by.values()))) if tau_by else None

        m_mean, m_sd, p_m_gt0 = _summ(m_draw)
        s_mean, s_sd, p_s_gt0 = _summ(s_draw)

        m_draws_by[d.name] = m_draw
        rows.append(
            {
                "run_id": d.name,
                "path": str(d),
                "m_mean": m_mean,
                "m_sd": m_sd,
                "p_m_gt0": p_m_gt0,
                "s_mean": s_mean,
                "s_sd": s_sd,
                "p_s_gt0": p_s_gt0,
                "accept_mean": float(accept) if accept is not None else None,
                "tau_est": float(tau_est) if tau_est is not None else None,
                "ess_min": float(ess_min) if ess_min is not None else None,
            }
        )

    # Choose reference for Wasserstein distances.
    ref_id = "M0_start101" if "M0_start101" in m_draws_by else rows[0]["run_id"]
    ref = m_draws_by[ref_id]
    for r in rows:
        if wasserstein_distance is None:
            r["wass_to_ref"] = None
        else:
            r["wass_to_ref"] = float(wasserstein_distance(ref, m_draws_by[r["run_id"]]))

    # Write CSV.
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "m_mean",
        "m_sd",
        "p_m_gt0",
        "s_mean",
        "s_sd",
        "p_s_gt0",
        "accept_mean",
        "tau_est",
        "ess_min",
        "wass_to_ref",
    ]
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    # Write markdown summary.
    n = len(rows)
    n_pos = sum(1 for r in rows if r["p_m_gt0"] is not None and float(r["p_m_gt0"]) > 0.5)
    n_neg = sum(1 for r in rows if r["p_m_gt0"] is not None and float(r["p_m_gt0"]) < 0.5)

    md = []
    md.append("# Multi-start summary (M0)\n")
    md.append(f"Runs: {n}\n")
    md.append(f"Reference run for Wasserstein: `{ref_id}`\n")
    md.append(f"Runs with P(m>0)>0.5: {n_pos}/{n}\n")
    md.append(f"Runs with P(m>0)<0.5: {n_neg}/{n}\n")
    md.append("\n| run_id | m_mean | m_sd | P(m>0) | s_mean | s_sd | accept | tau_est | ESS_min | W1(m,ref) |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        md.append(
            "| {run_id} | {m_mean:.4f} | {m_sd:.4f} | {p_m_gt0:.3f} | {s_mean:.4f} | {s_sd:.4f} | {accept} | {tau} | {ess} | {wass} |\n".format(
                run_id=r["run_id"],
                m_mean=float(r["m_mean"]),
                m_sd=float(r["m_sd"]),
                p_m_gt0=float(r["p_m_gt0"]),
                s_mean=float(r["s_mean"]),
                s_sd=float(r["s_sd"]),
                accept=f"{r['accept_mean']:.3f}" if r["accept_mean"] is not None else "NA",
                tau=f"{r['tau_est']:.1f}" if r["tau_est"] is not None else "NA",
                ess=f"{r['ess_min']:.0f}" if r["ess_min"] is not None else "NA",
                wass=f"{r['wass_to_ref']:.4f}" if r["wass_to_ref"] is not None else "NA",
            )
        )
    args.out_md.write_text("".join(md), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

