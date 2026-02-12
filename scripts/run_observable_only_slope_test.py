#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from math import erf, sqrt
from pathlib import Path
from typing import Any

import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.ingest_rsd_single_survey import load_rsd_single_survey


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
        raise ValueError("Expected at least one dataset.")
    return vals


def _block_diag(covs: list[np.ndarray]) -> np.ndarray:
    total = int(sum(int(c.shape[0]) for c in covs))
    out = np.zeros((total, total), dtype=float)
    pos = 0
    for c in covs:
        n = int(c.shape[0])
        out[pos : pos + n, pos : pos + n] = c
        pos += n
    return out


def _weighted_center(x: np.ndarray, cov: np.ndarray) -> np.ndarray:
    diag = np.clip(np.diag(cov), 1e-12, np.inf)
    w = 1.0 / diag
    x0 = float(np.sum(w * x) / np.sum(w))
    return x - x0


def _fit_gls(y: np.ndarray, x: np.ndarray, dataset_id: np.ndarray, cov: np.ndarray) -> dict[str, Any]:
    n = int(y.size)
    n_ds = int(np.max(dataset_id)) + 1
    if x.shape != (n,) or dataset_id.shape != (n,) or cov.shape != (n, n):
        raise ValueError("Input shape mismatch for GLS fit.")

    X = np.zeros((n, n_ds + 1), dtype=float)
    X[np.arange(n), dataset_id] = 1.0
    X[:, -1] = x

    inv_cov = np.linalg.inv(cov)
    xt_cinv = X.T @ inv_cov
    fisher = xt_cinv @ X
    fisher_inv = np.linalg.inv(fisher)
    beta_hat = fisher_inv @ (xt_cinv @ y)

    y_hat = X @ beta_hat
    resid = y - y_hat
    chi2 = float(resid @ inv_cov @ resid)

    slope = float(beta_hat[-1])
    slope_sigma = float(np.sqrt(max(fisher_inv[-1, -1], 0.0)))
    z_score = float(slope / slope_sigma) if slope_sigma > 0 else float("nan")
    p_pos = float(0.5 * (1.0 + erf(slope / (sqrt(2.0) * slope_sigma)))) if slope_sigma > 0 else float("nan")
    p_neg = float(1.0 - p_pos) if np.isfinite(p_pos) else float("nan")

    return {
        "X": X,
        "inv_cov": inv_cov,
        "fisher_inv": fisher_inv,
        "beta_hat": beta_hat,
        "chi2": chi2,
        "slope": slope,
        "slope_sigma": slope_sigma,
        "z_score": z_score,
        "p_slope_gt_0": p_pos,
        "p_slope_lt_0": p_neg,
    }


def _null_mc(
    *,
    y: np.ndarray,
    x: np.ndarray,
    dataset_id: np.ndarray,
    cov: np.ndarray,
    fit: dict[str, Any],
    n_mc: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))

    # Null model: keep per-dataset offsets but force common slope = 0.
    n = int(y.size)
    n_ds = int(np.max(dataset_id)) + 1
    X0 = np.zeros((n, n_ds), dtype=float)
    X0[np.arange(n), dataset_id] = 1.0
    inv_cov = fit["inv_cov"]

    fisher0 = X0.T @ inv_cov @ X0
    beta0 = np.linalg.inv(fisher0) @ (X0.T @ inv_cov @ y)
    y0 = X0 @ beta0

    # Slope estimator is linear in y: b_hat = w @ y
    X = fit["X"]
    w = (fit["fisher_inv"] @ (X.T @ inv_cov))[-1, :]
    b_obs = float(fit["slope"])

    draws = rng.multivariate_normal(mean=y0, cov=cov, size=int(n_mc))
    b_draw = draws @ w

    b_null_mean = float(np.mean(b_draw))
    b_null_sd = float(np.std(b_draw, ddof=1)) if b_draw.size > 1 else 0.0
    if b_obs >= b_null_mean:
        p_one_sided = float(np.mean(b_draw >= b_obs))
    else:
        p_one_sided = float(np.mean(b_draw <= b_obs))
    p_two_sided = float(np.mean(np.abs(b_draw - b_null_mean) >= abs(b_obs - b_null_mean)))

    q16, q50, q84 = [float(q) for q in np.percentile(b_draw, [16.0, 50.0, 84.0])]
    return {
        "n_mc": int(n_mc),
        "seed": int(seed),
        "b_null_mean": b_null_mean,
        "b_null_sd": b_null_sd,
        "b_null_q16": q16,
        "b_null_q50": q50,
        "b_null_q84": q84,
        "p_one_sided_extreme": p_one_sided,
        "p_two_sided_extreme": p_two_sided,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="H-agnostic observable-level slope test on RSD fσ8(z) datasets."
    )
    ap.add_argument(
        "--datasets",
        default="sdss_dr12_consensus_fs,sdss_dr16_lrg_fsbao_dmdhfs8",
        help="Comma-separated single-survey RSD datasets.",
    )
    ap.add_argument(
        "--x-mode",
        choices=["log1p", "z"],
        default="log1p",
        help="Redshift coordinate for common slope (default: log1p).",
    )
    ap.add_argument("--n-mc", type=int, default=20000, help="Null Monte Carlo draws.")
    ap.add_argument("--seed", type=int, default=20260210)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    datasets = _parse_csv_str(args.datasets)
    out_dir = Path(args.out) if args.out else Path("outputs") / f"observable_only_slope_{_utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = DataPaths(repo_root=Path.cwd())
    z_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    cov_parts: list[np.ndarray] = []
    ds_id_parts: list[np.ndarray] = []
    ds_meta: list[dict[str, Any]] = []

    for ds_i, ds_name in enumerate(datasets):
        rsd = load_rsd_single_survey(paths=paths, dataset=ds_name)
        z = np.asarray(rsd.z, dtype=float)
        y = np.asarray(rsd.fs8, dtype=float)
        cov = np.asarray(rsd.cov, dtype=float)
        if z.size == 0:
            raise RuntimeError(f"Dataset '{ds_name}' returned zero points.")
        if np.any(y <= 0.0):
            raise RuntimeError(f"Dataset '{ds_name}' has non-positive fσ8 values.")
        z_parts.append(z)
        y_parts.append(y)
        cov_parts.append(cov)
        ds_id_parts.append(np.full(z.size, int(ds_i), dtype=int))
        ds_meta.append({"name": ds_name, "n_points": int(z.size)})

    z_all = np.concatenate(z_parts)
    y_all = np.concatenate(y_parts)
    cov_all = _block_diag(cov_parts)
    ds_id = np.concatenate(ds_id_parts)

    if args.x_mode == "log1p":
        x_raw = np.log1p(z_all)
    else:
        x_raw = z_all.copy()
    x = _weighted_center(x_raw, cov_all)

    fit = _fit_gls(y=y_all, x=x, dataset_id=ds_id, cov=cov_all)
    null = _null_mc(
        y=y_all,
        x=x,
        dataset_id=ds_id,
        cov=cov_all,
        fit=fit,
        n_mc=int(args.n_mc),
        seed=int(args.seed),
    )

    summary = {
        "created_utc": _utc_stamp(),
        "out_dir": str(out_dir),
        "datasets": ds_meta,
        "n_total_points": int(y_all.size),
        "x_mode": str(args.x_mode),
        "fit": {
            "slope": float(fit["slope"]),
            "slope_sigma": float(fit["slope_sigma"]),
            "z_score": float(fit["z_score"]),
            "p_slope_gt_0": float(fit["p_slope_gt_0"]),
            "p_slope_lt_0": float(fit["p_slope_lt_0"]),
            "chi2": float(fit["chi2"]),
            "dof": int(y_all.size - (len(ds_meta) + 1)),
            "beta_hat": [float(x) for x in np.asarray(fit["beta_hat"]).tolist()],
        },
        "null_calibration": null,
    }

    _write_json_atomic(out_dir / "summary.json", summary)
    (out_dir / "report.md").write_text(
        "\n".join(
            [
                "# Observable-only slope test",
                "",
                f"- Datasets: `{', '.join(d['name'] for d in ds_meta)}`",
                f"- Total points: `{int(y_all.size)}`",
                f"- x-mode: `{args.x_mode}`",
                "",
                "## Fit",
                "",
                f"- slope = `{summary['fit']['slope']:.6f}` +/- `{summary['fit']['slope_sigma']:.6f}`",
                f"- z-score = `{summary['fit']['z_score']:.3f}`",
                f"- P(slope > 0) = `{summary['fit']['p_slope_gt_0']:.3f}`",
                f"- P(slope < 0) = `{summary['fit']['p_slope_lt_0']:.3f}`",
                "",
                "## Null MC",
                "",
                f"- n_mc = `{summary['null_calibration']['n_mc']}`",
                f"- null mean/sd = `{summary['null_calibration']['b_null_mean']:.6f}` / `{summary['null_calibration']['b_null_sd']:.6f}`",
                f"- one-sided tail p = `{summary['null_calibration']['p_one_sided_extreme']:.4f}`",
                f"- two-sided tail p = `{summary['null_calibration']['p_two_sided_extreme']:.4f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "[observable-only] slope={:.6f} +/- {:.6f}, P(s<0)={:.3f}, null two-sided p={:.4f}".format(
            summary["fit"]["slope"],
            summary["fit"]["slope_sigma"],
            summary["fit"]["p_slope_lt_0"],
            summary["null_calibration"]["p_two_sided_extreme"],
        ),
        flush=True,
    )
    print(f"[observable-only] wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
