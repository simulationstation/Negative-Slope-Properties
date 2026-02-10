#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def _load_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    deltas = []
    labels = []
    for row in rows:
        if "delta_score_relative_minus_absolute" not in row or "truth_mode" not in row:
            continue
        deltas.append(float(row["delta_score_relative_minus_absolute"]))
        labels.append(1 if str(row["truth_mode"]) == "relative" else 0)
    x = np.asarray(deltas, dtype=float)
    y = np.asarray(labels, dtype=int)
    if x.size == 0:
        raise RuntimeError(f"No usable rows in {path}")
    return x, y


def _candidate_thresholds(x: np.ndarray) -> np.ndarray:
    xs = np.unique(np.asarray(x, dtype=float))
    if xs.size == 1:
        return np.asarray([xs[0]], dtype=float)
    mids = (xs[:-1] + xs[1:]) * 0.5
    return np.concatenate(([xs[0] - 1e-9], mids, [xs[-1] + 1e-9]))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    n = int(y_true.size)
    acc = float((tp + tn) / max(1, n))
    tpr = float(tp / max(1, tp + fn))  # relative recall
    tnr = float(tn / max(1, tn + fp))  # absolute recall
    bal_acc = 0.5 * (tpr + tnr)
    return {
        "n": n,
        "tp_relative": tp,
        "tn_absolute": tn,
        "fp_relative": fp,
        "fn_absolute": fn,
        "accuracy": acc,
        "recall_relative": tpr,
        "recall_absolute": tnr,
        "balanced_accuracy": float(bal_acc),
    }


def _eval_threshold(x: np.ndarray, y: np.ndarray, thr: float) -> dict[str, Any]:
    y_pred = (x >= thr).astype(int)
    m = _metrics(y, y_pred)
    m["threshold"] = float(thr)
    return m


def _choose_best_threshold(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    cands = _candidate_thresholds(x)
    rows = [_eval_threshold(x, y, float(t)) for t in cands.tolist()]
    # Primary: balanced accuracy. Secondary: overall accuracy. Tertiary: closeness to 0.
    best = max(
        rows,
        key=lambda r: (
            float(r["balanced_accuracy"]),
            float(r["accuracy"]),
            -abs(float(r["threshold"])),
        ),
    )
    return {"best": best, "scan": rows}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Calibrate and validate threshold for relative-vs-absolute identifiability scores."
    )
    ap.add_argument("--train-rows", required=True, help="Path to rows_all.json used for threshold calibration.")
    ap.add_argument("--test-rows", default="", help="Optional holdout rows_all.json for validation.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    train_path = Path(args.train_rows).resolve()
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train rows: {train_path}")
    x_train, y_train = _load_rows(train_path)
    fit = _choose_best_threshold(x_train, y_train)
    best_thr = float(fit["best"]["threshold"])
    baseline_zero = _eval_threshold(x_train, y_train, 0.0)

    out: dict[str, Any] = {
        "created_utc": _utc_stamp(),
        "train_rows": str(train_path),
        "train_n": int(x_train.size),
        "train_delta_stats": {
            "mean": float(np.mean(x_train)),
            "sd": float(np.std(x_train, ddof=1)) if x_train.size > 1 else 0.0,
            "p16": float(np.percentile(x_train, 16.0)),
            "p50": float(np.percentile(x_train, 50.0)),
            "p84": float(np.percentile(x_train, 84.0)),
            "min": float(np.min(x_train)),
            "max": float(np.max(x_train)),
        },
        "threshold_fit": {
            "best_by_balanced_accuracy": fit["best"],
            "baseline_at_zero_threshold": baseline_zero,
        },
    }

    if str(args.test_rows).strip():
        test_path = Path(args.test_rows).resolve()
        if not test_path.exists():
            raise FileNotFoundError(f"Missing test rows: {test_path}")
        x_test, y_test = _load_rows(test_path)
        out["test_rows"] = str(test_path)
        out["test_n"] = int(x_test.size)
        out["test_eval"] = {
            "at_calibrated_threshold": _eval_threshold(x_test, y_test, best_thr),
            "at_zero_threshold": _eval_threshold(x_test, y_test, 0.0),
        }

    out_dir = Path(args.out) if args.out else Path("outputs") / f"growth_mapping_identifiability_calibration_{_utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(out_dir / "summary.json", out)

    lines = [
        "# Growth mapping threshold calibration",
        "",
        f"- Train rows: `{train_path}`",
        f"- Train N: `{x_train.size}`",
        "",
        "## Train fit",
        "",
        f"- best threshold (balanced accuracy): `{best_thr:.6f}`",
        f"- train balanced accuracy at best: `{float(fit['best']['balanced_accuracy']):.6f}`",
        f"- train balanced accuracy at 0: `{float(baseline_zero['balanced_accuracy']):.6f}`",
        "",
    ]
    if "test_eval" in out:
        test_cal = out["test_eval"]["at_calibrated_threshold"]
        test_zero = out["test_eval"]["at_zero_threshold"]
        lines.extend(
            [
                "## Holdout",
                "",
                f"- test balanced accuracy at calibrated threshold: `{float(test_cal['balanced_accuracy']):.6f}`",
                f"- test balanced accuracy at 0 threshold: `{float(test_zero['balanced_accuracy']):.6f}`",
                f"- test recall(relative) at calibrated threshold: `{float(test_cal['recall_relative']):.6f}`",
                f"- test recall(absolute) at calibrated threshold: `{float(test_cal['recall_absolute']):.6f}`",
            ]
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote threshold calibration outputs: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

