#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/restart_negative_slope_decision_grade_256c.sh [OUT_DIR] [extra args...]
# Example:
#   scripts/restart_negative_slope_decision_grade_256c.sh \
#     outputs/REPORTS/negative_slope_decision_grade_20260222_083326UTC

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${1:-outputs/REPORTS/negative_slope_decision_grade_20260222_083326UTC}"
shift || true

SEED="${SEED:-20260222}"
MU_PROCS="${MU_PROCS:-128}"
SBC_PROCS="${SBC_PROCS:-256}"
SBC_CHUNK_SIZE="${SBC_CHUNK_SIZE:-256}"
SBC_WORKER_MAX_TASKS="${SBC_WORKER_MAX_TASKS:-4}"
MAX_RSS_MB="${MAX_RSS_MB:-1536}"

# Prevent nested BLAS/OpenMP oversubscription when running many SBC workers.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

.venv/bin/python scripts/run_negative_slope_decision_grade.py \
  --out "$OUT_DIR" \
  --seed "$SEED" \
  --resume \
  --progress \
  --mu-procs "$MU_PROCS" \
  --sbc-procs "$SBC_PROCS" \
  --sbc-chunk-size "$SBC_CHUNK_SIZE" \
  --sbc-progress-every 1 \
  --sbc-worker-events \
  --sbc-worker-heartbeat-s 30 \
  --sbc-inner-quiet \
  --sbc-limit-blas-threads \
  --sbc-worker-max-tasks "$SBC_WORKER_MAX_TASKS" \
  --max-rss-mb "$MAX_RSS_MB" \
  "$@"
