#!/usr/bin/env bash
set -euo pipefail

out_dir="outputs/multistart_info_plus"
log_file="outputs/multistart_info_plus/queue.log"
mkdir -p "${out_dir}"

echo "[queue] $(date -Is) waiting for run_realdata_recon.py to be idle" | tee -a "$log_file"
while pgrep -f "run_realdata_recon.py" >/dev/null; do
  echo "[queue] $(date -Is) still running; sleeping 60s" | tee -a "$log_file"
  sleep 60
 done

echo "[queue] $(date -Is) launching Info+ multistart" | tee -a "$log_file"

scripts/run_info_plus_multistart.sh "$out_dir"

echo "[queue] $(date -Is) multistart launched" | tee -a "$log_file"
