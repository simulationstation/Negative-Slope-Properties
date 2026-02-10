#!/usr/bin/env bash
set -euo pipefail

base="outputs/multistart_info_plus"
log="${base}/postprocess.log"
mkdir -p "$base"

echo "[post] $(date -Is) waiting for multistart outputs" | tee -a "$log"
while true; do
  if ! pgrep -f "run_realdata_recon.py" >/dev/null; then
    count=$(ls -1 ${base}/M0_start*/tables/summary.json 2>/dev/null | wc -l | tr -d ' ')
    if [ "$count" -ge 5 ]; then
      break
    fi
  fi
  echo "[post] $(date -Is) waiting; summaries=$count" | tee -a "$log"
  sleep 60
 done

echo "[post] $(date -Is) collecting summary" | tee -a "$log"
python3 scripts/collect_finalization_summary.py --base "$base" --out FINDINGS/info_plus_summary.tsv

# Convert to JSON for aggregator input
python3 - <<'PY'
import csv, json
from pathlib import Path
src = Path('FINDINGS/info_plus_summary.tsv')
rows = []
with src.open() as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        rows.append({
            'seed': int(row['seed']),
            'omega_p50': float(row['omega_p50']),
            'omega_p16': float(row['omega_p16']),
            'omega_p84': float(row['omega_p84']),
            'm_mean': float(row['m_mean']),
            'm_std': float(row['m_std']),
        })
Path('FINDINGS/info_plus_summary.json').write_text(json.dumps({'rows': rows}, indent=2))
PY

python3 scripts/aggregate_multistart_m.py --in FINDINGS/info_plus_summary.json --outdir FINDINGS/

# Rename outputs for clarity
mv -f FINDINGS/m_aggregate.json FINDINGS/info_plus_m_aggregate.json
mv -f FINDINGS/m_aggregate.md FINDINGS/info_plus_m_aggregate.md

echo "[post] $(date -Is) done" | tee -a "$log"
