#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/launch_growth_mapping_identifiability_single_nohup.sh <mode>
  scripts/launch_growth_mapping_identifiability_single_nohup.sh <out_root> <mode>

Modes:
  smoke | pilot | full
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -eq 0 ]]; then
  usage
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

timestamp="$(date -u +%Y%m%d_%H%M%SUTC)"
arg1="${1:-smoke}"
arg2="${2:-}"
if [[ -n "$arg2" ]]; then
  out_root="$arg1"
  mode="$arg2"
else
  mode="$arg1"
  out_root="outputs/growth_mapping_identifiability_${mode}_${timestamp}"
fi

case "$mode" in
  smoke|pilot|full) ;;
  *)
    echo "ERROR: unknown mode '$mode' (expected smoke|pilot|full)" >&2
    exit 2
    ;;
esac

if [[ ! -x ".venv/bin/python" ]]; then
  echo "ERROR: missing .venv/bin/python" >&2
  exit 2
fi

n_total="$(nproc)"
if [[ "$n_total" -lt 1 ]]; then
  n_total=1
fi
cpuset="${CPUSET:-0-$((n_total - 1))}"
workers="${WORKERS:-$n_total}"

grid_dir="${GRID_DIR:-outputs/hubble_tension_mg_transfer_map_full_v1_20260208_045425UTC}"
datasets="${DATASETS:-sdss_dr12_consensus_fs,sdss_dr16_lrg_fsbao_dmdhfs8}"
truth_spec="${TRUTH_SPEC:-relative:0.7,absolute:0.0}"
nu_grid="${NU_GRID:-0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
seed="${SEED:-20260208}"
heartbeat_sec="${HEARTBEAT_SEC:-60}"
offset_prior_sigma="${OFFSET_PRIOR_SIGMA:-0.03}"
reps_per_truth=16
draws_per_run=8

case "$mode" in
  smoke)
    reps_per_truth=16
    draws_per_run=8
    workers="${WORKERS:-32}"
    ;;
  pilot)
    reps_per_truth=64
    draws_per_run=24
    workers="${WORKERS:-$n_total}"
    ;;
  full)
    reps_per_truth=256
    draws_per_run=48
    workers="${WORKERS:-$n_total}"
    ;;
esac

mkdir -p "$out_root"
job_sh="$out_root/job.sh"
run_log="$out_root/run.log"
pid_file="$out_root/pid.txt"
manifest="$out_root/launcher_manifest.json"

cat > "$manifest" <<JSON
{
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "repo_root": "$repo_root",
  "out_root": "$out_root",
  "mode": "$mode",
  "cpuset": "$cpuset",
  "workers": $workers,
  "grid_dir": "$grid_dir",
  "datasets": "$datasets",
  "truth_spec": "$truth_spec",
  "nu_grid": "$nu_grid",
  "reps_per_truth": $reps_per_truth,
  "draws_per_run": $draws_per_run,
  "offset_prior_sigma": $offset_prior_sigma,
  "seed": $seed,
  "heartbeat_sec": $heartbeat_sec
}
JSON

cat > "$job_sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$repo_root"
echo "[job] started \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[job] mode=$mode out_root=$out_root"
PYTHONPATH=src .venv/bin/python scripts/run_growth_mapping_identifiability_injection.py \\
  --grid-dir "$grid_dir" \\
  --datasets "$datasets" \\
  --truth-spec "$truth_spec" \\
  --nu-grid "$nu_grid" \\
  --reps-per-truth "$reps_per_truth" \\
  --draws-per-run "$draws_per_run" \\
  --offset-prior-sigma "$offset_prior_sigma" \\
  --workers "$workers" \\
  --seed "$seed" \\
  --heartbeat-sec "$heartbeat_sec" \\
  --resume \\
  --out "$out_root"
echo "[job] finished \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF
chmod +x "$job_sh"

env \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  PYTHONUNBUFFERED=1 \
  setsid taskset -c "$cpuset" bash "$job_sh" > "$run_log" 2>&1 < /dev/null &

pid="$!"
echo "$pid" > "$pid_file"
echo "[launcher] started mode=$mode"
echo "[launcher] out_root=$out_root"
echo "[launcher] pid=$pid"
echo "[launcher] run_log=$run_log"

