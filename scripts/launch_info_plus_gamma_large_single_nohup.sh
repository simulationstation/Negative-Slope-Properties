#!/usr/bin/env bash
set -u -o pipefail

# Launch a larger INFO+ gamma-mode suite on this host with robust detached seeds.
#
# Usage:
#   out_base="outputs/finalization/info_plus_gamma_large_$(date -u +%Y%m%d_%H%M%S)"
#   mkdir -p "$out_base"
#   nohup bash scripts/launch_info_plus_gamma_large_single_nohup.sh "$out_base" M0 > "$out_base/launcher.log" 2>&1 &
#
# Optional env overrides:
#   INFO_PLUS_SEEDS="101,202,303"
#   INFO_PLUS_GAMMA="0.55"
#   INFO_PLUS_STEPS="2200"
#   INFO_PLUS_BURN="700"
#   INFO_PLUS_DRAWS="1000"
#   INFO_PLUS_WALKERS="64"
#   INFO_PLUS_TOTAL_CORES="256"
#   INFO_PLUS_CORES_PER_JOB="128"
#   INFO_PLUS_MU_PROCS="128"
#   INFO_PLUS_CHECKPOINT_EVERY="100"
#   INFO_PLUS_PT_NTEMPS="4"
#   INFO_PLUS_PT_TMAX="10"
#
# Notes:
# - Requests ptemcee by default, but run_realdata_recon.py now auto-falls back to emcee
#   when MPI runtime libs are unavailable.
# - Uses all detected cores by default, split across selected seeds/jobs.
# - Per-job --mu-procs is pinned explicitly to avoid oversubscription on high-core hosts.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Positional parsing:
#   [out_base] [mapping_variant] [omega_k0_prior_for_M2] [-- extra run_realdata_recon args]
# Robust to direct passthrough flags (e.g. --no-ptemcee-fallback).
out_base_default="outputs/finalization/info_plus_gamma_large"
out_base="$out_base_default"
mapping_variant="M0"   # M0|M1|M2
omega_k0_prior=""      # only used for M2

if [ $# -ge 1 ] && [[ "${1:-}" != -* ]]; then
  out_base="$1"
  shift
fi
if [ $# -ge 1 ] && [[ "${1:-}" =~ ^M[012]$ ]]; then
  mapping_variant="$1"
  shift
fi
if [ "$mapping_variant" = "M2" ] && [ $# -ge 1 ] && [[ "${1:-}" != -* ]]; then
  omega_k0_prior="$1"
  shift
fi

mkdir -p "$out_base"
echo "$$" > "$out_base/queue.pid"
extra_args=("$@")

if [ ! -x ".venv/bin/python" ]; then
  echo "ERROR: missing .venv/bin/python (create venv + install deps first)" >&2
  exit 2
fi

common_env=(
  OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  PYTHONUNBUFFERED=1
  NO_LENSING=1
  EHR_CAMB_CACHE_DIR="${EHR_CAMB_CACHE_DIR:-data/cache/camb}"
)

extra_map_args=()
case "$mapping_variant" in
  M0)
    ;;
  M1)
    extra_map_args+=(--mapping-variant M1)
    ;;
  M2)
    extra_map_args+=(--mapping-variant M2)
    if [ -n "${omega_k0_prior}" ]; then
      # shellcheck disable=SC2206
      extra_map_args+=(--omega-k0-prior ${omega_k0_prior})
    fi
    ;;
  *)
    echo "ERROR: mapping_variant must be one of: M0, M1, M2 (got: ${mapping_variant})" >&2
    exit 2
    ;;
esac

seed_csv="${INFO_PLUS_SEEDS:-101,202}"
IFS=',' read -r -a seeds <<< "$seed_csv"
if [ "${#seeds[@]}" -eq 0 ]; then
  echo "ERROR: no seeds provided." >&2
  exit 2
fi

total_cores="${INFO_PLUS_TOTAL_CORES:-$(nproc)}"
if ! [[ "$total_cores" =~ ^[0-9]+$ ]] || [ "$total_cores" -lt 1 ]; then
  echo "ERROR: INFO_PLUS_TOTAL_CORES must be a positive integer (got: $total_cores)" >&2
  exit 2
fi

jobs="${#seeds[@]}"
if [ "$jobs" -gt "$total_cores" ]; then
  jobs="$total_cores"
  seeds=("${seeds[@]:0:$jobs}")
fi

cores_per_job_env="${INFO_PLUS_CORES_PER_JOB:-}"
if [ -n "$cores_per_job_env" ]; then
  if ! [[ "$cores_per_job_env" =~ ^[0-9]+$ ]] || [ "$cores_per_job_env" -lt 1 ]; then
    echo "ERROR: INFO_PLUS_CORES_PER_JOB must be a positive integer (got: $cores_per_job_env)" >&2
    exit 2
  fi
  if [ "$cores_per_job_env" -gt "$total_cores" ]; then
    cores_per_job_env="$total_cores"
  fi
  max_jobs=$((total_cores / cores_per_job_env))
  if [ "$max_jobs" -lt 1 ]; then
    max_jobs=1
  fi
  if [ "$jobs" -gt "$max_jobs" ]; then
    jobs="$max_jobs"
    seeds=("${seeds[@]:0:$jobs}")
  fi
  cores_per="$cores_per_job_env"
  remainder=0
else
  cores_per=$((total_cores / jobs))
  remainder=$((total_cores % jobs))
fi

steps="${INFO_PLUS_STEPS:-2200}"
burn="${INFO_PLUS_BURN:-700}"
draws="${INFO_PLUS_DRAWS:-1000}"
walkers="${INFO_PLUS_WALKERS:-64}"
gamma="${INFO_PLUS_GAMMA:-0.55}"
checkpoint_every="${INFO_PLUS_CHECKPOINT_EVERY:-100}"
pt_ntemps="${INFO_PLUS_PT_NTEMPS:-4}"
pt_tmax="${INFO_PLUS_PT_TMAX:-10}"
mu_procs_env="${INFO_PLUS_MU_PROCS:-}"

common_args=(
  --mu-sampler ptemcee
  --pt-ntemps "$pt_ntemps"
  --pt-tmax "$pt_tmax"
  --mu-steps "$steps"
  --mu-burn "$burn"
  --mu-draws "$draws"
  --mu-walkers "$walkers"
  --checkpoint-every "$checkpoint_every"
  --include-rsd
  --rsd-mode dr16_lrg_fs8
  --drop-bao dr16
  --include-planck-lensing-clpp
  --clpp-backend camb
  --include-fullshape-pk
  --growth-mode gamma
  --growth-gamma "$gamma"
  --skip-ablations
  --skip-hz-recon
  --gp-procs 1
)

pids=()
pids_path="$out_base/pids.txt"
: > "$pids_path"

watchdog_enable="${INFO_PLUS_WATCHDOG:-1}"
watchdog_kill_on_fail="${INFO_PLUS_WATCHDOG_KILL_ON_FAIL:-1}"
watchdog_interval="${INFO_PLUS_WATCHDOG_INTERVAL:-30}"
watchdog_stall_secs="${INFO_PLUS_WATCHDOG_STALL_SECS:-1800}"
watchdog_hard_stall_secs="${INFO_PLUS_WATCHDOG_HARD_STALL_SECS:-7200}"
watchdog_init_stall_secs="${INFO_PLUS_WATCHDOG_INIT_STALL_SECS:-900}"
watchdog_zero_proc_grace_secs="${INFO_PLUS_WATCHDOG_ZERO_PROC_GRACE_SECS:-180}"
watchdog_warmup_secs="${INFO_PLUS_WATCHDOG_WARMUP_SECS:-300}"
watchdog_min_active_cpu_pct="${INFO_PLUS_WATCHDOG_MIN_ACTIVE_CPU_PCT:-20}"

echo "[launcher] repo_root=$repo_root"
echo "[launcher] out_base=$out_base"
echo "[launcher] seeds=${seeds[*]}"
echo "[launcher] total_cores=$total_cores jobs=$jobs"
echo "[launcher] steps=$steps burn=$burn draws=$draws walkers=$walkers gamma=$gamma checkpoint_every=$checkpoint_every"
echo "[launcher] pt_ntemps=$pt_ntemps pt_tmax=$pt_tmax"

if [ "$watchdog_enable" != "0" ]; then
  watchdog_args=(
    --out-base "$out_base"
    --interval "$watchdog_interval"
    --stall-secs "$watchdog_stall_secs"
    --hard-stall-secs "$watchdog_hard_stall_secs"
    --init-stall-secs "$watchdog_init_stall_secs"
    --zero-proc-grace-secs "$watchdog_zero_proc_grace_secs"
    --warmup-secs "$watchdog_warmup_secs"
    --min-active-cpu-pct "$watchdog_min_active_cpu_pct"
  )
  if [ "$watchdog_kill_on_fail" != "0" ]; then
    watchdog_args+=(--kill-on-fail)
  fi
  env PYTHONUNBUFFERED=1 \
    setsid .venv/bin/python scripts/watch_out_base.py "${watchdog_args[@]}" \
    > "$out_base/watchdog.log" 2>&1 < /dev/null &
  watchdog_pid="$!"
  echo "$watchdog_pid" > "$out_base/watchdog.pid"
  echo "[launcher] watchdog pid=$watchdog_pid interval=${watchdog_interval}s stall=${watchdog_stall_secs}s hard_stall=${watchdog_hard_stall_secs}s init_stall=${watchdog_init_stall_secs}s min_active_cpu=${watchdog_min_active_cpu_pct}%"
fi

start_core=0
for i in "${!seeds[@]}"; do
  s="${seeds[$i]}"
  extra=0
  if [ "$i" -lt "$remainder" ]; then
    extra=1
  fi
  core_count=$((cores_per + extra))
  end_core=$((start_core + core_count - 1))
  cpuset="${start_core}-${end_core}"
  start_core=$((end_core + 1))

  mu_procs="$core_count"
  if [ -n "$mu_procs_env" ]; then
    if ! [[ "$mu_procs_env" =~ ^[0-9]+$ ]] || [ "$mu_procs_env" -lt 1 ]; then
      echo "ERROR: INFO_PLUS_MU_PROCS must be a positive integer (got: $mu_procs_env)" >&2
      exit 2
    fi
    mu_procs="$mu_procs_env"
  fi
  if [ "$mu_procs" -gt "$core_count" ]; then
    mu_procs="$core_count"
  fi

  out="$out_base/${mapping_variant}_start${s}"
  mkdir -p "$out"

  echo "[launcher] seed=$s cpuset=$cpuset mu_procs=$mu_procs out=$out"
  {
    echo ""
    echo "[launcher] $(date -u +%Y-%m-%dT%H:%M:%SZ) seed=$s cpuset=$cpuset mu_procs=$mu_procs out=$out"
    echo "[launcher] cmd=.venv/bin/python scripts/run_realdata_recon.py --out $out --seed $s --mu-init-seed $s ..."
  } >> "$out/run.log"

  env "${common_env[@]}" \
    setsid taskset -c "$cpuset" \
    .venv/bin/python scripts/run_realdata_recon.py \
      --out "$out" \
      --seed "$s" \
      --mu-init-seed "$s" \
      --save-chain "$out/samples/mu_chain.npz" \
      "${extra_map_args[@]}" \
      --mu-procs "$mu_procs" \
      "${common_args[@]}" \
      "${extra_args[@]}" \
    >> "$out/run.log" 2>&1 < /dev/null &
  pid="$!"
  pids+=("$pid")
  echo "$pid" >> "$pids_path"
  echo "[launcher] seed=$s pid=$pid"
done

echo "[launcher] pids written to $pids_path"
echo "[launcher] launched all seeds (detached)."
echo "[launcher] monitor with: .venv/bin/python scripts/status_out_base.py \"$out_base\""
exit 0
