#!/usr/bin/env bash
set -u -o pipefail

# Stability-gate launcher for INFO+ gamma-mode runs.
#
# Designed for one detached top-level launch:
#   out_base="outputs/finalization/info_plus_gamma_stability_gate_256c_$(date -u +%Y%m%d_%H%M%S)"
#   mkdir -p "$out_base"
#   nohup bash scripts/launch_info_plus_gamma_stability_gate_single_nohup.sh "$out_base" M0 \
#     --no-ptemcee-fallback > "$out_base/launcher.log" 2>&1 &
#
# Default gate:
# - seeds: 101,202,303
# - steps: 300, burn: 100, draws: 250
# - sampler: ptemcee with pt_ntemps=6, pt_tmax=10
# - execution mode: sequential (set INFO_PLUS_MODE=parallel to run all seeds concurrently)
#
# Optional env overrides:
#   INFO_PLUS_SEEDS="101,202,303"
#   INFO_PLUS_STEPS="300"
#   INFO_PLUS_BURN="100"
#   INFO_PLUS_DRAWS="250"
#   INFO_PLUS_WALKERS="48"
#   INFO_PLUS_GAMMA="0.55"
#   INFO_PLUS_PT_NTEMPS="6"
#   INFO_PLUS_PT_TMAX="10"
#   INFO_PLUS_TOTAL_CORES="256"
#   INFO_PLUS_MU_PROCS="256"
#   INFO_PLUS_CHECKPOINT_EVERY="100"
#   INFO_PLUS_MODE="sequential"   # or "parallel"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_base_default="outputs/finalization/info_plus_gamma_stability_gate_256c_$(date -u +%Y%m%d_%H%M%S)"
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
  EHR_CAMB_DISK_CACHE="${EHR_CAMB_DISK_CACHE:-0}"
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
    if [ -n "$omega_k0_prior" ]; then
      # shellcheck disable=SC2206
      extra_map_args+=(--omega-k0-prior ${omega_k0_prior})
    fi
    ;;
  *)
    echo "ERROR: mapping_variant must be one of: M0, M1, M2 (got: $mapping_variant)" >&2
    exit 2
    ;;
esac

seed_csv="${INFO_PLUS_SEEDS:-101,202,303}"
IFS=',' read -r -a seeds <<< "$seed_csv"
if [ "${#seeds[@]}" -eq 0 ]; then
  echo "ERROR: no seeds provided." >&2
  exit 2
fi
mode="${INFO_PLUS_MODE:-sequential}"
if [ "$mode" != "sequential" ] && [ "$mode" != "parallel" ]; then
  echo "ERROR: INFO_PLUS_MODE must be 'sequential' or 'parallel' (got: $mode)" >&2
  exit 2
fi

steps="${INFO_PLUS_STEPS:-300}"
burn="${INFO_PLUS_BURN:-100}"
draws="${INFO_PLUS_DRAWS:-250}"
walkers="${INFO_PLUS_WALKERS:-48}"
gamma="${INFO_PLUS_GAMMA:-0.55}"
checkpoint_every="${INFO_PLUS_CHECKPOINT_EVERY:-50}"
pt_ntemps="${INFO_PLUS_PT_NTEMPS:-6}"
pt_tmax="${INFO_PLUS_PT_TMAX:-10}"
total_cores="${INFO_PLUS_TOTAL_CORES:-$(nproc)}"
mu_procs="${INFO_PLUS_MU_PROCS:-$total_cores}"

if ! [[ "$total_cores" =~ ^[0-9]+$ ]] || [ "$total_cores" -lt 1 ]; then
  echo "ERROR: INFO_PLUS_TOTAL_CORES must be a positive integer (got: $total_cores)" >&2
  exit 2
fi
if ! [[ "$mu_procs" =~ ^[0-9]+$ ]] || [ "$mu_procs" -lt 1 ]; then
  echo "ERROR: INFO_PLUS_MU_PROCS must be a positive integer (got: $mu_procs)" >&2
  exit 2
fi
if [ "$mu_procs" -gt "$total_cores" ]; then
  mu_procs="$total_cores"
fi

cpuset="0-$((total_cores - 1))"
pids_path="$out_base/pids.txt"
queue_log="$out_base/queue.log"
: > "$pids_path"
: > "$queue_log"

watchdog_enable="${INFO_PLUS_WATCHDOG:-1}"
watchdog_kill_on_fail="${INFO_PLUS_WATCHDOG_KILL_ON_FAIL:-1}"
watchdog_interval="${INFO_PLUS_WATCHDOG_INTERVAL:-30}"
watchdog_stall_secs="${INFO_PLUS_WATCHDOG_STALL_SECS:-1800}"
watchdog_hard_stall_secs="${INFO_PLUS_WATCHDOG_HARD_STALL_SECS:-7200}"
watchdog_init_stall_secs="${INFO_PLUS_WATCHDOG_INIT_STALL_SECS:-900}"
watchdog_zero_proc_grace_secs="${INFO_PLUS_WATCHDOG_ZERO_PROC_GRACE_SECS:-180}"
watchdog_warmup_secs="${INFO_PLUS_WATCHDOG_WARMUP_SECS:-300}"
watchdog_min_active_cpu_pct="${INFO_PLUS_WATCHDOG_MIN_ACTIVE_CPU_PCT:-20}"

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

echo "[gate] repo_root=$repo_root" | tee -a "$queue_log"
echo "[gate] out_base=$out_base" | tee -a "$queue_log"
echo "[gate] mapping_variant=$mapping_variant" | tee -a "$queue_log"
echo "[gate] mode=$mode seeds=${seeds[*]}" | tee -a "$queue_log"
echo "[gate] cpuset=$cpuset total_cores=$total_cores mu_procs=$mu_procs" | tee -a "$queue_log"
echo "[gate] steps=$steps burn=$burn draws=$draws walkers=$walkers gamma=$gamma checkpoint_every=$checkpoint_every" | tee -a "$queue_log"
echo "[gate] pt_ntemps=$pt_ntemps pt_tmax=$pt_tmax" | tee -a "$queue_log"

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
  echo "[gate] watchdog pid=$watchdog_pid interval=${watchdog_interval}s stall=${watchdog_stall_secs}s hard_stall=${watchdog_hard_stall_secs}s init_stall=${watchdog_init_stall_secs}s min_active_cpu=${watchdog_min_active_cpu_pct}%" | tee -a "$queue_log"
fi

launch_seed() {
  local s="$1"
  local cpuset_seed="$2"
  local mu_procs_seed="$3"
  out="$out_base/${mapping_variant}_start${s}"
  mkdir -p "$out"
  echo "[gate] $(date -u +%Y-%m-%dT%H:%M:%SZ) launching seed=$s cpuset=$cpuset_seed mu_procs=$mu_procs_seed out=$out" | tee -a "$queue_log"

  {
    echo ""
    echo "[launcher] $(date -u +%Y-%m-%dT%H:%M:%SZ) seed=$s cpuset=$cpuset_seed mu_procs=$mu_procs_seed out=$out"
    echo "[launcher] cmd=.venv/bin/python scripts/run_realdata_recon.py --out $out --seed $s --mu-init-seed $s ..."
  } >> "$out/run.log"

  env "${common_env[@]}" \
    setsid taskset -c "$cpuset_seed" \
    .venv/bin/python scripts/run_realdata_recon.py \
      --out "$out" \
      --seed "$s" \
      --mu-init-seed "$s" \
      --save-chain "$out/samples/mu_chain.npz" \
      "${extra_map_args[@]}" \
      --mu-procs "$mu_procs_seed" \
      "${common_args[@]}" \
      "${extra_args[@]}" \
    >> "$out/run.log" 2>&1 < /dev/null &
  pid="$!"
  echo "$pid" >> "$pids_path"
  echo "[gate] seed=$s pid=$pid cpuset=$cpuset_seed mu_procs=$mu_procs_seed" | tee -a "$queue_log"
}

if [ "$mode" = "sequential" ]; then
  for s in "${seeds[@]}"; do
    launch_seed "$s" "$cpuset" "$mu_procs"
    wait "$pid"
    rc=$?
    echo "[gate] $(date -u +%Y-%m-%dT%H:%M:%SZ) seed=$s exit_code=$rc" | tee -a "$queue_log"
    if [ "$rc" -ne 0 ]; then
      echo "[gate] stopping sequence on failed seed=$s" | tee -a "$queue_log"
      exit "$rc"
    fi
  done
else
  n_seeds="${#seeds[@]}"
  if [ "$total_cores" -lt "$n_seeds" ]; then
    echo "ERROR: parallel mode requires total_cores >= number of seeds ($total_cores < $n_seeds)" | tee -a "$queue_log"
    exit 2
  fi
  base=$(( total_cores / n_seeds ))
  rem=$(( total_cores % n_seeds ))
  start=0
  pids=()
  seed_names=()
  for i in "${!seeds[@]}"; do
    s="${seeds[$i]}"
    cores_i=$base
    if [ "$i" -lt "$rem" ]; then
      cores_i=$((cores_i + 1))
    fi
    end=$((start + cores_i - 1))
    if [ "$start" -eq "$end" ]; then
      cpuset_i="$start"
    else
      cpuset_i="$start-$end"
    fi
    mu_procs_i="$cores_i"
    launch_seed "$s" "$cpuset_i" "$mu_procs_i"
    pids+=("$pid")
    seed_names+=("$s")
    start=$((end + 1))
  done
  rc_any=0
  for i in "${!pids[@]}"; do
    p="${pids[$i]}"
    s="${seed_names[$i]}"
    wait "$p"
    rc=$?
    echo "[gate] $(date -u +%Y-%m-%dT%H:%M:%SZ) seed=$s exit_code=$rc" | tee -a "$queue_log"
    if [ "$rc" -ne 0 ]; then
      rc_any="$rc"
    fi
  done
  if [ "$rc_any" -ne 0 ]; then
    echo "[gate] one or more parallel seeds failed." | tee -a "$queue_log"
    exit "$rc_any"
  fi
fi

echo "[gate] $(date -u +%Y-%m-%dT%H:%M:%SZ) all seeds completed." | tee -a "$queue_log"
echo "[gate] monitor summary: .venv/bin/python scripts/status_out_base.py \"$out_base\"" | tee -a "$queue_log"
exit 0
