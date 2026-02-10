#!/usr/bin/env bash
set -u -o pipefail

# Launch the full 256-core "info+" run in a way that survives session disconnects.
#
# Run THIS script under a single nohup (do not wrap each seed in its own nohup):
#
#   out_base="outputs/finalization/info_plus_full_256"
#   mkdir -p "$out_base"
#   nohup bash scripts/launch_full_info_plus_256_single_nohup.sh "$out_base" M0 > "$out_base/launcher.log" 2>&1 &
#
# Mapping sensitivity variants:
#   nohup bash scripts/launch_full_info_plus_256_single_nohup.sh "$out_base" M1 > "$out_base/launcher_M1.log" 2>&1 &
#   nohup bash scripts/launch_full_info_plus_256_single_nohup.sh "$out_base" M2 "-0.2 0.2" > "$out_base/launcher_M2.log" 2>&1 &
#
# Status:
#   .venv/bin/python scripts/status_out_base.py "$out_base"
#
# Stop:
#   xargs -r kill -TERM < "$out_base/pids.txt"
#
# NOTE:
# - Each seed process is started via `setsid ... < /dev/null &` so it is robustly detached
#   from the launching shell/session (this pattern proved more reliable than per-seed nohup).

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_base="${1:-outputs/finalization/info_plus_full_256}"
mapping_variant="${2:-M0}"   # M0|M1|M2
omega_k0_prior="${3:-}"     # only used for M2, e.g. "-0.2 0.2"
mkdir -p "$out_base"
echo "$$" > "$out_base/queue.pid"

if [ $# -ge 3 ]; then
  shift 3
elif [ $# -eq 2 ]; then
  shift 2
elif [ $# -eq 1 ]; then
  shift 1
fi
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
  # Use the full Cℓ^{φφ} likelihood, not the 1D compressed lensing proxy.
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

common_args=(
  --mu-sampler ptemcee
  --pt-ntemps 4
  --pt-tmax 10
  --mu-steps 1500
  --mu-burn 500
  --mu-draws 800
  --mu-walkers 64
  --mu-procs 0
  --checkpoint-every 100
  --include-rsd
  --rsd-mode dr16_lrg_fs8
  # Avoid (fs8, BAO distance) correlation double-counting from the same DR16 LRG covariance.
  --drop-bao dr16
  --include-planck-lensing-clpp
  --clpp-backend camb
  --include-fullshape-pk
  --skip-ablations
  --skip-hz-recon
  --gp-procs 1
)

# 5 seeds pinned to all 256 cores (52 + 51*4 = 256).
seeds=(101 202 303 404 505)
coresets=("0-51" "52-102" "103-153" "154-204" "205-255")

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
echo "[launcher] starting $((${#seeds[@]})) seeds..."

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

for i in "${!seeds[@]}"; do
  s="${seeds[$i]}"
  c="${coresets[$i]}"
  out="$out_base/${mapping_variant}_start${s}"
  mkdir -p "$out"

  echo "[launcher] seed=$s cores=$c out=$out"
  {
    echo ""
    echo "[launcher] $(date -u +%Y-%m-%dT%H:%M:%SZ) seed=$s cores=$c out=$out"
    echo "[launcher] cmd=.venv/bin/python scripts/run_realdata_recon.py --out $out --seed $s --mu-init-seed $s ..."
  } >> "$out/run.log"
  env "${common_env[@]}" \
    setsid taskset -c "$c" \
    .venv/bin/python scripts/run_realdata_recon.py \
      --out "$out" \
      --seed "$s" \
      --mu-init-seed "$s" \
      --save-chain "$out/samples/mu_chain.npz" \
      "${extra_map_args[@]}" \
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
echo "[launcher] use scripts/status_out_base.py to monitor; pids in $pids_path"
exit 0
