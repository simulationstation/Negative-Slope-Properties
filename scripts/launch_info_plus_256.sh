#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
out_base="${2:-}"

if [ -z "$mode" ]; then
  echo "Usage: $0 <smoke|pilot|full> [out_base]" >&2
  exit 2
fi

py=".venv/bin/python"
if [ ! -x "$py" ]; then
  py="python3"
fi

common_env=(
  OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  PYTHONUNBUFFERED=1
  NO_LENSING=1
)

common_args=(
  --mu-sampler ptemcee
  --pt-ntemps 4
  --pt-tmax 10
  --include-rsd
  --rsd-mode dr16_lrg_fs8
  --include-planck-lensing-clpp
  --clpp-backend camb
  --include-fullshape-pk
  --skip-ablations
  --skip-hz-recon
  --gp-procs 1
)

case "$mode" in
  smoke)
    out_base="${out_base:-outputs/smoke_scaling_256}"
    out="$out_base/M0_start203"
    mkdir -p "$out"
    nohup env "${common_env[@]}" \
      "$py" scripts/run_realdata_recon.py \
        --out "$out" \
        --seed 203 --mu-init-seed 203 \
        --mu-steps 30 --mu-burn 5 --mu-draws 10 \
        --mu-walkers 256 --mu-procs 256 \
        --checkpoint-every 10 \
        --save-chain "$out/samples/mu_chain.npz" \
        "${common_args[@]}" \
      > "$out/run.log" 2>&1 &
    echo "Started scaling smoke (pid=$!) out=$out"
    ;;
  pilot)
    out_base="${out_base:-outputs/pilot_info_plus_256}"
    out="$out_base/M0_start101"
    mkdir -p "$out"
    nohup env "${common_env[@]}" \
      "$py" scripts/run_realdata_recon.py \
        --out "$out" \
        --seed 101 --mu-init-seed 101 \
        --mu-steps 600 --mu-burn 200 --mu-draws 300 \
        --mu-walkers 256 --mu-procs 256 \
        --checkpoint-every 100 \
        --save-chain "$out/samples/mu_chain.npz" \
        "${common_args[@]}" \
      > "$out/run.log" 2>&1 &
    echo "Started pilot (pid=$!) out=$out"
    ;;
  full)
    out_base="${out_base:-outputs/finalization/info_plus_full_256}"
    mkdir -p "$out_base"

    # 5 seeds, each pinned to 51 cores => uses 255/256 cores.
    # Adjust if CPU numbering differs.
    coresets=("0-50" "51-101" "102-152" "153-203" "204-254")
    seeds=(101 202 303 404 505)

    for i in "${!seeds[@]}"; do
      s="${seeds[$i]}"
      c="${coresets[$i]}"
      out="$out_base/M0_start${s}"
      mkdir -p "$out"
      nohup env "${common_env[@]}" \
        taskset -c "$c" \
        "$py" scripts/run_realdata_recon.py \
          --out "$out" \
          --seed "$s" --mu-init-seed "$s" \
          --mu-steps 1500 --mu-burn 500 --mu-draws 800 \
          --mu-walkers 64 --mu-procs 0 \
          --checkpoint-every 100 \
          --save-chain "$out/samples/mu_chain.npz" \
          "${common_args[@]}" \
        > "$out/run.log" 2>&1 &
      echo "Started seed $s on cores $c (pid=$!) out=$out"
    done
    ;;
  *)
    echo "Unknown mode: $mode (expected: smoke|pilot|full)" >&2
    exit 2
    ;;
esac
