#!/usr/bin/env bash
set -u -o pipefail

# Wrapper tuned for a 128-core / 256-thread host.
#
# Default behavior is one high-fidelity run using all 256 logical CPUs.
#
# Usage:
#   out_base="outputs/finalization/info_plus_gamma_256c_$(date -u +%Y%m%d_%H%M%S)"
#   mkdir -p "$out_base"
#   nohup bash scripts/launch_info_plus_gamma_256c_nohup.sh "$out_base" M0 --no-ptemcee-fallback > "$out_base/launcher.log" 2>&1 &
#
# Optional env:
#   INFO_PLUS_LAYOUT=single|dual|quad   (default: single)
#   INFO_PLUS_SEEDS="101,202,..."
#   INFO_PLUS_STEPS / INFO_PLUS_BURN / INFO_PLUS_DRAWS / INFO_PLUS_WALKERS / INFO_PLUS_GAMMA

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

layout="${INFO_PLUS_LAYOUT:-single}"
total_cores="${INFO_PLUS_TOTAL_CORES:-256}"

case "$layout" in
  single)
    : "${INFO_PLUS_SEEDS:=101}"
    : "${INFO_PLUS_CORES_PER_JOB:=256}"
    : "${INFO_PLUS_MU_PROCS:=256}"
    ;;
  dual)
    : "${INFO_PLUS_SEEDS:=101,202}"
    : "${INFO_PLUS_CORES_PER_JOB:=128}"
    : "${INFO_PLUS_MU_PROCS:=128}"
    ;;
  quad)
    : "${INFO_PLUS_SEEDS:=101,202,303,404}"
    : "${INFO_PLUS_CORES_PER_JOB:=64}"
    : "${INFO_PLUS_MU_PROCS:=64}"
    ;;
  *)
    echo "ERROR: INFO_PLUS_LAYOUT must be one of: single, dual, quad (got: $layout)" >&2
    exit 2
    ;;
esac

export INFO_PLUS_TOTAL_CORES="$total_cores"
export INFO_PLUS_SEEDS
export INFO_PLUS_CORES_PER_JOB
export INFO_PLUS_MU_PROCS

echo "[256c-wrapper] layout=$layout total_cores=$INFO_PLUS_TOTAL_CORES seeds=$INFO_PLUS_SEEDS cores_per_job=$INFO_PLUS_CORES_PER_JOB mu_procs=$INFO_PLUS_MU_PROCS"

exec bash scripts/launch_info_plus_gamma_large_single_nohup.sh "$@"

