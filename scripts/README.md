# Scripts

- `run_realdata_recon.py`: end-to-end real-data pipeline + report
- `run_synthetic_closure.py`: generate mock data and validate recovery
- `run_ablation_suite.py`: kernel/prior/covariance/domain ablations
- `run_optical_bias_smoke.py`: optical-bias smoke test (mock or minimal)
- `run_optical_bias_realdata.py`: optical-bias real-data pipeline (Track A+B)
- `run_optical_bias_ablation.py`: optical-bias ablation suite

## Resource-safe defaults

All scripts set `OMP_NUM_THREADS=1` (and related BLAS/OpenMP env vars) to avoid nested parallelism.

Useful flags:
- `--cpu-cores N`: best-effort CPU affinity limiter (`0` = use all detected cores; respects pre-set task affinity masks for auto sizing)
- `--mu-procs N`, `--gp-procs N`, `--procs N`: multiprocessing workers (`0` = auto; `mu-procs` cap is walkers for `emcee`, walkers*temps for `ptemcee`)
- `--max-rss-mb MB`: per-process memory cap (best-effort; raises `MemoryError` instead of OOM)
- `run_synthetic_closure.py --fast`: minimal-cost settings for debugging (also skips the GP derivative ablation by default)
