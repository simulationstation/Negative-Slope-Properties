# Negative-Slope-Properties

Clean, referee-facing software repository for the negative-slope / info+ analysis pipeline.

## Scope

This repository packages the core code used to:

- run `info+` reconstruction jobs (`run_realdata_recon.py`);
- perform identifiability injections and threshold calibration;
- run power-map sensitivity checks used as pre-decision gates.

Large local artifacts (multi-GB/ TB caches, processed catalogs, and long-run outputs) are intentionally excluded.

## Included

- `src/entropy_horizon_recon/` core inference and likelihood code.
- `scripts/` launchers + status tools + analysis scripts for info+ and growth-mapping gates.
- `run_cards/` reference run cards used in production-style runs.
- `tests/` smoke and unit-style checks.
- `plan_forward_info+.md` short pipeline roadmap.

## Not Included

- Heavy `data/cache`, `data/processed`, and `outputs/` directories.
- Precomputed long-run artifacts.

The code is configured to regenerate required inputs through existing ingest/fetch paths when available.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

## Minimal Validation

Run the fast smoke checks first:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_info_plus_smoke.py --out outputs/smoke_info_plus
PYTHONPATH=src .venv/bin/python scripts/run_info_plus_full_smoke.py --out outputs/smoke_info_plus_full
```

Run a small identifiability gate:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_growth_mapping_identifiability_injection.py \
  --out outputs/growth_mapping_identifiability_smoke \
  --reps 16 --draws-per-run 12 --workers 32
```

Then run a small power map:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_growth_mapping_power_map.py \
  --grid-dir outputs/hubble_tension_mg_transfer_map_full_v1_20260208_045425UTC \
  --datasets sdss_dr16_lrg_fsbao_dmdhfs8 \
  --truth-grid relative:0.2,relative:0.5,relative:0.8,absolute:0.0 \
  --holdouts 4 --reps-per-truth 32 --draws-per-run 16 --workers 64 \
  --calibration-train-rows outputs/growth_mapping_identifiability_full_20260208_080051UTC/rows_all.json \
  --out outputs/growth_mapping_power_map_smoke
```

## Decision-Grade Pattern

1. Run identifiability injections.
2. Fit threshold from calibration rows.
3. Run higher-stat power map.
4. Promote to large info+ runs only if power-map separation is acceptable.

Use detached launchers in `scripts/launch_*_single_nohup.sh` for long runs and monitor with `scripts/status_out_base.py`.

## Repository Layout

- `src/entropy_horizon_recon`: core package
- `scripts`: run/launch/status tooling
- `run_cards`: run definitions
- `tests`: smoke + unit checks
- `data/README.md`: data directory notes

## Notes

- Default runtime settings avoid oversubscription (`OMP_NUM_THREADS=1`, BLAS threads pinned to 1) and rely on process-level parallelism.
- This repository is intended as a clean software handoff for reproducibility review.
