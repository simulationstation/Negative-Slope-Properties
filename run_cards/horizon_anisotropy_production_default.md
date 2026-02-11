# Horizon Anisotropy Production Run Card (Default)

This run card executes the directional production workflow with:
- out-of-sample axis selection (`crossfit`);
- survey-aware matching (`match_mode=survey_z`);
- look-elsewhere null calibration (sharded hemisphere null battery);
- multi-seed directional stability checks;
- free growth-index gamma (`growth_mode=gamma`, sampled from prior).

## Command

```bash
PYTHONPATH=src .venv/bin/python scripts/run_horizon_anisotropy_production.py \
  --out-base outputs/horizon_anisotropy_production_default \
  --seeds 123,321,2024 \
  --kfold 5 \
  --null-reps 400 \
  --null-shard-size 100 \
  --mu-procs 1 \
  --axis-jobs 0 \
  --null-axis-jobs 0 \
  --train-axis-jobs 0 \
  --z-threshold 1.5 \
  --null-p-threshold 0.05 \
  --axis-consistency-max-angle-deg 45
```

## Outputs

- `acceptance_criteria.json`: pre-registered thresholds used by the run.
- `seed_*/crossfit/crossfit_summary.json`: fold-wise out-of-sample directional summary.
- `seed_*/hemisphere_null/null_summary.json`: look-elsewhere calibrated p-value summary.
- `production_summary.json`: aggregate pass/fail report, including seed stability angles.
