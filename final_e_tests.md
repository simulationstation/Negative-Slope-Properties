# Final E-Tests Plan (All Three Families, Post Current Run)

## Status Gate

Do not start this plan until the current production run fully finishes and writes its final v1 bundle.

## Scope

Run all three genuinely different families as `decision_grade_v2`, with pre-registered definitions and no post-hoc stat changes:

1. Predictive necessity (held-out log score).
2. Calibrated deviance improvement (BH-fixed vs free-slope).
3. Bayes factor / evidence difference (free vs BH-fixed).

## Pre-Registered Global Rules

1. Freeze model definitions, priors, mappings, sampler settings, and seed schedule before execution.
2. Freeze all split policies before execution.
3. Freeze calibration counts before execution.
4. No swapping primary/secondary labels after seeing results.
5. No replacing stats after seeing outcomes.
6. All three tests are reported regardless of result.

## Models

Use consistent model pair across all three families:

1. `M0` = BH-fixed model (`s = 0`).
2. `M1` = free-slope model (existing free mapping pipeline).

## Family 1: Predictive Necessity

Question:

1. Does slope improve out-of-sample prediction?

Protocol:

1. Pre-specify holdout folds once.
2. For each fold, fit `M0` and `M1` on train.
3. Score held-out data for each model.

Statistic:

1. `DeltaLPD = L_test(M1) - L_test(M0)` (sum over held-out points, then aggregate over folds).

Calibration under BH-null:

1. Simulate BH datasets.
2. Apply identical folds and scoring.
3. Compute `DeltaLPD_sim`.
4. One-sided p-value: `p_pred = P(DeltaLPD_sim >= DeltaLPD_obs)`.

## Family 2: Calibrated Deviance Improvement

Question:

1. Does free slope improve fit quality beyond BH-null expectation?

Protocol:

1. Fit `M0` and `M1` on full data.
2. Extract best attained log-likelihood for each model.

Statistic:

1. `DeltaD = 2 * (ell1_star - ell0_star)`.

Calibration under BH-null:

1. Simulate BH datasets.
2. Refit `M0` and `M1` per replicate.
3. Compute `DeltaD_sim`.
4. One-sided p-value: `p_dev = P(DeltaD_sim >= DeltaD_obs)`.

## Family 3: Bayes Factor / Evidence

Question:

1. Does model evidence support free slope after complexity penalty?

Protocol:

1. Compute `logZ(M0)` and `logZ(M1)` with a pre-specified evidence method.
2. Use one method consistently across all real/null runs.

Statistic:

1. `logBF = logZ(M1) - logZ(M0)`.

Calibration under BH-null:

1. Simulate BH datasets.
2. Compute `logBF_sim`.
3. One-sided p-value: `p_bf = P(logBF_sim >= logBF_obs)`.

Also report conventional BF interpretation as descriptive context only.

## Calibration Counts

Pre-register:

1. `N_BH = 2000` BH-null replicates for calibration of each family.
2. Deterministic seed map shared across families where applicable.
3. Chunked resumable execution with per-replicate persistence.

## Multiple-Testing Control

Use familywise control across the three p-values:

1. Apply Holm-Bonferroni at `alpha_family = 0.05` to `{p_pred, p_dev, p_bf}`.
2. Report adjusted decisions and raw p-values with Wilson CI.

## Decision Rule (v2)

`smoking_gun` only if all are true:

1. At least two of three families reject BH under Holm-Bonferroni at familywise 0.05.
2. Directions are coherent (`DeltaLPD_obs > 0`, `DeltaD_obs > 0`, `logBF_obs > 0` for rejected families).
3. Mapping variant sign/convergence gate passes.
4. Calibration integrity checks pass (no missing replicates, deterministic seeds, no silent worker failures).

Else:

1. `suggestive_preference`.

## Implementation Plan (After Current Run)

### Phase 1: v2 Scaffolding

1. Add a dedicated `decision_grade_v2` driver script.
2. Add shared statistic interfaces for `DeltaLPD`, `DeltaD`, `logBF`.
3. Add robust checkpoint format for all three families.
4. Add deterministic seed ledger file in output bundle.

### Phase 2: Real-Data Observed Stats

1. Compute `DeltaLPD_obs`.
2. Compute `DeltaD_obs`.
3. Compute `logBF_obs`.
4. Persist observed-stat checkpoint.

### Phase 3: BH-Null Calibration Runs

1. Run BH-null replicates with resumable chunks.
2. Compute family-specific replicate stats.
3. Persist per-replicate and per-chunk outputs for all families.

### Phase 4: Final Aggregation and Report

Write:

1. `REPORTS/negative_slope_decision_grade_v2_<timestamp>/master_report.md`
2. `REPORTS/negative_slope_decision_grade_v2_<timestamp>/summary.json`
3. Figures:
   - `pred_null_hist_with_obs.png`
   - `dev_null_hist_with_obs.png`
   - `bf_null_hist_with_obs.png`
   - `combined_pvalue_panel.png`
   - `runtime_qc_panel.png`

## Summary JSON Minimum Keys

1. `observed.delta_lpd`
2. `observed.deltaD`
3. `observed.logBF`
4. `null.pred.n_reps`
5. `null.dev.n_reps`
6. `null.bf.n_reps`
7. `pvalues.raw.pred`
8. `pvalues.raw.dev`
9. `pvalues.raw.bf`
10. `pvalues.holm_adjusted`
11. `decision.classification`
12. `decision.criteria`
13. `config.pipeline_fingerprint`
14. `config.seed_ledger`

## Resource Plan

Target machine:

1. 256 logical cores.
2. 512 GB RAM.

Execution constraints:

1. BLAS/OpenMP threads pinned to 1 in workers.
2. High process parallelism with bounded worker recycling.
3. Full resumability and frequent checkpoint writes.

## Explicit Do-Not-Do

Before v2 starts:

1. Do not alter v2 definitions after seeing any interim v2 metric.
2. Do not drop any family because of unfavorable preliminary results.
3. Do not relabel secondary results as primary post hoc.

