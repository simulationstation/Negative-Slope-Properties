# Plan Forward: INFO+

## Objective
Establish whether the INFO+ negative-slope signal is robust, reproducible, and publication-worthy under realistic model/systematics stress.

## Current state
- 5-seed `120`-step mode-map gate is running (`101,202,303,404,505`).
- Purpose of gate: detect obvious seed pathologies before expensive long runs.

## Path forward
1. Finish the `120`-step gate and score seed health.
   - Check for: failed seeds, prior-edge hugging, seed-level contradictions, unstable nuisance behavior.
   - Decision: pass/fail gate.

2. Run a medium stability pass (`250`-`400` steps, 5 seeds).
   - Goal: confirm early agreement is not a short-chain artifact.
   - Deliverables: per-seed summary table (`H0`, `Omega_m0`, `r_d`, `sigma8`, slope stats, sign probabilities).

3. Run decision-grade INFO+ (`800`-`1000` steps, 5 seeds).
   - Keep same data/model choices used in gate.
   - Target outputs: merged posterior summaries, seed spread diagnostics, convergence/ESS metrics, chain-health report.

4. Stress the claim with minimal high-value alternatives (not a giant matrix).
   - Mapping sensitivity: `M0` baseline plus `M1` and `M2` checks.
   - Nuisance sensitivity: one tighter and one wider prior set for key nuisance channels.
   - Goal: test whether qualitative conclusion survives defensible modeling choices.

5. Run one orthogonal cross-check.
   - Use an independent estimator/summary path on the same INFO+ outputs.
   - Goal: check directional consistency, not exact numeric identity.

6. Decide paper posture.
   - If signal is stable across seeds and alternatives: frame as robust anomaly under tested assumptions.
   - If unstable or mode-dependent: frame as pilot/diagnostic and report what breaks the claim.

## Practical stop rule
Stop development and write once these are true:
- 5-seed long run completed cleanly.
- No single seed drives the conclusion.
- Main conclusion survives `M0/M1/M2` and minimal nuisance-variation checks.
- Orthogonal cross-check is non-contradictory.
