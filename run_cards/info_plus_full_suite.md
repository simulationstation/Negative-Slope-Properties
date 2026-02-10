# Info+ full-likelihood suite (paper target)

This run card defines the **paper-grade** “info+” configuration used to anchor the background-only reconstruction with growth + lensing + full-shape information, and to test mapping sensitivity under **M0/M1/M2**.

## Likelihood stack (Info+)

Enabled in `scripts/run_realdata_recon.py`:
- **SN**: Pantheon+ (cosmology subset), binned for forward inference.
- **CC**: cosmic chronometers.
- **BAO (distance-only)**: SDSS DR12 consensus BAO + DESI 2024 BAO (“ALL”).
- **RSD**: DR16 LRG `fσ8(z)` extracted from the DR16 LRG FSBAO covariance (`--rsd-mode dr16_lrg_fs8`).
- **CMB lensing**: Planck 2018 Cℓ^{φφ} bandpowers with a CAMB backend (`--include-planck-lensing-clpp --clpp-backend camb`).
- **Full-shape**: Shapefit BOSS DR12 monopole P(k) (`--include-fullshape-pk`).

Important: when using `--rsd-mode dr16_lrg_fs8`, we drop DR16 BAO distances to avoid treating correlated pieces of the same DR16 covariance as independent:
- `--drop-bao dr16`

## Environment knobs (avoid oversubscription)

Set for all runs:
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`
- `PYTHONUNBUFFERED=1`
- `NO_LENSING=1` (prevents adding the **compressed** lensing proxy when using full Cℓ^{φφ})
- `EHR_CAMB_CACHE_DIR=...` (recommended disk cache for CAMB products)

## Launch (robust detached; run one variant at a time)

Use the single top-level launcher (it starts each seed via `setsid ... < /dev/null &` and writes `pids.txt`):

```bash
out_root="outputs/finalization/info_plus_full_256_$(date -u +%Y%m%d_%H%M%S)UTC"
mkdir -p "$out_root"

# record environment snapshot (recommended)
.venv/bin/python scripts/write_env_manifest.py --out-dir "$out_root/env" --repo-root .

# M0 baseline
out_base="$out_root/M0"
nohup bash scripts/launch_full_info_plus_256_single_nohup.sh "$out_base" M0 > "$out_base/launcher.log" 2>&1 &
```

Mapping sensitivity suites (run after M0 finishes):

```bash
out_base="$out_root/M1"
nohup bash scripts/launch_full_info_plus_256_single_nohup.sh "$out_base" M1 > "$out_base/launcher.log" 2>&1 &

out_base="$out_root/M2"
nohup bash scripts/launch_full_info_plus_256_single_nohup.sh "$out_base" M2 "-0.2 0.2" > "$out_base/launcher.log" 2>&1 &
```

## Monitoring

```bash
.venv/bin/python scripts/status_out_base.py "$out_base"
```

## Convergence criteria + extension loop

Paper target: ensure (per seed) the post-burn chain length satisfies `n_post >= 50 * tau_max` (where `tau_max` is taken over the monitored scalar projections in `tables/summary.json`).

```bash
.venv/bin/python scripts/check_out_base_convergence.py "$out_base" --target-mult 50 --print-extend
```

## Resuming / extending

The sampler supports checkpoints (`--checkpoint-every` is enabled in the launcher). To extend chains, rerun the same launcher but append an override (later flags win), e.g.:

```bash
nohup bash scripts/launch_full_info_plus_256_single_nohup.sh "$out_base" M0 "" --mu-steps 4000 > "$out_base/launcher_extend.log" 2>&1 &
```

