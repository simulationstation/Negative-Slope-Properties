# Data layout

This repository downloads and caches public cosmology datasets locally.

- `data/cache/`: immutable file cache managed by `pooch`
- `data/processed/`: parsed/derived datasets (small)

Large raw downloads are *not* committed to git.

