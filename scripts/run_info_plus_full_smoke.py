#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path


def _run(out_dir: Path, steps: int, burn: int, draws: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    cmd = [
        str(Path('.venv/bin/python').resolve()),
        "scripts/run_realdata_recon.py",
        "--out",
        str(out_dir),
        "--seed",
        "203",
        "--mu-init-seed",
        "203",
        "--mu-sampler",
        "ptemcee",
        "--pt-ntemps",
        "4",
        "--pt-tmax",
        "10",
        "--mu-steps",
        str(steps),
        "--mu-burn",
        str(burn),
        "--mu-draws",
        str(draws),
        "--mu-walkers",
        "32",
        "--cpu-cores",
        "50",
        "--mu-procs",
        "50",
        "--gp-procs",
        "1",
        "--include-rsd",
        "--rsd-mode",
        "dr16_lrg_fs8",
        "--include-planck-lensing-clpp",
        "--clpp-backend",
        "camb",
        "--include-fullshape-pk",
        "--skip-ablations",
        "--skip-hz-recon",
    ]
    subprocess.run(cmd, check=True, env=env)


def main() -> int:
    base = Path("outputs") / "smoke_info_plus_full"
    _run(base / "small", steps=40, burn=10, draws=20)
    _run(base / "medium", steps=200, burn=50, draws=100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
