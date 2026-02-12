from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_anisotropy_scan_uses_repo_src_and_supports_gamma_prior() -> None:
    """Guard against importing a different checkout installed in the environment.

    We previously saw `entropy_horizon_recon` resolving to a separate path, which broke
    `growth_gamma_prior` support at runtime.
    """

    repo_root = Path(__file__).resolve().parents[1]
    code = r"""
import inspect
import pathlib

import scripts.run_horizon_anisotropy_scan  # noqa: F401
import entropy_horizon_recon
import entropy_horizon_recon.inversion as inv

print(pathlib.Path(entropy_horizon_recon.__file__).resolve())
print(pathlib.Path(inv.__file__).resolve())
print("growth_gamma_prior" in inspect.signature(inv.infer_logmu_forward).parameters)
"""
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 3

    pkg_path = Path(lines[0]).resolve()
    inv_path = Path(lines[1]).resolve()
    assert pkg_path.is_relative_to(repo_root / "src")
    assert inv_path.is_relative_to(repo_root / "src")
    assert lines[2] == "True"

