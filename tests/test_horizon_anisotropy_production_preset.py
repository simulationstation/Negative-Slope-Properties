from __future__ import annotations

from argparse import Namespace
import importlib.util
from pathlib import Path
import sys


def _load_scan_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "run_horizon_anisotropy_scan.py"
    spec = importlib.util.spec_from_file_location("run_horizon_anisotropy_scan", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Needed for dataclasses with postponed evaluation of annotations.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_production_preset_sets_free_gamma_and_fast_train_defaults():
    mod = _load_scan_module()
    args = Namespace(preset="production")
    mod._apply_production_preset(args)

    assert args.match_z == "bin_downsample"
    assert args.match_mode == "survey_z"
    assert args.mu_sampler == "emcee"
    assert args.growth_mode == "gamma"
    assert args.growth_gamma_free is True
    assert args.train_mu_sampler == "emcee"
    assert args.train_mu_walkers == 24


def test_production_preset_respects_explicit_train_overrides():
    mod = _load_scan_module()
    args = Namespace(preset="production", train_mu_sampler="ptemcee", train_mu_walkers=64)
    mod._apply_production_preset(args)

    assert args.train_mu_sampler == "ptemcee"
    assert args.train_mu_walkers == 64
