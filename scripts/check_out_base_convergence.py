#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_run_dirs(out_base: Path) -> list[Path]:
    # Accept either a single run dir (contains tables/summary.json) or a base with M*_start*/.
    if (out_base / "tables" / "summary.json").exists():
        return [out_base]
    return sorted([p for p in out_base.glob("M*_start*") if p.is_dir()])


def _parse_run_name(name: str) -> tuple[str, str]:
    m = re.match(r"^(?P<map>M\d+)_start(?P<seed>\d+)$", name)
    if not m:
        return "?", name
    return m.group("map"), m.group("seed")


def main() -> int:
    ap = argparse.ArgumentParser(description="Quick convergence checks from tables/summary.json across an out_base.")
    ap.add_argument("out_base", type=Path, help="Either a run dir (contains tables/summary.json) or a base with M*_start*/.")
    ap.add_argument("--target-mult", type=float, default=50.0, help="Target: post-burn steps >= target_mult * tau_max.")
    ap.add_argument("--min-ess", type=float, default=0.0, help="Optional ESS floor (0 disables).")
    ap.add_argument("--print-extend", action="store_true", help="Print suggested --mu-steps values for runs that miss targets.")
    args = ap.parse_args()

    out_base = args.out_base
    run_dirs = _iter_run_dirs(out_base)
    if not run_dirs:
        raise SystemExit(f"No run dirs found under {out_base}")

    target_mult = float(args.target_mult)
    min_ess = float(args.min_ess)

    print(f"out_base={out_base}  target_mult={target_mult:g}  min_ess={min_ess:g}")
    print("map seed  post_steps  tau_max   ess_min   ok  run_dir")

    for rd in run_dirs:
        summary_path = rd / "tables" / "summary.json"
        if not summary_path.exists():
            map_tag, seed = _parse_run_name(rd.name)
            print(f"{map_tag:>2} {seed:>4}  {'-':>9}  {'-':>7}  {'-':>7}  {'-':>2}  {rd} (missing summary.json)")
            continue

        s = _read_json(summary_path)
        settings = s.get("settings", {}) or {}
        mu_sampler = s.get("mu_sampler", {}) or {}

        mu_steps = int(settings.get("mu_steps", 0) or 0)
        mu_burn = int(settings.get("mu_burn", 0) or 0)
        post_steps = max(0, mu_steps - mu_burn)

        tau_by = mu_sampler.get("tau_by", {}) or {}
        tau_vals = [float(v) for v in tau_by.values() if v is not None]
        tau_max = max(tau_vals) if tau_vals else float("nan")
        ess_min_val = float(mu_sampler.get("ess_min", float("nan")))

        required_post = int(math.ceil(target_mult * tau_max)) if math.isfinite(tau_max) else None
        ok_tau = required_post is None or post_steps >= required_post
        ok_ess = (min_ess <= 0.0) or (math.isfinite(ess_min_val) and ess_min_val >= min_ess)
        ok = ok_tau and ok_ess

        map_tag, seed = _parse_run_name(rd.name)
        tau_s = f"{tau_max:7.1f}" if math.isfinite(tau_max) else f"{'-':>7}"
        ess_s = f"{ess_min_val:7.1f}" if math.isfinite(ess_min_val) else f"{'-':>7}"
        print(f"{map_tag:>2} {seed:>4}  {post_steps:9d}  {tau_s}  {ess_s}  {int(ok):>2}  {rd}")

        if args.print_extend and not ok and required_post is not None:
            want_mu_steps = int(mu_burn + required_post)
            ckpt_state = rd / "samples" / "mu_checkpoint" / "state.npz"
            ckpt_note = "" if ckpt_state.exists() else "  [no checkpoint state.npz found]"
            print(f"  -> suggest: --mu-steps {want_mu_steps}{ckpt_note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

