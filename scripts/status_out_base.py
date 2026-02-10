#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np


def _is_realdata_process(cmd: str) -> bool:
    cmd = cmd.strip()
    # We want the actual Python worker/master processes, not wrapper shells (tmux/bash).
    return (
        "run_realdata_recon.py" in cmd
        and cmd.startswith(("python", "python3", ".venv/bin/python", "/home/primary/PROJECT/.venv/bin/python"))
    )


def _tail_text(path: Path, *, max_bytes: int = 200_000) -> str:
    try:
        size = path.stat().st_size
    except Exception:
        return ""
    try:
        with path.open("rb") as f:
            f.seek(max(0, int(size) - int(max_bytes)))
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _read_init_progress(run_log: Path) -> tuple[int, int] | None:
    if not run_log.exists():
        return None
    text = _tail_text(run_log)
    if not text:
        return None
    pat = re.compile(r"\[init\] walker (\d+)/(\d+) (?:starting|ok)")
    matches = list(pat.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    try:
        i = int(m.group(1))
        n = int(m.group(2))
    except Exception:
        return None
    if i <= 0 or n <= 0:
        return None
    return i, n


def _read_argv(runtime_log: Path) -> list[str] | None:
    if not runtime_log.exists():
        return None
    # runtime.log can contain multiple START lines if a run is resumed/extended. Use the latest.
    text = _tail_text(runtime_log)
    if not text:
        return None
    for line in reversed(text.splitlines()):
        if "argv=" not in line:
            continue
        m = re.search(r"argv=(\[.*\])", line)
        if not m:
            continue
        try:
            return list(ast.literal_eval(m.group(1)))
        except Exception:
            continue
    return None


def _argv_get(argv: list[str] | None, flag: str) -> str | None:
    if not argv:
        return None
    try:
        i = argv.index(flag)
    except ValueError:
        return None
    if i + 1 >= len(argv):
        return None
    return argv[i + 1]


def _checkpoint_dir(seed_out: Path, argv: list[str] | None) -> Path:
    # Mirrors default behavior in scripts/run_realdata_recon.py.
    p = _argv_get(argv, "--checkpoint-path")
    if p:
        return Path(p)
    return seed_out / "samples" / "mu_checkpoint"


def _read_steps_done(checkpoint_dir: Path) -> int | None:
    state = checkpoint_dir / "state.npz"
    if not state.exists():
        return None
    try:
        with np.load(state, allow_pickle=True) as npz:
            return int(npz.get("steps_done", 0))
    except Exception:
        return None


def _read_steps_done_from_chunks(checkpoint_dir: Path) -> int | None:
    # Fallback when state.npz is missing/stale: infer progress from chunk filenames.
    # Expected form: chain_chunk_<start>_<end>.npz
    pat = re.compile(r"^chain_chunk_(\d+)_(\d+)\.npz$")
    best_end: int | None = None
    try:
        for p in checkpoint_dir.glob("chain_chunk_*_*.npz"):
            m = pat.match(p.name)
            if not m:
                continue
            end = int(m.group(2))
            if best_end is None or end > best_end:
                best_end = end
    except Exception:
        return None
    return best_end


def _latest_chunk_age_seconds(checkpoint_dir: Path, now: float) -> float | None:
    latest_mtime: float | None = None
    try:
        for p in checkpoint_dir.glob("chain_chunk_*_*.npz"):
            mt = p.stat().st_mtime
            if latest_mtime is None or mt > latest_mtime:
                latest_mtime = mt
    except Exception:
        return None
    if latest_mtime is None:
        return None
    return now - latest_mtime


def _ps_snapshot() -> list[tuple[float, float, str]]:
    # Returns (cpu_pct, mem_pct, args) per process line.
    out = subprocess.check_output(["ps", "-eo", "pcpu,pmem,args"], text=True)
    lines = out.splitlines()
    rows: list[tuple[float, float, str]] = []
    for line in lines[1:]:
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            cpu = float(parts[0])
            mem = float(parts[1])
        except ValueError:
            continue
        rows.append((cpu, mem, parts[2]))
    return rows


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def main() -> int:
    ap = argparse.ArgumentParser(description="Quick status for a multiseed out_base directory.")
    ap.add_argument("out_base", type=Path, help="Base dir containing M*_start*/")
    args = ap.parse_args()

    out_base = args.out_base
    seeds = sorted([p for p in out_base.glob("M*_start*") if p.is_dir()])

    load1, load5, load15 = os.getloadavg()
    ps_rows = _ps_snapshot()

    # Overall CPU/mem for any process mentioning out_base in args.
    out_base_s = str(out_base)
    cpu_sum = 0.0
    mem_sum = 0.0
    proc_count = 0
    for cpu, mem, cmd in ps_rows:
        if out_base_s in cmd and _is_realdata_process(cmd):
            cpu_sum += cpu
            mem_sum += mem
            proc_count += 1

    cores_used = cpu_sum / 100.0
    print(
        f"out_base={out_base}  procs={proc_count}  cpu={cpu_sum:.1f}% (~{cores_used:.1f} cores)  mem={mem_sum:.2f}%  loadavg={load1:.1f},{load5:.1f},{load15:.1f}"
    )

    if not seeds:
        print("no seed dirs found")
        return 0

    now = time.time()
    for seed_out in seeds:
        m = re.match(r"^(?P<map>M\d+)_start(?P<seed>\d+)$", seed_out.name)
        map_tag = m.group("map") if m else "?"
        seed = m.group("seed") if m else seed_out.name
        summary = seed_out / "tables" / "summary.json"
        runtime_log = seed_out / "runtime.log"
        run_log = seed_out / "run.log"
        argv = _read_argv(runtime_log)

        mu_steps_s = _argv_get(argv, "--mu-steps")
        mu_steps = int(mu_steps_s) if (mu_steps_s and mu_steps_s.isdigit()) else None

        ckpt_dir = _checkpoint_dir(seed_out, argv)
        steps_done = _read_steps_done(ckpt_dir)
        if steps_done is None or steps_done <= 0:
            steps_from_chunks = _read_steps_done_from_chunks(ckpt_dir)
            if steps_from_chunks is not None and steps_from_chunks > 0:
                steps_done = steps_from_chunks
        init_prog = _read_init_progress(run_log) if steps_done is None else None
        pct = None
        if mu_steps and steps_done is not None and mu_steps > 0:
            pct = 100.0 * float(steps_done) / float(mu_steps)
        elif init_prog is not None:
            i, n = init_prog
            pct = 100.0 * float(i) / float(n)
        state_path = ckpt_dir / "state.npz"
        age_s = None
        if state_path.exists():
            try:
                age_s = now - state_path.stat().st_mtime
            except Exception:
                age_s = None
        if age_s is None:
            age_s = _latest_chunk_age_seconds(ckpt_dir, now)

        # CPU/mem per seed: match full seed_out path in cmdline.
        seed_s = str(seed_out)
        seed_cpu = 0.0
        seed_mem = 0.0
        seed_procs = 0
        for cpu, mem, cmd in ps_rows:
            if seed_s in cmd and _is_realdata_process(cmd):
                seed_cpu += cpu
                seed_mem += mem
                seed_procs += 1
        seed_cores = seed_cpu / 100.0

        done = summary.exists()
        pct_s = f"{pct:.1f}%" if pct is not None else "-"
        stage = "done" if done else ("sample" if steps_done is not None else ("init" if init_prog is not None else "start"))
        steps_s = (
            f"{steps_done}/{mu_steps}"
            if (steps_done is not None and mu_steps is not None)
            else (f"{steps_done}/?" if steps_done is not None else "-")
        )
        if steps_done is None and init_prog is not None:
            i, n = init_prog
            steps_s = f"{i}/{n}"

        print(
            f"map={map_tag:<2} seed={seed:<3} stage={stage:<6} done={int(done)}  pct={pct_s:<7}  steps={steps_s:<12}  cpu={seed_cpu:>7.1f}% (~{seed_cores:>5.1f})  procs={seed_procs:<3}  state_age={_fmt_age(age_s)}  out={seed_out}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
