#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path


ERROR_PATTERNS_DEFAULT = [
    r"Traceback \(most recent call last\)",
    r"MemoryError",
    r"Segmentation fault",
    r"No space left on device",
    r"Killed",
    r"FloatingPointError",
    r"invalid value encountered",
]


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _read_tail(path: Path, max_bytes: int = 300_000) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _read_pid(path: Path) -> int | None:
    try:
        s = path.read_text(encoding="utf-8").strip()
        return int(s) if s else None
    except Exception:
        return None


def _pid_exists(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _kill_pid(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except Exception:
        return


def _load_pids(path: Path) -> list[int]:
    if not path.exists():
        return []
    pids: list[int] = []
    for line in _read_text(path).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return pids


def _parse_ps() -> list[tuple[int, float, str]]:
    out = subprocess.check_output(["ps", "-eo", "pid,pcpu,args"], text=True)
    rows: list[tuple[int, float, str]] = []
    for line in out.splitlines()[1:]:
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            pcpu = float(parts[1])
        except ValueError:
            continue
        rows.append((pid, pcpu, parts[2]))
    return rows


def _is_realdata_process(cmd: str) -> bool:
    cmd = cmd.strip()
    return (
        "run_realdata_recon.py" in cmd
        and cmd.startswith(("python", "python3", ".venv/bin/python", "/home/primary/PROJECT/.venv/bin/python"))
    )


def _latest_progress_mtime(seed_out: Path) -> float | None:
    ckpt_dir = seed_out / "samples" / "mu_checkpoint"
    latest = None
    state = ckpt_dir / "state.npz"
    if state.exists():
        try:
            latest = state.stat().st_mtime
        except Exception:
            latest = None
    try:
        for p in ckpt_dir.glob("chain_chunk_*_*.npz"):
            mt = p.stat().st_mtime
            if latest is None or mt > latest:
                latest = mt
    except Exception:
        pass
    if latest is None:
        run_log = seed_out / "run.log"
        if run_log.exists():
            try:
                latest = run_log.stat().st_mtime
            except Exception:
                latest = None
    return latest


def _scan_errors(seed_out: Path, patterns: list[re.Pattern[str]], tail_bytes: int) -> str | None:
    run_log = seed_out / "run.log"
    if not run_log.exists():
        return None
    text = _read_tail(run_log, max_bytes=tail_bytes)
    for pat in patterns:
        if pat.search(text):
            return pat.pattern
    return None


def _seed_status(seed_out: Path, ps_rows: list[tuple[int, float, str]], now: float) -> dict:
    seed_path = str(seed_out)
    procs: list[int] = []
    cpu_pct = 0.0
    for pid, pcpu, cmd in ps_rows:
        if seed_path in cmd and _is_realdata_process(cmd):
            procs.append(pid)
            cpu_pct += float(pcpu)
    done = (seed_out / "tables" / "summary.json").exists()
    progress_mtime = _latest_progress_mtime(seed_out)
    progress_age_s = None if progress_mtime is None else max(0.0, now - progress_mtime)
    run_log = seed_out / "run.log"
    run_log_age_s = None
    if run_log.exists():
        try:
            run_log_age_s = max(0.0, now - run_log.stat().st_mtime)
        except Exception:
            run_log_age_s = None
    return {
        "seed_out": str(seed_out),
        "done": bool(done),
        "procs": procs,
        "n_procs": len(procs),
        "cpu_pct": float(cpu_pct),
        "progress_age_s": progress_age_s,
        "run_log_age_s": run_log_age_s,
    }


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
        f.flush()


def _kill_run(out_base: Path, *, include_queue: bool = True) -> None:
    pids = _load_pids(out_base / "pids.txt")
    if include_queue:
        q = _read_pid(out_base / "queue.pid")
        if q is not None:
            pids.append(q)
    pids = sorted(set(p for p in pids if p > 0))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in pids:
            _kill_pid(p, sig)
        time.sleep(1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Watchdog for multiseed out_base runs.")
    ap.add_argument("--out-base", required=True, type=Path)
    ap.add_argument("--interval", type=float, default=30.0, help="Polling interval in seconds.")
    ap.add_argument("--stall-secs", type=float, default=1800.0, help="Max allowed progress staleness once warmed up.")
    ap.add_argument(
        "--hard-stall-secs",
        type=float,
        default=7200.0,
        help="Hard cap on staleness even if CPU remains active (<=0 disables).",
    )
    ap.add_argument("--init-stall-secs", type=float, default=900.0, help="Max allowed log staleness during init/start.")
    ap.add_argument("--zero-proc-grace-secs", type=float, default=180.0, help="Grace window for transient zero-proc states.")
    ap.add_argument("--warmup-secs", type=float, default=300.0, help="Ignore stall checks during initial warmup.")
    ap.add_argument(
        "--min-active-cpu-pct",
        type=float,
        default=20.0,
        help="Treat a seed as active (not stalled) when CPU usage exceeds this threshold.",
    )
    ap.add_argument("--tail-bytes", type=int, default=300_000, help="Per-seed run.log tail bytes to scan for error patterns.")
    ap.add_argument("--error-pattern", action="append", default=[], help="Additional regex pattern treated as fatal.")
    ap.add_argument("--kill-on-fail", action="store_true", help="Kill run pids on fail.")
    ap.add_argument("--once", action="store_true", help="Evaluate one cycle then exit.")
    args = ap.parse_args()

    out_base = args.out_base
    out_base.mkdir(parents=True, exist_ok=True)
    heartbeat_path = out_base / "watchdog_heartbeat.jsonl"
    report_path = out_base / "watchdog_report.json"
    fail_flag = out_base / "watchdog.FAIL"
    done_flag = out_base / "watchdog.DONE"
    run_start = time.time()

    pats = [re.compile(p) for p in ERROR_PATTERNS_DEFAULT + list(args.error_pattern)]

    while True:
        now = time.time()
        ps_rows = _parse_ps()
        seed_dirs = sorted(p for p in out_base.glob("M*_start*") if p.is_dir())
        statuses = [_seed_status(s, ps_rows, now) for s in seed_dirs]
        queue_pid = _read_pid(out_base / "queue.pid")
        queue_alive = _pid_exists(queue_pid)
        all_done = bool(statuses) and all(s["done"] for s in statuses)
        any_running = any((not s["done"]) and s["n_procs"] > 0 for s in statuses)
        elapsed_s = max(0.0, now - run_start)

        reason = None
        reason_detail = None

        for s in statuses:
            if s["done"]:
                continue
            bad = _scan_errors(Path(s["seed_out"]), pats, int(args.tail_bytes))
            if bad:
                reason = "log_error_pattern"
                reason_detail = {"seed_out": s["seed_out"], "pattern": bad}
                break

        if reason is None:
            for s in statuses:
                if s["done"]:
                    continue
                pa = s["progress_age_s"]
                if pa is not None and elapsed_s >= float(args.warmup_secs) and pa > float(args.stall_secs):
                    if float(s.get("cpu_pct", 0.0)) >= float(args.min_active_cpu_pct):
                        continue
                    reason = "progress_stall"
                    reason_detail = {
                        "seed_out": s["seed_out"],
                        "progress_age_s": pa,
                        "cpu_pct": float(s.get("cpu_pct", 0.0)),
                    }
                    break

        if reason is None:
            for s in statuses:
                if s["done"]:
                    continue
                pa = s["progress_age_s"]
                if (
                    pa is not None
                    and float(args.hard_stall_secs) > 0.0
                    and elapsed_s >= float(args.warmup_secs)
                    and pa > float(args.hard_stall_secs)
                ):
                    reason = "hard_progress_stall"
                    reason_detail = {
                        "seed_out": s["seed_out"],
                        "progress_age_s": pa,
                        "cpu_pct": float(s.get("cpu_pct", 0.0)),
                    }
                    break

        if reason is None:
            for s in statuses:
                if s["done"]:
                    continue
                if s["n_procs"] == 0:
                    rla = s["run_log_age_s"]
                    if rla is not None and rla > float(args.zero_proc_grace_secs):
                        reason = "zero_process_stall"
                        reason_detail = {"seed_out": s["seed_out"], "run_log_age_s": rla}
                        break
                elif (
                    s["progress_age_s"] is None
                    and s["run_log_age_s"] is not None
                    and s["run_log_age_s"] > float(args.init_stall_secs)
                    and float(s.get("cpu_pct", 0.0)) < float(args.min_active_cpu_pct)
                ):
                    reason = "init_stall"
                    reason_detail = {
                        "seed_out": s["seed_out"],
                        "run_log_age_s": s["run_log_age_s"],
                        "cpu_pct": float(s.get("cpu_pct", 0.0)),
                    }
                    break

        if reason is None and statuses and (not all_done) and (not any_running) and (not queue_alive):
            reason = "queue_and_workers_dead_before_done"
            reason_detail = {"queue_pid": queue_pid, "queue_alive": queue_alive}

        hb = {
            "utc": _now_utc(),
            "out_base": str(out_base),
            "queue_pid": queue_pid,
            "queue_alive": queue_alive,
            "n_seeds": len(statuses),
            "n_done": sum(1 for s in statuses if s["done"]),
            "n_running": sum(1 for s in statuses if (not s["done"]) and s["n_procs"] > 0),
            "elapsed_s": elapsed_s,
            "reason": reason,
            "reason_detail": reason_detail,
        }
        _append_jsonl(heartbeat_path, hb)

        if all_done and reason is None:
            done_flag.write_text(_now_utc() + "\n", encoding="utf-8")
            _write_json(
                report_path,
                {
                    "status": "done",
                    "utc": _now_utc(),
                    "out_base": str(out_base),
                    "elapsed_s": elapsed_s,
                    "seeds": statuses,
                },
            )
            return 0

        if reason is not None:
            payload = {
                "status": "fail",
                "utc": _now_utc(),
                "out_base": str(out_base),
                "reason": reason,
                "reason_detail": reason_detail,
                "elapsed_s": elapsed_s,
                "queue_pid": queue_pid,
                "queue_alive": queue_alive,
                "seeds": statuses,
            }
            _write_json(report_path, payload)
            fail_flag.write_text(_now_utc() + "\n", encoding="utf-8")
            if args.kill_on_fail:
                _kill_run(out_base, include_queue=True)
            return 2

        if args.once:
            _write_json(
                report_path,
                {
                    "status": "ok",
                    "utc": _now_utc(),
                    "out_base": str(out_base),
                    "elapsed_s": elapsed_s,
                    "queue_pid": queue_pid,
                    "queue_alive": queue_alive,
                    "seeds": statuses,
                },
            )
            return 0

        time.sleep(float(args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
