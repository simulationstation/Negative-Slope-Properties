#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import os
import platform
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, text=True, stderr=subprocess.STDOUT)
    except Exception:
        return None


def _yaml_quote(s: str | None) -> str:
    if s is None:
        return "null"
    s = str(s)
    s = s.replace("'", "''")
    return f"'{s}'"


def _yaml_block(s: str | None, *, indent: int = 2) -> str:
    if not s:
        return "null"
    pad = " " * indent
    lines = s.rstrip("\n").splitlines()
    body = "\n".join(f"{pad}{line}" for line in lines) if lines else f"{pad}"
    return "|\n" + body + "\n"


def _git_info(repo_root: Path) -> tuple[str | None, str | None, bool | None]:
    sha = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    sha = sha.strip() if sha else None
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    branch = branch.strip() if branch else None
    dirty_out = _run(["git", "status", "--porcelain=v1"], cwd=repo_root)
    dirty = None if dirty_out is None else bool(dirty_out.strip())
    return sha, branch, dirty


def _cpu_count_effective() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return int(len(os.sched_getaffinity(0)))
        except Exception:
            pass
    return int(os.cpu_count() or 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Write a lightweight environment manifest for reproducibility.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory to write manifest.yaml + pip_freeze.txt into.")
    ap.add_argument("--repo-root", type=Path, default=Path("."), help="Repo root (default: .)")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    repo_root: Path = args.repo_root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pip_freeze_path = out_dir / "pip_freeze.txt"
    pip_exe = repo_root / ".venv" / "bin" / "pip"
    if pip_exe.exists():
        pip_cmd = [str(pip_exe), "freeze"]
    else:
        pip_cmd = [sys.executable, "-m", "pip", "freeze"]
    pip_freeze = _run(pip_cmd, cwd=repo_root) or ""
    pip_freeze_path.write_text(pip_freeze, encoding="utf-8")

    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    sha, branch, dirty = _git_info(repo_root)

    lscpu = _run(["lscpu"], cwd=repo_root)
    mem = _run(["free", "-h"], cwd=repo_root)

    manifest_path = out_dir / "manifest.yaml"
    payload = []
    payload.append(f"timestamp: {_yaml_quote(ts)}")
    payload.append("os:")
    payload.append(f"  platform: {_yaml_quote(platform.platform())}")
    payload.append(f"  uname: {_yaml_quote(' '.join(platform.uname()))}")
    payload.append("python:")
    payload.append(f"  executable: {_yaml_quote(sys.executable)}")
    payload.append(f"  version: {_yaml_quote(platform.python_version())}")
    payload.append("git:")
    payload.append(f"  commit: {_yaml_quote(sha)}")
    payload.append(f"  branch: {_yaml_quote(branch)}")
    payload.append(f"  dirty: {('true' if dirty else 'false') if dirty is not None else 'null'}")
    payload.append("hardware:")
    payload.append(f"  cpu_count: {_cpu_count_effective()}")
    payload.append("  lscpu: " + _yaml_block(lscpu, indent=4).lstrip())
    payload.append("  mem: " + _yaml_block(mem, indent=4).lstrip())
    payload.append("packages:")
    payload.append(f"  pip_freeze_file: {_yaml_quote(str(pip_freeze_path))}")
    payload.append("commands: []")
    manifest_path.write_text("\n".join(payload) + "\n", encoding="utf-8")

    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

