from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect(base: Path) -> list[dict]:
    rows = []
    for run in sorted(base.glob("M0_start*/tables/summary.json")):
        seed = run.parent.parent.name.replace("M0_start", "")
        d = json.loads(run.read_text())
        omega = d.get("posterior", {}).get("omega_m0", {})
        m = d.get("departure", {}).get("m", {})
        rows.append(
            {
                "seed": int(seed),
                "omega_p16": omega.get("p16"),
                "omega_p50": omega.get("p50"),
                "omega_p84": omega.get("p84"),
                "m_mean": m.get("mean"),
                "m_std": m.get("std"),
                "p_m_gt0": m.get("p_gt0"),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = collect(args.base)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write("seed\tomega_p50\tomega_p16\tomega_p84\tm_mean\tm_std\tP(m>0)\n")
        for r in rows:
            f.write(
                f"{r['seed']}\t{r['omega_p50']:.4f}\t{r['omega_p16']:.4f}\t{r['omega_p84']:.4f}\t"
                f"{r['m_mean']:.4f}\t{r['m_std']:.4f}\t{r['p_m_gt0']:.3f}\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
