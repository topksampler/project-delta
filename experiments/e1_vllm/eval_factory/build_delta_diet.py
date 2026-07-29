#!/usr/bin/env python3
"""Build a delta-overweight SFT diet from factory train probes."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from export_sft_messages import gold_answer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probes", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--delta-copies", type=int, default=40)
    p.add_argument("--stable-cap", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260728)
    args = p.parse_args()

    rows = [
        json.loads(line)
        for line in args.probes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_drift: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_drift[row.get("drift_type", "stable")].append(row)

    rng = random.Random(args.seed)
    selected: list[dict] = []
    for drift in ("added", "changed", "removed"):
        pool = by_drift.get(drift) or []
        if not pool:
            continue
        for _ in range(args.delta_copies):
            selected.extend(pool)
    stable = list(by_drift.get("stable") or [])
    rng.shuffle(stable)
    # Prefer version_delta stables a bit less; keep mix but cap total
    selected.extend(stable[: args.stable_cap])
    rng.shuffle(selected)

    slim = [
        {
            "messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": gold_answer(row.get("gold") or {})},
            ]
        }
        for row in selected
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in slim),
        encoding="utf-8",
    )
    drift_counts = Counter(r.get("drift_type", "stable") for r in selected)
    meta = {
        "schema": "delta.eval_factory.sft_delta_diet.v1",
        "source": str(args.probes),
        "n": len(slim),
        "delta_copies": args.delta_copies,
        "stable_cap": args.stable_cap,
        "seed": args.seed,
        "by_drift": dict(drift_counts),
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), **meta}, indent=2))


if __name__ == "__main__":
    main()
