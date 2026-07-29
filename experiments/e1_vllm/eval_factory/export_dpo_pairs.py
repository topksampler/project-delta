#!/usr/bin/env python3
"""Export factory probes → DPO preference pairs (remember/forget).

Row rules (Experiment A):
  changed/removed → chosen=gold_after, rejected=gold_before
  added           → chosen=gold_after, rejected=absent default
  stable          → chosen=gold, rejected=wrong default (small capped mix)

Diet selection mirrors build_delta_diet (delta overweight + stable cap).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .export_sft_messages import gold_answer
    from .forget_score import gold_before_for_probe
except ImportError:  # script: python export_dpo_pairs.py
    from export_sft_messages import gold_answer
    from forget_score import gold_before_for_probe

ABSENT_REJECTED = "Unknown or absent."
SCHEMA = "delta.eval_factory.dpo_pairs.v1"


def wrong_default_answer(probe: dict, rng: random.Random) -> str:
    """Small wrong-answer mix for stable canaries."""
    gold = probe.get("gold") or {}
    if gold.get("boolean"):
        flipped = "no" if str(gold["boolean"]).lower() == "yes" else "yes"
        return f"{flipped.capitalize()}."
    options = [
        ABSENT_REJECTED,
        "Unchanged.",
        "Removed.",
        "Added in a later release.",
        "Incorrect default.",
    ]
    return rng.choice(options)


def rejected_answer(probe: dict, rng: random.Random) -> str | None:
    drift = probe.get("drift_type") or "stable"
    if drift in {"changed", "removed"}:
        before = gold_before_for_probe(probe)
        if before is None:
            # Fall back: existence probes without a distinct before gold.
            gold = probe.get("gold") or {}
            if gold.get("boolean"):
                flipped = "no" if str(gold["boolean"]).lower() == "yes" else "yes"
                return f"{flipped.capitalize()}."
            return ABSENT_REJECTED
        return gold_answer(before)
    if drift == "added":
        return ABSENT_REJECTED
    if drift == "stable":
        return wrong_default_answer(probe, rng)
    return None


def select_diet(
    rows: list[dict],
    *,
    delta_copies: int,
    stable_cap: int,
    seed: int,
) -> list[dict]:
    by_drift: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_drift[row.get("drift_type", "stable")].append(row)

    rng = random.Random(seed)
    selected: list[dict] = []
    for drift in ("added", "changed", "removed"):
        pool = by_drift.get(drift) or []
        if not pool:
            continue
        for _ in range(delta_copies):
            selected.extend(pool)
    stable = list(by_drift.get("stable") or [])
    rng.shuffle(stable)
    selected.extend(stable[:stable_cap])
    rng.shuffle(selected)
    return selected


def to_dpo_row(probe: dict, rng: random.Random) -> dict | None:
    rejected = rejected_answer(probe, rng)
    if rejected is None:
        return None
    chosen = gold_answer(probe.get("gold") or {})
    prompt = probe["question"]
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
        "id": probe.get("id"),
        "claim_id": probe.get("claim_id"),
        "drift_type": probe.get("drift_type"),
    }


def convert(
    *,
    probes_path: Path,
    out_path: Path,
    delta_copies: int = 40,
    stable_cap: int = 200,
    seed: int = 20260729,
) -> dict:
    rows = [
        json.loads(line)
        for line in probes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = select_diet(
        rows,
        delta_copies=delta_copies,
        stable_cap=stable_cap,
        seed=seed,
    )
    rng = random.Random(seed)
    pairs: list[dict] = []
    for probe in selected:
        row = to_dpo_row(probe, rng)
        if row is not None:
            pairs.append(row)

    # Trainer needs prompt/chosen/rejected only.
    slim = [
        {
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        }
        for row in pairs
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in slim),
        encoding="utf-8",
    )
    meta = {
        "schema": SCHEMA,
        "source": str(probes_path),
        "n": len(slim),
        "delta_copies": delta_copies,
        "stable_cap": stable_cap,
        "seed": seed,
        "by_drift": dict(Counter(row.get("drift_type", "stable") for row in pairs)),
        "absent_rejected": ABSENT_REJECTED,
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"out": str(out_path), "meta": str(meta_path), **meta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--delta-copies", type=int, default=40)
    parser.add_argument("--stable-cap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    print(
        json.dumps(
            convert(
                probes_path=args.probes,
                out_path=args.out,
                delta_copies=args.delta_copies,
                stable_cap=args.stable_cap,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
