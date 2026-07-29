#!/usr/bin/env python3
"""Export SFT JSONL with evidence context in the user turn (RAG-aware FT)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_evidence_map import build_evidence_map
from export_sft_messages import gold_answer


def _select_delta_diet(
    rows: list[dict], *, delta_copies: int, stable_cap: int, seed: int
) -> list[dict]:
    by_drift: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_drift[row.get("drift_type", "stable")].append(row)
    rng = random.Random(seed)
    selected: list[dict] = []
    for drift in ("added", "changed", "removed"):
        pool = by_drift.get(drift) or []
        for _ in range(delta_copies):
            selected.extend(pool)
    stable = list(by_drift.get("stable") or [])
    rng.shuffle(stable)
    selected.extend(stable[:stable_cap])
    rng.shuffle(selected)
    return selected


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probes", type=Path, required=True)
    p.add_argument(
        "--snapshots-root",
        type=Path,
        default=Path("data/experiments/e1_vllm/snapshots"),
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--evidence-map-out", type=Path, required=True)
    p.add_argument("--delta-copies", type=int, default=40)
    p.add_argument("--stable-cap", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260728)
    p.add_argument("--max-chars", type=int, default=3500)
    args = p.parse_args()

    version_to_tag = {
        "0.20.0": "v0.20.0",
        "0.21.0": "v0.21.0",
        "0.22.0": "v0.22.0",
        "0.23.0": "v0.23.0",
        "0.24.0": "v0.24.0",
    }
    evidence_payload = build_evidence_map(
        probes_path=args.probes,
        snapshots_root=args.snapshots_root,
        version_to_tag=version_to_tag,
        max_chars=args.max_chars,
        era="fresh",
    )
    args.evidence_map_out.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_map_out.write_text(
        json.dumps(evidence_payload, indent=2) + "\n", encoding="utf-8"
    )
    by_probe = evidence_payload["by_probe"]

    rows = [
        json.loads(line)
        for line in args.probes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = _select_delta_diet(
        rows,
        delta_copies=args.delta_copies,
        stable_cap=args.stable_cap,
        seed=args.seed,
    )

    slim: list[dict] = []
    missing_ctx = 0
    for row in selected:
        ctx = (by_probe.get(row["id"]) or {}).get("context") or ""
        if not ctx:
            missing_ctx += 1
        user = f"{ctx}\n\nQuestion: {row['question']}" if ctx else row["question"]
        slim.append(
            {
                "messages": [
                    {"role": "user", "content": user},
                    {
                        "role": "assistant",
                        "content": gold_answer(row.get("gold") or {}),
                    },
                ]
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in slim),
        encoding="utf-8",
    )
    meta = {
        "schema": "delta.eval_factory.sft_evidence_diet.v1",
        "source": str(args.probes),
        "evidence_map": str(args.evidence_map_out),
        "n": len(slim),
        "missing_context": missing_ctx,
        "delta_copies": args.delta_copies,
        "stable_cap": args.stable_cap,
        "seed": args.seed,
        "by_drift": dict(Counter(r.get("drift_type", "stable") for r in selected)),
        "evidence_coverage": {
            "n_with_context": evidence_payload["n_with_context"],
            "n_probes": evidence_payload["n_probes"],
            "n_missing_spans": evidence_payload["n_missing_spans"],
        },
    }
    args.out.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
