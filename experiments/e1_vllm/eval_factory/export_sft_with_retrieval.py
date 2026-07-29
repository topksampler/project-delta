#!/usr/bin/env python3
"""Export delta-diet SFT with non-oracle retrieval context in the user turn."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_sft_messages import gold_answer
from lab.eval_vllm_qa import RetrievalBundle


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
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("symbol", "bm25"), default="symbol")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--symbol-index", type=Path, default=None)
    p.add_argument("--delta-copies", type=int, default=40)
    p.add_argument("--stable-cap", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260728)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--max-chars", type=int, default=3500)
    args = p.parse_args()

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
    retriever = RetrievalBundle(
        corpus_path=args.corpus,
        index_path=args.index,
        mode=args.mode,
        top_k=args.top_k,
        max_chars=args.max_chars,
        seed=args.seed,
        symbol_index_path=args.symbol_index,
    )

    slim: list[dict] = []
    nonempty = 0
    for row in selected:
        ctx, _ids = retriever.retrieve(row["question"], example=row)
        if ctx:
            nonempty += 1
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
        "schema": "delta.eval_factory.sft_retrieval_diet.v1",
        "source": str(args.probes),
        "mode": args.mode,
        "corpus": str(args.corpus),
        "index": str(args.index),
        "symbol_index": str(args.symbol_index) if args.symbol_index else None,
        "n": len(slim),
        "nonempty_context": nonempty,
        "delta_copies": args.delta_copies,
        "stable_cap": args.stable_cap,
        "seed": args.seed,
        "by_drift": dict(Counter(r.get("drift_type", "stable") for r in selected)),
    }
    args.out.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
