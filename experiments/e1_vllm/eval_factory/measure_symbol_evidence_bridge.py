#!/usr/bin/env python3
"""Measure symbol↔evidence path overlap and symbol_expand coverage."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from lab.eval_vllm_qa import RetrievalBundle


def gold_paths(row: dict) -> set[str]:
    out: set[str] = set()
    for cid in row.get("chunk_ids") or []:
        bits = str(cid).split(":")
        # evidence:v0.23.0:path:start-end
        if len(bits) >= 4:
            out.add(bits[2])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probes", type=Path, required=True)
    p.add_argument("--evidence-map", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--symbol-index", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=4)
    args = p.parse_args()

    probes = [
        json.loads(line)
        for line in args.probes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    evidence = json.loads(args.evidence_map.read_text(encoding="utf-8"))
    by_probe = evidence.get("by_probe") or {}

    symbol = RetrievalBundle(
        corpus_path=args.corpus,
        index_path=args.index,
        mode="symbol",
        top_k=args.top_k,
        max_chars=3500,
        symbol_index_path=args.symbol_index,
    )
    expand = RetrievalBundle(
        corpus_path=args.corpus,
        index_path=args.index,
        mode="symbol_expand",
        top_k=args.top_k,
        max_chars=3500,
        symbol_index_path=args.symbol_index,
    )

    def path_hit(retriever: RetrievalBundle, probe: dict, gold: set[str]) -> bool:
        _ctx, ids = retriever.retrieve(probe["question"], example=probe)
        got = {retriever.cid_to_path.get(i, "") for i in ids}
        return bool(gold & got)

    rows = []
    by_mode_drift: dict[str, Counter] = defaultdict(Counter)
    by_mode_hit: dict[str, Counter] = defaultdict(Counter)
    for probe in probes:
        gold = gold_paths(by_probe.get(probe["id"]) or {})
        if not gold:
            continue
        drift = probe.get("drift_type") or "?"
        for name, ret in (("symbol", symbol), ("symbol_expand", expand)):
            hit = path_hit(ret, probe, gold)
            by_mode_drift[name][drift] += 1
            if hit:
                by_mode_hit[name][drift] += 1
            rows.append(
                {
                    "id": probe["id"],
                    "drift_type": drift,
                    "mode": name,
                    "path_hit": hit,
                    "n_gold_paths": len(gold),
                }
            )

    summary = {
        "schema": "delta.eval_factory.symbol_evidence_bridge.v1",
        "n_probes_with_gold_paths": sum(
            1 for r in rows if r["mode"] == "symbol"
        ),
        "path_hit_rate": {
            mode: round(
                sum(by_mode_hit[mode].values()) / max(1, sum(by_mode_drift[mode].values())),
                4,
            )
            for mode in ("symbol", "symbol_expand")
        },
        "path_hit_rate_by_drift": {
            mode: {
                drift: round(
                    by_mode_hit[mode][drift] / max(1, by_mode_drift[mode][drift]),
                    4,
                )
                for drift in sorted(by_mode_drift[mode])
            }
            for mode in ("symbol", "symbol_expand")
        },
        "counts_by_drift": {
            mode: dict(by_mode_drift[mode]) for mode in ("symbol", "symbol_expand")
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    detail = args.out.with_suffix(".jsonl")
    detail.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
