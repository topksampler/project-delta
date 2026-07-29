#!/usr/bin/env python3
"""Offline path-hit: any RetrievalBundle mode vs gold evidence paths."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from lab.eval_vllm_qa import RetrievalBundle


def gold_paths(row: dict) -> set[str]:
    out: set[str] = set()
    for cid in row.get("chunk_ids") or []:
        bits = str(cid).split(":")
        if len(bits) >= 4:
            out.add(bits[2])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probes", type=Path, required=True)
    p.add_argument("--evidence-map", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--mode", default="bm25")
    p.add_argument("--symbol-index", type=Path, default=None)
    p.add_argument("--diff-paths", type=Path, default=None)
    p.add_argument("--diff-corpus", type=Path, default=None)
    p.add_argument("--diff-index", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--label", default=None, help="name in summary (default: mode)")
    args = p.parse_args()
    label = args.label or args.mode

    probes = [
        json.loads(line)
        for line in args.probes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    evidence = json.loads(args.evidence_map.read_text(encoding="utf-8"))
    by_probe = evidence.get("by_probe") or {}

    retriever = RetrievalBundle(
        corpus_path=args.corpus,
        index_path=args.index,
        mode=args.mode,
        top_k=args.top_k,
        max_chars=3500,
        symbol_index_path=args.symbol_index,
        diff_paths_path=args.diff_paths,
        diff_corpus_path=args.diff_corpus,
        diff_index_path=args.diff_index,
    )

    by_drift = Counter()
    by_hit = Counter()
    rows = []
    for probe in probes:
        gold = gold_paths(by_probe.get(probe["id"]) or {})
        if not gold:
            continue
        drift = probe.get("drift_type") or "?"
        _ctx, ids = retriever.retrieve(probe["question"], example=probe)
        got = {retriever.cid_to_path.get(i, "") for i in ids}
        hit = bool(gold & got)
        by_drift[drift] += 1
        if hit:
            by_hit[drift] += 1
        rows.append(
            {
                "id": probe["id"],
                "drift_type": drift,
                "mode": label,
                "path_hit": hit,
                "n_gold_paths": len(gold),
                "retrieved_paths": sorted(p for p in got if p),
            }
        )

    n = sum(by_drift.values())
    summary = {
        "schema": "delta.eval_factory.retrieval_path_hit.v1",
        "label": label,
        "mode": args.mode,
        "corpus": str(args.corpus),
        "index": str(args.index),
        "top_k": args.top_k,
        "n_probes_with_gold_paths": n,
        "path_hit_rate": round(sum(by_hit.values()) / max(1, n), 4),
        "path_hit_rate_by_drift": {
            d: round(by_hit[d] / max(1, by_drift[d]), 4) for d in sorted(by_drift)
        },
        "counts_by_drift": dict(by_drift),
        "hits_by_drift": dict(by_hit),
        "routes": dict(retriever.route_log) if retriever.route_log else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.out.with_suffix(".jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
