#!/usr/bin/env python3
"""Build a BM25 index from an e1_vllm corpus JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bm25 import BM25Index

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data" / "experiments" / "e1_vllm" / "indexes"


def load_corpus(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build(corpus_path: Path, out_path: Path) -> dict:
    rows = load_corpus(corpus_path)
    chunk_ids = [r["chunk_id"] for r in rows]
    texts = [r["text"] for r in rows]
    index = BM25Index.build(chunk_ids, texts)

    payload = {
        "format": "bm25_v1",
        "corpus_path": str(corpus_path),
        "n_docs": len(chunk_ids),
        "index": index.to_dict(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return {"out": str(out_path), "n_docs": len(chunk_ids)}


def main() -> None:
    p = argparse.ArgumentParser(description="corpus JSONL → BM25 index JSON")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: data/experiments/e1_vllm/indexes/<stem>_bm25.json",
    )
    args = p.parse_args()
    corpus = args.corpus
    if not corpus.is_absolute():
        corpus = REPO_ROOT / corpus
    out = args.out
    if out is None:
        out = DEFAULT_OUT / f"{corpus.stem}_bm25.json"
    elif not out.is_absolute():
        out = REPO_ROOT / out
    summary = build(corpus, out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
