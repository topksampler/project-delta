#!/usr/bin/env python3
"""Build chunk inverted index: lowercase needle → chunk_ids (for symbol routing)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--needles-from-probes",
        type=Path,
        action="append",
        default=[],
        help="Probe JSONL files; index only display_entity / entity strings",
    )
    args = p.parse_args()
    needles: set[str] = set()
    for path in args.needles_from_probes:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("display_entity", "entity"):
                val = row.get(key)
                if val:
                    needles.add(str(val).lower())
    if not needles:
        raise SystemExit("no needles — pass --needles-from-probes")

    postings: dict[str, list[str]] = defaultdict(list)
    path_postings: dict[str, list[str]] = defaultdict(list)
    n_docs = 0
    with args.corpus.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            n_docs += 1
            cid = row["chunk_id"]
            text = (row.get("text") or "").lower()
            sp = (row.get("source_path") or "").lower()
            for needle in needles:
                if needle and needle in text:
                    postings[needle].append(cid)
                # path hint: last path component contains needle fragment
                if needle and needle.lstrip("-").replace(".", "_") in sp.replace("/", "_"):
                    path_postings[needle].append(cid)

    payload = {
        "schema": "delta.eval_factory.symbol_index.v1",
        "corpus_path": str(args.corpus),
        "n_docs": n_docs,
        "n_needles": len(needles),
        "text_postings": {k: v for k, v in sorted(postings.items())},
        "path_postings": {k: v for k, v in sorted(path_postings.items())},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    hit = sum(1 for v in postings.values() if v)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n_docs": n_docs,
                "n_needles": len(needles),
                "needles_with_text_hit": hit,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
