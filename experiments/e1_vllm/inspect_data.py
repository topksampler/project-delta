#!/usr/bin/env python3
"""Inspect e1_vllm corpora and validate eval gold against chunked docs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA_DIR = REPO_ROOT / "data" / "experiments" / "e1_vllm"
FIXTURES_DIR = EXPERIMENT_DIR / "fixtures"

CORPORA = {
    "doc_0": DATA_DIR / "corpus_doc_0.jsonl",
    "doc_8": DATA_DIR / "corpus_doc_8.jsonl",
}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_corpus(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing corpus: {path} (run build_corpus.py first)")
    return load_jsonl(path)


def corpus_stats(rows: list[dict], label: str) -> None:
    by_top = Counter()
    by_source = Counter()
    lengths = [len(r["text"]) for r in rows]
    for r in rows:
        sp = r["source_path"]
        top = sp.split("/")[0]
        by_top[top] += 1
        by_source[sp] += 1
    print(f"\n=== {label} ===")
    print(f"chunks: {len(rows)}")
    print(f"chars: min={min(lengths)} med={sorted(lengths)[len(lengths)//2]} max={max(lengths)}")
    print("by top-level:", dict(by_top.most_common()))
    print("top files:")
    for src, n in by_source.most_common(8):
        print(f"  {n:3d}  {src}")


def compare_sources(rows_0: list[dict], rows_8: list[dict]) -> None:
    s0 = {r["source_path"] for r in rows_0}
    s8 = {r["source_path"] for r in rows_8}
    only_0 = sorted(s0 - s8)
    only_8 = sorted(s8 - s0)
    print("\n=== source path drift ===")
    print(f"common: {len(s0 & s8)}  only doc_0: {len(only_0)}  only doc_8: {len(only_8)}")
    if only_8:
        print("only in doc_8 (v0.23.0):")
        for p in only_8:
            print(f"  {p}")
    if only_0:
        print("only in doc_0 (v0.22.0):")
        for p in only_0:
            print(f"  {p}")


def search_corpus(rows: list[dict], terms: list[str], *, limit: int) -> list[dict]:
    hits: list[dict] = []
    for row in rows:
        text = row["text"].lower()
        if all(t.lower() in text for t in terms):
            hits.append(row)
            if len(hits) >= limit:
                break
    return hits


def cmd_stats(_: argparse.Namespace) -> None:
    rows_0 = load_corpus(CORPORA["doc_0"])
    rows_8 = load_corpus(CORPORA["doc_8"])
    corpus_stats(rows_0, "doc_0 (v0.22.0)")
    corpus_stats(rows_8, "doc_8 (v0.23.0)")
    compare_sources(rows_0, rows_8)


def cmd_search(args: argparse.Namespace) -> None:
    corpus = load_corpus(CORPORA[args.corpus])
    terms = args.terms
    hits = search_corpus(corpus, terms, limit=args.limit)
    print(f"search {terms!r} in {args.corpus}: {len(hits)} hit(s)")
    for h in hits:
        print(f"\n--- {h['source_path']} chunk {h['chunk_index']}/{h['chunk_total']} ---")
        print(h["text"][: args.max_chars])
        if len(h["text"]) > args.max_chars:
            print("...")


def gold_in_corpus(rows: list[dict], needles: list[str]) -> tuple[bool, list[str], list[dict]]:
    hits = search_corpus(rows, needles, limit=3)
    missing = [n for n in needles if not any(n.lower() in r["text"].lower() for r in rows)]
    return (not missing, missing, hits)


def cmd_validate_eval(args: argparse.Namespace) -> None:
    eval_path = Path(args.eval)
    if not eval_path.is_absolute():
        eval_path = (REPO_ROOT / eval_path) if not eval_path.exists() else eval_path
    if not eval_path.exists():
        eval_path = FIXTURES_DIR / args.eval
    items = load_jsonl(eval_path)

    corpora = {
        "0.22.0": load_corpus(CORPORA["doc_0"]),
        "0.23.0": load_corpus(CORPORA["doc_8"]),
    }

    print(f"eval: {eval_path} ({len(items)} items)")
    print("corpus: per-item requires_doc (doc_0 / doc_8)")
    print()
    ok = 0
    skipped = 0
    for item in items:
        req = item.get("requires_doc", args.against)
        gold = item.get("gold") or {}
        if gold.get("abstain_if_unknown"):
            skipped += 1
            print(
                f"  {item['id']:14} requires={req:6}  SKIP abstain  "
                f"hint={item.get('source_hint', '-')}"
            )
            continue

        rows = corpora.get(req) or corpora[args.against]
        needles = gold["must_contain"]
        valid, missing, hits = gold_in_corpus(rows, needles)
        mark = "OK" if valid else f"MISSING {missing}"
        if valid:
            ok += 1
        hint = item.get("source_hint", "-")
        ec = item.get("eval_class", "-")
        print(f"  {item['id']:14} class={ec} requires={req:6}  {mark:20}  hint={hint}")
        if args.verbose and hits:
            h = hits[0]
            print(f"    e.g. {h['source_path']} [{h['chunk_index']}]")

    checked = len(items) - skipped
    print(f"\n{ok}/{checked} non-abstain items grounded; {skipped} abstain skipped")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect e1_vllm corpora and eval sets.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats", help="corpus size + source drift summary").set_defaults(func=cmd_stats)

    search = sub.add_parser("search", help="grep-like search in a corpus")
    search.add_argument("terms", nargs="+", help="all terms must appear in chunk")
    search.add_argument("--corpus", choices=list(CORPORA), default="doc_8")
    search.add_argument("--limit", type=int, default=3)
    search.add_argument("--max-chars", type=int, default=500)
    search.set_defaults(func=cmd_search)

    val = sub.add_parser("validate-eval", help="check must_contain gold against corpus")
    val.add_argument("eval", nargs="?", default="eval_v2.jsonl")
    val.add_argument("--against", choices=["0.22.0", "0.23.0"], default="0.23.0")
    val.add_argument("--verbose", action="store_true")
    val.set_defaults(func=cmd_validate_eval)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
