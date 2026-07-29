#!/usr/bin/env python3
"""Export Experiment C teacher-context SFT + closed-book consolidation.

Invariant:
  Teacher context (hybrid_diff / oracle evidence) is train-time only.
  Curriculum ends with a closed-book consolidation phase on the same claims.
  Assistant target is always gold_after.

Phases:
  teacher     — user may include retrieved/oracle context; also emit bare
                paraphrase/no-context pairs for the same claim_ids
  consolidate — context dropped; bare question → gold_after only
"""

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

SCHEMA_TEACHER = "delta.eval_factory.sft_teacher_context.v1"
SCHEMA_CONSOLIDATE = "delta.eval_factory.sft_cb_consolidate.v1"


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


def _message_row(user: str, gold: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": gold_answer(gold)},
        ]
    }


def _build_retriever(args: argparse.Namespace):
    if args.teacher in {"none", "bare"}:
        return None
    from lab.eval_vllm_qa import RetrievalBundle

    if args.teacher == "oracle":
        return RetrievalBundle(
            mode="evidence",
            evidence_map_path=args.evidence_map,
            max_chars=args.max_chars,
            seed=args.seed,
        )
    if args.teacher == "hybrid_diff":
        return RetrievalBundle(
            corpus_path=args.corpus,
            index_path=args.index,
            mode="hybrid_diff",
            top_k=args.top_k,
            max_chars=args.max_chars,
            seed=args.seed,
            symbol_index_path=args.symbol_index,
            diff_corpus_path=args.diff_corpus,
            diff_index_path=args.diff_index,
        )
    raise ValueError(f"unknown teacher mode: {args.teacher}")


def _context_for(retriever, probe: dict) -> str:
    if retriever is None:
        return ""
    ctx, _ids = retriever.retrieve(probe["question"], example=probe)
    return ctx or ""


def export_teacher(
    *,
    probes: list[dict],
    selected: list[dict],
    retriever,
    bare_paraphrases: bool,
) -> tuple[list[dict], dict]:
    """Teacher phase: context rows + optional bare paraphrase rows."""
    slim: list[dict] = []
    nonempty = 0
    for row in selected:
        ctx = _context_for(retriever, row)
        if ctx:
            nonempty += 1
            user = f"{ctx}\n\nQuestion: {row['question']}"
        else:
            user = row["question"]
        slim.append(_message_row(user, row.get("gold") or {}))

    bare_n = 0
    if bare_paraphrases:
        claim_ids = {row.get("claim_id") for row in selected if row.get("claim_id")}
        seen_probe_ids: set[str] = set()
        for row in probes:
            cid = row.get("claim_id")
            pid = row.get("id")
            if cid not in claim_ids:
                continue
            if pid in seen_probe_ids:
                continue
            seen_probe_ids.add(str(pid))
            slim.append(_message_row(row["question"], row.get("gold") or {}))
            bare_n += 1

    stats = {
        "n_teacher_context": len(selected),
        "nonempty_context": nonempty,
        "n_bare_paraphrase": bare_n,
        "n": len(slim),
    }
    return slim, stats


def export_consolidate(*, selected: list[dict]) -> tuple[list[dict], dict]:
    """Consolidation phase: context dropped on the same diet rows."""
    slim = [_message_row(row["question"], row.get("gold") or {}) for row in selected]
    return slim, {"n": len(slim), "n_teacher_context": 0, "n_bare_paraphrase": len(slim)}


def convert(args: argparse.Namespace) -> dict:
    probes = [
        json.loads(line)
        for line in args.probes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = select_diet(
        probes,
        delta_copies=args.delta_copies,
        stable_cap=args.stable_cap,
        seed=args.seed,
    )
    retriever = None
    if args.phase == "teacher":
        retriever = _build_retriever(args)
        slim, phase_stats = export_teacher(
            probes=probes,
            selected=selected,
            retriever=retriever,
            bare_paraphrases=not args.no_bare_paraphrases,
        )
        schema = SCHEMA_TEACHER
    else:
        slim, phase_stats = export_consolidate(selected=selected)
        schema = SCHEMA_CONSOLIDATE

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in slim),
        encoding="utf-8",
    )
    route_log = dict(getattr(retriever, "route_log", {}) or {}) if retriever else {}
    meta = {
        "schema": schema,
        "phase": args.phase,
        "teacher": args.teacher if args.phase == "teacher" else "none",
        "source": str(args.probes),
        "delta_copies": args.delta_copies,
        "stable_cap": args.stable_cap,
        "seed": args.seed,
        "by_drift": dict(Counter(r.get("drift_type", "stable") for r in selected)),
        "route_log": route_log,
        **phase_stats,
    }
    if args.phase == "teacher" and args.teacher == "hybrid_diff":
        meta.update(
            {
                "corpus": str(args.corpus) if args.corpus else None,
                "index": str(args.index) if args.index else None,
                "symbol_index": str(args.symbol_index) if args.symbol_index else None,
                "diff_corpus": str(args.diff_corpus) if args.diff_corpus else None,
                "diff_index": str(args.diff_index) if args.diff_index else None,
            }
        )
    if args.phase == "teacher" and args.teacher == "oracle":
        meta["evidence_map"] = str(args.evidence_map) if args.evidence_map else None

    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"out": str(args.out), "meta": str(meta_path), **meta}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probes", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phase", choices=("teacher", "consolidate"), required=True)
    p.add_argument(
        "--teacher",
        choices=("hybrid_diff", "oracle", "none"),
        default="hybrid_diff",
        help="train-time teacher context (ignored for consolidate)",
    )
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--index", type=Path, default=None)
    p.add_argument("--symbol-index", type=Path, default=None)
    p.add_argument("--diff-corpus", type=Path, default=None)
    p.add_argument("--diff-index", type=Path, default=None)
    p.add_argument("--evidence-map", type=Path, default=None)
    p.add_argument("--delta-copies", type=int, default=40)
    p.add_argument("--stable-cap", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--max-chars", type=int, default=3500)
    p.add_argument(
        "--no-bare-paraphrases",
        action="store_true",
        help="teacher phase: skip bare paraphrase pairs",
    )
    args = p.parse_args()
    if args.phase == "teacher" and args.teacher == "hybrid_diff":
        for req in ("corpus", "index", "diff_corpus", "diff_index"):
            if getattr(args, req) is None:
                p.error(f"--{req.replace('_', '-')} required for hybrid_diff")
    if args.phase == "teacher" and args.teacher == "oracle" and args.evidence_map is None:
        p.error("--evidence-map required for oracle teacher")
    print(json.dumps(convert(args), indent=2))


if __name__ == "__main__":
    main()
