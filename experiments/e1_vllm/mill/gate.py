"""Stage 3 — gate: validate candidates, dedupe, optional eval denylist."""

from __future__ import annotations

import json
import re
from pathlib import Path

_WORD = re.compile(r"[a-z0-9]+", re.I)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _assistant(row: dict) -> str:
    for m in row.get("messages", []):
        if m.get("role") == "assistant":
            return m.get("content") or ""
    return ""


def _user(row: dict) -> str:
    for m in row.get("messages", []):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def _check_row(row: dict, chunks: dict[str, str]) -> str | None:
    """Return reject reason, or None if ok."""
    prov = row.get("provenance") or {}
    chunk_ids = prov.get("chunk_ids") or []
    if not chunk_ids:
        return "orphan: no chunk_ids"

    texts = []
    for cid in chunk_ids:
        if cid not in chunks:
            return f"orphan: missing chunk {cid}"
        texts.append(chunks[cid])
    blob_l = "\n".join(texts).lower()
    answer = _assistant(row)
    answer_l = answer.lower()
    gold = row.get("gold_check") or {}
    abstain = bool(gold.get("abstain"))

    for tok in gold.get("must_contain") or []:
        if tok.lower() not in answer_l:
            return f"ungrounded: {tok!r} not in answer"
        # abstain rows are not corpus-anchored facts
        if not abstain and tok.lower() not in blob_l:
            return f"ungrounded: {tok!r} not in chunk"

    for tok in gold.get("must_not_contain") or []:
        if tok.lower() in answer_l:
            return f"trap: answer contains {tok!r}"

    if len(answer.strip()) < 8:
        return "too_short"
    if not _user(row).strip():
        return "empty_question"
    return None


def gate(
    snapshot: Path,
    *,
    eval_denylist: Path | None = None,
    max_eval_jaccard: float = 0.85,
) -> dict:
    snapshot = snapshot.resolve()
    corpus_path = snapshot / "corpus.jsonl"
    cand_dir = snapshot / "candidates"
    if not corpus_path.is_file() or not cand_dir.is_dir():
        raise FileNotFoundError("need corpus.jsonl and candidates/")

    chunks = {r["chunk_id"]: r["text"] for r in _load_jsonl(corpus_path)}
    candidates: list[dict] = []
    for path in sorted(cand_dir.glob("*.jsonl")):
        candidates.extend(_load_jsonl(path))
    if not candidates:
        raise FileNotFoundError("no candidate jsonl files")

    eval_qs: list[set[str]] = []
    if eval_denylist and eval_denylist.is_file():
        for row in _load_jsonl(eval_denylist):
            q = row.get("question") or ""
            if q:
                eval_qs.append(_tokens(q))

    accepted: list[dict] = []
    rejected: list[dict] = []
    seen_q: set[str] = set()
    reasons: dict[str, int] = {}

    def reject(row: dict, reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1
        rejected.append({**row, "reject_reason": reason})

    for row in candidates:
        reason = _check_row(row, chunks)
        if reason:
            reject(row, reason)
            continue

        qn = _norm(_user(row))
        if qn in seen_q:
            reject(row, "dup_question")
            continue

        if eval_qs:
            qt = _tokens(_user(row))
            if any(_jaccard(qt, eq) >= max_eval_jaccard for eq in eval_qs):
                reject(row, "eval_contamination")
                continue

        seen_q.add(qn)
        accepted.append(row)

    out = snapshot / "gated"
    _write_jsonl(accepted, out / "accepted.jsonl")
    _write_jsonl(rejected, out / "rejected.jsonl")

    by_class: dict[str, int] = {}
    for row in accepted:
        c = row.get("train_class", "?")
        by_class[c] = by_class.get(c, 0) + 1

    return {
        "snapshot": str(snapshot),
        "accepted": str(out / "accepted.jsonl"),
        "rejected": str(out / "rejected.jsonl"),
        "n_in": len(candidates),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "by_class": by_class,
        "reject_reasons": reasons,
    }
