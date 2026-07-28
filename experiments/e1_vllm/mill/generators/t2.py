"""T2 — procedural how-to from numbered / imperative doc spans."""

from __future__ import annotations

import re

_STEP_LINE = re.compile(r"^\s*(?:(?:\d+[.)])|(?:[-*]\s+))\s+\S", re.M)
_TOPIC = re.compile(
    r"\b(how to|install|configure|run|serve|deploy|enable|setup|usage|"
    r"quickstart|getting started|set up)\b",
    re.I,
)


def _steps_from_chunk(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        if _STEP_LINE.match(line):
            cleaned = re.sub(r"\s+", " ", line).strip()
            if 12 <= len(cleaned) <= 240:
                out.append(cleaned)
    return out


def generate_t2(
    *,
    symbols: list[dict],
    chunks: dict[str, str],
    tag: str,
    limit: int | None = None,
) -> list[dict]:
    seen: set[str] = set()
    rows: list[dict] = []

    for sym in symbols:
        if sym.get("kind") != "heading":
            continue
        name = (sym.get("name") or "").strip()
        if not name or name in seen or not _TOPIC.search(name):
            continue
        chunk_id = sym.get("chunk_id")
        text = chunks.get(chunk_id or "")
        if not text:
            continue
        steps = _steps_from_chunk(text)
        if len(steps) < 2:
            continue
        token = next(
            (t for t in re.findall(r"[A-Za-z_]{4,}", steps[0]) if t.lower() in text.lower()),
            None,
        )
        if not token:
            continue
        seen.add(name)
        answer = " ".join(steps[:5])
        if not answer.endswith((".", "!", "?")):
            answer += "."
        rows.append(
            {
                "train_class": "T2",
                "messages": [
                    {
                        "role": "user",
                        "content": f"In vLLM, how do I: {name}?",
                    },
                    {"role": "assistant", "content": answer},
                ],
                "provenance": {
                    "tag": tag,
                    "chunk_ids": [chunk_id],
                    "source_paths": [sym.get("source_path", "")],
                    "generator": "t2_howto_v1",
                    "symbol": name,
                },
                "gold_check": {"must_contain": [token], "must_not_contain": []},
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows
