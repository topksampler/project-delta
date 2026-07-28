"""T7 — multi-hop: two consecutive chunks from the same doc path."""

from __future__ import annotations

import re
from collections import defaultdict


def generate_t7(
    *,
    symbols: list[dict],
    chunks: dict[str, str],
    tag: str,
    limit: int | None = None,
) -> list[dict]:
    """Pair consecutive chunk indices on one path; ask using a heading on chunk A."""
    by_path: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for cid, text in chunks.items():
        # chunk_id: {tag}:{rel}:{i}
        parts = cid.split(":")
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[-1])
        except ValueError:
            continue
        src = ":".join(parts[1:-1])
        by_path[src].append((idx, cid))

    headings_by_chunk: dict[str, list[str]] = defaultdict(list)
    for sym in symbols:
        if sym.get("kind") == "heading" and sym.get("chunk_id"):
            headings_by_chunk[sym["chunk_id"]].append(sym["name"])

    rows: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()

    for src, items in sorted(by_path.items()):
        items = sorted(items, key=lambda x: x[0])
        for i in range(len(items) - 1):
            _, cid_a = items[i]
            _, cid_b = items[i + 1]
            pair = (cid_a, cid_b)
            if pair in seen_pairs:
                continue
            heads = headings_by_chunk.get(cid_a) or headings_by_chunk.get(cid_b)
            if not heads:
                continue
            topic = heads[0].strip()
            if len(topic) < 4:
                continue
            text_a = chunks.get(cid_a, "").strip()
            text_b = chunks.get(cid_b, "").strip()
            if len(text_a) < 40 or len(text_b) < 40:
                continue

            # grounding token present in both halves of the combined answer span
            span_a = re.sub(r"\s+", " ", text_a)[:280].strip()
            span_b = re.sub(r"\s+", " ", text_b)[:280].strip()
            if not span_a.endswith((".", "!", "?")):
                span_a += "."
            if not span_b.endswith((".", "!", "?")):
                span_b += "."

            token = next(
                (
                    t
                    for t in re.findall(r"[A-Za-z_][A-Za-z0-9_-]{3,}", topic)
                    if t.lower() in (text_a + "\n" + text_b).lower()
                ),
                None,
            )
            if not token:
                continue

            seen_pairs.add(pair)
            rows.append(
                {
                    "train_class": "T7",
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"Using vLLM docs on `{src}`, summarize "
                                f"\"{topic}\" across consecutive sections."
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": f"{span_a} {span_b}",
                        },
                    ],
                    "provenance": {
                        "tag": tag,
                        "chunk_ids": [cid_a, cid_b],
                        "source_paths": [src],
                        "generator": "t7_multihop_v1",
                        "symbol": topic,
                    },
                    "gold_check": {"must_contain": [token], "must_not_contain": []},
                }
            )
            if limit is not None and len(rows) >= limit:
                return rows
    return rows
