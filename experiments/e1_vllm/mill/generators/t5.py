"""T5 — negatives / contrast: wrong flag vs right flag."""

from __future__ import annotations

from generators.common import unique_flags


def _near_miss(flag: str, pool: list[str]) -> str | None:
    """Pick another flag from the pool that shares a prefix token, else any other."""
    stem = flag.strip("-").split("-")[0]
    for other in pool:
        if other == flag:
            continue
        if other.strip("-").split("-")[0] == stem:
            return other
    for other in pool:
        if other != flag:
            return other
    return None


def generate_t5(
    *,
    symbols: list[dict],
    chunks: dict[str, str],
    tag: str,
    limit: int | None = None,
) -> list[dict]:
    flags = unique_flags(symbols)
    names = [s["name"] for s in flags]
    rows: list[dict] = []

    for sym in flags:
        right = sym["name"]
        wrong = _near_miss(right, names)
        if not wrong:
            continue
        chunk_id = sym.get("chunk_id")
        text = chunks.get(chunk_id or "")
        if not text or right not in text:
            continue

        rows.append(
            {
                "train_class": "T5",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"In vLLM {tag}, is `{wrong}` the flag that "
                            f"`{right}` is for? Answer briefly."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"No. Use `{right}`, not `{wrong}`."
                        ),
                    },
                ],
                "provenance": {
                    "tag": tag,
                    "chunk_ids": [chunk_id],
                    "source_paths": [sym["source_path"]],
                    "generator": "t5_contrast_v1",
                    "symbol": right,
                    "contrast": wrong,
                },
                "gold_check": {
                    "must_contain": [right],
                    "must_not_contain": [],
                },
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows
