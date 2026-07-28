"""T6 — cite-augmented: same as flag Q→A but answer names source_path."""

from __future__ import annotations

from generators.common import answers_for_flags, unique_flag_names


def generate_t6(
    *,
    symbols: list[dict],
    chunks: dict[str, str],
    tag: str,
    limit: int | None = None,
) -> list[dict]:
    path_for: dict[str, str] = {}
    for sym in symbols:
        if sym.get("kind") == "flag" and sym["name"] not in path_for:
            path_for[sym["name"]] = sym.get("source_path", "")

    names = unique_flag_names(symbols)
    answers = answers_for_flags(names, chunks)

    rows: list[dict] = []
    for name in names:
        hit = answers.get(name)
        if not hit:
            continue
        span, chunk_id = hit
        src = path_for.get(name, "")
        if not src:
            continue
        rows.append(
            {
                "train_class": "T6",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"What does `{name}` control in vLLM, "
                            f"and which doc path covers it?"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": f"{span} Source: `{src}`.",
                    },
                ],
                "provenance": {
                    "tag": tag,
                    "chunk_ids": [chunk_id],
                    "source_paths": [src],
                    "generator": "t6_cite_v1",
                    "symbol": name,
                },
                # path is metadata; do not require it in chunk text
                "gold_check": {"must_contain": [name], "must_not_contain": []},
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows
