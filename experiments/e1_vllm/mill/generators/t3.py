"""T3 — version-conditioned: same facts, tag glued into Q and A."""

from __future__ import annotations

from generators.common import answers_for_flags, unique_flag_names


def generate_t3(
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
        rows.append(
            {
                "train_class": "T3",
                "messages": [
                    {
                        "role": "user",
                        "content": f"In vLLM {tag}, what does `{name}` control?",
                    },
                    {
                        "role": "assistant",
                        "content": f"In vLLM {tag}: {span}",
                    },
                ],
                "provenance": {
                    "tag": tag,
                    "chunk_ids": [chunk_id],
                    "source_paths": [path_for.get(name, "")],
                    "generator": "t3_version_v2",
                    "symbol": name,
                },
                "gold_check": {"must_contain": [name], "must_not_contain": []},
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows
