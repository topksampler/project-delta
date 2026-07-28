"""T4 — abstain: ask about things not in this tag's docs."""

from __future__ import annotations

# Deliberately fake / other-product names. Gate checks they are absent from index.
FAKE_FLAGS = (
    "--enable-zentorch-fusion",
    "--tensorrt-llm-backend",
    "--mlx-metal-offload",
    "--deepspeed-zero3-serve",
    "--llama-cpp-n-gpu-layers",
    "--vllm-legacy-engine-v0-only",
    "--cuda-graphs-for-moe-expert-xyz",
    "--hypothetical-speculative-draft-model-path",
)


def generate_t4(
    *,
    symbols: list[dict],
    tag: str,
    limit: int | None = None,
) -> list[dict]:
    real = {s["name"] for s in symbols if s.get("kind") == "flag"}
    # pick any real chunk id only for provenance bookkeeping (not as gold)
    anchor = next((s for s in symbols if s.get("chunk_id")), None)
    if not anchor:
        return []

    rows: list[dict] = []
    for fake in FAKE_FLAGS:
        if fake in real:
            continue
        rows.append(
            {
                "train_class": "T4",
                "messages": [
                    {
                        "role": "user",
                        "content": f"What does `{fake}` control in vLLM {tag}?",
                    },
                    {
                        "role": "assistant",
                        "content": f"unknown — not documented in vLLM {tag}",
                    },
                ],
                "provenance": {
                    "tag": tag,
                    "chunk_ids": [anchor["chunk_id"]],
                    "source_paths": [anchor["source_path"]],
                    "generator": "t4_abstain_v1",
                    "symbol": fake,
                },
                "gold_check": {
                    "must_contain": ["unknown"],
                    "must_not_contain": [fake],
                    "abstain": True,
                },
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows
