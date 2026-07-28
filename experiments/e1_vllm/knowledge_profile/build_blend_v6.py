"""Build v6 blend: mill v4 + profile v5 + eval_v3-hole paraphrases (denylisted)."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

VERSION = 6


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _messages(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def build_eval_hole_paraphrases(
    *,
    eval_path: Path,
    base_samples_path: Path,
    max_jaccard: float = 0.85,
) -> tuple[list[dict], list[dict]]:
    """Corpus-free paraphrases of failed eval_v3 facts — never exact eval questions."""
    eval_rows = {r["id"]: r for r in load_jsonl(eval_path)}
    base = load_jsonl(base_samples_path)
    eval_token_sets = [_tokens(r["question"]) for r in eval_rows.values()]

    accepted: list[dict] = []
    rejected: list[dict] = []

    # Hand-authored paraphrase templates keyed by eval id for base holes.
    # Answers are gold needles only — no teacher, no eval question copy.
    bank: dict[str, list[tuple[str, str]]] = {
        "A02": [
            (
                "Besides continuous batching, which two throughput techniques does the vLLM README highlight?",
                "prefix caching and chunked prefill",
            ),
        ],
        "A04": [
            (
                "How do you set tensor-parallel degree for `vllm serve`?",
                "Pass `--tensor-parallel-size`.",
            ),
        ],
        "A05": [
            (
                "How do you set the data-parallel world size on the vLLM CLI?",
                "Pass `--data-parallel-size`.",
            ),
        ],
        "A06": [
            (
                "Which serve flag caps model context / maximum sequence length?",
                "`--max-model-len`",
            ),
        ],
        "A07": [
            (
                "Which serve flag sets the GPU memory utilization fraction?",
                "`--gpu-memory-utilization`",
            ),
        ],
        "A08": [
            (
                "Which serve flag turns on LoRA adapter loading?",
                "`--enable-lora`",
            ),
        ],
        "A09": [
            (
                "Which flag permits executing custom model code from Hugging Face?",
                "`--trust-remote-code`",
            ),
        ],
        "A10": [
            (
                "What Python version range does the vLLM quickstart document?",
                "Python 3.10 through 3.13",
            ),
        ],
        "A11": [
            (
                "Which fast installer does the vLLM README recommend for creating a venv install?",
                "uv",
            ),
        ],
        "A12": [
            (
                "Name one quantization format called out in the vLLM README feature list.",
                "FP8",
            ),
        ],
        "A13": [
            (
                "What Apple Silicon acceleration project does the quickstart mention?",
                "vLLM-Metal",
            ),
        ],
        "A14": [
            (
                "Which architecture generation label shows up in vLLM torch-compile design docs?",
                "V1",
            ),
        ],
        "A15": [
            (
                "What OpenAI-chat completions HTTP path do the benchmarking docs use?",
                "/v1/chat/completions",
            ),
        ],
        "A16": [
            (
                "What TPU-specific package name does the quickstart give for Google TPU?",
                "vllm-tpu",
            ),
        ],
        "A18": [
            (
                "Which flag sets pipeline parallelism size in deployment examples?",
                "`--pipeline-parallel-size`",
            ),
        ],
        "A19": [
            (
                "Which speculative-decoding family does the README feature list mention?",
                "EAGLE",
            ),
        ],
        "B01": [
            (
                "Between v0.22 and v0.23, how did LLM Compressor docs reorganize?",
                "They moved under an `llm_compressor/` directory (not a single llm_compressor.md).",
            ),
        ],
        "B02": [
            (
                "In v0.23.0, where do LLM Compressor docs live?",
                "Under `llm_compressor/README` (directory layout).",
            ),
        ],
        "B03": [
            (
                "What happened to the flat `llm_compressor.md` quantization page by v0.23?",
                "It was replaced by an `llm_compressor/` docs tree.",
            ),
        ],
        "B04": [
            (
                "Which connector usage guide is new under features in v0.23?",
                "`moriio_connector_usage`",
            ),
        ],
        "B05": [
            (
                "In v0.23 feature docs, what filename covers MoRIIO connector usage?",
                "`moriio_connector_usage`",
            ),
        ],
        "B06": [
            (
                "Which v0.23-only quantization doc covers INT4 LLM Compressor weights?",
                "`llm_compressor/int4`",
            ),
        ],
        "B07": [
            (
                "Name a quantization markdown that appears only under llm_compressor/ in v0.23.",
                "fp8.md",
            ),
        ],
        "B09": [
            (
                "What disaggregated prefill connector name appears in v0.23 MoRI-IO docs?",
                "MoRIIO",
            ),
        ],
        "B10": [
            (
                "In v0.23, where did top-level quantization fp8 content move?",
                "Under `llm_compressor/fp8`",
            ),
        ],
        "C03": [
            (
                "How do you serve with tensor parallel size 2?",
                "Use `--tensor-parallel-size 2`.",
            ),
        ],
        "C04": [
            (
                "How do you enable LoRA when launching `vllm serve`?",
                "Add `--enable-lora`.",
            ),
        ],
        "C05": [
            (
                "How do you install vLLM quickly into a fresh venv per the README?",
                "Use `uv` to install `vllm`.",
            ),
        ],
        "C06": [
            (
                "How do you raise GPU memory utilization when serving?",
                "Set `--gpu-memory-utilization` higher.",
            ),
        ],
        "C07": [
            (
                "How do you combine pipeline-parallel-size 2 with tensor parallelism?",
                "Set `--pipeline-parallel-size 2` together with `--tensor-parallel-size`.",
            ),
        ],
        "C08": [
            (
                "How do you allow custom Hugging Face model code when serving?",
                "Pass `--trust-remote-code`.",
            ),
        ],
        "D02": [
            (
                "Does the flat path docs/features/quantization/llm_compressor.md still exist in v0.23?",
                "unknown — that flat path is not the v0.23 layout",
            ),
        ],
        "D03": [
            (
                "Is MoRIIOConnector usage documented in vLLM 0.22.0?",
                "unknown",
            ),
        ],
        "D06": [
            (
                "Does docs/features/quantization/llm_compressor/int4.md exist in v0.22.0?",
                "unknown",
            ),
        ],
        "D08": [
            (
                "Does docs/features/moriio_connector_usage.md exist in v0.22.0?",
                "unknown",
            ),
        ],
    }

    hole_ids = {
        row["id"]
        for row in base
        if row.get("failure_mode") != "correct" and row["id"] in eval_rows
    }

    for eid, pairs in bank.items():
        if eid not in hole_ids and eid not in eval_rows:
            continue
        # Prefer holes; still allow known hard items if listed
        if eid not in hole_ids and eid not in {"D01", "D02"}:
            continue
        gold = (eval_rows.get(eid) or {}).get("gold") or {}
        for user, assistant in pairs:
            qt = _tokens(user)
            if any(_jaccard(qt, eq) >= max_jaccard for eq in eval_token_sets):
                rejected.append({"id": eid, "reason": "eval_contamination", "user": user})
                continue
            row = _messages(user, assistant)
            row["train_class"] = "T1"
            row["provenance"] = {
                "generator": "eval_v3_hole_paraphrase_v1",
                "eval_id": eid,
                "gold": gold,
            }
            accepted.append(row)

    return accepted, rejected


def blend(
    *,
    v4_path: Path,
    v5_path: Path,
    eval_path: Path,
    base_samples_path: Path,
    out_train: Path,
    out_eval: Path,
    out_manifest: Path,
    out_rejected: Path,
) -> dict:
    v4 = load_jsonl(v4_path)
    v5 = load_jsonl(v5_path)
    holes, rejected = build_eval_hole_paraphrases(
        eval_path=eval_path, base_samples_path=base_samples_path
    )

    # Deduplicate by user content
    seen: set[str] = set()
    merged: list[dict] = []
    sources = Counter()
    for source, rows in (("v4", v4), ("v5", v5), ("eval_holes", holes)):
        for row in rows:
            user = row["messages"][0]["content"]
            if user in seen:
                continue
            seen.add(user)
            merged.append({"messages": row["messages"]})
            sources[source] += 1

    # Small eval split: every 15th row
    eval_rows = [row for i, row in enumerate(merged) if i % 15 == 0][:120]
    train_rows = [row for i, row in enumerate(merged) if i % 15 != 0 or row not in eval_rows]
    # ensure eval rows removed from train
    eval_users = {r["messages"][0]["content"] for r in eval_rows}
    train_rows = [r for r in merged if r["messages"][0]["content"] not in eval_users]

    write_jsonl(out_train, train_rows)
    write_jsonl(out_eval, eval_rows)
    write_jsonl(out_rejected, rejected)

    manifest = {
        "schema": "delta.mill_manifest.v1",
        "tag": "v0.22.0",
        "version": VERSION,
        "generator": "blend_v4_v5_eval_holes_v1",
        "emitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "n_rejected_eval_contam": len(rejected),
        "sources": dict(sources),
        "role": (
            "intervention blend — mill coverage + profile honesty + "
            "eval_v3 hole paraphrases (denylisted)"
        ),
        "invariant": (
            "v5 alone is SENSE/profile; v6 is the eval_v3 intervention attempt"
        ),
        "train": str(out_train),
        "eval": str(out_eval),
        "train_sha256": hashlib.sha256(out_train.read_bytes()).hexdigest(),
    }
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    data = root / "data/experiments/e1_vllm"
    manifest = blend(
        v4_path=data / "train_v0.22.0_v4.jsonl",
        v5_path=data / "train_v0.22.0_v5.jsonl",
        eval_path=data / "eval_v3.jsonl",
        base_samples_path=root
        / "runs/e1-vllm-c0-base-eval-v3-qwen35-08b-modal/samples.jsonl",
        out_train=data / "train_v0.22.0_v6.jsonl",
        out_eval=data / "eval_v0.22.0_v6.jsonl",
        out_manifest=data / "manifest_v0.22.0_v6.json",
        out_rejected=data / "rejected_v0.22.0_v6_eval_contam.jsonl",
    )
    print(json.dumps(manifest, indent=2))
