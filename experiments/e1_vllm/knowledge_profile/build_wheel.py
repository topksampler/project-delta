"""Profile-driven data wheel: meanings + contrastive honesty + abstain.

Invariant: docs evidence spans define truth; the target model never labels gold.
Targets the 0.8B dense-profile hole: chance recognition + acquiescent yes-bias.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .dense import PROTOCOL, SEED, _distractors, _zone

WHEEL_VERSION = 5
GENERATOR = "profile_wheel_meanings_honesty_v1"


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _row(
    *,
    train_class: str,
    user: str,
    assistant: str,
    claim: dict,
    extra_prov: dict | None = None,
    gold_check: dict | None = None,
) -> dict:
    provenance = {
        "tag": claim["source_revision"],
        "chunk_ids": claim["evidence"].get("chunk_ids", []),
        "source_paths": claim["evidence"].get("source_paths", []),
        "generator": GENERATOR,
        "symbol": claim["entity"],
        "claim_id": claim["claim_id"],
        "zone": _zone(claim),
        "protocol": PROTOCOL,
    }
    if extra_prov:
        provenance.update(extra_prov)
    return {
        "train_class": train_class,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "provenance": provenance,
        "gold_check": gold_check
        or {
            "must_contain": [claim["entity"]],
            "must_not_contain": [],
        },
    }


def build_wheel_rows(
    claims: list[dict],
    *,
    seed: int = SEED,
    n_paraphrases: int = 3,
) -> list[dict]:
    labels = ["A", "B", "C"]
    rows: list[dict] = []

    for claim in claims:
        flag = claim["entity"]
        meaning = claim["object"]
        rng = random.Random(f"{seed}:wheel:{claim['claim_id']}")
        wrong = _distractors(claim, claims, rng, count=2)
        if len(wrong) < 2:
            continue

        recall_qs = [
            f"What does `{flag}` control in vLLM v0.22.0?",
            f"Explain the documented purpose of `{flag}` in vLLM version 0.22.0.",
            f"What behavior does the vLLM v0.22.0 option `{flag}` affect?",
        ]
        for question in recall_qs[:n_paraphrases]:
            rows.append(
                _row(
                    train_class="T1",
                    user=question,
                    assistant=meaning,
                    claim=claim,
                    extra_prov={"form": "free_recall"},
                    gold_check={
                        "must_contain": [flag],
                        "must_not_contain": [],
                    },
                )
            )

        for idx in range(n_paraphrases):
            choices = [meaning, wrong[0]["object"], wrong[1]["object"]]
            order = list(range(3))
            random.Random(f"{seed}:{claim['claim_id']}:choice:{idx}").shuffle(order)
            options = [(labels[pos], choices[src]) for pos, src in enumerate(order)]
            gold_label = labels[order.index(0)]
            rendered = "\n".join(f"{lab}. {text}" for lab, text in options)
            leads = [
                f"Which description best matches the vLLM v0.22.0 flag `{flag}`?",
                f"In vLLM 0.22.0, what is the documented purpose of `{flag}`?",
                f"Choose the accurate meaning of `{flag}` for vLLM version 0.22.0.",
            ]
            user = (
                f"{leads[idx]}\n{rendered}\nReply exactly as `ANSWER: <letter>`."
            )
            rows.append(
                _row(
                    train_class="T5",
                    user=user,
                    assistant=f"ANSWER: {gold_label}",
                    claim=claim,
                    extra_prov={
                        "form": "meaning_recognition",
                        "contrast_claim_ids": [c["claim_id"] for c in wrong],
                    },
                    gold_check={
                        "must_contain": [f"ANSWER: {gold_label}"],
                        "must_not_contain": [],
                    },
                )
            )

        true_qs = [
            (
                f"Is this description accurate for vLLM v0.22.0?\n"
                f"`{flag}`: {meaning}\n"
                "Answer yes or no. If you cannot verify it, answer unknown."
            ),
            (
                f"Does the documentation support this description for vLLM v0.22.0?\n"
                f"`{flag}`: {meaning}\n"
                "Answer yes or no. If you cannot verify it, answer unknown."
            ),
            (
                f"Is the following statement true for vLLM v0.22.0?\n"
                f"`{flag}`: {meaning}\n"
                "Answer yes or no. If you cannot verify it, answer unknown."
            ),
        ]
        for question in true_qs:
            rows.append(
                _row(
                    train_class="T1",
                    user=question,
                    assistant=f"Yes. `{flag}`: {meaning}",
                    claim=claim,
                    extra_prov={"form": "meaning_true"},
                    gold_check={
                        "must_contain": ["Yes", flag],
                        "must_not_contain": ["unknown"],
                    },
                )
            )

        for idx, question_base in enumerate(
            [
                "Is this description accurate",
                "Does the documentation support this description",
                "Is the following statement true",
            ]
        ):
            donor = wrong[idx % len(wrong)]
            false_meaning = donor["object"]
            question = (
                f"{question_base} for vLLM v0.22.0?\n"
                f"`{flag}`: {false_meaning}\n"
                "Answer yes or no. If you cannot verify it, answer unknown."
            )
            rows.append(
                _row(
                    train_class="T5",
                    user=question,
                    assistant=(
                        f"No. That description does not match `{flag}` "
                        f"in vLLM v0.22.0."
                    ),
                    claim=claim,
                    extra_prov={
                        "form": "meaning_false",
                        "contrast_claim_id": donor["claim_id"],
                    },
                    gold_check={
                        "must_contain": ["No"],
                        "must_not_contain": ["Yes"],
                    },
                )
            )

    # Name-boundary abstain: keep the skill that already works on 0.8B.
    fake_flags = [
        "--enable-zentorch-fusion",
        "--tensorrt-llm-backend",
        "--mlx-metal-offload",
        "--deepspeed-zero3-serve",
        "--llama-cpp-n-gpu-layers",
        "--hypothetical-speculative-draft-model-path",
    ]
    real = {c["entity"] for c in claims}
    anchor = claims[0]
    for fake in fake_flags:
        if fake in real:
            continue
        rows.append(
            {
                "train_class": "T4",
                "messages": [
                    {
                        "role": "user",
                        "content": f"What does `{fake}` control in vLLM v0.22.0?",
                    },
                    {
                        "role": "assistant",
                        "content": "unknown — not documented in vLLM v0.22.0",
                    },
                ],
                "provenance": {
                    "tag": "v0.22.0",
                    "chunk_ids": anchor["evidence"].get("chunk_ids", []),
                    "source_paths": anchor["evidence"].get("source_paths", []),
                    "generator": GENERATOR,
                    "symbol": fake,
                    "form": "name_abstain",
                    "protocol": PROTOCOL,
                },
                "gold_check": {
                    "must_contain": ["unknown"],
                    "must_not_contain": [fake],
                    "abstain": True,
                },
            }
        )

    rows.sort(
        key=lambda r: (
            r["train_class"],
            r["provenance"].get("claim_id", ""),
            r["provenance"].get("form", ""),
            r["messages"][0]["content"],
        )
    )
    return rows


def emit_wheel(
    *,
    claims_path: Path,
    profile_path: Path | None,
    out_train: Path,
    out_full: Path,
    out_manifest: Path,
    holdout_claim_ids: set[str] | None = None,
) -> dict:
    claims = load_jsonl(claims_path)
    holdout_claim_ids = holdout_claim_ids or set()

    priority: set[str] = set()
    if profile_path and profile_path.exists():
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        for claim in profile.get("claims", []):
            honesty = claim.get("honesty_status")
            meaning = claim.get("meaning_status")
            if honesty in {
                "acquiescent_hallucination",
                "unreliable",
                "rejection_biased",
            } or meaning in {"unknown", "weakly_elicitable", "recognized_not_recalled"}:
                priority.add(claim["claim_id"])

    # Prefer hole claims; keep a small known set for stability.
    hole = [c for c in claims if c["claim_id"] in priority]
    known = [c for c in claims if c["claim_id"] not in priority]
    selected = hole + known
    # Drop human-audit holdout claims from train entirely.
    selected = [c for c in selected if c["claim_id"] not in holdout_claim_ids]

    rows = build_wheel_rows(selected)
    train_messages = [{"messages": row["messages"]} for row in rows]
    write_jsonl(out_train, train_messages)
    write_jsonl(out_full, rows)

    by_class = Counter(row["train_class"] for row in rows)
    by_form = Counter(row["provenance"].get("form", "?") for row in rows)
    manifest = {
        "schema": "delta.mill_manifest.v1",
        "tag": "v0.22.0",
        "version": WHEEL_VERSION,
        "generator": GENERATOR,
        "protocol": PROTOCOL,
        "emitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_claims_selected": len(selected),
        "n_claims_holdout": len(holdout_claim_ids),
        "n_priority_hole_claims": len([c for c in selected if c["claim_id"] in priority]),
        "n_train": len(train_messages),
        "by_class": dict(sorted(by_class.items())),
        "by_form": dict(sorted(by_form.items())),
        "claims_path": str(claims_path),
        "profile_path": str(profile_path) if profile_path else None,
        "train": str(out_train),
        "train_full": str(out_full),
        "train_sha256": _hash_bytes(out_train.read_bytes()),
        "holdout_claim_ids": sorted(holdout_claim_ids),
    }
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    audit_rows = load_jsonl(
        root / "artifacts/reports/dense_v1_grader_audit_50.jsonl"
    ) if (root / "artifacts/reports/dense_v1_grader_audit_50.jsonl").exists() else []
    # Hold out one claim per audit item so grader review stays clean.
    holdout = {row["claim_id"] for row in audit_rows}
    # Cap holdout so train still covers most of the bank (~10 claims).
    holdout = set(sorted(holdout)[:10])

    manifest = emit_wheel(
        claims_path=root
        / "data/experiments/e1_vllm/profile/dense_v1/claims_verified.jsonl",
        profile_path=root
        / "artifacts/reports/topic_knowledge_profile_dense_v1_verified_v0.22.0.json",
        out_train=root / "data/experiments/e1_vllm/train_v0.22.0_v5.jsonl",
        out_full=root / "data/experiments/e1_vllm/train_v0.22.0_v5_full.jsonl",
        out_manifest=root / "data/experiments/e1_vllm/manifest_v0.22.0_v5.json",
        holdout_claim_ids=holdout,
    )
    print(json.dumps(manifest, indent=2))
