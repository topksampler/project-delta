"""Held-out paraphrase probes — same claims, new surface forms (anti-memorization)."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .dense import PROTOCOL, _distractors, write_dense_artifacts
from .probes import _probe, write_probes

PARA_SEED = 20260727
PARA_PROTOCOL = "e1_profile_meaning_honesty_v1_paraphrase_holdout"


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_paraphrase_probes(
    claims: list[dict],
    *,
    seed: int = PARA_SEED,
    n_paraphrases: int = 3,
) -> list[dict]:
    """Same gold/forms as dense v1, deliberately different question templates."""
    probes: list[dict] = []
    labels = ["A", "B", "C"]

    for claim in claims:
        flag = claim["entity"]
        meaning = claim["object"]
        keywords = claim["truth"]["keywords"]
        # Keep same distractor partners as dense seed family for fairness,
        # but reshuffle option order with PARA_SEED.
        rng_dist = random.Random(f"20260720:{claim['claim_id']}")
        wrong = _distractors(claim, claims, rng_dist)
        if len(wrong) < 2:
            continue

        recall_qs = [
            f"According to the vLLM 0.22.0 docs, what is `{flag}` used for?",
            f"Summarize the role of `{flag}` when running vLLM at version 0.22.0.",
            f"In the v0.22.0 documentation, which setting does `{flag}` change?",
        ]
        for idx, question in enumerate(recall_qs[:n_paraphrases]):
            probes.append(
                _probe(
                    claim=claim,
                    probe_form="free_recall",
                    paraphrase_idx=idx,
                    question=question,
                    gold={"must_contain_any": [keywords], "advisory": True},
                )
            )

        for idx in range(n_paraphrases):
            choices = [meaning, wrong[0]["object"], wrong[1]["object"]]
            order = list(range(3))
            random.Random(f"{seed}:{claim['claim_id']}:choice:{idx}").shuffle(order)
            options = [(labels[pos], choices[src]) for pos, src in enumerate(order)]
            gold_label = labels[order.index(0)]
            leads = [
                f"Select the docs-correct gloss for `{flag}` (vLLM 0.22.0):",
                f"For vLLM v0.22.0, which option correctly defines `{flag}`?",
                f"Match `{flag}` to its documented meaning in v0.22.0:",
            ]
            rendered = "\n".join(f"{lab}. {text}" for lab, text in options)
            question = (
                f"{leads[idx]}\n{rendered}\n"
                "Respond with only `ANSWER: <letter>`."
            )
            probe = _probe(
                claim=claim,
                probe_form="meaning_recognition",
                paraphrase_idx=idx,
                question=question,
                gold={"choice": gold_label},
            )
            probe["contrast_claim_ids"] = [c["claim_id"] for c in wrong]
            probe["paraphrase_bank"] = "holdout_v1"
            probes.append(probe)

        true_leads = [
            "Given vLLM v0.22.0 docs, should we accept this gloss?",
            "For version 0.22.0, is this an accepted description?",
            "Would a careful reader of the v0.22.0 docs call this correct?",
        ]
        false_leads = [
            "Given vLLM v0.22.0 docs, should we accept this gloss?",
            "For version 0.22.0, is this an accepted description?",
            "Would a careful reader of the v0.22.0 docs call this correct?",
        ]
        for idx in range(n_paraphrases):
            probes.append(
                _probe(
                    claim=claim,
                    probe_form="meaning_true",
                    paraphrase_idx=idx,
                    question=(
                        f"{true_leads[idx]}\n`{flag}`: {meaning}\n"
                        "Reply yes, no, or unknown."
                    ),
                    gold={"boolean": "yes"},
                )
            )
            donor = wrong[idx % len(wrong)]
            probe = _probe(
                claim=claim,
                probe_form="meaning_false",
                paraphrase_idx=idx,
                question=(
                    f"{false_leads[idx]}\n`{flag}`: {donor['object']}\n"
                    "Reply yes, no, or unknown."
                ),
                gold={
                    "boolean": "no_or_unknown",
                    "abstain_if_unknown": True,
                },
            )
            probe["contrast_claim_id"] = donor["claim_id"]
            probes.append(probe)

    probes.sort(key=lambda p: p["id"])
    return probes


def overlap_exact(train_path: Path, probes: list[dict]) -> int:
    train_qs = set()
    for line in train_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        msgs = row.get("messages") or []
        if msgs:
            train_qs.add(msgs[0]["content"].strip())
    return sum(1 for p in probes if p["question"].strip() in train_qs)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    claims = load_jsonl(
        root / "data/experiments/e1_vllm/profile/dense_v1/claims_verified.jsonl"
    )
    probes = build_paraphrase_probes(claims)
    out = root / "data/experiments/e1_vllm/profile/dense_v1_paraphrase"
    out.mkdir(parents=True, exist_ok=True)
    write_probes(probes, out / "probes.jsonl")
    # reuse verified claims file by copy reference in manifest
    (out / "claims_verified.jsonl").write_text(
        (root / "data/experiments/e1_vllm/profile/dense_v1/claims_verified.jsonl").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    train = root / "data/experiments/e1_vllm/train_v0.22.0_v5_full.jsonl"
    n_overlap = overlap_exact(train, probes) if train.exists() else -1
    # also check messages-only train
    train2 = root / "data/experiments/e1_vllm/train_v0.22.0_v5.jsonl"
    n_overlap2 = overlap_exact(train2, probes) if train2.exists() else -1
    manifest = {
        "protocol": PARA_PROTOCOL,
        "parent_protocol": PROTOCOL,
        "source_revision": "v0.22.0",
        "n_claims": len(claims),
        "n_probes": len(probes),
        "exact_question_overlap_vs_v5_full": n_overlap,
        "exact_question_overlap_vs_v5_train": n_overlap2,
        "probes_sha256": hashlib.sha256((out / "probes.jsonl").read_bytes()).hexdigest(),
        "status": "frozen",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
