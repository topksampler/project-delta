"""Claim bank → probe bundle (paraphrases × behavioral forms)."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:120]


def _fake_flag(name: str, rng: random.Random) -> str:
    """Nearby-looking but nonexistent CLI flag."""
    base = name.lstrip("-")
    mutations = [
        f"--{base}-ex",
        f"--{base.replace('parallel', 'fractal', 1)}" if "parallel" in base else f"--pseudo-{base}",
        f"--{base}-v2x",
        f"--no-{base}",
    ]
    # ensure we don't accidentally invent a real one later; caller filters
    return rng.choice(mutations)


def _fake_env(name: str, rng: random.Random) -> str:
    return rng.choice(
        [
            f"{name}_EXPERIMENTAL",
            name.replace("VLLM_", "VLLMX_", 1),
            f"{name}_SHADOW",
        ]
    )


def _fake_path(path: str, rng: random.Random) -> str:
    stem = Path(path).stem
    parent = str(Path(path).parent)
    return rng.choice(
        [
            f"{parent}/{stem}_v9.md",
            f"{parent}/legacy_{stem}.md",
            f"docs/missing/{stem}.md",
        ]
    )


def _hard_negative_flags(entity: str, universe: list[str], rng: random.Random, k: int = 2) -> list[str]:
    others = [f for f in universe if f != entity]
    if len(others) <= k:
        return others
    # prefer lexicographically nearby
    others_sorted = sorted(others, key=lambda x: abs(ord(x[2:3] or "a") - ord((entity[2:3] or "a"))))
    nearby = others_sorted[:20]
    return rng.sample(nearby, min(k, len(nearby)))


def _recall_templates_flag(entity: str) -> list[str]:
    return [
        f"Does vLLM v0.22.0 documentation include the CLI flag `{entity}`? Answer yes or no.",
        f"Is `{entity}` a documented vLLM command-line argument in version 0.22.0?",
        f"According to vLLM 0.22.0 docs, can users pass `{entity}` to the CLI?",
    ]


def _recall_templates_env(entity: str) -> list[str]:
    return [
        f"Does vLLM v0.22.0 documentation mention the environment variable `{entity}`?",
        f"Is `{entity}` a documented vLLM environment variable in 0.22.0?",
        f"Can `{entity}` be set to configure vLLM according to the 0.22.0 docs?",
    ]


def _recall_templates_path(entity: str) -> list[str]:
    return [
        f"Does the vLLM v0.22.0 documentation tree include the file `{entity}`?",
        f"Is `{entity}` present in the vLLM 0.22.0 docs snapshot?",
        f"In vLLM 0.22.0, does path `{entity}` exist under the documentation?",
    ]


def _meaning_templates(entity: str) -> list[str]:
    return [
        f"What does `{entity}` control in vLLM?",
        f"In one or two sentences, explain the purpose of the vLLM flag `{entity}`.",
        f"What behavior does setting `{entity}` change when running vLLM?",
    ]


def _recognition_templates(entity: str, negatives: list[str]) -> list[str]:
    options = [entity] + negatives
    # fixed order per paraphrase via sorted then rotate — caller shuffles once
    listed = ", ".join(f"`{o}`" for o in options)
    return [
        f"Which of these is a real documented vLLM CLI flag: {listed}? Reply with the flag name.",
        f"Pick the authentic vLLM flag from: {listed}.",
        f"Only one of these flags is documented in vLLM: {listed}. Which one?",
    ]


def _negation_templates(fake: str, kind: str) -> list[str]:
    if kind == "flag":
        return [
            f"Does vLLM v0.22.0 documentation include the CLI flag `{fake}`? Answer yes or no.",
            f"Is `{fake}` a documented vLLM command-line argument in version 0.22.0?",
            f"According to vLLM 0.22.0 docs, can users pass `{fake}` to the CLI?",
        ]
    if kind == "env":
        return [
            f"Does vLLM v0.22.0 documentation mention the environment variable `{fake}`?",
            f"Is `{fake}` a documented vLLM environment variable in 0.22.0?",
            f"Can `{fake}` be set to configure vLLM according to the 0.22.0 docs?",
        ]
    return [
        f"Does the vLLM v0.22.0 documentation tree include the file `{fake}`?",
        f"Is `{fake}` present in the vLLM 0.22.0 docs snapshot?",
        f"In vLLM 0.22.0, does path `{fake}` exist under the documentation?",
    ]


def _probe(
    *,
    claim: dict,
    probe_form: str,
    paraphrase_idx: int,
    question: str,
    gold: dict,
) -> dict:
    cid = claim["claim_id"]
    pid = f"{_slug(cid)}:{probe_form}:{paraphrase_idx}"
    return {
        "id": pid,
        "claim_id": cid,
        "probe_form": probe_form,
        "paraphrase_idx": paraphrase_idx,
        "eval_class": "P",
        "type": claim["claim_type"],
        "question": question,
        "gold": gold,
        "requires_doc": claim["source_revision"].lstrip("v"),
        "source_hint": (claim.get("evidence") or {}).get("source_paths", [""])[0],
        "centrality": claim.get("centrality"),
        "zone": claim.get("zone"),
    }


def build_probes(
    claims: list[dict],
    *,
    seed: int = 20260718,
    n_paraphrases: int = 3,
    flag_universe: list[str] | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    if flag_universe is None:
        flag_universe = sorted(
            {c["entity"] for c in claims if c["claim_type"] in ("flag_exists", "flag_meaning")}
        )
    else:
        flag_universe = sorted(flag_universe)
    real_flags = set(flag_universe)
    real_envs = {c["entity"] for c in claims if c["claim_type"] == "env_exists"}
    real_paths = {c["entity"] for c in claims if c["claim_type"] == "path_exists"}

    probes: list[dict] = []
    for claim in claims:
        ctype = claim["claim_type"]
        entity = claim["entity"]

        if ctype == "flag_exists":
            templates = _recall_templates_flag(entity)
            for i, q in enumerate(templates[:n_paraphrases]):
                probes.append(
                    _probe(
                        claim=claim,
                        probe_form="recall",
                        paraphrase_idx=i,
                        question=q,
                        gold={
                            "must_contain_any": [[entity, "yes", "true"]],
                            "must_not_contain": ["unknown"],
                        },
                    )
                )
            negs = _hard_negative_flags(entity, flag_universe, rng, k=2)
            # shuffle option order per claim
            options = [entity] + negs
            rng.shuffle(options)
            listed = ", ".join(f"`{o}`" for o in options)
            recog = [
                f"Which of these is a real documented vLLM CLI flag: {listed}? Reply with the flag name.",
                f"Pick the authentic vLLM flag from: {listed}.",
                f"Only one of these flags is documented in vLLM: {listed}. Which one?",
            ]
            for i, q in enumerate(recog[:n_paraphrases]):
                probes.append(
                    _probe(
                        claim=claim,
                        probe_form="recognition",
                        paraphrase_idx=i,
                        question=q,
                        gold={"must_contain": [entity]},
                    )
                )
            fake = _fake_flag(entity, rng)
            while fake in real_flags:
                fake = _fake_flag(entity, rng)
            for i, q in enumerate(_negation_templates(fake, "flag")[:n_paraphrases]):
                probes.append(
                    _probe(
                        claim=claim,
                        probe_form="negation",
                        paraphrase_idx=i,
                        question=q,
                        gold={
                            "must_contain_any": [
                                ["no", "unknown", "not", "does not", "doesn't"]
                            ],
                            "abstain_if_unknown": True,
                        },
                    )
                )

        elif ctype == "env_exists":
            for i, q in enumerate(_recall_templates_env(entity)[:n_paraphrases]):
                probes.append(
                    _probe(
                        claim=claim,
                        probe_form="recall",
                        paraphrase_idx=i,
                        question=q,
                        gold={
                            "must_contain_any": [[entity, "yes", "true"]],
                            "must_not_contain": ["unknown"],
                        },
                    )
                )
            fake = _fake_env(entity, rng)
            while fake in real_envs:
                fake = _fake_env(entity, rng)
            for i, q in enumerate(_negation_templates(fake, "env")[:n_paraphrases]):
                probes.append(
                    _probe(
                        claim=claim,
                        probe_form="negation",
                        paraphrase_idx=i,
                        question=q,
                        gold={
                            "must_contain_any": [
                                ["no", "unknown", "not", "does not", "doesn't"]
                            ],
                            "abstain_if_unknown": True,
                        },
                    )
                )

        elif ctype == "path_exists":
            for i, q in enumerate(_recall_templates_path(entity)[:n_paraphrases]):
                probes.append(
                    _probe(
                        claim=claim,
                        probe_form="recall",
                        paraphrase_idx=i,
                        question=q,
                        gold={
                            "must_contain_any": [
                                [entity, Path(entity).name, "yes", "true"]
                            ],
                            "must_not_contain": ["unknown"],
                        },
                    )
                )
            fake = _fake_path(entity, rng)
            while fake in real_paths:
                fake = _fake_path(entity, rng)
            for i, q in enumerate(_negation_templates(fake, "path")[:n_paraphrases]):
                probes.append(
                    _probe(
                        claim=claim,
                        probe_form="negation",
                        paraphrase_idx=i,
                        question=q,
                        gold={
                            "must_contain_any": [
                                ["no", "unknown", "not", "does not", "doesn't"]
                            ],
                            "abstain_if_unknown": True,
                        },
                    )
                )

        elif ctype == "flag_meaning":
            needles = claim.get("truth", {}).get("must_contain") or [entity]
            for i, q in enumerate(_meaning_templates(entity)[:n_paraphrases]):
                probes.append(
                    _probe(
                        claim=claim,
                        probe_form="recall",
                        paraphrase_idx=i,
                        question=q,
                        gold={"must_contain": needles},
                    )
                )

    probes.sort(key=lambda p: p["id"])
    return probes


def write_probes(probes: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in probes:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def probe_bank_hash(probes: list[dict]) -> str:
    payload = "\n".join(p["id"] for p in probes)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
