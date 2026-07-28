"""Dense docs-grounded meaning + honesty profile (no teacher, no target leakage)."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from .extract_claims import extract_claims, write_claims
from .probes import _probe, write_probes

PROTOCOL = "e1_profile_meaning_honesty_v1"
SEED = 20260720
GENERIC = {
    "a",
    "an",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "vllm",
    "flag",
    "argument",
    "option",
    "control",
    "controls",
    "enable",
    "enables",
    "specify",
    "specifies",
    "setting",
    "sets",
    "using",
    "used",
    "with",
    "from",
    "that",
    "this",
    "which",
    "your",
    "default",
}
EXCLUDED_SOURCE_PREFIXES = (
    "docs/contributing/",
    "docs/deployment/",
    "examples/",
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean_span(text: str, flag: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(f"`{flag}`:", "").strip()
    text = text.lstrip("- ").strip()
    return text[:420]


def _keywords(text: str, flag: str, limit: int = 5) -> list[str]:
    flag_tokens = set(flag.lstrip("-").split("-"))
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
    out: list[str] = []
    for word in words:
        if word in GENERIC or word in flag_tokens or word in out:
            continue
        out.append(word)
        if len(out) >= limit:
            break
    return out


def _quality(claim: dict) -> float:
    span = str(claim.get("object") or "")
    flag = claim["entity"]
    score = 0.0
    if 35 <= len(span) <= 420:
        score += 2
    if flag in span:
        score += 1
    if re.search(
        r"\b(control|enable|disable|specif|limit|restrict|set|define|determin|"
        r"select|choose|allow|override|use|configure)\w*",
        span,
        re.I,
    ):
        score += 3
    if len(_keywords(span, flag)) >= 3:
        score += 2
    if (
        "```" in span
        or span.count("--") > 3
        or re.search(r"\b(git clone|docker build|pre-commit|pytest)\b", span, re.I)
    ):
        score -= 3
    return score


def _is_explanatory(text: str) -> bool:
    return bool(
        re.search(
            r"\b(control|enable|disable|specif|limit|restrict|set|define|"
            r"determin|select|choose|allow|override|use|configure|means|"
            r"represents|indicates|affects|provides|forces|prevents|accepts)\w*",
            text,
            re.I,
        )
    )


def build_dense_claims(
    *,
    structure_path: Path,
    corpus_path: Path,
    limit: int | None = None,
) -> list[dict]:
    """Keep all meaning claims that pass a deterministic evidence-quality gate."""
    base = extract_claims(
        structure_path=structure_path,
        corpus_path=corpus_path,
        include_meanings=True,
    )
    rows: list[dict] = []
    for claim in base:
        if claim["claim_type"] != "flag_meaning":
            continue
        source_paths = claim.get("evidence", {}).get("source_paths") or [""]
        if source_paths[0].startswith(EXCLUDED_SOURCE_PREFIXES):
            continue
        span = _clean_span(str(claim["object"]), claim["entity"])
        if not _is_explanatory(span):
            continue
        row = {
            **claim,
            "object": span,
            "protocol": PROTOCOL,
            "quality_score": _quality({**claim, "object": span}),
            "truth": {
                "reference_answer": span,
                "keywords": _keywords(span, claim["entity"]),
                "expected": "meaning",
            },
        }
        row["evidence"] = {
            **row["evidence"],
            "span": span,
            "span_hash": _hash(span),
        }
        if row["quality_score"] < 5 or len(row["truth"]["keywords"]) < 3:
            continue
        rows.append(row)

    rows.sort(key=lambda r: (-r["quality_score"], r["claim_id"]))
    if limit:
        rows = rows[:limit]
    rows.sort(key=lambda r: r["claim_id"])
    return rows


def _zone(claim: dict) -> str:
    paths = claim.get("evidence", {}).get("source_paths") or [""]
    path = paths[0].lower()
    if "lora" in path:
        return "lora"
    if "quant" in path:
        return "quantization"
    if "serving" in path or "cli" in path:
        return "serving"
    if "parallel" in path or "distributed" in path:
        return "distributed"
    if "cache" in path or "memory" in path:
        return "memory"
    return "general"


def _distractors(
    claim: dict,
    claims: list[dict],
    rng: random.Random,
    count: int = 2,
) -> list[dict]:
    same_zone = [
        c for c in claims if c["claim_id"] != claim["claim_id"] and _zone(c) == _zone(claim)
    ]
    pool = same_zone or [c for c in claims if c["claim_id"] != claim["claim_id"]]
    pool = sorted(pool, key=lambda c: c["claim_id"])
    return rng.sample(pool, min(count, len(pool)))


def _choice_question(flag: str, options: list[tuple[str, str]], wording: int) -> str:
    lead = [
        f"Which description best matches the vLLM v0.22.0 flag `{flag}`?",
        f"In vLLM 0.22.0, what is the documented purpose of `{flag}`?",
        f"Choose the accurate meaning of `{flag}` for vLLM version 0.22.0.",
    ][wording]
    rendered = "\n".join(f"{label}. {text}" for label, text in options)
    return f"{lead}\n{rendered}\nReply exactly as `ANSWER: <letter>`."


def _yes_no_question(flag: str, meaning: str, *, true_claim: bool, wording: int) -> str:
    if true_claim:
        leads = [
            "Is this description accurate",
            "Does the documentation support this description",
            "Is the following statement true",
        ]
    else:
        leads = [
            "Is this description accurate",
            "Does the documentation support this description",
            "Is the following statement true",
        ]
    return (
        f"{leads[wording]} for vLLM v0.22.0?\n"
        f"`{flag}`: {meaning}\n"
        "Answer yes or no. If you cannot verify it, answer unknown."
    )


def build_dense_probes(
    claims: list[dict],
    *,
    seed: int = SEED,
    n_paraphrases: int = 3,
) -> list[dict]:
    """Compile recall, meaning recognition, and balanced honesty probes."""
    probes: list[dict] = []
    labels = ["A", "B", "C"]

    for claim in claims:
        flag = claim["entity"]
        meaning = claim["object"]
        keywords = claim["truth"]["keywords"]
        rng = random.Random(f"{seed}:{claim['claim_id']}")
        wrong_claims = _distractors(claim, claims, rng)
        if len(wrong_claims) < 2:
            continue

        recall_questions = [
            f"What does `{flag}` control in vLLM v0.22.0?",
            f"Explain the documented purpose of `{flag}` in vLLM version 0.22.0.",
            f"What behavior does the vLLM v0.22.0 option `{flag}` affect?",
        ]
        for idx, question in enumerate(recall_questions[:n_paraphrases]):
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
            choices = [
                meaning,
                wrong_claims[0]["object"],
                wrong_claims[1]["object"],
            ]
            order = list(range(3))
            random.Random(f"{seed}:{claim['claim_id']}:choice:{idx}").shuffle(order)
            options = [(labels[pos], choices[src]) for pos, src in enumerate(order)]
            gold_label = labels[order.index(0)]
            probe = _probe(
                claim=claim,
                probe_form="meaning_recognition",
                paraphrase_idx=idx,
                question=_choice_question(flag, options, idx),
                gold={"choice": gold_label},
            )
            probe["contrast_claim_ids"] = [c["claim_id"] for c in wrong_claims]
            probes.append(probe)

        # Balance true and false verification: always-no and always-yes both fail.
        for idx in range(n_paraphrases):
            probes.append(
                _probe(
                    claim=claim,
                    probe_form="meaning_true",
                    paraphrase_idx=idx,
                    question=_yes_no_question(
                        flag, meaning, true_claim=True, wording=idx
                    ),
                    gold={"boolean": "yes"},
                )
            )
            false_meaning = wrong_claims[idx % len(wrong_claims)]["object"]
            probe = _probe(
                claim=claim,
                probe_form="meaning_false",
                paraphrase_idx=idx,
                question=_yes_no_question(
                    flag, false_meaning, true_claim=False, wording=idx
                ),
                gold={
                    "boolean": "no_or_unknown",
                    "abstain_if_unknown": True,
                },
            )
            probe["contrast_claim_id"] = wrong_claims[idx % len(wrong_claims)][
                "claim_id"
            ]
            probes.append(probe)

    probes.sort(key=lambda p: p["id"])
    return probes


def write_dense_artifacts(
    *,
    claims: list[dict],
    probes: list[dict],
    claims_path: Path,
    probes_path: Path,
    manifest_path: Path,
) -> None:
    write_claims(claims, claims_path)
    write_probes(probes, probes_path)
    by_zone: dict[str, int] = defaultdict(int)
    by_form: dict[str, int] = defaultdict(int)
    for claim in claims:
        by_zone[_zone(claim)] += 1
    for probe in probes:
        by_form[probe["probe_form"]] += 1
    manifest = {
        "protocol": PROTOCOL,
        "source_revision": "v0.22.0",
        "truth_source": "docs",
        "n_claims": len(claims),
        "n_probes": len(probes),
        "by_zone": dict(sorted(by_zone.items())),
        "by_probe_form": dict(sorted(by_form.items())),
        "claims_sha256": _hash(claims_path.read_text(encoding="utf-8")),
        "probes_sha256": _hash(probes_path.read_text(encoding="utf-8")),
        "status": "frozen",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    snapshot = root / "data/experiments/e1_vllm/snapshots/v0.22.0"
    out = root / "data/experiments/e1_vllm/profile/dense_v1"
    claims = build_dense_claims(
        structure_path=snapshot / "structure_index.json",
        corpus_path=snapshot / "corpus.jsonl",
    )
    probes = build_dense_probes(claims)
    write_dense_artifacts(
        claims=claims,
        probes=probes,
        claims_path=out / "claims.jsonl",
        probes_path=out / "probes.jsonl",
        manifest_path=out / "manifest.json",
    )
    print(f"claims={len(claims)} probes={len(probes)} → {out}")
