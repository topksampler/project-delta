"""Assemble final eval surfaces without relaxing semantic gates."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .generate_surfaces import _clean, _validate_surface

VERSION_RE = re.compile(r"\b\d+\.\d+\.\d+\b")


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _ensure_boolean_answer_hint(question: str, gold: dict | None) -> str:
    gold = gold or {}
    if not gold.get("boolean"):
        return question
    q = question.rstrip()
    if re.search(r"\byes\s+or\s+no\b", q, flags=re.IGNORECASE):
        return q
    return f"{q} Answer yes or no."


def _neutral_delta(seed: dict) -> dict:
    versions = VERSION_RE.findall(seed["question"])
    if len(versions) != 2 and seed.get("source_before_version") and seed.get(
        "source_after_version"
    ):
        versions = [seed["source_before_version"], seed["source_after_version"]]
    if len(versions) != 2:
        raise ValueError(f"delta seed lacks two versions: {seed['id']}")
    entity = seed["display_entity"]
    templates = (
        (
            "Was `{entity}` the same in vLLM {before} and {after}, or did it "
            "differ? Summarize the status."
        ),
        (
            "Compare `{entity}` in vLLM {before} with {after}: was it the same, "
            "and if not, what differed?"
        ),
        (
            "Did `{entity}` stay the same between vLLM {before} and {after}? "
            "State any difference."
        ),
    )
    index = int(hashlib.sha256(seed["id"].encode()).hexdigest()[:8], 16) % len(
        templates
    )
    question = templates[index].format(
        entity=entity,
        before=versions[0],
        after=versions[1],
    )
    return {
        **seed,
        "id": f"{seed['id']}:neutral-template",
        "seed_id": seed["id"],
        "seed_question": seed["question"],
        "question": question,
        "generator": {
            "id": "neutral_delta_template_v1",
            "template_index": index,
        },
    }


def finalize(
    *,
    artifact_dir: Path,
    teacher_generated: Path,
    teacher_rejected: Path | None,
    teacher_metrics: Path | None,
) -> dict:
    artifact_dir = artifact_dir.resolve()
    seeds = _load_jsonl(artifact_dir / "probes_eval_seed.jsonl")
    teacher = _load_jsonl(teacher_generated.resolve())
    by_seed_id = {row["seed_id"]: row for row in teacher}
    metrics = (
        json.loads(teacher_metrics.resolve().read_text(encoding="utf-8"))
        if teacher_metrics
        else {}
    )
    recovered = 0
    if teacher_rejected:
        seed_by_id = {row["id"]: row for row in seeds}
        max_jaccard = float(metrics.get("max_seed_jaccard", 0.7))
        for rejected in _load_jsonl(teacher_rejected.resolve()):
            seed = seed_by_id.get(rejected["seed_id"])
            if seed is None:
                continue
            question = _clean(rejected["candidate_question"])
            if _validate_surface(seed, question, max_jaccard) is not None:
                continue
            by_seed_id[seed["id"]] = {
                **seed,
                "id": f"{seed['id']}:teacher-recovered",
                "seed_id": seed["id"],
                "seed_question": seed["question"],
                "question": question,
                "generator": {
                    "id": metrics.get("generator_id", "teacher_unknown"),
                    "model_id": metrics.get("model_id"),
                    "decoding": "greedy",
                    "prompt_sha256": rejected.get("prompt_sha256"),
                    "postprocess_id": "surface_cleanup_v2",
                },
            }
            recovered += 1
    final: list[dict] = []
    unresolved: list[dict] = []
    fallback = 0
    for seed in seeds:
        generated = by_seed_id.get(seed["id"])
        if generated is not None:
            generated = {
                **generated,
                "question": _ensure_boolean_answer_hint(
                    generated["question"], generated.get("gold") or seed.get("gold")
                ),
            }
            final.append(generated)
        elif seed["probe_form"] == "version_delta":
            row = _neutral_delta(seed)
            row["question"] = _ensure_boolean_answer_hint(row["question"], row.get("gold"))
            final.append(row)
            fallback += 1
        else:
            unresolved.append(
                {
                    "seed_id": seed["id"],
                    "reason": "teacher_rejected_non_delta",
                }
            )
    final.sort(key=lambda row: row["seed_id"])
    _write_jsonl(artifact_dir / "probes_eval.jsonl", final)
    _write_jsonl(artifact_dir / "surface_unresolved.jsonl", unresolved)
    if teacher_metrics:
        (artifact_dir / "surface_generation.json").write_bytes(
            teacher_metrics.resolve().read_bytes()
        )
    summary = {
        "schema": "delta.eval_factory.surface_finalize.v1",
        "n_seed": len(seeds),
        "n_teacher": len(teacher),
        "n_teacher_recovered": recovered,
        "n_neutral_delta_fallback": fallback,
        "n_final": len(final),
        "n_unresolved": len(unresolved),
        "coverage": len(final) / max(1, len(seeds)),
        "teacher_source": str(teacher_generated),
    }
    (artifact_dir / "surface_finalize.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
