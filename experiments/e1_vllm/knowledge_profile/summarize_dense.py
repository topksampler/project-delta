"""Reproducible summary for dense meaning+honesty profile."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _form_accuracy(rows: list[dict], form: str) -> float:
    selected = [row for row in rows if row["probe_form"] == form]
    return sum(row["failure_mode"] == "correct" for row in selected) / len(selected)


def _cluster_ci(
    rows: list[dict], form: str, *, seed: int = 20260720, samples: int = 2000
) -> list[float]:
    by_claim: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["probe_form"] == form:
            by_claim[row["claim_id"]].append(row)
    claim_ids = sorted(by_claim)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        chosen = [rng.choice(claim_ids) for _ in claim_ids]
        sampled = [row for claim_id in chosen for row in by_claim[claim_id]]
        values.append(
            sum(row["failure_mode"] == "correct" for row in sampled) / len(sampled)
        )
    values.sort()
    return [round(values[int(samples * 0.025)], 4), round(values[int(samples * 0.975)], 4)]


def summarize(
    *, profile_path: Path, samples_path: Path, claims_path: Path, cost_path: Path
) -> dict:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    claims = load_jsonl(claims_path)
    claim_ids = {claim["claim_id"] for claim in claims}
    rows = [
        row for row in load_jsonl(samples_path) if row.get("claim_id") in claim_ids
    ]
    forms = (
        "free_recall",
        "meaning_recognition",
        "meaning_true",
        "meaning_false",
    )
    cost = json.loads(cost_path.read_text(encoding="utf-8"))
    return {
        "schema": "delta.topic_knowledge_profile_summary.v1",
        "protocol": profile["protocol_version"],
        "run_id": profile["model_state"]["run_id"],
        "model": profile["model_state"]["name"],
        "source_revision": profile["source_revision"],
        "n_verified_claims": len(claims),
        "n_verified_probes": len(rows),
        "form_accuracy": {
            form: {
                "accuracy": round(_form_accuracy(rows, form), 4),
                "cluster_bootstrap_95_ci": _cluster_ci(rows, form),
            }
            for form in forms
        },
        "recognition_random_baseline": 0.3333,
        "meaning_status_counts": profile["aggregate"]["meaning_status_counts"],
        "honesty_status_counts": profile["aggregate"]["honesty_status_counts"],
        "meaning_known_rate": profile["aggregate"]["meaning_known_rate"],
        "honesty_failure_rate": profile["aggregate"]["honesty_failure_rate"],
        "estimated_cost_usd": cost["estimated_cost_usd"],
        "billable_seconds": cost["billable_seconds"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--claims", required=True)
    parser.add_argument("--cost", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = summarize(
        profile_path=Path(args.profile),
        samples_path=Path(args.samples),
        claims_path=Path(args.claims),
        cost_path=Path(args.cost),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
