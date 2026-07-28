"""Aggregate probe samples → TopicKnowledgeProfile."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import SCHEMA_PROFILE
from .extract_claims import claim_bank_hash


def _status(claim_rows: list[dict]) -> str:
    """Classify one claim from its probe outcomes."""
    if not claim_rows:
        return "unknown"

    by_form: dict[str, list[dict]] = defaultdict(list)
    for r in claim_rows:
        by_form[r.get("probe_form", "?")].append(r)

    def form_acc(form: str) -> float | None:
        rows = by_form.get(form) or []
        if not rows:
            return None
        return sum(1 for r in rows if r.get("failure_mode") == "correct") / len(rows)

    recall = form_acc("recall")
    recog = form_acc("recognition")
    neg = form_acc("negation")

    # paraphrase consistency on recall
    recall_rows = by_form.get("recall") or []
    if len(recall_rows) >= 2:
        correct_flags = [r.get("failure_mode") == "correct" for r in recall_rows]
        consistency = sum(correct_flags) / len(correct_flags)
        agreement = 1.0 if len(set(correct_flags)) == 1 else (
            max(sum(correct_flags), len(correct_flags) - sum(correct_flags)) / len(correct_flags)
        )
    else:
        consistency = recall if recall is not None else 0.0
        agreement = 1.0

    # stable misconception: consistently wrong on recall, and negation also wrong
    # (affirms fake things) or high confidence wrong — we approximate via consistency
    if recall is not None and recall == 0.0 and agreement >= 0.99 and len(recall_rows) >= 2:
        if neg is not None and neg < 0.34:
            return "stable_misconception"
        return "unknown"

    if agreement < 0.67 and recall_rows:
        return "unstable"

    # known: strong recall (+ recognition if present) and decent negation
    ok_recall = recall is not None and recall >= 0.67
    ok_recog = recog is None or recog >= 0.67
    ok_neg = neg is None or neg >= 0.67
    if ok_recall and ok_recog and ok_neg:
        return "known"

    if (recall or 0) > 0 or (recog or 0) > 0:
        return "partial"

    if neg is not None and neg >= 0.67 and (recall or 0) == 0:
        # correctly rejects fakes but cannot affirm real — treat as unknown boundary
        return "unknown"

    return "unknown"


def aggregate_profile(
    *,
    claims: list[dict],
    samples: list[dict],
    model_name: str,
    run_id: str,
    protocol_version: str = "e1_profile_v0",
    source_revision: str = "v0.22.0",
) -> dict:
    by_claim_samples: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        cid = s.get("claim_id")
        if not cid and ":" in s.get("id", ""):
            # fallback: id = slug(claim):form:idx
            cid = s["id"].rsplit(":", 2)[0]
        if cid:
            by_claim_samples[cid].append(s)

    claim_results = []
    status_counts: dict[str, int] = defaultdict(int)
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_zone: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for claim in claims:
        cid = claim["claim_id"]
        rows = by_claim_samples.get(cid, [])
        status = _status(rows)
        status_counts[status] += 1
        by_type[claim["claim_type"]][status] += 1
        by_zone[claim.get("zone", "?")][status] += 1

        n = len(rows)
        n_correct = sum(1 for r in rows if r.get("failure_mode") == "correct")
        claim_results.append(
            {
                "claim_id": cid,
                "claim_type": claim["claim_type"],
                "entity": claim["entity"],
                "zone": claim.get("zone"),
                "centrality": claim.get("centrality"),
                "status": status,
                "n_probes": n,
                "n_correct": n_correct,
                "probe_accuracy": round(n_correct / max(1, n), 4),
                "by_probe_form": {
                    form: round(
                        sum(1 for r in rows if r.get("probe_form") == form and r.get("failure_mode") == "correct")
                        / max(1, sum(1 for r in rows if r.get("probe_form") == form)),
                        4,
                    )
                    for form in sorted({r.get("probe_form", "?") for r in rows})
                },
            }
        )

    n_claims = len(claims)
    return {
        "schema": SCHEMA_PROFILE,
        "protocol_version": protocol_version,
        "model_state": {
            "name": model_name,
            "adapter": None,
            "run_id": run_id,
        },
        "subject": "vllm",
        "source_revision": source_revision,
        "legacy_alias": "doc_0",
        "mode": "closed_book",
        "claim_bank_hash": claim_bank_hash(claims),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "aggregate": {
            "n_claims": n_claims,
            "n_probes": len(samples),
            "status_counts": dict(status_counts),
            "status_rates": {
                k: round(v / max(1, n_claims), 4) for k, v in sorted(status_counts.items())
            },
            "by_claim_type": {k: dict(v) for k, v in sorted(by_type.items())},
            "by_zone": {k: dict(v) for k, v in sorted(by_zone.items())},
            "known_rate": round(status_counts.get("known", 0) / max(1, n_claims), 4),
        },
        "claims": claim_results,
    }


def write_profile(profile: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
