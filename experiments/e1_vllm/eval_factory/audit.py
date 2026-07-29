"""Build a deterministic, delta-oversampled human audit packet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

AUDIT_SEED = "e1_eval_factory_v1:audit:20260728"


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _order(row: dict) -> str:
    return hashlib.sha256(f"{AUDIT_SEED}:{row['claim_id']}".encode()).hexdigest()


def build_audit(artifact_dir: Path, *, size: int = 50) -> dict:
    artifact_dir = artifact_dir.resolve()
    claims = _load_jsonl(artifact_dir / "claims_verified.jsonl")
    delta = sorted(
        [row for row in claims if row["status"] != "stable"],
        key=_order,
    )
    stable = sorted(
        [row for row in claims if row["status"] == "stable"],
        key=_order,
    )
    selected = (delta + stable)[:size]
    packet = [
        {
            "audit_id": f"audit:{index:03d}",
            "claim_id": row["claim_id"],
            "entity": row["entity"],
            "status": row["status"],
            "value_before": row["value_before"],
            "value_after": row["value_after"],
            "evidence_before": row["evidence_before"],
            "evidence_after": row["evidence_after"],
            "truth_correct": None,
            "version_label_correct": None,
            "answerable": None,
            "notes": "",
        }
        for index, row in enumerate(selected, start=1)
    ]
    path = artifact_dir / "audit_50.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in packet),
        encoding="utf-8",
    )
    return {
        "audit_path": str(path),
        "n": len(packet),
        "n_delta": sum(row["status"] != "stable" for row in selected),
        "n_stable": sum(row["status"] == "stable" for row in selected),
        "seed": AUDIT_SEED,
    }
