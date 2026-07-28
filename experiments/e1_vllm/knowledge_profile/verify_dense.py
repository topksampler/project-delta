"""Audit dense docs claims with code existence; preserve exact evaluated probes."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parent.parent
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

from knowledge_profile.truthsource.extract_code import extract_full_flags  # noqa: E402


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def verify(*, snapshot: Path, source_dir: Path) -> dict:
    claims_path = source_dir / "claims.jsonl"
    probes_path = source_dir / "probes.jsonl"
    claims = load_jsonl(claims_path)
    probes = load_jsonl(probes_path)
    code_flags = {row["name"] for row in extract_full_flags(snapshot)}
    kept = [claim for claim in claims if claim["entity"] in code_flags]
    kept_ids = {claim["claim_id"] for claim in kept}
    kept_probes = [probe for probe in probes if probe["claim_id"] in kept_ids]
    dropped = [claim for claim in claims if claim["claim_id"] not in kept_ids]

    write_jsonl(source_dir / "claims_verified.jsonl", kept)
    write_jsonl(source_dir / "probes_verified.jsonl", kept_probes)
    manifest = {
        "schema": "delta.profile_claim_audit.v1",
        "parent_claims_sha256": _hash(claims_path),
        "parent_probes_sha256": _hash(probes_path),
        "verifier": "vllm/**/*.py add_argument(--...)",
        "n_parent_claims": len(claims),
        "n_verified_claims": len(kept),
        "n_dropped_claims": len(dropped),
        "n_verified_probes": len(kept_probes),
        "dropped_entities": [claim["entity"] for claim in dropped],
        "claims_verified_sha256": _hash(source_dir / "claims_verified.jsonl"),
        "probes_verified_sha256": _hash(source_dir / "probes_verified.jsonl"),
    }
    (source_dir / "verification.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    result = verify(
        snapshot=root / "data/experiments/e1_vllm/snapshots/v0.22.0",
        source_dir=root / "data/experiments/e1_vllm/profile/dense_v1",
    )
    print(json.dumps(result, indent=2))
