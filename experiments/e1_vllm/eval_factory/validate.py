"""Mandatory integrity gates for an eval-factory artifact."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tokens(text: str) -> set[str]:
    return set(WORD_RE.findall(text.lower()))


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def validate(artifact_dir: Path) -> dict:
    artifact_dir = artifact_dir.resolve()
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claims = _load_jsonl(artifact_dir / "claims_verified.jsonl")
    train = _load_jsonl(artifact_dir / "probes_train.jsonl")
    dev = _load_jsonl(artifact_dir / "probes_dev.jsonl")
    generated_eval_path = artifact_dir / "probes_eval.jsonl"
    seed_eval = _load_jsonl(artifact_dir / "probes_eval_seed.jsonl")
    eval_source = (
        generated_eval_path
        if generated_eval_path.is_file()
        else artifact_dir / "probes_eval_seed.jsonl"
    )
    eval_rows = _load_jsonl(eval_source)
    generated_seed_ids = {row.get("seed_id") for row in eval_rows}
    expected_seed_ids = {row["id"] for row in seed_eval}
    generated_coverage = (
        len(generated_seed_ids & expected_seed_ids) / len(expected_seed_ids)
        if expected_seed_ids
        else 0.0
    )

    hashes_match = all(
        (artifact_dir / name).is_file()
        and _sha256(artifact_dir / name) == expected
        for name, expected in manifest.get("artifact_sha256", {}).items()
    )
    claim_ids = [row["claim_id"] for row in claims]
    claim_ids_unique = len(claim_ids) == len(set(claim_ids))
    claim_split = {row["claim_id"]: row["split"] for row in claims}
    probes_match_claim_split = all(
        claim_split.get(row["claim_id"]) == row["split"]
        for row in train + dev + eval_rows
    )

    exact_train = {row["question"].strip().lower() for row in train}
    exact_overlap = sorted(
        row["id"]
        for row in eval_rows
        if row["question"].strip().lower() in exact_train
    )
    near_overlap: list[dict] = []
    max_seen = 0.0
    for eval_row in eval_rows:
        for train_row in train:
            score = _jaccard(eval_row["question"], train_row["question"])
            max_seen = max(max_seen, score)
            if score >= 0.85:
                near_overlap.append(
                    {
                        "eval_id": eval_row["id"],
                        "train_id": train_row["id"],
                        "jaccard": round(score, 4),
                    }
                )

    by_status: dict[str, int] = {}
    by_family_status: dict[str, dict[str, int]] = {}
    for row in claims:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        family = row["entity_type"]
        family_status = by_family_status.setdefault(family, {})
        family_status[row["status"]] = family_status.get(row["status"], 0) + 1
    changed_population = sum(
        by_status.get(status, 0) for status in ("added", "removed", "changed")
    )

    audit_path = artifact_dir / "audit_50.jsonl"
    audit_rows = _load_jsonl(audit_path)
    audited = [
        row
        for row in audit_rows
        if row.get("truth_correct") is not None
        and row.get("version_label_correct") is not None
        and row.get("answerable") is not None
    ]
    audit_agree = (
        sum(
            bool(row["truth_correct"])
            and bool(row["version_label_correct"])
            and bool(row["answerable"])
            for row in audited
        )
        / len(audited)
        if audited
        else None
    )

    gates = {
        "minimum_100_executable_claims": {
            "pass": len(claims) >= 100,
            "observed": len(claims),
        },
        "minimum_40_changed_claims": {
            "pass": changed_population >= 40,
            "observed": changed_population,
        },
        "verifier_rerun_equal": {
            "pass": bool(manifest.get("verifier_rerun_equal")),
            "observed": manifest.get("verifier_rerun_equal"),
        },
        "artifact_hashes_match": {
            "pass": hashes_match,
            "observed": hashes_match,
        },
        "claim_ids_unique": {
            "pass": claim_ids_unique,
            "observed": claim_ids_unique,
        },
        "probe_claim_splits_isolated": {
            "pass": probes_match_claim_split,
            "observed": probes_match_claim_split,
        },
        "no_exact_train_eval_overlap": {
            "pass": not exact_overlap,
            "observed": exact_overlap,
        },
        "no_jaccard_train_eval_overlap_gte_0_85": {
            "pass": not near_overlap,
            "observed": {
                "n_overlaps": len(near_overlap),
                "max_jaccard": round(max_seen, 4),
                "examples": near_overlap[:20],
            },
        },
        "human_audit_50_at_least_90pct": {
            "pass": len(audited) >= 50 and audit_agree is not None and audit_agree >= 0.9,
            "observed": {
                "n_audited": len(audited),
                "agreement": round(audit_agree, 4) if audit_agree is not None else None,
            },
        },
        "independent_generated_surfaces_present": {
            "pass": any(
                row.get("generator", {}).get("id")
                != "deterministic_claim_probe_v1"
                for row in eval_rows
            ),
            "observed": sorted(
                {
                    row.get("generator", {}).get("id", "unknown")
                    for row in eval_rows
                }
            ),
        },
        "generated_surface_coverage_at_least_90pct": {
            "pass": generated_eval_path.is_file() and generated_coverage >= 0.9,
            "observed": {
                "coverage": round(generated_coverage, 4),
                "n_expected": len(expected_seed_ids),
                "n_generated": len(generated_seed_ids & expected_seed_ids),
            },
        },
    }
    result = {
        "schema": "delta.eval_factory.validation.v1",
        "artifact_dir": str(artifact_dir),
        "freeze_ready": all(gate["pass"] for gate in gates.values()),
        "counts": {
            "claims": len(claims),
            "train_probes": len(train),
            "dev_probes": len(dev),
            "eval_probes": len(eval_rows),
            "eval_source": eval_source.name,
            "by_status": dict(sorted(by_status.items())),
            "by_family_status": {
                family: dict(sorted(statuses.items()))
                for family, statuses in sorted(by_family_status.items())
            },
        },
        "gates": gates,
    }
    (artifact_dir / "validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def seal(artifact_dir: Path) -> dict:
    """Freeze a passing environment by hashing every derived artifact."""
    artifact_dir = artifact_dir.resolve()
    result = validate(artifact_dir)
    if not result["freeze_ready"]:
        failed = [name for name, gate in result["gates"].items() if not gate["pass"]]
        raise RuntimeError(f"cannot seal; failed gates: {', '.join(failed)}")
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["freeze"]["ready"] = True
    manifest["freeze"]["validation_schema"] = result["schema"]
    manifest["artifact_sha256"] = {
        path.name: _sha256(path)
        for path in sorted(artifact_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "artifact_dir": str(artifact_dir),
        "sealed": True,
        "n_hashed_artifacts": len(manifest["artifact_sha256"]),
    }
