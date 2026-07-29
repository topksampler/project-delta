from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


FEATURE_DELTA_SCHEMA = "delta.feature_delta.v1"
RESULT_SCHEMA = "delta.behavior_probe_result.v1"
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
DEVELOPMENT_ROLES = {"development_before", "development_after"}


class FeaturePromotionError(ValueError):
    """A FeatureDelta promotion record is malformed or unsupported by probes."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FeaturePromotionError(f"{field} must be a mapping")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise FeaturePromotionError(f"{field} must be a non-empty string")
    return value


def _unique_strings(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise FeaturePromotionError(f"{field} must be a list of strings")
    if len(value) != len(set(value)):
        raise FeaturePromotionError(f"{field} must not contain duplicates")
    return value


def load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FeaturePromotionError(f"cannot read YAML: {path}") from exc
    return _mapping(payload, str(path))


def validate_feature_delta(record: Mapping[str, Any]) -> None:
    if record.get("schema") != FEATURE_DELTA_SCHEMA:
        raise FeaturePromotionError("unsupported FeatureDelta schema")
    if not _string(record.get("feature_id"), "feature_id").startswith("feature:"):
        raise FeaturePromotionError("feature_id must use feature: prefix")
    if not _string(
        record.get("source_candidate_id"),
        "source_candidate_id",
    ).startswith("feature-candidate:"):
        raise FeaturePromotionError(
            "source_candidate_id must use feature-candidate: prefix"
        )
    if record.get("transition") != "development":
        raise FeaturePromotionError("FeatureDelta may only use development")
    if record.get("status") not in {"stable", "added", "removed", "changed"}:
        raise FeaturePromotionError("unsupported FeatureDelta status")
    evidence_ids = _unique_strings(record.get("evidence_ids"), "evidence_ids")
    if not any(item.startswith("fact:") for item in evidence_ids):
        raise FeaturePromotionError("FeatureDelta requires atomic-fact evidence")
    if not any(item.startswith("file:") for item in evidence_ids):
        raise FeaturePromotionError("FeatureDelta requires FileDelta evidence")
    if not any(item.startswith("pull-request:") for item in evidence_ids):
        raise FeaturePromotionError("FeatureDelta requires PR provenance")
    probe_ids = _unique_strings(
        record.get("behavior_probe_ids"),
        "behavior_probe_ids",
    )
    if not probe_ids or any(
        not probe_id.startswith("behavior-probe:") for probe_id in probe_ids
    ):
        raise FeaturePromotionError("FeatureDelta requires BehaviorProbe IDs")

    verification = _mapping(record.get("verification"), "verification")
    if verification.get("old_revision") != "unavailable":
        raise FeaturePromotionError(
            "added capability must be unavailable on the old revision"
        )
    if verification.get("new_revision") != "pass":
        raise FeaturePromotionError("new revision must pass verification")
    _string(verification.get("recipe"), "verification.recipe")
    probe_results = _mapping(
        verification.get("probe_results"),
        "verification.probe_results",
    )
    if set(probe_results) != set(probe_ids):
        raise FeaturePromotionError(
            "probe result declarations must match behavior_probe_ids"
        )
    scopes: set[str] = set()
    for probe_id, raw_result in probe_results.items():
        result = _mapping(raw_result, f"verification.probe_results.{probe_id}")
        digest = _string(result.get("sha256"), f"{probe_id}.sha256")
        if not HEX_SHA256.fullmatch(digest):
            raise FeaturePromotionError(f"invalid result SHA-256 for {probe_id}")
        scopes.add(_string(result.get("claim_scope"), f"{probe_id}.claim_scope"))
    if "complete-candidate" not in scopes:
        raise FeaturePromotionError(
            "promotion requires a complete-candidate BehaviorProbe"
        )
    if record.get("environment_status") != "development-only-unfrozen":
        raise FeaturePromotionError(
            "FeatureDelta must not claim an EvalEnvironment freeze"
        )


def load_probe_result(path: Path) -> tuple[Mapping[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise FeaturePromotionError(f"cannot read probe result: {path}") from exc
    result = _mapping(payload, str(path))
    if result.get("schema") != RESULT_SCHEMA:
        raise FeaturePromotionError(f"unsupported probe result schema: {path}")
    return result, hashlib.sha256(raw).hexdigest()


def audit_promotion(
    candidate: Mapping[str, Any],
    feature_delta: Mapping[str, Any],
    probe_results: Mapping[str, tuple[Mapping[str, Any], str]],
) -> dict[str, Any]:
    validate_feature_delta(feature_delta)
    candidate_id = _string(candidate.get("candidate_id"), "candidate_id")
    if feature_delta["source_candidate_id"] != candidate_id:
        raise FeaturePromotionError("FeatureDelta links the wrong candidate")
    candidate_evidence = set(
        _unique_strings(candidate.get("evidence_ids"), "candidate.evidence_ids")
    )
    feature_evidence = set(feature_delta["evidence_ids"])
    if feature_evidence != candidate_evidence:
        raise FeaturePromotionError(
            "FeatureDelta evidence must exactly preserve candidate evidence"
        )

    declared_results = feature_delta["verification"]["probe_results"]
    expected_probe_ids = set(feature_delta["behavior_probe_ids"])
    if set(probe_results) != expected_probe_ids:
        raise FeaturePromotionError("provided probe result set is incomplete")

    audits: list[dict[str, Any]] = []
    for probe_id in feature_delta["behavior_probe_ids"]:
        result, observed_sha256 = probe_results[probe_id]
        declared = declared_results[probe_id]
        sources = result.get("sources")
        if not isinstance(sources, Mapping) or set(sources) != DEVELOPMENT_ROLES:
            raise FeaturePromotionError(
                f"{probe_id} did not use exactly the development snapshots"
            )
        checks = {
            "result_id_matches": result.get("probe_id") == probe_id,
            "candidate_id_matches": result.get("candidate_id") == candidate_id,
            "status_passes": result.get("status") == "pass",
            "claim_scope_matches": (
                result.get("claim_scope") == declared["claim_scope"]
            ),
            "sha256_matches": observed_sha256 == declared["sha256"],
        }
        audits.append(
            {
                "probe_id": probe_id,
                "observed_sha256": observed_sha256,
                "checks": checks,
                "status": "pass" if all(checks.values()) else "fail",
            }
        )

    status = "pass" if all(audit["status"] == "pass" for audit in audits) else "fail"
    return {
        "schema": "delta.feature_promotion_audit.v1",
        "feature_id": feature_delta["feature_id"],
        "source_candidate_id": candidate_id,
        "status": status,
        "probe_audits": audits,
        "acceptance_accessed": False,
        "eval_environment_frozen": False,
    }


def _parse_result_argument(value: str) -> tuple[str, Path]:
    probe_id, separator, raw_path = value.partition("=")
    if not separator or not probe_id or not raw_path:
        raise argparse.ArgumentTypeError(
            "probe result must have form behavior-probe:id=/path/result.json"
        )
    return probe_id, Path(raw_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a delta_v2 FeatureCandidate promotion."
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--feature-delta", type=Path, required=True)
    parser.add_argument(
        "--probe-result",
        action="append",
        type=_parse_result_argument,
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    loaded_results: dict[str, tuple[Mapping[str, Any], str]] = {}
    for probe_id, path in args.probe_result:
        if probe_id in loaded_results:
            raise FeaturePromotionError(f"duplicate probe result: {probe_id}")
        loaded_results[probe_id] = load_probe_result(path)
    result = audit_promotion(
        load_yaml(args.candidate),
        load_yaml(args.feature_delta),
        loaded_results,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
