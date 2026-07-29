from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA = "delta.atomic_fact_families.v1"
TRANSITION = "development"
SOURCE_SELECTION = "complete-snapshot-tree"
PARSER_IMPLEMENTATION = "cpython.ast"
PARSER_FEATURE_VERSION = "3.11"
CANONICAL_FORM = "ast-structural-json-v1"
ENCODING = "utf-8"
COMPARISON_STATUSES = {
    "absent_present": "added",
    "present_absent": "removed",
    "equal_equal": "stable",
    "unequal_unequal": "changed",
}
JOIN_KEY = ("family_id", "semantic_key")
EVIDENCE_FIELDS = (
    "snapshot_role",
    "path",
    "line_start",
    "line_end",
    "git_blob_sha1",
    "source_sha256",
    "extractor_version",
)
FAMILY_ID = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+\.v[1-9][0-9]*$")
REQUIRED_FAMILY_FIELDS = (
    "id",
    "description",
    "rationale",
    "paths",
    "selector",
    "semantic_key",
    "normalized_value",
    "inclusion_rules",
    "rejection_rules",
)


class FactFamilyManifestError(ValueError):
    """The atomic-fact family preregistration is incomplete or inconsistent."""


@dataclass(frozen=True)
class FactFamily:
    family_id: str
    description: str
    semantic_key: str
    normalized_value: tuple[str, ...]
    inclusion_rules: tuple[str, ...]
    rejection_rules: tuple[str, ...]


@dataclass(frozen=True)
class FactFamilyManifest:
    source_manifest: str
    transition: str
    parser_implementation: str
    parser_feature_version: str
    families: tuple[FactFamily, ...]


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FactFamilyManifestError(f"{field} must be a mapping")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FactFamilyManifestError(f"{field} must be a non-empty string")
    return value


def _string_list(
    value: Any,
    field: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise FactFamilyManifestError(f"{field} must be a list")
    if not allow_empty and not value:
        raise FactFamilyManifestError(f"{field} must not be empty")
    result = tuple(
        _string(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        raise FactFamilyManifestError(f"{field} must not contain duplicates")
    return result


def _require_fields(
    record: Mapping[str, Any],
    fields: Sequence[str],
    label: str,
) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise FactFamilyManifestError(f"{label} missing required fields: {missing}")


def _load_family(raw_family: Any, index: int) -> FactFamily:
    label = f"families[{index}]"
    family = _mapping(raw_family, label)
    _require_fields(family, REQUIRED_FAMILY_FIELDS, label)

    family_id = _string(family["id"], f"{label}.id")
    if not FAMILY_ID.fullmatch(family_id):
        raise FactFamilyManifestError(f"invalid family id: {family_id}")

    paths = _mapping(family["paths"], f"{label}.paths")
    _string_list(paths.get("include"), f"{label}.paths.include")
    _string_list(paths.get("exclude"), f"{label}.paths.exclude", allow_empty=True)

    selector = _mapping(family["selector"], f"{label}.selector")
    _string(selector.get("container"), f"{label}.selector.container")
    _string(selector.get("node"), f"{label}.selector.node")

    normalized_value = _string_list(
        family["normalized_value"],
        f"{label}.normalized_value",
    )
    inclusion_rules = _string_list(
        family["inclusion_rules"],
        f"{label}.inclusion_rules",
    )
    rejection_rules = _string_list(
        family["rejection_rules"],
        f"{label}.rejection_rules",
    )
    if not any("Reject" in rule for rule in rejection_rules):
        raise FactFamilyManifestError(
            f"{label}.rejection_rules must state explicit rejection behavior"
        )

    return FactFamily(
        family_id=family_id,
        description=_string(family["description"], f"{label}.description"),
        semantic_key=_string(family["semantic_key"], f"{label}.semantic_key"),
        normalized_value=normalized_value,
        inclusion_rules=inclusion_rules,
        rejection_rules=rejection_rules,
    )


def load_fact_family_manifest(path: Path) -> FactFamilyManifest:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FactFamilyManifestError(
            f"cannot read fact-family manifest: {path}"
        ) from exc

    root = _mapping(payload, "manifest")
    schema = _string(root.get("schema"), "schema")
    if schema != SCHEMA:
        raise FactFamilyManifestError(f"unsupported schema: {schema}")

    source_manifest = _string(root.get("source_manifest"), "source_manifest")
    if Path(source_manifest).is_absolute():
        raise FactFamilyManifestError("source_manifest must be a relative path")
    source_manifest_path = path.parent / source_manifest
    if not source_manifest_path.is_file():
        raise FactFamilyManifestError(
            f"source_manifest does not exist: {source_manifest_path}"
        )

    transition = _string(root.get("transition"), "transition")
    if transition != TRANSITION:
        raise FactFamilyManifestError(
            "fact-family preregistration may inspect only the development transition"
        )

    extraction = _mapping(root.get("extraction"), "extraction")
    if extraction.get("source_selection") != SOURCE_SELECTION:
        raise FactFamilyManifestError(
            "source_selection must be complete-snapshot-tree"
        )
    if extraction.get("extract_each_snapshot_independently") is not True:
        raise FactFamilyManifestError(
            "snapshots must be extracted independently before comparison"
        )
    if extraction.get("syntax_error_policy") != "fail-snapshot":
        raise FactFamilyManifestError("syntax errors must fail the snapshot")
    if extraction.get("duplicate_semantic_key_policy") != "fail-family-snapshot":
        raise FactFamilyManifestError("duplicate semantic keys must fail the family")
    if extraction.get("rejected_candidate_policy") != "record-reason-without-fact":
        raise FactFamilyManifestError(
            "rejected candidates must be recorded without emitting facts"
        )

    parser = _mapping(extraction.get("parser"), "extraction.parser")
    parser_implementation = _string(
        parser.get("implementation"),
        "extraction.parser.implementation",
    )
    if parser_implementation != PARSER_IMPLEMENTATION:
        raise FactFamilyManifestError(
            f"parser implementation must be {PARSER_IMPLEMENTATION}"
        )
    parser_feature_version = _string(
        parser.get("feature_version"),
        "extraction.parser.feature_version",
    )
    if parser_feature_version != PARSER_FEATURE_VERSION:
        raise FactFamilyManifestError(
            f"parser feature version must be {PARSER_FEATURE_VERSION}"
        )
    canonical_form = _string(
        parser.get("canonical_form"),
        "extraction.parser.canonical_form",
    )
    if canonical_form != CANONICAL_FORM:
        raise FactFamilyManifestError(
            f"parser canonical form must be {CANONICAL_FORM}"
        )
    if extraction.get("encoding") != ENCODING:
        raise FactFamilyManifestError(f"encoding must be {ENCODING}")

    comparison = _mapping(root.get("comparison"), "comparison")
    if tuple(comparison.get("join_key", ())) != JOIN_KEY:
        raise FactFamilyManifestError(f"comparison.join_key must be {JOIN_KEY}")
    statuses = dict(
        _mapping(comparison.get("statuses"), "comparison.statuses")
    )
    if statuses != COMPARISON_STATUSES:
        raise FactFamilyManifestError("comparison statuses do not match the contract")

    evidence = _mapping(root.get("evidence"), "evidence")
    required_evidence = _string_list(
        evidence.get("required_fields"),
        "evidence.required_fields",
    )
    if required_evidence != EVIDENCE_FIELDS:
        raise FactFamilyManifestError(
            f"evidence.required_fields must be {EVIDENCE_FIELDS}"
        )

    raw_families = root.get("families")
    if not isinstance(raw_families, list) or not raw_families:
        raise FactFamilyManifestError("families must be a non-empty list")
    families = tuple(
        _load_family(raw_family, index)
        for index, raw_family in enumerate(raw_families)
    )
    family_ids = [family.family_id for family in families]
    if len(family_ids) != len(set(family_ids)):
        raise FactFamilyManifestError("family ids must be unique")

    return FactFamilyManifest(
        source_manifest=source_manifest,
        transition=transition,
        parser_implementation=parser_implementation,
        parser_feature_version=parser_feature_version,
        families=families,
    )


def manifest_summary(manifest: FactFamilyManifest) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "source_manifest": manifest.source_manifest,
        "transition": manifest.transition,
        "parser": {
            "implementation": manifest.parser_implementation,
            "feature_version": manifest.parser_feature_version,
        },
        "family_count": len(manifest.families),
        "family_ids": [family.family_id for family in manifest.families],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the delta_v2 atomic-fact family preregistration.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("fact_families.yaml"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_fact_family_manifest(args.manifest)
    print(json.dumps(manifest_summary(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
