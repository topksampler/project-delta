from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from experiments.delta_v2.feature_promotion import (
    FeaturePromotionError,
    validate_feature_delta,
)


CONTRACT_SCHEMA = "delta.source_split_contract.v1"
FACT_SCHEMA = "delta.atomic_fact_delta.v1"
FEATURE_SCHEMA = "delta.feature_delta.v1"
ROW_SCHEMA = "delta.source_split_assignment.v1"
SUMMARY_SCHEMA = "delta.source_split_audit.v1"
SPLITS = ("train", "dev", "eval")
STATUSES = {"stable", "added", "removed", "changed"}
UINT64_SPACE = 1 << 64
TRAIN_UPPER_EXCLUSIVE = (UINT64_SPACE * 4) // 5


class SourceSplitError(ValueError):
    """The split contract or verified-source corpus is invalid."""


@dataclass(frozen=True)
class Source:
    source_id: str
    source_kind: str
    status: str
    family: str | None
    referenced_fact_ids: tuple[str, ...] = ()


class DisjointSet:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        if item not in self.parent:
            raise SourceSplitError(f"unknown split source: {item}")
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        anchor = min(left_root, right_root)
        other = max(left_root, right_root)
        self.parent[other] = anchor


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceSplitError(f"{field} must be a mapping")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourceSplitError(f"{field} must be a non-empty string")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise SourceSplitError(f"{field} must be a list of strings")
    if len(value) != len(set(value)):
        raise SourceSplitError(f"{field} must not contain duplicates")
    return value


def load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SourceSplitError(f"cannot read YAML: {path}") from exc
    return _mapping(payload, str(path))


def validate_contract(
    contract: Mapping[str, Any],
    *,
    transition: str = "development",
    allow_acceptance: bool = False,
) -> None:
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise SourceSplitError("unsupported source split contract schema")
    _string(contract.get("contract_id"), "contract_id")
    _string(contract.get("repository"), "repository")
    if contract.get("transition") != "development":
        raise SourceSplitError("source split contract must originate on development")
    if transition not in {"development", "acceptance"}:
        raise SourceSplitError(f"unsupported split transition: {transition}")
    if transition == "acceptance" and not allow_acceptance:
        raise SourceSplitError(
            "acceptance transition is sealed; pass allow_acceptance only "
            "after the development recipe freezes"
        )

    eligible = _mapping(contract.get("eligible_sources"), "eligible_sources")
    facts = _mapping(
        eligible.get("atomic_fact_delta"),
        "eligible_sources.atomic_fact_delta",
    )
    features = _mapping(
        eligible.get("feature_delta"),
        "eligible_sources.feature_delta",
    )
    if facts.get("schema") != FACT_SCHEMA:
        raise SourceSplitError("unexpected atomic-fact schema")
    if features.get("schema") != FEATURE_SCHEMA:
        raise SourceSplitError("unexpected feature schema")
    if set(_string_list(facts.get("statuses"), "fact statuses")) != STATUSES:
        raise SourceSplitError("atomic-fact statuses must be complete")
    if set(_string_list(features.get("statuses"), "feature statuses")) != STATUSES:
        raise SourceSplitError("feature statuses must be complete")

    units = _mapping(contract.get("split_units"), "split_units")
    if units.get("algorithm") != "referenced-source-connected-components-v1":
        raise SourceSplitError("unsupported split-unit algorithm")
    if units.get("anchor") != "lexicographically-smallest-source-id":
        raise SourceSplitError("unsupported split-unit anchor")
    if units.get("edges") != ["feature-delta-to-referenced-atomic-fact"]:
        raise SourceSplitError("unsupported split-unit edges")

    assignment = _mapping(
        contract.get("development_assignment"),
        "development_assignment",
    )
    if assignment.get("algorithm") != "sha256-anchor-threshold-v1":
        raise SourceSplitError("unsupported assignment algorithm")
    _string(assignment.get("salt"), "development_assignment.salt")
    thresholds = _mapping(assignment.get("thresholds"), "thresholds")
    train = _mapping(thresholds.get("train"), "thresholds.train")
    dev = _mapping(thresholds.get("dev"), "thresholds.dev")
    expected = {
        "train": {
            "lower_inclusive": "0",
            "upper_exclusive": str(TRAIN_UPPER_EXCLUSIVE),
        },
        "dev": {
            "lower_inclusive": str(TRAIN_UPPER_EXCLUSIVE),
            "upper_exclusive": str(UINT64_SPACE),
        },
    }
    if dict(train) != expected["train"] or dict(dev) != expected["dev"]:
        raise SourceSplitError("development thresholds must encode exact 80/20")
    if assignment.get("eval_sources") != "forbidden":
        raise SourceSplitError("development eval sources must be forbidden")

    acceptance = _mapping(
        contract.get("acceptance_assignment"),
        "acceptance_assignment",
    )
    if (
        acceptance.get("state") != "sealed"
        or acceptance.get("unlock_condition")
        != "validated-development-recipe-freeze"
        or acceptance.get("algorithm") != "all-sources-eval-v1"
        or acceptance.get("future_split") != "eval"
        or acceptance.get("development_builder_behavior") != "reject"
    ):
        raise SourceSplitError("acceptance must remain sealed and reserved for eval")
    wording = _mapping(contract.get("wording_policy"), "wording_policy")
    if wording.get("assignment_key") != "source_id":
        raise SourceSplitError("wording must inherit assignment from source_id")
    if contract.get("freeze_state") != "unfrozen":
        raise SourceSplitError("development split must not claim a freeze")


def load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SourceSplitError(f"cannot read JSONL: {path}") from exc
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SourceSplitError(
                f"invalid JSON at {path}:{line_number}"
            ) from exc
        records.append(_mapping(payload, f"{path}:{line_number}"))
    return records


def source_from_fact(record: Mapping[str, Any]) -> Source:
    if record.get("schema") != FACT_SCHEMA:
        raise SourceSplitError("only AtomicFactDelta records are eligible")
    fact_id = _string(record.get("fact_id"), "fact_id")
    if not fact_id.startswith("fact:"):
        raise SourceSplitError("fact_id must use fact: prefix")
    status = _string(record.get("status"), "fact.status")
    if status not in STATUSES:
        raise SourceSplitError(f"unsupported fact status: {status}")
    family = _string(record.get("family"), "fact.family")
    verifier = _mapping(record.get("verifier"), "fact.verifier")
    _string(verifier.get("extractor_version"), "fact.verifier.extractor_version")
    if status in {"stable", "changed"}:
        if record.get("evidence_before") is None or record.get("evidence_after") is None:
            raise SourceSplitError(f"{status} fact requires evidence on both sides")
    elif status == "added":
        if record.get("evidence_before") is not None or record.get("evidence_after") is None:
            raise SourceSplitError("added fact requires only after evidence")
    elif status == "removed":
        if record.get("evidence_before") is None or record.get("evidence_after") is not None:
            raise SourceSplitError("removed fact requires only before evidence")
    return Source(
        source_id=fact_id,
        source_kind="atomic_fact_delta",
        status=status,
        family=family,
    )


def source_from_feature(record: Mapping[str, Any]) -> Source:
    try:
        validate_feature_delta(record)
    except FeaturePromotionError as exc:
        raise SourceSplitError(str(exc)) from exc
    feature_id = _string(record.get("feature_id"), "feature_id")
    fact_ids = tuple(
        sorted(
            evidence_id
            for evidence_id in record["evidence_ids"]
            if evidence_id.startswith("fact:")
        )
    )
    if not fact_ids and record.get("transition") != "acceptance":
        raise SourceSplitError(
            f"FeatureDelta has no referenced AtomicFactDelta: {feature_id}"
        )
    return Source(
        source_id=feature_id,
        source_kind="feature_delta",
        status=str(record["status"]),
        family=None,
        referenced_fact_ids=fact_ids,
    )


def collect_sources(
    fact_records: Iterable[Mapping[str, Any]],
    feature_records: Iterable[Mapping[str, Any]],
) -> dict[str, Source]:
    sources: dict[str, Source] = {}
    for record in fact_records:
        source = source_from_fact(record)
        if source.source_id in sources:
            raise SourceSplitError(f"duplicate source_id: {source.source_id}")
        sources[source.source_id] = source
    for record in feature_records:
        source = source_from_feature(record)
        if source.source_id in sources:
            raise SourceSplitError(f"duplicate source_id: {source.source_id}")
        sources[source.source_id] = source
    if not sources:
        raise SourceSplitError("source corpus must not be empty")
    return sources


def build_units(sources: Mapping[str, Source]) -> dict[str, tuple[str, ...]]:
    disjoint = DisjointSet(sources)
    fact_ids = {
        source_id
        for source_id, source in sources.items()
        if source.source_kind == "atomic_fact_delta"
    }
    for source in sources.values():
        if source.source_kind != "feature_delta":
            continue
        for fact_id in source.referenced_fact_ids:
            if fact_id not in fact_ids:
                raise SourceSplitError(
                    f"{source.source_id} references missing fact: {fact_id}"
                )
            disjoint.union(source.source_id, fact_id)

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for source_id in sorted(sources):
        members_by_root[disjoint.find(source_id)].append(source_id)
    units: dict[str, tuple[str, ...]] = {}
    for members in members_by_root.values():
        anchor = min(members)
        units[anchor] = tuple(sorted(members))
    return dict(sorted(units.items()))


def assignment_for_anchor(anchor: str, salt: str) -> dict[str, str]:
    digest = hashlib.sha256(f"{salt}\0{anchor}".encode("utf-8")).hexdigest()
    value = int(digest[:16], 16)
    split = "train" if value < TRAIN_UPPER_EXCLUSIVE else "dev"
    return {
        "split": split,
        "assignment_hash": f"sha256:{digest}",
        "hash_prefix_u64": str(value),
    }


def build_assignments(
    contract: Mapping[str, Any],
    sources: Mapping[str, Source],
    *,
    transition: str = "development",
    allow_acceptance: bool = False,
) -> list[dict[str, Any]]:
    validate_contract(
        contract,
        transition=transition,
        allow_acceptance=allow_acceptance,
    )
    units = build_units(sources)
    salt = str(contract["development_assignment"]["salt"])
    rows: list[dict[str, Any]] = []
    for anchor, members in units.items():
        assignment = (
            assignment_for_anchor(anchor, salt)
            if transition == "development"
            else {
                "split": "eval",
                "assignment_hash": "policy:all-sources-eval-v1",
                "hash_prefix_u64": "not-applicable",
            }
        )
        unit_identity = "\0".join(
            (
                "delta.source_split_unit.v1",
                str(contract["repository"]),
                transition,
                anchor,
            )
        )
        unit_id = "source-unit:" + hashlib.sha256(
            unit_identity.encode("utf-8")
        ).hexdigest()
        for source_id in members:
            source = sources[source_id]
            rows.append(
                {
                    "schema": ROW_SCHEMA,
                    "contract_id": contract["contract_id"],
                    "source_id": source_id,
                    "source_kind": source.source_kind,
                    "status": source.status,
                    "family": source.family,
                    "transition": transition,
                    "split_unit_id": unit_id,
                    "unit_anchor_id": anchor,
                    "unit_member_count": len(members),
                    **assignment,
                }
            )
    return sorted(rows, key=lambda row: row["source_id"])


def audit_assignments(
    contract: Mapping[str, Any],
    sources: Mapping[str, Source],
    rows: Sequence[Mapping[str, Any]],
    *,
    transition: str = "development",
    allow_acceptance: bool = False,
) -> dict[str, Any]:
    validate_contract(
        contract,
        transition=transition,
        allow_acceptance=allow_acceptance,
    )
    source_ids = [str(row.get("source_id")) for row in rows]
    duplicate_source_ids = len(source_ids) != len(set(source_ids))
    missing_sources = sorted(set(sources) - set(source_ids))
    extra_sources = sorted(set(source_ids) - set(sources))
    allowed_splits = (
        {"train", "dev"} if transition == "development" else {"eval"}
    )
    invalid_splits = sorted(
        {
            str(row.get("split"))
            for row in rows
            if row.get("split") not in allowed_splits
        }
    )
    wrong_transitions = sorted(
        {
            str(row.get("transition"))
            for row in rows
            if row.get("transition") != transition
        }
    )

    splits_by_unit: dict[str, set[str]] = defaultdict(set)
    members_by_unit: Counter[str] = Counter()
    for row in rows:
        unit_id = _string(row.get("split_unit_id"), "split_unit_id")
        split = _string(row.get("split"), "split")
        splits_by_unit[unit_id].add(split)
        members_by_unit[unit_id] += 1
    cross_split_units = sorted(
        unit_id
        for unit_id, splits in splits_by_unit.items()
        if len(splits) != 1
    )

    split_counts = Counter(str(row["split"]) for row in rows)
    kind_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    status_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    family_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        split = str(row["split"])
        kind_split_counts[str(row["source_kind"])][split] += 1
        status_split_counts[str(row["status"])][split] += 1
        if row.get("family") is not None:
            family_split_counts[str(row["family"])][split] += 1

    audit_ok = not (
        duplicate_source_ids
        or missing_sources
        or extra_sources
        or invalid_splits
        or wrong_transitions
        or cross_split_units
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "contract_id": contract["contract_id"],
        "transition": transition,
        "status": "pass" if audit_ok else "fail",
        "sources": len(rows),
        "units": len(splits_by_unit),
        "multi_source_units": sum(
            1 for count in members_by_unit.values() if count > 1
        ),
        "split_counts": dict(sorted(split_counts.items())),
        "source_kind_split_counts": {
            key: dict(sorted(value.items()))
            for key, value in sorted(kind_split_counts.items())
        },
        "status_split_counts": {
            key: dict(sorted(value.items()))
            for key, value in sorted(status_split_counts.items())
        },
        "family_split_counts": {
            key: dict(sorted(value.items()))
            for key, value in sorted(family_split_counts.items())
        },
        "duplicate_source_ids": duplicate_source_ids,
        "missing_sources": missing_sources,
        "extra_sources": extra_sources,
        "invalid_splits": invalid_splits,
        "wrong_transitions": wrong_transitions,
        "cross_split_units": cross_split_units,
        "eval_sources": split_counts.get("eval", 0),
        "acceptance_accessed": transition == "acceptance",
        "deterministic_order": source_ids == sorted(source_ids),
    }


def serialize_rows(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    ).encode("utf-8")


def write_outputs(
    rows: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    inputs: Mapping[str, Any],
    output_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    serialized = serialize_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(serialized)
    summary = {
        **dict(audit),
        "inputs": dict(inputs),
        "output": {
            "filename": output_path.name,
            "rows": len(rows),
            "sha256": hashlib.sha256(serialized).hexdigest(),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split verified delta_v2 sources before wording generation."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument(
        "--transition",
        choices=("development", "acceptance"),
        default="development",
    )
    parser.add_argument(
        "--allow-acceptance",
        action="store_true",
        help="Unlock acceptance only after the development recipe freezes.",
    )
    parser.add_argument(
        "--feature-delta",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = load_yaml(args.contract)
    feature_paths = sorted(args.feature_delta, key=lambda path: str(path))
    inputs = {
        "contract": {
            "filename": args.contract.name,
            "sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest(),
        },
        "facts": {
            "filename": args.facts.name,
            "sha256": hashlib.sha256(args.facts.read_bytes()).hexdigest(),
        },
        "feature_deltas": [
            {
                "filename": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in feature_paths
        ],
    }
    sources = collect_sources(
        load_jsonl(args.facts),
        (load_yaml(path) for path in feature_paths),
    )
    rows = build_assignments(
        contract,
        sources,
        transition=args.transition,
        allow_acceptance=args.allow_acceptance,
    )
    audit = audit_assignments(
        contract,
        sources,
        rows,
        transition=args.transition,
        allow_acceptance=args.allow_acceptance,
    )
    summary = write_outputs(rows, audit, inputs, args.out, args.summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
