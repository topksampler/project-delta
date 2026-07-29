from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from experiments.delta_v2.feature_promotion import validate_feature_delta


CONTRACT_SCHEMA = "delta.eval_item_contract.v1"
FACT_SCHEMA = "delta.atomic_fact_delta.v1"
FEATURE_SCHEMA = "delta.feature_delta.v1"
SPLIT_SCHEMA = "delta.source_split_assignment.v1"
PROBE_RESULT_SCHEMA = "delta.behavior_probe_result.v1"
EVAL_SCHEMA = "delta.eval_item.v1"
AUDIT_ROW_SCHEMA = "delta.human_eval_audit_item.v1"
SUMMARY_SCHEMA = "delta.eval_item_build_audit.v1"
FACT_STATUSES = ("stable", "added", "removed", "changed")
NONSTABLE_STATUSES = {"added", "removed", "changed"}
ALLOWED_SPLITS = {"train", "dev"}
FEATURE_TASK = "feature_behavior_matrix"
FACT_TASK = "atomic_fact_change_status"


class EvalBuildError(ValueError):
    """The EvalItem contract, verified sources, or generated rows are invalid."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvalBuildError(f"{field} must be a mapping")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvalBuildError(f"{field} must be a non-empty string")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise EvalBuildError(f"{field} must be a list of strings")
    if len(value) != len(set(value)):
        raise EvalBuildError(f"{field} must not contain duplicates")
    return value


def load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvalBuildError(f"cannot read YAML: {path}") from exc
    return _mapping(payload, str(path))


def load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalBuildError(f"cannot read JSON: {path}") from exc
    return _mapping(payload, str(path))


def load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvalBuildError(f"cannot read JSONL: {path}") from exc
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalBuildError(
                f"invalid JSON at {path}:{line_number}"
            ) from exc
        records.append(_mapping(payload, f"{path}:{line_number}"))
    return records


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise EvalBuildError("unsupported EvalItem contract schema")
    _string(contract.get("contract_id"), "contract_id")
    if contract.get("transition") != "development":
        raise EvalBuildError("EvalItem builder may only use development")
    if contract.get("source_split_contract_id") != (
        "delta-v2-development-source-split-v1"
    ):
        raise EvalBuildError("unexpected source split contract")

    eligibility = _mapping(contract.get("eligibility"), "eligibility")
    facts = _mapping(
        eligibility.get("atomic_fact_delta"),
        "eligibility.atomic_fact_delta",
    )
    if set(
        _string_list(
            facts.get("include_nonstable_statuses"),
            "include_nonstable_statuses",
        )
    ) != NONSTABLE_STATUSES:
        raise EvalBuildError("all nonstable fact statuses must be eligible")
    controls = _mapping(facts.get("stable_controls"), "stable_controls")
    if (
        controls.get("ratio_to_nonstable") != "1:1"
        or controls.get("strata") != ["split", "family"]
        or controls.get("ranking_algorithm") != "sha256-source-id-v1"
    ):
        raise EvalBuildError("unsupported stable-control selection")
    _string(controls.get("salt"), "stable_controls.salt")

    features = _mapping(eligibility.get("feature_delta"), "feature_delta")
    if (
        features.get("include") != "all-promoted"
        or features.get("required_probe_scope") != "complete-candidate"
    ):
        raise EvalBuildError("only promoted features may be eligible")

    templates = _mapping(contract.get("templates"), "templates")
    fact_template = _mapping(
        templates.get(FACT_TASK),
        f"templates.{FACT_TASK}",
    )
    if fact_template.get("response_labels") != list(FACT_STATUSES):
        raise EvalBuildError("fact response labels must be exact")
    if fact_template.get("scorer") != "exact-enum-v1":
        raise EvalBuildError("unsupported fact scorer")
    feature_template = _mapping(
        templates.get(FEATURE_TASK),
        f"templates.{FEATURE_TASK}",
    )
    if feature_template.get("scorer") != "exact-structured-json-v1":
        raise EvalBuildError("unsupported feature scorer")
    if feature_template.get("feature_id") != (
        "feature:vllm-prefix-cache-retention"
    ):
        raise EvalBuildError("unexpected feature template")
    response_schema = _mapping(
        feature_template.get("response_schema"),
        "feature response_schema",
    )
    expected_feature_gold = {
        "unset": "dense",
        "positive_aligned_interval": "sparse_interval_plus_latest_boundary",
        "zero": "latest_boundary_only",
        "negative": "rejected",
        "misaligned": "rejected",
    }
    if dict(response_schema) != expected_feature_gold:
        raise EvalBuildError("feature response schema changed")

    generator = _mapping(contract.get("generator"), "generator")
    if (
        generator.get("method") != "deterministic-template-only"
        or generator.get("teacher_role") != "forbidden"
    ):
        raise EvalBuildError("teacher models must not define EvalItems")
    revisions = _mapping(generator.get("revisions"), "generator.revisions")
    if dict(revisions) != {"before": "v0.22.0", "after": "v0.23.0"}:
        raise EvalBuildError("unexpected development revisions")

    split_policy = _mapping(contract.get("split_policy"), "split_policy")
    if (
        split_policy.get("invariant") != "every-item-inherits-source-split"
        or split_policy.get("development_eval_items") != "forbidden"
        or split_policy.get("acceptance_future_split") != "eval"
    ):
        raise EvalBuildError("invalid EvalItem split policy")

    human_audit = _mapping(contract.get("human_audit"), "human_audit")
    if (
        human_audit.get("algorithm") != "stratified-hash-sample-v1"
        or human_audit.get("sample_size") != 32
        or human_audit.get("judgments")
        != ["truth", "version_status", "answerability"]
        or human_audit.get("pass_threshold_per_judgment") != 0.90
    ):
        raise EvalBuildError("human audit gate changed")
    if contract.get("freeze_state") != "unfrozen":
        raise EvalBuildError("EvalItem contract must not claim a freeze")


def index_splits(
    records: Iterable[Mapping[str, Any]],
    expected_contract_id: str,
) -> dict[str, str]:
    splits: dict[str, str] = {}
    for record in records:
        if record.get("schema") != SPLIT_SCHEMA:
            raise EvalBuildError("unexpected source split schema")
        if record.get("contract_id") != expected_contract_id:
            raise EvalBuildError("source split contract mismatch")
        source_id = _string(record.get("source_id"), "split.source_id")
        split = _string(record.get("split"), "split")
        if split not in ALLOWED_SPLITS:
            raise EvalBuildError("development source cannot be assigned to eval")
        if source_id in splits:
            raise EvalBuildError(f"duplicate split source: {source_id}")
        splits[source_id] = split
    return splits


def _selection_hash(source_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}\0{source_id}".encode("utf-8")).hexdigest()


def select_fact_records(
    fact_records: Iterable[Mapping[str, Any]],
    splits: Mapping[str, str],
    contract: Mapping[str, Any],
) -> list[tuple[Mapping[str, Any], str]]:
    facts: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for record in fact_records:
        if record.get("schema") != FACT_SCHEMA:
            raise EvalBuildError("only AtomicFactDelta records are eligible")
        fact_id = _string(record.get("fact_id"), "fact_id")
        if fact_id in seen:
            raise EvalBuildError(f"duplicate fact_id: {fact_id}")
        seen.add(fact_id)
        if fact_id not in splits:
            raise EvalBuildError(f"fact has no source split: {fact_id}")
        status = _string(record.get("status"), "fact.status")
        if status not in FACT_STATUSES:
            raise EvalBuildError(f"unsupported fact status: {status}")
        _string(record.get("family"), "fact.family")
        _string(record.get("semantic_key"), "fact.semantic_key")
        facts.append(record)

    nonstable: list[tuple[Mapping[str, Any], str]] = []
    nonstable_counts: Counter[tuple[str, str]] = Counter()
    stable_by_stratum: dict[
        tuple[str, str],
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    for record in facts:
        fact_id = str(record["fact_id"])
        split = splits[fact_id]
        family = str(record["family"])
        stratum = (split, family)
        if record["status"] in NONSTABLE_STATUSES:
            nonstable.append((record, "nonstable"))
            nonstable_counts[stratum] += 1
        else:
            stable_by_stratum[stratum].append(record)

    salt = str(
        contract["eligibility"]["atomic_fact_delta"]["stable_controls"]["salt"]
    )
    selected_controls: list[tuple[Mapping[str, Any], str]] = []
    for stratum, quota in sorted(nonstable_counts.items()):
        ranked = sorted(
            stable_by_stratum.get(stratum, []),
            key=lambda record: (
                _selection_hash(str(record["fact_id"]), salt),
                str(record["fact_id"]),
            ),
        )
        if len(ranked) < quota:
            raise EvalBuildError(
                f"not enough stable controls for stratum {stratum}: "
                f"need {quota}, have {len(ranked)}"
            )
        selected_controls.extend(
            (record, "stable_control") for record in ranked[:quota]
        )
    return sorted(
        nonstable + selected_controls,
        key=lambda pair: str(pair[0]["fact_id"]),
    )


def _display_fact(record: Mapping[str, Any]) -> tuple[str, str]:
    family_labels = {
        "python.cli_option.v1": "Python CLI option",
        "python.config_field.v1": "Python configuration field",
        "python.environment_variable.v1": "Python environment variable",
        "python.literal_domain.v1": "Python Literal domain",
    }
    family = str(record["family"])
    if family not in family_labels:
        raise EvalBuildError(f"no prompt label for fact family: {family}")
    semantic_key = str(record["semantic_key"])
    _, separator, display_name = semantic_key.partition(":")
    if not separator or not display_name:
        raise EvalBuildError(f"invalid semantic key: {semantic_key}")
    return family_labels[family], display_name


def _eval_id(source_id: str, task_kind: str, generator_version: str) -> str:
    identity = "\0".join(
        ("delta.eval_item.v1", source_id, task_kind, generator_version)
    )
    return "eval:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def build_fact_item(
    record: Mapping[str, Any],
    *,
    split: str,
    eligibility_reason: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    family_label, display_name = _display_fact(record)
    generator_version = str(contract["generator"]["version"])
    template = contract["templates"][FACT_TASK]
    prompt = (
        "Across vLLM v0.22.0 to v0.23.0, how did the "
        f"{family_label} `{display_name}` change? Answer with exactly one "
        "label: stable, added, removed, or changed."
    )
    return {
        "schema": EVAL_SCHEMA,
        "eval_id": _eval_id(str(record["fact_id"]), FACT_TASK, generator_version),
        "source_id": record["fact_id"],
        "source_kind": "atomic_fact_delta",
        "split": split,
        "task_kind": FACT_TASK,
        "prompt": prompt,
        "gold": {"status": record["status"]},
        "scorer": template["scorer"],
        "generator": {
            "version": generator_version,
            "template_version": template["version"],
            "method": contract["generator"]["method"],
        },
        "eligibility_reason": eligibility_reason,
        "provenance": {
            "family": record["family"],
            "semantic_key": record["semantic_key"],
            "evidence_before": record.get("evidence_before"),
            "evidence_after": record.get("evidence_after"),
            "verifier": record["verifier"],
        },
    }


def _feature_gold_from_probe(
    result: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, str]:
    if result.get("schema") != PROBE_RESULT_SCHEMA:
        raise EvalBuildError("unsupported feature probe result")
    if result.get("status") != "pass":
        raise EvalBuildError("feature probe result must pass")
    template = contract["templates"][FEATURE_TASK]
    if result.get("probe_id") != template["probe_id"]:
        raise EvalBuildError("feature probe ID mismatch")
    if result.get("claim_scope") != "complete-candidate":
        raise EvalBuildError("feature probe must cover complete candidate")
    cases = {
        str(case["case_id"]): case
        for case in result.get("cases", [])
        if isinstance(case, Mapping)
    }
    required = {
        "dense_default",
        "interval_64",
        "latest_only",
        "negative_rejected",
        "misaligned_rejected",
    }
    if set(cases) != required or any(
        case.get("status") != "pass" for case in cases.values()
    ):
        raise EvalBuildError("feature probe cases are incomplete or failed")
    dense = cases["dense_default"]["observed_after"]
    interval = cases["interval_64"]["observed_after"]
    latest = cases["latest_only"]["observed_after"]
    negative = cases["negative_rejected"]["observed_after"]
    misaligned = cases["misaligned_rejected"]["observed_after"]
    if (
        dense.get("kind") != "cache_state"
        or dense.get("cached_indices") != list(range(16))
        or interval.get("kind") != "cache_state"
        or interval.get("cached_indices") != [3, 7, 11, 14, 15]
        or latest.get("kind") != "cache_state"
        or latest.get("cached_indices") != [14]
        or negative.get("kind") != "construction_error"
        or misaligned.get("kind") != "construction_error"
    ):
        raise EvalBuildError("feature probe observations do not support template")
    return dict(template["response_schema"])


def build_feature_item(
    record: Mapping[str, Any],
    *,
    split: str,
    probe_result: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    validate_feature_delta(record)
    feature_id = str(record["feature_id"])
    template = contract["templates"][FEATURE_TASK]
    if feature_id != template["feature_id"]:
        raise EvalBuildError(f"no feature template for {feature_id}")
    generator_version = str(contract["generator"]["version"])
    gold = _feature_gold_from_probe(probe_result, contract)
    prompt = (
        "For vLLM v0.23.0, return a JSON object describing the verified "
        "`VLLM_PREFIX_CACHE_RETENTION_INTERVAL` behavior. Use exactly these "
        "keys: `unset`, `positive_aligned_interval`, `zero`, `negative`, and "
        "`misaligned`. Use only the behavior labels `dense`, "
        "`sparse_interval_plus_latest_boundary`, `latest_boundary_only`, or "
        "`rejected`."
    )
    return {
        "schema": EVAL_SCHEMA,
        "eval_id": _eval_id(feature_id, FEATURE_TASK, generator_version),
        "source_id": feature_id,
        "source_kind": "feature_delta",
        "split": split,
        "task_kind": FEATURE_TASK,
        "prompt": prompt,
        "gold": gold,
        "scorer": template["scorer"],
        "generator": {
            "version": generator_version,
            "template_version": template["version"],
            "method": contract["generator"]["method"],
        },
        "eligibility_reason": "promoted_feature",
        "provenance": {
            "status": record["status"],
            "evidence_ids": record["evidence_ids"],
            "behavior_probe_ids": record["behavior_probe_ids"],
            "verification": record["verification"],
        },
    }


def build_items(
    *,
    contract: Mapping[str, Any],
    fact_records: Iterable[Mapping[str, Any]],
    feature_records: Iterable[Mapping[str, Any]],
    split_records: Iterable[Mapping[str, Any]],
    probe_results: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validate_contract(contract)
    splits = index_splits(split_records, str(contract["source_split_contract_id"]))
    items: list[dict[str, Any]] = []
    for fact, reason in select_fact_records(fact_records, splits, contract):
        items.append(
            build_fact_item(
                fact,
                split=splits[str(fact["fact_id"])],
                eligibility_reason=reason,
                contract=contract,
            )
        )
    for feature in feature_records:
        validate_feature_delta(feature)
        feature_id = str(feature["feature_id"])
        if feature_id not in splits:
            raise EvalBuildError(f"feature has no source split: {feature_id}")
        probe_id = str(contract["templates"][FEATURE_TASK]["probe_id"])
        if probe_id not in probe_results:
            raise EvalBuildError(f"missing feature probe result: {probe_id}")
        items.append(
            build_feature_item(
                feature,
                split=splits[feature_id],
                probe_result=probe_results[probe_id],
                contract=contract,
            )
        )
    return sorted(items, key=lambda item: item["eval_id"])


def score_response(item: Mapping[str, Any], response: str) -> bool:
    scorer = item.get("scorer")
    if scorer == "exact-enum-v1":
        expected = _mapping(item.get("gold"), "item.gold").get("status")
        return response.strip().lower() == expected
    if scorer == "exact-structured-json-v1":
        try:
            observed = json.loads(response)
        except json.JSONDecodeError:
            return False
        return observed == item.get("gold")
    raise EvalBuildError(f"unsupported scorer: {scorer}")


def audit_items(
    items: Sequence[Mapping[str, Any]],
    split_records: Iterable[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    splits = index_splits(split_records, str(contract["source_split_contract_id"]))
    eval_ids = [str(item.get("eval_id")) for item in items]
    source_ids = [str(item.get("source_id")) for item in items]
    split_mismatches = sorted(
        source_id
        for source_id, item in zip(source_ids, items)
        if source_id not in splits or item.get("split") != splits[source_id]
    )
    invalid_splits = sorted(
        {
            str(item.get("split"))
            for item in items
            if item.get("split") not in ALLOWED_SPLITS
        }
    )
    selected_counts = Counter(
        (str(item["split"]), str(item["provenance"].get("family")))
        for item in items
        if item["source_kind"] == "atomic_fact_delta"
        and item["eligibility_reason"] == "nonstable"
    )
    control_counts = Counter(
        (str(item["split"]), str(item["provenance"].get("family")))
        for item in items
        if item["source_kind"] == "atomic_fact_delta"
        and item["eligibility_reason"] == "stable_control"
    )
    control_balance_matches = selected_counts == control_counts
    audit_ok = (
        len(eval_ids) == len(set(eval_ids))
        and len(source_ids) == len(set(source_ids))
        and not split_mismatches
        and not invalid_splits
        and control_balance_matches
        and eval_ids == sorted(eval_ids)
    )
    split_counts = Counter(str(item["split"]) for item in items)
    task_counts = Counter(str(item["task_kind"]) for item in items)
    status_counts = Counter(
        str(item["gold"]["status"])
        for item in items
        if item["task_kind"] == FACT_TASK
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "contract_id": contract["contract_id"],
        "status": "pass" if audit_ok else "fail",
        "items": len(items),
        "unique_eval_ids": len(eval_ids) == len(set(eval_ids)),
        "unique_source_ids": len(source_ids) == len(set(source_ids)),
        "split_mismatches": split_mismatches,
        "invalid_splits": invalid_splits,
        "control_balance_matches": control_balance_matches,
        "split_counts": dict(sorted(split_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "fact_status_counts": dict(sorted(status_counts.items())),
        "eval_items": split_counts.get("eval", 0),
        "acceptance_accessed": False,
        "deterministic_order": eval_ids == sorted(eval_ids),
    }


def _audit_hash(eval_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}\0{eval_id}".encode("utf-8")).hexdigest()


def build_human_audit_packet(
    items: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    items = list(items)
    audit_contract = contract["human_audit"]
    sample_size = int(audit_contract["sample_size"])
    salt = str(audit_contract["fill_ranking"]["salt"])
    by_eval_id = {str(item["eval_id"]): item for item in items}
    selected: set[str] = {
        str(item["eval_id"])
        for item in items
        if item["source_kind"] == "feature_delta"
    }
    strata: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        if item["source_kind"] != "atomic_fact_delta":
            continue
        stratum = (
            str(item["split"]),
            str(item["provenance"]["family"]),
            str(item["gold"]["status"]),
        )
        strata[stratum].append(item)
    for stratum_items in strata.values():
        chosen = min(
            stratum_items,
            key=lambda item: (
                _audit_hash(str(item["eval_id"]), salt),
                str(item["eval_id"]),
            ),
        )
        selected.add(str(chosen["eval_id"]))
    if len(selected) > sample_size:
        raise EvalBuildError(
            f"mandatory audit strata exceed sample size: {len(selected)}"
        )
    remaining = sorted(
        (item for item in items if str(item["eval_id"]) not in selected),
        key=lambda item: (
            _audit_hash(str(item["eval_id"]), salt),
            str(item["eval_id"]),
        ),
    )
    for item in remaining[: sample_size - len(selected)]:
        selected.add(str(item["eval_id"]))
    selected_items = sorted(
        (by_eval_id[eval_id] for eval_id in selected),
        key=lambda item: (
            _audit_hash(str(item["eval_id"]), salt),
            str(item["eval_id"]),
        ),
    )
    packet: list[dict[str, Any]] = []
    for index, item in enumerate(selected_items, start=1):
        packet.append(
            {
                "schema": AUDIT_ROW_SCHEMA,
                "audit_index": index,
                "eval_id": item["eval_id"],
                "source_id": item["source_id"],
                "split": item["split"],
                "task_kind": item["task_kind"],
                "prompt": item["prompt"],
                "proposed_gold": item["gold"],
                "provenance": item["provenance"],
                "review": {
                    "truth": None,
                    "version_status": None,
                    "answerability": None,
                    "notes": None,
                },
            }
        )
    return packet


def _serialize_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def write_outputs(
    *,
    items: Sequence[Mapping[str, Any]],
    packet: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    inputs: Mapping[str, Any],
    items_path: Path,
    packet_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    item_bytes = _serialize_jsonl(items)
    packet_bytes = _serialize_jsonl(packet)
    for path in (items_path, packet_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    items_path.write_bytes(item_bytes)
    packet_path.write_bytes(packet_bytes)
    summary = {
        **dict(audit),
        "inputs": dict(inputs),
        "outputs": {
            "eval_items": {
                "filename": items_path.name,
                "rows": len(items),
                "sha256": hashlib.sha256(item_bytes).hexdigest(),
            },
            "human_audit_packet": {
                "filename": packet_path.name,
                "rows": len(packet),
                "sha256": hashlib.sha256(packet_bytes).hexdigest(),
            },
        },
        "human_audit": {
            "status": "pending",
            "required_sample_size": len(packet),
            "judgments": ["truth", "version_status", "answerability"],
            "pass_threshold_per_judgment": 0.90,
        },
        "freeze_state": "unfrozen",
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_probe_result(value: str) -> tuple[str, Path]:
    probe_id, separator, raw_path = value.partition("=")
    if not separator or not probe_id or not raw_path:
        raise argparse.ArgumentTypeError("probe result must be ID=/path/result.json")
    return probe_id, Path(raw_path)


def _file_record(path: Path) -> dict[str, str]:
    return {
        "filename": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate split-inheriting delta_v2 EvalItems."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument(
        "--feature-delta",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument(
        "--probe-result",
        action="append",
        type=_parse_probe_result,
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--human-audit-packet", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    feature_paths = sorted(args.feature_delta, key=lambda path: str(path))
    probe_paths = dict(args.probe_result)
    if len(probe_paths) != len(args.probe_result):
        raise EvalBuildError("duplicate probe result ID")
    contract = load_yaml(args.contract)
    fact_records = load_jsonl(args.facts)
    feature_records = [load_yaml(path) for path in feature_paths]
    split_records = load_jsonl(args.splits)
    probe_results = {
        probe_id: load_json(path)
        for probe_id, path in sorted(probe_paths.items())
    }
    items = build_items(
        contract=contract,
        fact_records=fact_records,
        feature_records=feature_records,
        split_records=split_records,
        probe_results=probe_results,
    )
    audit = audit_items(items, split_records, contract)
    packet = build_human_audit_packet(items, contract)
    inputs = {
        "contract": _file_record(args.contract),
        "facts": _file_record(args.facts),
        "features": [_file_record(path) for path in feature_paths],
        "source_splits": _file_record(args.splits),
        "probe_results": [
            {
                "probe_id": probe_id,
                **_file_record(path),
            }
            for probe_id, path in sorted(probe_paths.items())
        ],
    }
    summary = write_outputs(
        items=items,
        packet=packet,
        audit=audit,
        inputs=inputs,
        items_path=args.out,
        packet_path=args.human_audit_packet,
        summary_path=args.summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
