from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiments.delta_v2.build_eval import (
    AUDIT_ROW_SCHEMA,
    FACT_TASK,
    FEATURE_TASK,
    EvalBuildError,
    _feature_gold_from_probe,
    build_fact_item,
    build_feature_item,
    load_json,
    load_jsonl,
    load_yaml,
)
from experiments.delta_v2.feature_promotion import validate_feature_delta


CONTRACT_SCHEMA = "delta.llm_judge_contract.v1"
EVIDENCE_SCHEMA = "delta.llm_judge_evidence.v1"
ATTESTATION_SCHEMA = "delta.llm_judge_attestation.v1"
VERDICT_SCHEMA = "delta.llm_judge_verdict.v1"
SUMMARY_SCHEMA = "delta.llm_judge_audit.v1"
DIMENSIONS = ("truth", "version_status", "answerability")
VERDICTS = {"pass", "fail", "abstain"}
FACT_SCHEMA = "delta.atomic_fact_delta.v1"
FEATURE_SCHEMA = "delta.feature_delta.v1"


class JudgeAuditError(ValueError):
    """The evidence-grounded judge contract or one of its inputs is invalid."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise JudgeAuditError(f"{field} must be a mapping")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise JudgeAuditError(f"{field} must be a non-empty string")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise JudgeAuditError(f"{field} must be a list of strings")
    if len(value) != len(set(value)):
        raise JudgeAuditError(f"{field} must not contain duplicates")
    return value


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_json_bytes(dict(row)) for row in rows)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_json_bytes(value))


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise JudgeAuditError("unsupported LLM judge contract schema")
    _string(contract.get("contract_id"), "contract_id")
    transition = contract.get("transition")
    if (
        transition not in {"development", "acceptance"}
        or contract.get("expected_sample_size") != 32
        or contract.get("input_audit_schema") != AUDIT_ROW_SCHEMA
    ):
        raise JudgeAuditError("judge contract must bind one known sample")

    amendment = _mapping(
        contract.get("protocol_amendment"),
        "protocol_amendment",
    )
    if (
        amendment.get("owner_direction")
        != "use-llm-as-judge-rooted-in-actual-truth"
        or amendment.get("timing")
        != {
            "development": (
                "before-development-freeze-and-before-acceptance-unseal"
            ),
            "acceptance": "audit-method-chosen-before-development-freeze",
        }[str(transition)]
        or amendment.get("replaces_review_actor") != "human"
        or amendment.get("preserves_judgments") != list(DIMENSIONS)
        or amendment.get("preserves_threshold") != 0.90
    ):
        raise JudgeAuditError("protocol amendment changed")
    if transition == "acceptance":
        if amendment.get("implementation_timing") != (
            "validator-parameterized-after-acceptance-unseal"
        ):
            raise JudgeAuditError("acceptance audit timing is not explicit")
        contract_id = str(contract["contract_id"])
        if contract_id == "delta-v2-acceptance-evidence-grounded-llm-judge-v2":
            if amendment.get("source_or_gold_changed") is not False:
                raise JudgeAuditError("v2 acceptance amendment changed")
        elif contract_id == (
            "delta-v2-acceptance-evidence-grounded-llm-judge-v3"
        ):
            if (
                amendment.get("source_or_gold_changed") is not True
                or amendment.get("selection_changed") is not True
                or amendment.get("change_scope")
                != "one-promoted-fact-frozen-acceptance-feature"
            ):
                raise JudgeAuditError(
                    "v3 acceptance feature amendment is not explicit"
                )
        else:
            raise JudgeAuditError("unsupported acceptance judge contract")

    authority = _mapping(contract.get("truth_authority"), "truth_authority")
    if (
        authority.get("llm_role") != "review-only"
        or authority.get("llm_may_change_gold") is not False
        or authority.get("conflict_action") != "fail-and-return-to-build"
    ):
        raise JudgeAuditError("LLM cannot own or rewrite ground truth")

    output = _mapping(contract.get("judge_output"), "judge_output")
    if (
        set(
            _string_list(
                output.get("allowed_verdicts"),
                "allowed_verdicts",
            )
        )
        != VERDICTS
        or output.get("required_dimensions") != list(DIMENSIONS)
        or output.get("require_reason") is not True
        or output.get("bind_exact_evidence_bundle") is not True
        or output.get("abstention_counts_as_failure") is not True
        or output.get("minimum_pass_rate_per_dimension") != 0.90
    ):
        raise JudgeAuditError("unsupported judge-output policy")

    identity = _mapping(contract.get("judge_identity"), "judge_identity")
    if (
        identity.get("kind") != "llm"
        or identity.get("target_model") is not False
    ):
        raise JudgeAuditError("judge must be an LLM distinct from target model")
    _string(identity.get("runtime"), "judge_identity.runtime")
    _string(identity.get("model_family"), "judge_identity.model_family")
    if contract.get("freeze_state") != "unfrozen":
        raise JudgeAuditError("judge contract must not claim a freeze")


def _index(
    records: Iterable[Mapping[str, Any]],
    *,
    id_field: str,
    schema: str,
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if record.get("schema") != schema:
            raise JudgeAuditError(f"unexpected schema for {id_field}")
        record_id = _string(record.get(id_field), id_field)
        if record_id in result:
            raise JudgeAuditError(f"duplicate {id_field}: {record_id}")
        result[record_id] = record
    return result


def _expected_presence(status: str) -> tuple[bool, bool]:
    expected = {
        "stable": (True, True),
        "added": (False, True),
        "removed": (True, False),
        "changed": (True, True),
    }
    try:
        return expected[status]
    except KeyError as exc:
        raise JudgeAuditError(f"unsupported fact status: {status}") from exc


def _fact_evidence(
    row: Mapping[str, Any],
    fact: Mapping[str, Any],
    eval_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    fact_id = _string(fact.get("fact_id"), "fact_id")
    status = _string(fact.get("status"), "fact.status")
    if row.get("proposed_gold") != {"status": status}:
        raise JudgeAuditError(f"proposed gold disagrees with {fact_id}")
    before_expected, after_expected = _expected_presence(status)
    before = fact.get("evidence_before")
    after = fact.get("evidence_after")
    if (before is not None) != before_expected:
        raise JudgeAuditError(f"before evidence disagrees with status: {fact_id}")
    if (after is not None) != after_expected:
        raise JudgeAuditError(f"after evidence disagrees with status: {fact_id}")
    if status == "stable" and fact.get("value_before") != fact.get("value_after"):
        raise JudgeAuditError(f"stable values disagree: {fact_id}")
    if status == "changed" and fact.get("value_before") == fact.get("value_after"):
        raise JudgeAuditError(f"changed values are equal: {fact_id}")

    reason = "stable_control" if status == "stable" else "nonstable"
    regenerated = build_fact_item(
        fact,
        split=_string(row.get("split"), "audit.split"),
        eligibility_reason=reason,
        contract=eval_contract,
    )
    if row.get("prompt") != regenerated["prompt"]:
        raise JudgeAuditError(f"prompt disagrees with source: {fact_id}")
    if row.get("eval_id") != regenerated["eval_id"]:
        raise JudgeAuditError(f"eval_id disagrees with source: {fact_id}")

    evidence = [
        {
            "evidence_id": f"atomic-fact:{fact_id}:record",
            "kind": "verified-atomic-fact",
            "payload": dict(fact),
        }
    ]
    for role, source in (("before", before), ("after", after)):
        if source is not None:
            evidence.append(
                {
                    "evidence_id": f"atomic-fact:{fact_id}:{role}",
                    "kind": "pinned-source-span",
                    "payload": {
                        "role": role,
                        "value": fact[f"value_{role}"],
                        "source": source,
                    },
                }
            )
    return evidence


def _probe_gold(
    probe: Mapping[str, Any],
    eval_contract: Mapping[str, Any],
) -> dict[str, str]:
    try:
        return _feature_gold_from_probe(probe, eval_contract)
    except EvalBuildError as exc:
        raise JudgeAuditError(str(exc)) from exc


def _feature_evidence(
    row: Mapping[str, Any],
    feature: Mapping[str, Any],
    probes: Mapping[str, Mapping[str, Any]],
    eval_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    validate_feature_delta(feature)
    feature_id = _string(feature.get("feature_id"), "feature_id")
    template = eval_contract["templates"][FEATURE_TASK]
    probe_id = _string(template.get("probe_id"), "feature probe_id")
    if probe_id not in probes:
        raise JudgeAuditError(f"missing feature probe: {probe_id}")
    probe = probes[probe_id]
    gold = _probe_gold(probe, eval_contract)
    if row.get("proposed_gold") != gold:
        raise JudgeAuditError(f"proposed feature gold disagrees: {feature_id}")
    regenerated = build_feature_item(
        feature,
        split=_string(row.get("split"), "audit.split"),
        probe_result=probe,
        contract=eval_contract,
    )
    if row.get("prompt") != regenerated["prompt"]:
        raise JudgeAuditError(f"feature prompt disagrees: {feature_id}")
    if row.get("eval_id") != regenerated["eval_id"]:
        raise JudgeAuditError(f"feature eval_id disagrees: {feature_id}")

    evidence = [
        {
            "evidence_id": f"feature:{feature_id}:record",
            "kind": "promoted-feature",
            "payload": dict(feature),
        }
    ]
    for case in probe["cases"]:
        case_id = _string(case.get("case_id"), "case_id")
        evidence.append(
            {
                "evidence_id": f"probe:{probe_id}:case:{case_id}",
                "kind": "executable-after-observation",
                "payload": dict(case),
            }
        )
    return evidence


def build_evidence_bundle(
    *,
    audit_rows: Sequence[Mapping[str, Any]],
    fact_records: Sequence[Mapping[str, Any]],
    feature_records: Sequence[Mapping[str, Any]],
    probe_results: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
    eval_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    validate_contract(contract)
    expected_size = int(contract["expected_sample_size"])
    if len(audit_rows) != expected_size:
        raise JudgeAuditError(
            f"audit packet has {len(audit_rows)} rows, expected {expected_size}"
        )
    facts = _index(fact_records, id_field="fact_id", schema=FACT_SCHEMA)
    features = _index(
        feature_records,
        id_field="feature_id",
        schema=FEATURE_SCHEMA,
    )
    bundle: list[dict[str, Any]] = []
    seen_eval_ids: set[str] = set()
    for expected_index, row in enumerate(audit_rows, start=1):
        if row.get("schema") != contract["input_audit_schema"]:
            raise JudgeAuditError("unexpected audit-row schema")
        if row.get("audit_index") != expected_index:
            raise JudgeAuditError("audit indices must be contiguous")
        eval_id = _string(row.get("eval_id"), "eval_id")
        source_id = _string(row.get("source_id"), "source_id")
        if eval_id in seen_eval_ids:
            raise JudgeAuditError(f"duplicate eval_id: {eval_id}")
        seen_eval_ids.add(eval_id)
        task_kind = _string(row.get("task_kind"), "task_kind")
        if task_kind == FACT_TASK:
            if source_id not in facts:
                raise JudgeAuditError(f"unresolved fact source: {source_id}")
            evidence = _fact_evidence(row, facts[source_id], eval_contract)
        elif task_kind == FEATURE_TASK:
            if source_id not in features:
                raise JudgeAuditError(f"unresolved feature source: {source_id}")
            evidence = _feature_evidence(
                row,
                features[source_id],
                probe_results,
                eval_contract,
            )
        else:
            raise JudgeAuditError(f"unsupported task kind: {task_kind}")
        bundle.append(
            {
                "schema": EVIDENCE_SCHEMA,
                "audit_index": expected_index,
                "eval_id": eval_id,
                "source_id": source_id,
                "task_kind": task_kind,
                "prompt": row["prompt"],
                "proposed_gold": row["proposed_gold"],
                "root_check": "pass",
                "evidence": evidence,
            }
        )
    return bundle


def _reviewed_ids_hash(bundle: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_json([row["eval_id"] for row in bundle])


def apply_attestation(
    *,
    bundle: Sequence[Mapping[str, Any]],
    attestation: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if attestation.get("schema") != ATTESTATION_SCHEMA:
        raise JudgeAuditError("unsupported judge attestation schema")
    if attestation.get("contract_id") != contract["contract_id"]:
        raise JudgeAuditError("judge attestation contract mismatch")
    bundle_bytes = _jsonl_bytes(bundle)
    bundle_hash = _sha256_bytes(bundle_bytes)
    if attestation.get("evidence_bundle_sha256") != bundle_hash:
        raise JudgeAuditError("judge attestation evidence hash mismatch")
    if attestation.get("reviewed_eval_ids_sha256") != _reviewed_ids_hash(bundle):
        raise JudgeAuditError("judge attestation reviewed-ID hash mismatch")
    if attestation.get("reviewed_items") != len(bundle):
        raise JudgeAuditError("judge attestation item count mismatch")

    judge_run = _mapping(attestation.get("judge_run"), "judge_run")
    judge_run_id = _string(judge_run.get("judge_run_id"), "judge_run_id")
    expected_identity = contract["judge_identity"]
    for field in ("kind", "runtime", "model_family", "target_model"):
        if judge_run.get(field) != expected_identity[field]:
            raise JudgeAuditError(f"judge identity mismatch: {field}")

    defaults = _mapping(
        attestation.get("default_dimensions"),
        "default_dimensions",
    )
    if set(defaults) != set(DIMENSIONS):
        raise JudgeAuditError("attestation must decide every dimension")
    for dimension in DIMENSIONS:
        decision = _mapping(defaults[dimension], dimension)
        if decision.get("verdict") not in VERDICTS:
            raise JudgeAuditError(f"invalid verdict for {dimension}")
        _string(decision.get("reason"), f"{dimension}.reason")

    exceptions_raw = attestation.get("exceptions")
    if not isinstance(exceptions_raw, list):
        raise JudgeAuditError("exceptions must be a list")
    valid_eval_ids = {str(row["eval_id"]) for row in bundle}
    exceptions: dict[tuple[str, str], Mapping[str, Any]] = {}
    for exception in exceptions_raw:
        exception = _mapping(exception, "exception")
        eval_id = _string(exception.get("eval_id"), "exception.eval_id")
        dimension = _string(exception.get("dimension"), "exception.dimension")
        if eval_id not in valid_eval_ids or dimension not in DIMENSIONS:
            raise JudgeAuditError("exception target is not in the bundle")
        key = (eval_id, dimension)
        if key in exceptions:
            raise JudgeAuditError("duplicate attestation exception")
        if exception.get("verdict") not in VERDICTS:
            raise JudgeAuditError("invalid exception verdict")
        _string(exception.get("reason"), "exception.reason")
        exceptions[key] = exception

    verdicts: list[dict[str, Any]] = []
    counts: dict[str, Counter[str]] = {
        dimension: Counter() for dimension in DIMENSIONS
    }
    for row in bundle:
        eval_id = str(row["eval_id"])
        evidence_ids = [
            str(item["evidence_id"]) for item in row["evidence"]
        ]
        dimensions: dict[str, dict[str, Any]] = {}
        for dimension in DIMENSIONS:
            decision = exceptions.get((eval_id, dimension), defaults[dimension])
            verdict = str(decision["verdict"])
            counts[dimension][verdict] += 1
            dimensions[dimension] = {
                "verdict": verdict,
                "reason": str(decision["reason"]),
                "evidence_ids": evidence_ids,
            }
        verdicts.append(
            {
                "schema": VERDICT_SCHEMA,
                "contract_id": contract["contract_id"],
                "judge_run_id": judge_run_id,
                "audit_index": row["audit_index"],
                "eval_id": eval_id,
                "source_id": row["source_id"],
                "evidence_row_sha256": _sha256_json(row),
                "dimensions": dimensions,
            }
        )

    threshold = float(
        contract["judge_output"]["minimum_pass_rate_per_dimension"]
    )
    rates = {
        dimension: counts[dimension]["pass"] / len(bundle)
        for dimension in DIMENSIONS
    }
    status = "pass" if all(rate >= threshold for rate in rates.values()) else "fail"
    summary = {
        "schema": SUMMARY_SCHEMA,
        "contract_id": contract["contract_id"],
        "judge_run": dict(judge_run),
        "status": status,
        "items": len(bundle),
        "root_checks": "pass",
        "evidence_bundle_sha256": bundle_hash,
        "reviewed_eval_ids_sha256": _reviewed_ids_hash(bundle),
        "dimension_counts": {
            dimension: dict(sorted(counts[dimension].items()))
            for dimension in DIMENSIONS
        },
        "dimension_pass_rates": rates,
        "minimum_pass_rate_per_dimension": threshold,
        "acceptance_accessed": contract["transition"] == "acceptance",
        "freeze_state": "unfrozen",
    }
    return verdicts, summary


def _file_hash(path: Path) -> dict[str, str]:
    return {
        "filename": path.name,
        "sha256": _sha256_bytes(path.read_bytes()),
    }


def write_outputs(
    *,
    bundle: Sequence[Mapping[str, Any]],
    verdicts: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    inputs: Mapping[str, Any],
    bundle_path: Path,
    verdicts_path: Path,
    summary_path: Path,
) -> None:
    bundle_bytes = _jsonl_bytes(bundle)
    verdict_bytes = _jsonl_bytes(verdicts)
    for path in (bundle_path, verdicts_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_bytes(bundle_bytes)
    verdicts_path.write_bytes(verdict_bytes)
    output_summary = {
        **dict(summary),
        "inputs": dict(inputs),
        "outputs": {
            "evidence_bundle": {
                "filename": bundle_path.name,
                "rows": len(bundle),
                "sha256": _sha256_bytes(bundle_bytes),
            },
            "verdicts": {
                "filename": verdicts_path.name,
                "rows": len(verdicts),
                "sha256": _sha256_bytes(verdict_bytes),
            },
        },
    }
    summary_path.write_text(
        json.dumps(output_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_probe_result(value: str) -> tuple[str, Path]:
    probe_id, separator, raw_path = value.partition("=")
    if not separator or not probe_id or not raw_path:
        raise argparse.ArgumentTypeError("probe result must be ID=/path/result.json")
    return probe_id, Path(raw_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an evidence-grounded LLM audit of EvalItems."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--eval-contract", type=Path, required=True)
    parser.add_argument("--audit-packet", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument(
        "--feature-delta",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--probe-result",
        action="append",
        type=_parse_probe_result,
        default=[],
    )
    parser.add_argument("--attestation", type=Path)
    parser.add_argument("--evidence-out", type=Path, required=True)
    parser.add_argument("--verdicts-out", type=Path)
    parser.add_argument("--summary", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = load_yaml(args.contract)
    eval_contract = load_yaml(args.eval_contract)
    audit_rows = load_jsonl(args.audit_packet)
    facts = load_jsonl(args.facts)
    feature_paths = sorted(args.feature_delta, key=lambda path: str(path))
    features = [load_yaml(path) for path in feature_paths]
    probe_paths = dict(args.probe_result)
    if len(probe_paths) != len(args.probe_result):
        raise JudgeAuditError("duplicate probe result ID")
    probes = {
        probe_id: load_json(path)
        for probe_id, path in sorted(probe_paths.items())
    }
    bundle = build_evidence_bundle(
        audit_rows=audit_rows,
        fact_records=facts,
        feature_records=features,
        probe_results=probes,
        contract=contract,
        eval_contract=eval_contract,
    )
    if args.attestation is None:
        if args.verdicts_out is not None or args.summary is not None:
            raise JudgeAuditError(
                "--verdicts-out and --summary require --attestation"
            )
        bundle_bytes = _jsonl_bytes(bundle)
        args.evidence_out.parent.mkdir(parents=True, exist_ok=True)
        args.evidence_out.write_bytes(bundle_bytes)
        print(
            json.dumps(
                {
                    "schema": EVIDENCE_SCHEMA,
                    "rows": len(bundle),
                    "sha256": _sha256_bytes(bundle_bytes),
                    "reviewed_eval_ids_sha256": _reviewed_ids_hash(bundle),
                    "root_checks": "pass",
                    "acceptance_accessed": (
                        contract["transition"] == "acceptance"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.verdicts_out is None or args.summary is None:
        raise JudgeAuditError(
            "--verdicts-out and --summary are required with --attestation"
        )
    attestation = load_yaml(args.attestation)
    verdicts, summary = apply_attestation(
        bundle=bundle,
        attestation=attestation,
        contract=contract,
    )
    inputs = {
        "contract": _file_hash(args.contract),
        "eval_contract": _file_hash(args.eval_contract),
        "audit_packet": _file_hash(args.audit_packet),
        "facts": _file_hash(args.facts),
        "features": [_file_hash(path) for path in feature_paths],
        "probe_results": [
            {"probe_id": probe_id, **_file_hash(path)}
            for probe_id, path in sorted(probe_paths.items())
        ],
        "attestation": _file_hash(args.attestation),
    }
    write_outputs(
        bundle=bundle,
        verdicts=verdicts,
        summary=summary,
        inputs=inputs,
        bundle_path=args.evidence_out,
        verdicts_path=args.verdicts_out,
        summary_path=args.summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
