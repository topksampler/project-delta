from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2.baseline_protocol import (
    EXPECTED_MODEL_REVISION,
    REQUEST_SCHEMA,
    BaselineProtocolError,
    load_yaml,
)
from experiments.delta_v2.oracle_source_context import (
    PROTOCOL_PATH as ORACLE_PROTOCOL_PATH,
    _message,
    audit_oracle_requests,
    validate_oracle_protocol,
)


PROTOCOL_SCHEMA = "delta.source_comparison_control_protocol.v1"
PROTOCOL_ID = "delta-v2-c3-source-comparison-qwen35-08b-v1"
PROTOCOL_PATH = Path("experiments/delta_v2/source_comparison_protocol.yaml")
ORACLE_PROTOCOL_SHA256 = (
    "c16136937d87bb6b51269ae39f6f4ab5f4e37fa6ead48aa49a0d9239294c322a"
)
C2_REQUEST_SHA256 = (
    "b4b39f0e325b502d3cf8fb7d5e7fb9329b92ce1301d26e4f2758619c91a98023"
)
C2_METRICS_SHA256 = (
    "3da05d8420a5dc32ee146c05293edd5517ce49823aee17be9287dd83e7356bc6"
)
C2_SAMPLES_SHA256 = (
    "b292ac14f618f29bf073067cb5405d8cbaf338104fa3d8d230c304ebecb740e3"
)
FACT_PROJECTION = (
    "before_present",
    "after_present",
    "canonical_equal",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BaselineProtocolError(f"{field} must be a mapping")
    return value


def validate_source_comparison_protocol(
    protocol: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[
    Mapping[str, Any],
    list[Mapping[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, bool],
]:
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("experiment_id") != "delta_v2"
        or protocol.get("condition_id") != "c3_source_comparison"
        or protocol.get("state") != "preregistered-for-execution"
    ):
        raise BaselineProtocolError("source-comparison identity changed")
    trigger = _mapping(protocol.get("decision_trigger"), "decision_trigger")
    if trigger != {
        "run_id": "delta-v2-c2-oracle-source-context-qwen35-08b-modal-v1",
        "metrics_sha256": C2_METRICS_SHA256,
        "samples_sha256": C2_SAMPLES_SHA256,
        "fact_response": "changed",
        "fact_response_count": 42,
        "feature_correct": True,
        "rationale": trigger.get("rationale"),
    } or not isinstance(trigger.get("rationale"), str):
        raise BaselineProtocolError("source-comparison trigger changed")

    oracle = _mapping(protocol.get("oracle_contract"), "oracle_contract")
    if oracle != {
        "path": str(ORACLE_PROTOCOL_PATH),
        "sha256": ORACLE_PROTOCOL_SHA256,
        "request_sha256": C2_REQUEST_SHA256,
        "truth_sources_unchanged": True,
        "acceptance_items_unchanged": True,
    }:
        raise BaselineProtocolError("source-comparison oracle binding changed")
    oracle_path = repo_root / ORACLE_PROTOCOL_PATH
    if (
        not oracle_path.is_file()
        or _sha256(oracle_path) != ORACLE_PROTOCOL_SHA256
    ):
        raise BaselineProtocolError("source-comparison oracle bytes changed")
    oracle_protocol = load_yaml(oracle_path)
    (
        baseline_protocol,
        items,
        oracle_evidence,
        feature_evidence,
    ) = validate_oracle_protocol(oracle_protocol, repo_root=repo_root)

    representation = _mapping(
        protocol.get("representation"),
        "representation",
    )
    if (
        representation.get("role")
        != "deterministic-source-comparison-control"
        or representation.get("fact_projection") != list(FACT_PROJECTION)
        or set(representation.get("excluded") or [])
        != {
            "canonical_sha256_values",
            "semantic_key",
            "gold",
            "status",
            "provenance",
            "scorer",
        }
        or representation.get("feature_projection")
        != "inherited-unchanged-from-c2"
    ):
        raise BaselineProtocolError(
            "source-comparison representation changed"
        )
    projected: dict[str, dict[str, Any]] = {}
    for eval_id, evidence in oracle_evidence.items():
        before = bool(evidence["before_present"])
        after = bool(evidence["after_present"])
        projected[eval_id] = {
            "before_present": before,
            "after_present": after,
            "canonical_equal": bool(
                before
                and after
                and evidence["before_canonical_sha256"]
                == evidence["after_canonical_sha256"]
            ),
        }

    adaptation = _mapping(protocol.get("adaptation"), "adaptation")
    if (
        adaptation.get("method")
        != "deterministic-source-comparison-features-v1"
        or adaptation.get("role") != "upper-bound-control"
        or adaptation.get("production_retrieval_claim") != "forbidden"
        or adaptation.get("weight_updates") is not False
        or adaptation.get("feature_instruction")
        != oracle_protocol["adaptation"]["feature_instruction"]
        or not isinstance(adaptation.get("fact_instruction"), str)
    ):
        raise BaselineProtocolError(
            "source-comparison adaptation changed"
        )
    if protocol.get("generation") != baseline_protocol.get("generation"):
        raise BaselineProtocolError("source-comparison generation changed")
    reporting = _mapping(
        protocol.get("scoring_and_reporting"),
        "scoring_and_reporting",
    )
    if (
        reporting.get("inherited_from")
        != "delta-v2-c0-base-qwen35-08b-v1"
        or reporting.get("target_scorers_unchanged") is not True
        or reporting.get("pooled_overall_accuracy") != "forbidden"
        or reporting.get("required_strata")
        != {
            "atomic_fact_added": 21,
            "atomic_fact_stable": 21,
            "feature_added": 1,
        }
        or reporting.get("feature_signal") != "report-separately"
    ):
        raise BaselineProtocolError("source-comparison reporting changed")
    boundary = _mapping(
        protocol.get("execution_boundary"),
        "execution_boundary",
    )
    if (
        boundary.get("expected_request_sha256")
        != "080337920a74694131463385915ddc4237ff25071ac13eff2deb9fe746dbcab1"
        or boundary.get("model_invocations_completed") != 0
        or boundary.get("target_model_results_exist") is not False
        or boundary.get("modal_authorized") is not True
        or boundary.get("training") != "forbidden"
        or boundary.get("fine_tuning") != "forbidden"
    ):
        raise BaselineProtocolError(
            "source-comparison execution boundary changed"
        )
    return baseline_protocol, items, projected, feature_evidence


def scoring_protocol(
    baseline_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    adapted = dict(baseline_protocol)
    adapted["protocol_id"] = PROTOCOL_ID
    return adapted


def _request_id(eval_id: str, repeat_index: int) -> str:
    identity = "\0".join(
        (REQUEST_SCHEMA, PROTOCOL_ID, eval_id, str(repeat_index))
    )
    return "target-request:" + hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()


def build_source_comparison_requests(
    protocol: Mapping[str, Any],
    *,
    baseline_protocol: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    evidence_by_eval: Mapping[str, Mapping[str, Any]],
    feature_evidence: Mapping[str, bool],
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    generation = protocol["generation"]
    for item in items:
        eval_id = str(item["eval_id"])
        if item["source_kind"] == "atomic_fact_delta":
            instruction = str(protocol["adaptation"]["fact_instruction"])
            evidence = evidence_by_eval[eval_id]
        else:
            instruction = str(protocol["adaptation"]["feature_instruction"])
            evidence = feature_evidence
        content = _message(
            instruction=instruction,
            evidence=evidence,
            prompt=str(item["prompt"]),
        ).replace("SOURCE_EVIDENCE=", "SOURCE_COMPARISON=", 1)
        for repeat_index in range(1, int(generation["repeats"]) + 1):
            requests.append(
                {
                    "schema": REQUEST_SCHEMA,
                    "request_id": _request_id(eval_id, repeat_index),
                    "protocol_id": PROTOCOL_ID,
                    "eval_id": eval_id,
                    "repeat_index": repeat_index,
                    "model": {
                        "repository": baseline_protocol["target_model"][
                            "repository"
                        ],
                        "revision": EXPECTED_MODEL_REVISION,
                    },
                    "model_input": {
                        "messages": [{"role": "user", "content": content}],
                        "add_generation_prompt": True,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    "generation": {
                        "do_sample": False,
                        "max_new_tokens": generation["max_new_tokens"],
                        "num_return_sequences": 1,
                        "seed": generation["seed"],
                    },
                }
            )
    return requests


def audit_source_comparison_requests(
    requests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rewritten = []
    for request in requests:
        cloned = dict(request)
        model_input = dict(request["model_input"])
        messages = [dict(message) for message in model_input["messages"]]
        messages[0]["content"] = messages[0]["content"].replace(
            "SOURCE_COMPARISON=",
            "SOURCE_EVIDENCE=",
            1,
        )
        model_input["messages"] = messages
        cloned["model_input"] = model_input
        rewritten.append(cloned)
    base_audit = audit_oracle_requests(rewritten)
    serialized = json.dumps(requests, sort_keys=True)
    for forbidden in (
        "before_canonical_sha256",
        "after_canonical_sha256",
        "semantic_key",
        '"gold"',
        '"status"',
        '"provenance"',
        '"scorer"',
    ):
        if forbidden in serialized:
            raise BaselineProtocolError(
                "excluded source field entered comparison request"
            )
    return {
        "schema": "delta.source_comparison_request_audit.v1",
        "protocol_id": PROTOCOL_ID,
        "requests": len(requests),
        "acceptance_items": base_audit["acceptance_items"],
        "fact_projection": list(FACT_PROJECTION),
        "feature_projection": "inherited-unchanged-from-c2",
        "gold_label_visible": False,
        "status_field_visible": False,
        "canonical_hashes_visible": False,
        "production_retrieval_claim": False,
        "status": "pass",
    }
