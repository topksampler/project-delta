from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2.baseline_protocol import (
    EXPECTED_DATASET_SHA256,
    EXPECTED_MODEL_REVISION,
    EXPECTED_PROTOCOL_ID,
    REQUEST_SCHEMA,
    BaselineProtocolError,
    load_jsonl,
    load_yaml,
    validate_protocol,
)


PROTOCOL_SCHEMA = "delta.oracle_source_context_protocol.v1"
PROTOCOL_ID = "delta-v2-c2-oracle-source-context-qwen35-08b-v1"
PROTOCOL_PATH = Path(
    "experiments/delta_v2/oracle_source_context_protocol.yaml"
)
BASELINE_PROTOCOL_PATH = Path(
    "experiments/delta_v2/target_baseline_protocol.yaml"
)
BASELINE_PROTOCOL_SHA256 = (
    "e26985e78b94365123cb8f4c0d1cf34d377d6b488f70f7ca618946336eb9b98d"
)
FACT_DELTAS_PATH = Path(
    "data/experiments/delta_v2/acceptance_attempt_2/facts/"
    "atomic_fact_deltas_acceptance.jsonl"
)
FACT_DELTAS_SHA256 = (
    "52f5f8bfa247550ef50277657900725db6ed8cc99bc3fe21d8a112629ffec75a"
)
FEATURE_PROBE_PATH = Path(
    "data/experiments/delta_v2/acceptance_attempt_2/probes/"
    "endpoint_plugins_framework.result.json"
)
FEATURE_PROBE_SHA256 = (
    "3352044178ac07f2988d0fb70ce66536ab485755b788a1ed773f6293819921f5"
)
C1_METRICS_SHA256 = (
    "6dab2fc6870bb14a8a18aa9a3ea98816d751e4832e8560ce4b575dff1ae6f934"
)
C1_SAMPLES_SHA256 = (
    "6b02d5342e57626d70e8a81dfe20adc8547c5ef4f28db3b3e693bac0005de58d"
)
FACT_PROJECTION = (
    "semantic_key",
    "before_present",
    "before_canonical_sha256",
    "after_present",
    "after_canonical_sha256",
)
FEATURE_PROJECTION = (
    "default_off",
    "allowlisted_task_match_loads",
    "required_task_miss_skips",
    "factory_failure_isolated",
    "route_phase_attaches",
    "attach_router_hook",
    "state_phase_initializes",
    "async_init_state_hook",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BaselineProtocolError(f"{field} must be a mapping")
    return value


def _canonical_hash(value: Any) -> str | None:
    if value is None:
        return None
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fact_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    before = record.get("value_before")
    after = record.get("value_after")
    return {
        "semantic_key": record["semantic_key"],
        "before_present": before is not None,
        "before_canonical_sha256": _canonical_hash(before),
        "after_present": after is not None,
        "after_canonical_sha256": _canonical_hash(after),
    }


def _feature_evidence(probe: Mapping[str, Any]) -> dict[str, bool]:
    cases = probe.get("cases")
    if not isinstance(cases, list) or len(cases) != 1:
        raise BaselineProtocolError("feature probe case count changed")
    observed = _mapping(cases[0].get("observed_after"), "observed_after")
    evidence = {key: observed.get(key) for key in FEATURE_PROJECTION}
    if any(value is not True for value in evidence.values()):
        raise BaselineProtocolError("feature evidence projection changed")
    return dict(evidence)


def _status_from_evidence(evidence: Mapping[str, Any]) -> str:
    before = evidence["before_present"]
    after = evidence["after_present"]
    if before and after:
        return (
            "stable"
            if evidence["before_canonical_sha256"]
            == evidence["after_canonical_sha256"]
            else "changed"
        )
    if not before and after:
        return "added"
    if before and not after:
        return "removed"
    raise BaselineProtocolError("fact absent in both source snapshots")


def validate_oracle_protocol(
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
        or protocol.get("condition_id") != "c2_oracle_source_context"
        or protocol.get("state") != "preregistered-for-execution"
    ):
        raise BaselineProtocolError("oracle source-context identity changed")

    trigger = _mapping(protocol.get("decision_trigger"), "decision_trigger")
    if (
        trigger.get("run_id")
        != "delta-v2-c1-task-calibration-qwen35-08b-modal-v1"
        or trigger.get("metrics_sha256") != C1_METRICS_SHA256
        or trigger.get("samples_sha256") != C1_SAMPLES_SHA256
        or trigger.get("state")
        != "baseline-incapable-of-localizing-drift"
        or trigger.get("observed_fact_responses")
        != {"added": 14, "changed": 26, "removed": 2, "stable": 0}
        or trigger.get("feature_correct") is not True
    ):
        raise BaselineProtocolError("oracle decision trigger changed")

    baseline = _mapping(protocol.get("baseline_contract"), "baseline_contract")
    if baseline != {
        "path": str(BASELINE_PROTOCOL_PATH),
        "sha256": BASELINE_PROTOCOL_SHA256,
        "acceptance_eval_items_sha256": EXPECTED_DATASET_SHA256,
        "acceptance_rows": 43,
    }:
        raise BaselineProtocolError("oracle baseline binding changed")
    baseline_path = repo_root / BASELINE_PROTOCOL_PATH
    if (
        not baseline_path.is_file()
        or _sha256(baseline_path) != BASELINE_PROTOCOL_SHA256
    ):
        raise BaselineProtocolError("oracle baseline bytes changed")
    baseline_protocol = load_yaml(baseline_path)
    items = list(validate_protocol(baseline_protocol, repo_root=repo_root))

    truth = _mapping(protocol.get("truth_sources"), "truth_sources")
    facts_binding = _mapping(
        truth.get("atomic_fact_deltas"),
        "atomic_fact_deltas",
    )
    probe_binding = _mapping(truth.get("feature_probe"), "feature_probe")
    if (
        facts_binding.get("path") != str(FACT_DELTAS_PATH)
        or facts_binding.get("sha256") != FACT_DELTAS_SHA256
        or facts_binding.get("routing") != "exact-evalitem-source-id"
        or facts_binding.get("model_visible_projection")
        != list(FACT_PROJECTION)
        or probe_binding.get("path") != str(FEATURE_PROBE_PATH)
        or probe_binding.get("sha256") != FEATURE_PROBE_SHA256
        or probe_binding.get("probe_id")
        != "behavior-probe:vllm-endpoint-plugins-framework-v1"
        or probe_binding.get("routing") != "sole-verified-feature-item"
        or probe_binding.get("model_visible_projection")
        != list(FEATURE_PROJECTION)
        or set(truth.get("forbidden_model_inputs") or [])
        != {
            "gold",
            "status",
            "provenance",
            "scorer",
            "fact_id",
            "verifier",
            "expected_probe_values",
            "source_paths",
            "development_items",
        }
    ):
        raise BaselineProtocolError("oracle truth-source binding changed")

    facts_path = repo_root / FACT_DELTAS_PATH
    probe_path = repo_root / FEATURE_PROBE_PATH
    if (
        not facts_path.is_file()
        or _sha256(facts_path) != FACT_DELTAS_SHA256
        or not probe_path.is_file()
        or _sha256(probe_path) != FEATURE_PROBE_SHA256
    ):
        raise BaselineProtocolError("oracle truth-source bytes changed")
    records = {
        str(row["fact_id"]): row for row in load_jsonl(facts_path)
    }
    evidence_by_eval: dict[str, dict[str, Any]] = {}
    feature_items = 0
    for item in items:
        if item["source_kind"] == "atomic_fact_delta":
            source_id = str(item["source_id"])
            if source_id not in records:
                raise BaselineProtocolError("oracle fact routing failed")
            evidence = _fact_evidence(records[source_id])
            gold = str(item["gold"]["status"])
            if _status_from_evidence(evidence) != gold:
                raise BaselineProtocolError(
                    "oracle fact evidence disagrees with frozen gold"
                )
            evidence_by_eval[str(item["eval_id"])] = evidence
        else:
            feature_items += 1
    if len(evidence_by_eval) != 42 or feature_items != 1:
        raise BaselineProtocolError("oracle item routing coverage changed")
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    feature_evidence = _feature_evidence(_mapping(probe, "feature_probe"))

    adaptation = _mapping(protocol.get("adaptation"), "adaptation")
    if (
        adaptation.get("method")
        != "oracle-routed-frozen-source-context-v1"
        or adaptation.get("role") != "upper-bound-control"
        or adaptation.get("production_retrieval_claim") != "forbidden"
        or adaptation.get("weight_updates") is not False
    ):
        raise BaselineProtocolError("oracle adaptation method changed")
    for field in ("fact_instruction", "feature_instruction"):
        if not isinstance(adaptation.get(field), str) or not adaptation[field]:
            raise BaselineProtocolError(f"adaptation.{field} changed")
    if protocol.get("generation") != baseline_protocol.get("generation"):
        raise BaselineProtocolError("oracle generation changed")
    reporting = _mapping(
        protocol.get("scoring_and_reporting"),
        "scoring_and_reporting",
    )
    if (
        reporting.get("inherited_from") != EXPECTED_PROTOCOL_ID
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
        raise BaselineProtocolError("oracle reporting changed")
    boundary = _mapping(
        protocol.get("execution_boundary"),
        "execution_boundary",
    )
    if (
        boundary.get("expected_request_sha256")
        != "b4b39f0e325b502d3cf8fb7d5e7fb9329b92ce1301d26e4f2758619c91a98023"
        or boundary.get("model_invocations_completed") != 0
        or boundary.get("target_model_results_exist") is not False
        or boundary.get("modal_authorized") is not True
        or boundary.get("training") != "forbidden"
        or boundary.get("fine_tuning") != "forbidden"
    ):
        raise BaselineProtocolError("oracle execution boundary changed")
    return baseline_protocol, items, evidence_by_eval, feature_evidence


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


def _message(
    *,
    instruction: str,
    evidence: Mapping[str, Any],
    prompt: str,
) -> str:
    evidence_json = json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        f"{instruction}\n\n"
        f"SOURCE_EVIDENCE={evidence_json}\n\n"
        f"FINAL_QUESTION={prompt}"
    )


def build_oracle_requests(
    protocol: Mapping[str, Any],
    *,
    baseline_protocol: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    evidence_by_eval: Mapping[str, Mapping[str, Any]],
    feature_evidence: Mapping[str, bool],
) -> list[dict[str, Any]]:
    generation = protocol["generation"]
    adaptation = protocol["adaptation"]
    requests: list[dict[str, Any]] = []
    for item in items:
        eval_id = str(item["eval_id"])
        if item["source_kind"] == "atomic_fact_delta":
            instruction = str(adaptation["fact_instruction"])
            evidence = evidence_by_eval[eval_id]
        else:
            instruction = str(adaptation["feature_instruction"])
            evidence = feature_evidence
        content = _message(
            instruction=instruction,
            evidence=evidence,
            prompt=str(item["prompt"]),
        )
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


def audit_oracle_requests(
    requests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    forbidden = (
        '"gold"',
        '"status"',
        '"provenance"',
        '"scorer"',
        '"fact_id"',
        '"verifier"',
        '"expected_after"',
        '"expected_before"',
    )
    for request in requests:
        messages = request["model_input"]["messages"]
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise BaselineProtocolError("oracle message shape changed")
        content = str(messages[0]["content"])
        if "SOURCE_EVIDENCE=" not in content or "FINAL_QUESTION=" not in content:
            raise BaselineProtocolError("oracle message sections changed")
        if any(token in content for token in forbidden):
            raise BaselineProtocolError(
                "forbidden truth field entered oracle request"
            )
    return {
        "schema": "delta.oracle_source_context_request_audit.v1",
        "protocol_id": PROTOCOL_ID,
        "requests": len(requests),
        "acceptance_items": len(requests) // 2,
        "routing": "oracle-exact-source-id",
        "fact_evidence_projection": list(FACT_PROJECTION),
        "feature_evidence_projection": list(FEATURE_PROJECTION),
        "gold_label_visible": False,
        "status_field_visible": False,
        "provenance_visible": False,
        "production_retrieval_claim": False,
        "status": "pass",
    }
