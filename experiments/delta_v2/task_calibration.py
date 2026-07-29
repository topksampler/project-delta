from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2.baseline_protocol import (
    EXPECTED_DATASET_SHA256,
    EXPECTED_ENVIRONMENT_ID,
    EXPECTED_MODEL_REVISION,
    EXPECTED_PROTOCOL_ID,
    REQUEST_SCHEMA,
    BaselineProtocolError,
    load_jsonl,
    load_yaml,
    validate_protocol,
)


PROTOCOL_SCHEMA = "delta.prompt_adaptation_protocol.v1"
PROTOCOL_ID = "delta-v2-c1-task-calibration-qwen35-08b-v1"
PROTOCOL_PATH = Path("experiments/delta_v2/task_calibration_protocol.yaml")
BASELINE_PROTOCOL_PATH = Path(
    "experiments/delta_v2/target_baseline_protocol.yaml"
)
BASELINE_PROTOCOL_SHA256 = (
    "e26985e78b94365123cb8f4c0d1cf34d377d6b488f70f7ca618946336eb9b98d"
)
DEVELOPMENT_PATH = Path(
    "data/experiments/delta_v2/eval/development_eval_items.jsonl"
)
DEVELOPMENT_SHA256 = (
    "db169943888ec2258d6b43638a07e5c4c85998a5a50524eec3709258096c3059"
)
FACT_LABEL_ORDER = ("stable", "added", "removed", "changed")
FACT_EXAMPLES_PER_LABEL = 2
FACT_EVAL_IDS = (
    "eval:19e5e309801b953eca3af7f870259051a02579ec2399ec0a1b0c65cf490c7181",
    "eval:1d046b9019fcf7880d59c58713e6aae726427b729760cc15b263f130df42909f",
    "eval:46436fffd50f44d08033c5e96479ae35a2087e3cc958ced27c3f8bf6f0dd21d8",
    "eval:5116c995a4cb171216aa18cff136d164f38632f8c8c987149e59c3c8d93439ce",
    "eval:00dfe4bc9446d781da22f9ac6ef955dcf8bd5519a61a67d9cb3df95e13fc0124",
    "eval:52cd98b563a45c038c3374c2443556efd8438817c3fa83eb0ce1219e65f966d0",
    "eval:0de08e8382d338e99a667e258cb216e9e07000611efab9f72d513f1d6c00d753",
    "eval:26d915f2d4a19eef10c2b054a81abafcf41c518c6a77d8424057b68cd932f8ce",
)
FEATURE_EVAL_ID = (
    "eval:379be8c847d38e071735dc3f8add0051a1693eb42294d39b8604f0d2b55cdaed"
)
BASELINE_METRICS_SHA256 = (
    "9a218d184b25f1c80356b6ef01a38f40b728b2783b10b8f382e833f1dfc8b981"
)
BASELINE_SAMPLES_SHA256 = (
    "605482c56c7482171670db72ea5d04975106d7575cb0fc4863444dab7e2e8696"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BaselineProtocolError(f"{field} must be a mapping")
    return value


def _selected_development_examples(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    by_id = {str(row.get("eval_id")): row for row in rows}
    expected_ids = set(FACT_EVAL_IDS) | {FEATURE_EVAL_ID}
    if not expected_ids.issubset(by_id):
        raise BaselineProtocolError("calibration example is missing")
    facts = [by_id[eval_id] for eval_id in FACT_EVAL_IDS]
    feature = by_id[FEATURE_EVAL_ID]
    if any(
        row.get("split") != "train"
        or row.get("source_kind") != "atomic_fact_delta"
        for row in facts
    ):
        raise BaselineProtocolError(
            "fact calibration example crossed the train boundary"
        )
    if (
        feature.get("split") != "train"
        or feature.get("source_kind") != "feature_delta"
    ):
        raise BaselineProtocolError(
            "feature calibration example crossed the train boundary"
        )
    labels = [
        str(_mapping(row.get("gold"), "development.gold").get("status"))
        for row in facts
    ]
    expected_labels = [
        label
        for label in FACT_LABEL_ORDER
        for _ in range(FACT_EXAMPLES_PER_LABEL)
    ]
    if labels != expected_labels:
        raise BaselineProtocolError("calibration label balance changed")
    return facts, feature


def validate_task_calibration_protocol(
    protocol: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[
    Mapping[str, Any],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    Mapping[str, Any],
]:
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("experiment_id") != "delta_v2"
        or protocol.get("condition_id") != "c1_task_calibration"
        or protocol.get("state") != "preregistered-for-execution"
    ):
        raise BaselineProtocolError("task-calibration identity changed")

    trigger = _mapping(protocol.get("decision_trigger"), "decision_trigger")
    if (
        trigger.get("run_id")
        != "delta-v2-c0-base-qwen35-08b-modal-v3"
        or trigger.get("metrics_sha256") != BASELINE_METRICS_SHA256
        or trigger.get("samples_sha256") != BASELINE_SAMPLES_SHA256
        or trigger.get("state")
        != "baseline-incapable-of-localizing-drift"
        or trigger.get("observed_fact_responses")
        != {"added": 33, "changed": 9, "removed": 0, "stable": 0}
    ):
        raise BaselineProtocolError("task-calibration trigger changed")

    baseline_binding = _mapping(
        protocol.get("baseline_contract"),
        "baseline_contract",
    )
    if baseline_binding != {
        "path": str(BASELINE_PROTOCOL_PATH),
        "sha256": BASELINE_PROTOCOL_SHA256,
        "acceptance_eval_items_sha256": EXPECTED_DATASET_SHA256,
        "acceptance_rows": 43,
    }:
        raise BaselineProtocolError("baseline contract binding changed")
    baseline_path = repo_root / BASELINE_PROTOCOL_PATH
    if (
        not baseline_path.is_file()
        or _sha256(baseline_path) != BASELINE_PROTOCOL_SHA256
    ):
        raise BaselineProtocolError("baseline protocol bytes changed")
    baseline_protocol = load_yaml(baseline_path)
    acceptance_items = list(
        validate_protocol(baseline_protocol, repo_root=repo_root)
    )

    development = _mapping(
        protocol.get("development_context"),
        "development_context",
    )
    selection = _mapping(development.get("selection"), "selection")
    forbidden = development.get("forbidden_model_inputs")
    if (
        development.get("path") != str(DEVELOPMENT_PATH)
        or development.get("sha256") != DEVELOPMENT_SHA256
        or development.get("source_transition")
        != "vllm-v0.22.0-to-v0.23.0"
        or development.get("allowed_split") != "train"
        or selection.get("algorithm")
        != "eval-id-ascending-first-two-per-label-v1"
        or selection.get("fact_label_order") != list(FACT_LABEL_ORDER)
        or selection.get("fact_examples_per_label")
        != FACT_EXAMPLES_PER_LABEL
        or selection.get("fact_eval_ids") != list(FACT_EVAL_IDS)
        or selection.get("feature_eval_id") != FEATURE_EVAL_ID
        or development.get("model_visible_fields")
        != [
            "selected_development_prompt",
            "selected_development_answer",
            "acceptance_prompt",
        ]
        or set(forbidden or [])
        != {
            "acceptance_gold",
            "acceptance_provenance",
            "acceptance_scorer",
            "acceptance_source_snapshot",
            "acceptance_atomic_fact_record",
            "acceptance_feature_delta_record",
            "acceptance_behavior_probe_result",
            "acceptance_judge_evidence",
            "development_dev_items",
        }
    ):
        raise BaselineProtocolError("development context boundary changed")
    development_path = repo_root / DEVELOPMENT_PATH
    if (
        not development_path.is_file()
        or _sha256(development_path) != DEVELOPMENT_SHA256
    ):
        raise BaselineProtocolError("development context bytes changed")
    development_rows = load_jsonl(development_path)
    fact_examples, feature_example = _selected_development_examples(
        development_rows
    )

    adaptation = _mapping(protocol.get("adaptation"), "adaptation")
    if (
        adaptation.get("method")
        != "development-only-in-context-task-calibration-v1"
        or adaptation.get("weight_updates") is not False
        or adaptation.get("retrieval") is not False
        or adaptation.get("source_evidence_injection") is not False
        or adaptation.get("target_rendering")
        != {"role": "user", "content": "acceptance-prompt-only"}
    ):
        raise BaselineProtocolError("task-calibration method changed")
    for field in (
        "fact_instruction",
        "fact_acknowledgement",
        "feature_instruction",
        "feature_acknowledgement",
    ):
        if not isinstance(adaptation.get(field), str) or not adaptation[field]:
            raise BaselineProtocolError(f"adaptation.{field} changed")

    if protocol.get("generation") != baseline_protocol.get("generation"):
        raise BaselineProtocolError("adapted generation settings changed")
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
        raise BaselineProtocolError("adapted scoring boundary changed")
    boundary = _mapping(
        protocol.get("execution_boundary"),
        "execution_boundary",
    )
    if boundary != {
        "expected_request_sha256": (
            "96e209a30ca9b723f8e28e2af9c233f2a48b277c913bbb5da5bc4e7019ffbe77"
        ),
        "model_invocations_completed": 0,
        "target_model_results_exist": False,
        "modal_authorized": True,
        "training": "forbidden",
        "fine_tuning": "forbidden",
    }:
        raise BaselineProtocolError("adapted execution boundary changed")
    return (
        baseline_protocol,
        acceptance_items,
        fact_examples,
        feature_example,
    )


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


def _fact_messages(
    protocol: Mapping[str, Any],
    *,
    examples: Sequence[Mapping[str, Any]],
    target_prompt: str,
) -> list[dict[str, str]]:
    adaptation = protocol["adaptation"]
    messages = [
        {"role": "user", "content": adaptation["fact_instruction"]},
        {
            "role": "assistant",
            "content": adaptation["fact_acknowledgement"],
        },
    ]
    for example in examples:
        messages.extend(
            (
                {"role": "user", "content": str(example["prompt"])},
                {
                    "role": "assistant",
                    "content": str(example["gold"]["status"]),
                },
            )
        )
    messages.append({"role": "user", "content": target_prompt})
    return messages


def _feature_messages(
    protocol: Mapping[str, Any],
    *,
    example: Mapping[str, Any],
    target_prompt: str,
) -> list[dict[str, str]]:
    adaptation = protocol["adaptation"]
    answer = json.dumps(
        example["gold"],
        sort_keys=True,
        separators=(",", ":"),
    )
    return [
        {"role": "user", "content": adaptation["feature_instruction"]},
        {
            "role": "assistant",
            "content": adaptation["feature_acknowledgement"],
        },
        {"role": "user", "content": str(example["prompt"])},
        {"role": "assistant", "content": answer},
        {"role": "user", "content": target_prompt},
    ]


def build_task_calibration_requests(
    protocol: Mapping[str, Any],
    *,
    baseline_protocol: Mapping[str, Any],
    acceptance_items: Sequence[Mapping[str, Any]],
    fact_examples: Sequence[Mapping[str, Any]],
    feature_example: Mapping[str, Any],
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    generation = protocol["generation"]
    for item in acceptance_items:
        eval_id = str(item["eval_id"])
        if item["source_kind"] == "atomic_fact_delta":
            messages = _fact_messages(
                protocol,
                examples=fact_examples,
                target_prompt=str(item["prompt"]),
            )
        else:
            messages = _feature_messages(
                protocol,
                example=feature_example,
                target_prompt=str(item["prompt"]),
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
                        "messages": messages,
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


def audit_request_context(
    *,
    requests: Sequence[Mapping[str, Any]],
    acceptance_items: Sequence[Mapping[str, Any]],
    fact_examples: Sequence[Mapping[str, Any]],
    feature_example: Mapping[str, Any],
) -> dict[str, Any]:
    acceptance_prompts = {
        str(item["eval_id"]): str(item["prompt"])
        for item in acceptance_items
    }
    allowed_example_prompts = {
        str(item["prompt"]) for item in fact_examples
    } | {str(feature_example["prompt"])}
    allowed_example_answers = {
        str(item["gold"]["status"]) for item in fact_examples
    } | {
        json.dumps(
            feature_example["gold"],
            sort_keys=True,
            separators=(",", ":"),
        )
    }
    assistant_contents: set[str] = set()
    for request in requests:
        eval_id = str(request["eval_id"])
        messages = request["model_input"]["messages"]
        if messages[-1] != {
            "role": "user",
            "content": acceptance_prompts[eval_id],
        }:
            raise BaselineProtocolError(
                "acceptance target was not the final user message"
            )
        for message in messages[2:-1]:
            if message["role"] == "user":
                if message["content"] not in allowed_example_prompts:
                    raise BaselineProtocolError(
                        "unselected development prompt entered context"
                    )
            else:
                assistant_contents.add(str(message["content"]))
                if message["content"] not in allowed_example_answers:
                    raise BaselineProtocolError(
                        "unselected development answer entered context"
                    )
    expected_assistant = Counter(
        str(item["gold"]["status"]) for item in fact_examples
    )
    return {
        "schema": "delta.prompt_adaptation_request_audit.v1",
        "protocol_id": PROTOCOL_ID,
        "requests": len(requests),
        "acceptance_items": len(acceptance_items),
        "development_fact_examples": len(fact_examples),
        "development_fact_label_counts": dict(sorted(expected_assistant.items())),
        "development_feature_examples": 1,
        "assistant_example_surfaces": len(assistant_contents),
        "acceptance_gold_visible": False,
        "acceptance_provenance_visible": False,
        "development_dev_items_visible": False,
        "status": "pass",
    }
