from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from experiments.delta_v2.build_eval import score_response


PROTOCOL_SCHEMA = "delta.target_model_baseline_protocol.v1"
REQUEST_SCHEMA = "delta.target_model_request.v1"
OUTPUT_SCHEMA = "delta.target_model_output.v1"
EVAL_SCHEMA = "delta.eval_item.v1"
EXPECTED_PROTOCOL_ID = "delta-v2-c0-base-qwen35-08b-v1"
EXPECTED_ENVIRONMENT_ID = "delta-v2-vllm-source-build-v2"
EXPECTED_ENVIRONMENT_SHA256 = (
    "8e12ebdb685eceb8f9b590c67c05c449d71cc4b8a95f7d2bf3c8f3b4064991c4"
)
EXPECTED_DATASET_SHA256 = (
    "1125a7a5429cf847d16475259ffeb6578b8d4841c81f2f01d20d4065f4c188cb"
)
EXPECTED_MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
EXPECTED_STRATA = {
    "atomic_fact_added": 21,
    "atomic_fact_stable": 21,
    "feature_added": 1,
}
FACT_LABELS = {"stable", "added", "removed", "changed"}


class BaselineProtocolError(ValueError):
    """The preregistered baseline or its offline evidence is invalid."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BaselineProtocolError(f"{field} must be a mapping")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise BaselineProtocolError(f"{field} must be a non-empty string")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise BaselineProtocolError(f"{field} must be a list of strings")
    if len(value) != len(set(value)):
        raise BaselineProtocolError(f"{field} must not contain duplicates")
    return value


def _relative_path(value: Any, field: str) -> Path:
    path = Path(_string(value, field))
    if path.is_absolute() or ".." in path.parts:
        raise BaselineProtocolError(f"{field} must stay within the repository")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BaselineProtocolError(f"cannot read YAML: {path}") from exc
    return _mapping(payload, str(path))


def load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BaselineProtocolError(f"cannot read JSONL: {path}") from exc
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BaselineProtocolError(
                f"invalid JSON at {path}:{line_number}"
            ) from exc
        rows.append(_mapping(payload, f"{path}:{line_number}"))
    return rows


def _stratum(item: Mapping[str, Any]) -> str:
    source_kind = item.get("source_kind")
    if source_kind == "atomic_fact_delta":
        status = _mapping(item.get("gold"), "item.gold").get("status")
        if status == "added":
            return "atomic_fact_added"
        if status == "stable":
            return "atomic_fact_stable"
        raise BaselineProtocolError(
            f"acceptance fact has unsupported status: {status}"
        )
    if (
        source_kind == "feature_delta"
        and _mapping(item.get("provenance"), "item.provenance").get("status")
        == "added"
    ):
        return "feature_added"
    raise BaselineProtocolError("acceptance item has unsupported source kind")


def _validate_items(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
) -> None:
    expected_rows = int(protocol["evaluation_input"]["rows"])
    if len(rows) != expected_rows:
        raise BaselineProtocolError(
            f"expected {expected_rows} EvalItems, found {len(rows)}"
        )
    eval_ids: list[str] = []
    source_ids: set[str] = set()
    strata: Counter[str] = Counter()
    for row in rows:
        if row.get("schema") != EVAL_SCHEMA:
            raise BaselineProtocolError("unexpected EvalItem schema")
        eval_id = _string(row.get("eval_id"), "eval_id")
        source_id = _string(row.get("source_id"), "source_id")
        if eval_id in eval_ids or source_id in source_ids:
            raise BaselineProtocolError("duplicate EvalItem identity")
        if row.get("split") != "eval":
            raise BaselineProtocolError("baseline may access eval items only")
        scorer = row.get("scorer")
        if scorer not in {"exact-enum-v1", "exact-structured-json-v1"}:
            raise BaselineProtocolError("unfrozen scorer entered baseline")
        _string(row.get("prompt"), "item.prompt")
        eval_ids.append(eval_id)
        source_ids.add(source_id)
        strata[_stratum(row)] += 1
    if eval_ids != sorted(eval_ids):
        raise BaselineProtocolError("EvalItems must be ordered by eval_id")
    if dict(sorted(strata.items())) != EXPECTED_STRATA:
        raise BaselineProtocolError("baseline strata changed")


def validate_protocol(
    protocol: Mapping[str, Any],
    *,
    repo_root: Path,
) -> list[Mapping[str, Any]]:
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID
        or protocol.get("experiment_id") != "delta_v2"
        or protocol.get("condition_id") != "c0_base"
        or protocol.get("state") != "preregistered-not-executed"
    ):
        raise BaselineProtocolError("baseline protocol identity changed")

    environment = _mapping(
        protocol.get("eval_environment"),
        "eval_environment",
    )
    environment_path = repo_root / _relative_path(
        environment.get("path"),
        "eval_environment.path",
    )
    if (
        environment.get("environment_id") != EXPECTED_ENVIRONMENT_ID
        or environment.get("sha256") != EXPECTED_ENVIRONMENT_SHA256
        or not environment_path.is_file()
        or _sha256(environment_path) != EXPECTED_ENVIRONMENT_SHA256
    ):
        raise BaselineProtocolError("frozen EvalEnvironment binding changed")
    environment_manifest = load_yaml(environment_path)
    environment_execution = _mapping(
        environment_manifest.get("execution"),
        "EvalEnvironment.execution",
    )
    if (
        environment_manifest.get("environment_id") != EXPECTED_ENVIRONMENT_ID
        or environment_manifest.get("freeze_state") != "frozen"
        or environment_execution.get("target_model_runs") != 0
        or environment_execution.get("training_runs") != 0
        or environment_execution.get("fine_tuning_runs") != 0
    ):
        raise BaselineProtocolError(
            "EvalEnvironment is not a zero-execution freeze"
        )

    model = _mapping(protocol.get("target_model"), "target_model")
    expected_model = {
        "repository": "Qwen/Qwen3.5-0.8B",
        "revision": EXPECTED_MODEL_REVISION,
        "training_stage": "post-trained",
        "role": "active-base-state",
        "adapter": "none",
        "quantization": "none",
        "parameter_class": "sub-1B",
        "temporal_relation": "checkpoint-predates-acceptance-transition",
    }
    if dict(model) != expected_model:
        raise BaselineProtocolError("target-model identity changed")

    runtime = _mapping(protocol.get("runtime"), "runtime")
    if (
        runtime.get("backend") != "transformers-generate-v1"
        or runtime.get("python") != "3.11.9"
        or runtime.get("packages")
        != {"torch": "2.10.0", "transformers": "5.14.1"}
        or runtime.get("device_policy") != "single-cuda-gpu"
        or runtime.get("dtype") != "bfloat16"
        or runtime.get("batch_size") != 1
        or runtime.get("deterministic_algorithms") != "required"
    ):
        raise BaselineProtocolError("runtime lock changed")

    evaluation_input = _mapping(
        protocol.get("evaluation_input"),
        "evaluation_input",
    )
    dataset_path = repo_root / _relative_path(
        evaluation_input.get("path"),
        "evaluation_input.path",
    )
    expected_forbidden = {
        "gold",
        "provenance",
        "scorer",
        "source_snapshot",
        "atomic_fact_record",
        "feature_delta_record",
        "behavior_probe_result",
        "judge_evidence",
        "development_train_items",
        "development_dev_items",
    }
    if (
        evaluation_input.get("sha256") != EXPECTED_DATASET_SHA256
        or not dataset_path.is_file()
        or _sha256(dataset_path) != EXPECTED_DATASET_SHA256
        or evaluation_input.get("rows") != 43
        or evaluation_input.get("split") != "eval"
        or evaluation_input.get("item_order") != "eval_id-ascending"
        or evaluation_input.get("model_visible_fields") != ["prompt"]
        or set(
            _string_list(
                evaluation_input.get("forbidden_model_inputs"),
                "forbidden_model_inputs",
            )
        )
        != expected_forbidden
    ):
        raise BaselineProtocolError("evaluation-input boundary changed")

    prompting = _mapping(protocol.get("prompting"), "prompting")
    if (
        prompting.get("interface") != "official-chat-template"
        or prompting.get("messages")
        != [{"role": "user", "content_from": "prompt"}]
        or prompting.get("add_generation_prompt") is not True
        or prompting.get("chat_template_kwargs") != {"enable_thinking": False}
        or prompting.get("system_prompt") != "none"
    ):
        raise BaselineProtocolError("prompt interface changed")

    generation = _mapping(protocol.get("generation"), "generation")
    if dict(generation) != {
        "method": "greedy",
        "do_sample": False,
        "temperature": "none",
        "max_new_tokens": 128,
        "num_return_sequences": 1,
        "repeats": 2,
        "seed": 20260730,
        "repeat_invariant": "byte-identical-raw-continuation",
    }:
        raise BaselineProtocolError("generation lock changed")

    scoring = _mapping(protocol.get("scoring"), "scoring")
    if (
        scoring.get("authority") != "frozen-deterministic-evalitem-scorer"
        or scoring.get("scorer_ids")
        != ["exact-enum-v1", "exact-structured-json-v1"]
        or scoring.get("input") != "raw-decoded-continuation"
        or scoring.get("invalid_or_unparseable") != "incorrect"
        or scoring.get("repair") != "forbidden"
        or scoring.get("llm_judge") != "forbidden"
    ):
        raise BaselineProtocolError("target scoring boundary changed")

    reporting = _mapping(protocol.get("reporting"), "reporting")
    localization = _mapping(
        reporting.get("drift_localization"),
        "drift_localization",
    )
    if (
        reporting.get("pooled_overall_accuracy") != "forbidden"
        or reporting.get("required_strata") != EXPECTED_STRATA
        or reporting.get("required_metrics")
        != ["exact_accuracy", "parseable_rate", "invalid_response_count"]
        or localization
        != {
            "stable_null_accuracy": 0.25,
            "stable_capability_test": "one-sided-exact-binomial",
            "added_vs_stable_test": "one-sided-fisher-exact",
            "alpha": 0.05,
            "feature_signal": "report-separately",
        }
    ):
        raise BaselineProtocolError("baseline reporting contract changed")

    boundary = _mapping(
        protocol.get("execution_boundary"),
        "execution_boundary",
    )
    expected_actions = {
        "model-download",
        "local-model-evaluation",
        "modal-job",
        "lambda-job",
        "b2-write",
        "training",
        "fine-tuning",
    }
    if (
        boundary.get("model_invocations_completed") != 0
        or boundary.get("target_model_results_exist") is not False
        or boundary.get("run_config_state") != "absent-until-owner-review"
        or boundary.get("dispatch_authorized_by_this_file") is not False
        or set(
            _string_list(
                boundary.get("prohibited_actions"),
                "prohibited_actions",
            )
        )
        != expected_actions
    ):
        raise BaselineProtocolError("zero-execution boundary changed")

    rows = load_jsonl(dataset_path)
    _validate_items(rows, protocol=protocol)
    return rows


def _request_id(protocol_id: str, eval_id: str, repeat_index: int) -> str:
    identity = "\0".join(
        (
            REQUEST_SCHEMA,
            protocol_id,
            eval_id,
            str(repeat_index),
        )
    )
    return "target-request:" + hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()


def build_requests(
    protocol: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    generation = protocol["generation"]
    rows: list[dict[str, Any]] = []
    for item in items:
        eval_id = str(item["eval_id"])
        for repeat_index in range(1, int(generation["repeats"]) + 1):
            rows.append(
                {
                    "schema": REQUEST_SCHEMA,
                    "request_id": _request_id(
                        str(protocol["protocol_id"]),
                        eval_id,
                        repeat_index,
                    ),
                    "protocol_id": protocol["protocol_id"],
                    "eval_id": eval_id,
                    "repeat_index": repeat_index,
                    "model": {
                        "repository": protocol["target_model"]["repository"],
                        "revision": protocol["target_model"]["revision"],
                    },
                    "model_input": {
                        "messages": [
                            {
                                "role": "user",
                                "content": item["prompt"],
                            }
                        ],
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
    return rows


def serialize_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(dict(row), sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _is_parseable(item: Mapping[str, Any], response: str) -> bool:
    if item.get("scorer") == "exact-enum-v1":
        return response.strip().lower() in FACT_LABELS
    if item.get("scorer") == "exact-structured-json-v1":
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return False
        return isinstance(parsed, Mapping)
    raise BaselineProtocolError("unsupported scorer")


def _exact_binomial_upper_tail(
    *,
    correct: int,
    total: int,
    null_accuracy: float,
) -> float:
    return sum(
        math.comb(total, observed)
        * null_accuracy**observed
        * (1.0 - null_accuracy) ** (total - observed)
        for observed in range(correct, total + 1)
    )


def _one_sided_fisher_less(
    *,
    added_correct: int,
    stable_correct: int,
    group_size: int,
) -> float:
    total_correct = added_correct + stable_correct
    lower = max(0, total_correct - group_size)
    denominator = math.comb(group_size * 2, total_correct)
    return sum(
        (
            math.comb(group_size, candidate)
            * math.comb(group_size, total_correct - candidate)
            / denominator
        )
        for candidate in range(lower, added_correct + 1)
    )


def _interpret(
    *,
    added_correct: int,
    stable_correct: int,
    group_size: int,
    feature_correct: bool,
    alpha: float,
) -> dict[str, Any]:
    stable_p = _exact_binomial_upper_tail(
        correct=stable_correct,
        total=group_size,
        null_accuracy=0.25,
    )
    fisher_p = _one_sided_fisher_less(
        added_correct=added_correct,
        stable_correct=stable_correct,
        group_size=group_size,
    )
    if stable_p >= alpha:
        state = "baseline-incapable-of-localizing-drift"
    elif added_correct < stable_correct and fisher_p < alpha:
        state = "localized-added-fact-drift"
    elif added_correct < stable_correct:
        state = "observed-gap-not-statistically-localized"
    elif added_correct == group_size and feature_correct:
        state = "no-observed-acceptance-gap"
    else:
        state = "knowledge-gap-without-relative-drift"
    return {
        "state": state,
        "stable_vs_chance_p_value": stable_p,
        "added_less_than_stable_p_value": fisher_p,
        "added_minus_stable_accuracy": (
            (added_correct - stable_correct) / group_size
        ),
        "feature_correct": feature_correct,
        "decision_authority": "evidence-for-decide-not-promotion",
    }


def score_outputs(
    *,
    protocol: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_repeats = int(protocol["generation"]["repeats"])
    items_by_id = {str(item["eval_id"]): item for item in items}
    observed: dict[str, dict[int, Mapping[str, Any]]] = {}
    for output in outputs:
        if (
            output.get("schema") != OUTPUT_SCHEMA
            or output.get("protocol_id") != protocol["protocol_id"]
            or output.get("model_revision")
            != protocol["target_model"]["revision"]
        ):
            raise BaselineProtocolError("target output identity changed")
        eval_id = _string(output.get("eval_id"), "output.eval_id")
        if eval_id not in items_by_id:
            raise BaselineProtocolError(f"unknown output EvalItem: {eval_id}")
        repeat_index = output.get("repeat_index")
        if (
            not isinstance(repeat_index, int)
            or repeat_index < 1
            or repeat_index > expected_repeats
        ):
            raise BaselineProtocolError("invalid output repeat index")
        _string(output.get("raw_response"), "output.raw_response")
        repeats = observed.setdefault(eval_id, {})
        if repeat_index in repeats:
            raise BaselineProtocolError("duplicate target output repeat")
        repeats[repeat_index] = output
    if set(observed) != set(items_by_id) or any(
        set(repeats) != set(range(1, expected_repeats + 1))
        for repeats in observed.values()
    ):
        raise BaselineProtocolError("target output coverage is incomplete")

    strata: dict[str, Counter[str]] = {
        stratum: Counter() for stratum in EXPECTED_STRATA
    }
    for eval_id, item in items_by_id.items():
        raw_responses = [
            str(observed[eval_id][index]["raw_response"])
            for index in range(1, expected_repeats + 1)
        ]
        if len(set(raw_responses)) != 1:
            raise BaselineProtocolError(
                f"nondeterministic target response: {eval_id}"
            )
        response = raw_responses[0]
        stratum = _stratum(item)
        strata[stratum]["items"] += 1
        if _is_parseable(item, response):
            strata[stratum]["parseable"] += 1
        if score_response(item, response):
            strata[stratum]["correct"] += 1

    metrics: dict[str, dict[str, Any]] = {}
    for stratum, expected_count in EXPECTED_STRATA.items():
        counts = strata[stratum]
        if counts["items"] != expected_count:
            raise BaselineProtocolError("scored stratum count changed")
        metrics[stratum] = {
            "items": expected_count,
            "correct": counts["correct"],
            "exact_accuracy": counts["correct"] / expected_count,
            "parseable": counts["parseable"],
            "parseable_rate": counts["parseable"] / expected_count,
            "invalid_response_count": expected_count - counts["parseable"],
        }
    localization = protocol["reporting"]["drift_localization"]
    return {
        "schema": "delta.target_model_baseline_audit.v1",
        "protocol_id": protocol["protocol_id"],
        "status": "pass",
        "deterministic_repeats": True,
        "items": len(items),
        "model_outputs": len(outputs),
        "strata": metrics,
        "interpretation": _interpret(
            added_correct=metrics["atomic_fact_added"]["correct"],
            stable_correct=metrics["atomic_fact_stable"]["correct"],
            group_size=EXPECTED_STRATA["atomic_fact_added"],
            feature_correct=bool(metrics["feature_added"]["correct"]),
            alpha=float(localization["alpha"]),
        ),
        "pooled_overall_accuracy": None,
        "target_model_results_exist": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate delta_v2 baseline preregistration and emit model-safe "
            "request envelopes without invoking a model."
        )
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--requests-out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    protocol = load_yaml(args.protocol)
    items = validate_protocol(protocol, repo_root=repo_root)
    requests = build_requests(protocol, items)
    payload = serialize_jsonl(requests)
    args.requests_out.parent.mkdir(parents=True, exist_ok=True)
    args.requests_out.write_bytes(payload)
    print(
        json.dumps(
            {
                "schema": REQUEST_SCHEMA,
                "protocol_id": protocol["protocol_id"],
                "requests": len(requests),
                "eval_items": len(items),
                "repeats": protocol["generation"]["repeats"],
                "sha256": hashlib.sha256(payload).hexdigest(),
                "model_invocations_completed": 0,
                "target_model_results_exist": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
