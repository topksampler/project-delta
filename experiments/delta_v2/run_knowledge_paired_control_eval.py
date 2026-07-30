from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2.run_knowledge_eval import (
    DATASET_ID,
    EVAL_PATH,
    METRICS_SCHEMA,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODAL_APP_PATH,
    OUTPUT_SCHEMA,
    PROTOCOL_ID,
    RECEIPT_SCHEMA,
    RUNTIME_PACKAGES,
    TransformersKnowledgeBackend,
    _load_yaml,
    _mapping,
    _relative_path,
    _sha256,
    audit_requests,
    build_requests,
    score_outputs,
    serialize_jsonl,
    validate_protocol,
)


EXTENSION_SCHEMA = "delta.knowledge_eval_extension_protocol.v1"
CONFIG_SCHEMA = "delta.knowledge_eval_extension_config.v1"
EXTENSION_ID = "delta-v2-knowledge-paired-control-eval-v1"
CONDITION_ID = "d2_paired_data_sft_control_eval"
RUN_ID = (
    "delta-v2-d2-knowledge-paired-data-sft-control-"
    "eval-modal-v1"
)
TRAINING_RUN_ID = (
    "delta-v2-d2-knowledge-paired-data-sft-control-"
    "qwen35-08b-modal-v1"
)
MARGIN_RUN_ID = (
    "delta-v2-d2-knowledge-paired-data-sft-control-"
    "margin-modal-v1"
)
EXTENSION_PATH = Path(
    "experiments/delta_v2/knowledge_paired_control_eval_protocol.yaml"
)
PARENT_PATH = Path("experiments/delta_v2/knowledge_eval_protocol.yaml")
GATE_SOURCE_PATH = Path(
    "experiments/delta_v2/knowledge_lora_protocol_v2.yaml"
)
ADAPTER_PATH = Path("runs") / TRAINING_RUN_ID / "adapter"


class PairedControlEvalError(ValueError):
    """The unchanged full evaluation extension cannot run safely."""


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_extension(
    extension: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    if (
        extension.get("schema") != EXTENSION_SCHEMA
        or extension.get("protocol_id") != EXTENSION_ID
        or extension.get("experiment_id") != "delta_v2"
        or extension.get("condition_id") != CONDITION_ID
        or extension.get("state") != "preregistered-for-evaluation"
    ):
        raise PairedControlEvalError("eval extension identity changed")
    parent_binding = _mapping(extension.get("parent_eval"), "parent_eval")
    parent_path = repo_root / _relative_path(
        parent_binding.get("path"), "parent_eval.path"
    )
    if dict(parent_binding) != {
        "protocol_id": PROTOCOL_ID,
        "path": str(PARENT_PATH),
        "sha256": (
            "0ba449b993cd9afb356d35952283b582d81519960ae362137456173c3978c226"
        ),
        "only_extension": "target-condition-and-adapter-lineage",
        "dataset_unchanged": True,
        "prompting_unchanged": True,
        "generation_unchanged": True,
        "scoring_unchanged": True,
        "reporting_unchanged": True,
    } or parent_path != repo_root / PARENT_PATH:
        raise PairedControlEvalError("parent eval binding changed")
    if _sha256(parent_path) != parent_binding["sha256"]:
        raise PairedControlEvalError("parent eval bytes changed")
    parent = _load_yaml(parent_path)
    probes = validate_protocol(parent, repo_root=repo_root)

    trigger = _mapping(extension.get("decision_trigger"), "decision_trigger")
    if dict(trigger) != {
        "margin_run_id": MARGIN_RUN_ID,
        "margin_launch_commit": "ccb8609",
        "margin_receipt_sha256": (
            "f9cc242bee8bab98d29fbca7f683303360c37286d5c4245d80143f15f5dceb68"
        ),
        "margin_metrics_sha256": (
            "a954b4501fe9a60a2a8c842a69ad050fa278e9c28baa446300002aed5c02ce5b"
        ),
        "first_gate_passed": True,
        "truth_conditioned_pairs": "57/63",
    }:
        raise PairedControlEvalError("eval extension trigger changed")
    margin_root = repo_root / "runs" / MARGIN_RUN_ID
    if (
        _sha256(margin_root / "run_receipt.json")
        != trigger["margin_receipt_sha256"]
        or _sha256(margin_root / "metrics.json")
        != trigger["margin_metrics_sha256"]
    ):
        raise PairedControlEvalError("margin gate evidence changed")
    candidate = _mapping(extension.get("candidate"), "candidate")
    if dict(candidate) != {
        "training_run_id": TRAINING_RUN_ID,
        "objective_id": "assistant-only-generative-sft-v1",
        "adapter_sha256": (
            "9ef3caaf1f1d83a26df485ee0abcd365d1f932e0752cb9a99d9c5ac3510565a5"
        ),
        "base_revision": MODEL_REVISION,
        "initialization": "fresh-base",
    }:
        raise PairedControlEvalError("eval candidate lineage changed")
    adapter_file = (
        repo_root / ADAPTER_PATH / "adapter_model.safetensors"
    )
    if (
        not adapter_file.is_file()
        or _sha256(adapter_file) != candidate["adapter_sha256"]
    ):
        raise PairedControlEvalError("eval adapter bytes changed")

    comparison = _mapping(
        extension.get("comparison_gates"), "comparison_gates"
    )
    gate_binding = _mapping(
        comparison.get("source_protocol"),
        "comparison_gates.source_protocol",
    )
    gate_path = repo_root / _relative_path(
        gate_binding.get("path"), "comparison_gates.source_protocol.path"
    )
    if dict(gate_binding) != {
        "path": str(GATE_SOURCE_PATH),
        "sha256": (
            "73f7dde372ebdbeafd56cd25808b9dcc7135bbeafd829f91ff4d970abdd44331"
        ),
    } or gate_path != repo_root / GATE_SOURCE_PATH:
        raise PairedControlEvalError("comparison gate source changed")
    if _sha256(gate_path) != gate_binding["sha256"]:
        raise PairedControlEvalError("comparison gate bytes changed")
    gate_source = _load_yaml(gate_path)
    source_gates = _mapping(
        gate_source.get("verification_gates"), "verification_gates"
    )
    acquisition = _mapping(source_gates.get("acquisition"), "acquisition")
    retention = _mapping(source_gates.get("retention"), "retention")
    if (
        comparison.get("acquisition_choice_accuracy_minimum")
        != acquisition.get("choice_accuracy_minimum")
        or comparison.get("acquisition_boolean_true_accuracy_minimum")
        != acquisition.get("boolean_true_accuracy_minimum")
        or comparison.get("acquisition_boolean_false_accuracy_minimum")
        != acquisition.get("boolean_false_accuracy_minimum")
        or comparison.get("retention_boolean_false_accuracy_minimum")
        != retention.get("boolean_false_accuracy_minimum")
        or comparison.get("pooled_overall_score") != "forbidden"
        or comparison.get("recall") != "advisory"
        or comparison.get("feature") != "report-separately"
        or comparison.get("result") != "candidate-only-no-promotion"
    ):
        raise PairedControlEvalError("comparison gates changed")
    boundary = _mapping(
        extension.get("execution_boundary"), "execution_boundary"
    )
    if dict(boundary) != {
        "unchanged_full_eval_authorized": True,
        "model_updates": 0,
        "paired_objective_training_authorized": False,
        "qlora_authorized": False,
        "full_weight_authorized": False,
        "reinforcement_authorized": False,
        "promotion_authorized": False,
    }:
        raise PairedControlEvalError("eval extension boundary changed")
    return parent, probes


def validate_config(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    list[Mapping[str, Any]],
]:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("run_id") != RUN_ID
        or config.get("experiment_id") != "delta_v2"
        or config.get("condition_id") != CONDITION_ID
        or config.get("eval_environment_id") != DATASET_ID
        or config.get("task") != "eval"
    ):
        raise PairedControlEvalError("eval extension config changed")
    binding = _mapping(config.get("protocol"), "protocol")
    extension_path = repo_root / _relative_path(
        binding.get("path"), "protocol.path"
    )
    if (
        binding.get("protocol_id") != EXTENSION_ID
        or extension_path != repo_root / EXTENSION_PATH
        or _sha256(extension_path) != binding.get("sha256")
    ):
        raise PairedControlEvalError("eval extension binding changed")
    extension = _load_yaml(extension_path)
    parent, probes = validate_extension(extension, repo_root=repo_root)

    candidate = extension["candidate"]
    if dict(_mapping(config.get("model"), "model")) != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "adapter": {
            "kind": "lora",
            "run_id": TRAINING_RUN_ID,
            "path": str(ADAPTER_PATH),
            "sha256": candidate["adapter_sha256"],
        },
        "quantization": "none",
    }:
        raise PairedControlEvalError("eval extension model changed")
    runtime = _mapping(config.get("runtime"), "runtime")
    if (
        runtime.get("profile") != "delta-v2-knowledge-modal-v1"
        or runtime.get("python") != "3.11.9"
        or runtime.get("packages") != RUNTIME_PACKAGES
        or runtime.get("device") != "cuda"
        or runtime.get("dtype") != "bfloat16"
        or runtime.get("attention_implementation") != "eager"
        or runtime.get("deterministic_algorithms") is not True
    ):
        raise PairedControlEvalError("eval extension runtime changed")
    trigger = extension["decision_trigger"]
    if dict(_mapping(config.get("inputs"), "inputs")) != {
        "run_artifacts": [
            {
                "run_id": MARGIN_RUN_ID,
                "path": str(
                    Path("runs") / MARGIN_RUN_ID / "run_receipt.json"
                ),
                "sha256": trigger["margin_receipt_sha256"],
            },
            {
                "run_id": MARGIN_RUN_ID,
                "path": str(
                    Path("runs") / MARGIN_RUN_ID / "metrics.json"
                ),
                "sha256": trigger["margin_metrics_sha256"],
            },
        ]
    }:
        raise PairedControlEvalError("eval extension inputs changed")
    eval_block = _mapping(config.get("eval"), "eval")
    eval_path = repo_root / _relative_path(
        eval_block.get("path"), "eval.path"
    )
    if (
        eval_block.get("module")
        != "experiments.delta_v2.run_knowledge_paired_control_eval"
        or eval_path != repo_root / EVAL_PATH
        or _sha256(eval_path) != eval_block.get("sha256")
        or eval_block.get("sha256")
        != parent["dataset"]["eval_sha256"]
    ):
        raise PairedControlEvalError("eval extension data changed")
    if _relative_path(config["output"]["dir"], "output.dir") != (
        Path("runs") / RUN_ID
    ):
        raise PairedControlEvalError("eval extension output changed")
    if dict(_mapping(config.get("modal"), "modal")) != {
        "app_path": MODAL_APP_PATH,
        "gpu": "A10G",
        "timeout_seconds": 3600,
    }:
        raise PairedControlEvalError("eval extension Modal binding changed")
    return extension, parent, probes


def execute(
    *,
    config_path: Path,
    repo_root: Path,
    backend: Any | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    config = _load_yaml(config_path)
    extension, parent, probes = validate_config(
        config, repo_root=repo_root
    )
    requests = build_requests(
        config=config,
        protocol=parent,
        probes=probes,
    )
    request_audit = audit_requests(requests)
    request_bytes = serialize_jsonl(requests)
    active = backend or TransformersKnowledgeBackend(
        config=config,
        protocol=parent,
        repo_root=repo_root,
    )
    outputs: list[dict[str, Any]] = []
    for request in requests:
        raw_response, finish_reason = active.generate(request)
        outputs.append(
            {
                "schema": OUTPUT_SCHEMA,
                "request_id": request["request_id"],
                "run_id": RUN_ID,
                "protocol_id": PROTOCOL_ID,
                "probe_id": request["probe_id"],
                "repeat_index": request["repeat_index"],
                "raw_response": raw_response,
                "finish_reason": finish_reason,
            }
        )
    metrics = score_outputs(
        probes=probes,
        outputs=outputs,
        request_audit=request_audit,
    )
    output_bytes = serialize_jsonl(outputs)
    output_dir = repo_root / str(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    metrics_path = output_dir / "metrics.json"
    samples_path.write_bytes(output_bytes)
    _write_json(metrics_path, metrics)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "run_id": RUN_ID,
        "condition_id": CONDITION_ID,
        "protocol_id": PROTOCOL_ID,
        "extension_protocol_id": EXTENSION_ID,
        "dataset_id": DATASET_ID,
        "config_sha256": _sha256(config_path),
        "protocol_sha256": _sha256(repo_root / PARENT_PATH),
        "extension_protocol_sha256": _sha256(
            repo_root / EXTENSION_PATH
        ),
        "eval_sha256": _sha256(repo_root / EVAL_PATH),
        "adapter_sha256": extension["candidate"]["adapter_sha256"],
        "margin_receipt_sha256": extension["decision_trigger"][
            "margin_receipt_sha256"
        ],
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "samples_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "metrics_sha256": _sha256(metrics_path),
        "eval_items": len(probes),
        "model_outputs": len(outputs),
        "runtime": dict(active.runtime_receipt()),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "model_updates": 0,
        "status": "pass",
        "promotion_authorized": False,
    }
    receipt_path = output_dir / "run_receipt.json"
    _write_json(receipt_path, receipt)
    return {"metrics": metrics, "receipt": receipt}


def validate_only(config_path: Path, repo_root: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    _extension, parent, probes = validate_config(
        config, repo_root=repo_root
    )
    requests = build_requests(
        config=config,
        protocol=parent,
        probes=probes,
    )
    request_audit = audit_requests(requests)
    return {
        "run_id": RUN_ID,
        "protocol_id": PROTOCOL_ID,
        "extension_protocol_id": EXTENSION_ID,
        "eval_items": len(probes),
        "requests": len(requests),
        "request_sha256": hashlib.sha256(
            serialize_jsonl(requests)
        ).hexdigest(),
        "request_audit": request_audit,
        "model_invocations_completed": 0,
        "status": "valid-unexecuted",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    result = (
        validate_only(config_path, repo_root)
        if args.validate_only
        else execute(config_path=config_path, repo_root=repo_root)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
