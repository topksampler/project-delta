from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2.knowledge_paired_lora import (
    OBJECTIVE_PAIRED,
    build_training_units,
    train_paired_candidate,
)
from experiments.delta_v2.run_knowledge_eval import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODAL_APP_PATH,
    RUNTIME_PACKAGES,
)
from experiments.delta_v2.train_knowledge_lora import (
    TARGET_MODULES_REGEX,
    KnowledgeTrainError,
    _load_yaml,
    _mapping,
    _relative_path,
    _sha256,
)
from experiments.delta_v2.train_knowledge_paired_sft_control import (
    DATASET_ID,
    DEV_PATH,
    ENGINE_PATH,
    EVAL_PATH,
    OPTIMIZATION,
    PAIRS_PATH,
    TRAIN_PATH,
    validate_protocol as validate_control_protocol,
)


PROTOCOL_SCHEMA = "delta.knowledge_paired_lora_protocol.v1"
CONFIG_SCHEMA = "delta.knowledge_paired_lora_train_config.v1"
PROTOCOL_ID = "delta-v2-knowledge-paired-objective-v1"
CONDITION_ID = "d3_paired_objective_lora"
RUN_ID = "delta-v2-d3-knowledge-paired-objective-qwen35-08b-modal-v1"
PROTOCOL_PATH = Path(
    "experiments/delta_v2/knowledge_paired_objective_protocol.yaml"
)
CONTROL_PROTOCOL_PATH = Path(
    "experiments/delta_v2/knowledge_paired_sft_control_protocol.yaml"
)
VALIDATOR_PATH = Path(
    "experiments/delta_v2/train_knowledge_paired_objective.py"
)
CONTROL_TRAIN_RUN_ID = (
    "delta-v2-d2-knowledge-paired-data-sft-control-"
    "qwen35-08b-modal-v1"
)
CONTROL_EVAL_RUN_ID = (
    "delta-v2-d2-knowledge-paired-data-sft-control-"
    "eval-modal-v1"
)
CONTROL_TRAIN_RECEIPT_SHA256 = (
    "cff9d721dadbf48d8974504728ef0fba3b14e9b498a0a893f4dbdba8c32c8f07"
)
CONTROL_TRAIN_METRICS_SHA256 = (
    "4cc7035c23ff0b949810e7b7e2ad599243cbb574bb3d111a4a234d9ccfea27b2"
)
CONTROL_EVAL_RECEIPT_SHA256 = (
    "90b9b64b1cce0fa11e8f7af146f49700b54f8317daf6972f4e9de6a5f95c1f19"
)
CONTROL_EVAL_METRICS_SHA256 = (
    "c75b475e4af524948c3f7976b18f3b5a53a126d282bf47e7c47603706488e705"
)
CONTROL_ADAPTER_SHA256 = (
    "9ef3caaf1f1d83a26df485ee0abcd365d1f932e0752cb9a99d9c5ac3510565a5"
)
OBJECTIVE = {
    "objective_id": OBJECTIVE_PAIRED,
    "recall_loss": "teacher-forced-assistant-token-cross-entropy",
    "boolean_pair_loss": "truth-conditioned-logistic-plus-ranking",
    "pair_normalization": "one-pair-equals-one-training-unit",
    "truth_margin_term": (
        "mean-softplus-negative-true-and-positive-false"
    ),
    "pair_ranking_term": (
        "softplus-ranking-margin-minus-pair-separation"
    ),
    "ranking_margin": 1.0,
    "ranking_weight": 1.0,
}


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeTrainError(f"cannot read JSON: {path}") from exc
    return _mapping(payload, str(path))


def _evidence_paths(repo_root: Path) -> dict[str, Path]:
    return {
        "train_receipt": (
            repo_root / "runs" / CONTROL_TRAIN_RUN_ID / "run_receipt.json"
        ),
        "train_metrics": (
            repo_root / "runs" / CONTROL_TRAIN_RUN_ID / "train_metrics.json"
        ),
        "eval_receipt": (
            repo_root / "runs" / CONTROL_EVAL_RUN_ID / "run_receipt.json"
        ),
        "eval_metrics": (
            repo_root / "runs" / CONTROL_EVAL_RUN_ID / "metrics.json"
        ),
    }


def _validate_control_evidence(repo_root: Path) -> None:
    paths = _evidence_paths(repo_root)
    expected = {
        "train_receipt": CONTROL_TRAIN_RECEIPT_SHA256,
        "train_metrics": CONTROL_TRAIN_METRICS_SHA256,
        "eval_receipt": CONTROL_EVAL_RECEIPT_SHA256,
        "eval_metrics": CONTROL_EVAL_METRICS_SHA256,
    }
    if any(_sha256(paths[key]) != digest for key, digest in expected.items()):
        raise KnowledgeTrainError("paired objective trigger evidence changed")
    train_receipt = _load_json(paths["train_receipt"])
    train_metrics = _load_json(paths["train_metrics"])
    eval_receipt = _load_json(paths["eval_receipt"])
    eval_metrics = _load_json(paths["eval_metrics"])
    retention_false = _mapping(
        _mapping(eval_metrics.get("cells"), "eval_metrics.cells").get(
            "retention_stable.boolean_false"
        ),
        "retention_stable.boolean_false",
    )
    if (
        train_receipt.get("run_id") != CONTROL_TRAIN_RUN_ID
        or train_receipt.get("objective_id")
        != "assistant-only-generative-sft-v1"
        or train_receipt.get("adapter_sha256") != CONTROL_ADAPTER_SHA256
        or train_receipt.get("metrics_sha256")
        != CONTROL_TRAIN_METRICS_SHA256
        or train_metrics.get("steps") != 60
        or train_metrics.get("objective_id")
        != "assistant-only-generative-sft-v1"
        or eval_receipt.get("run_id") != CONTROL_EVAL_RUN_ID
        or eval_receipt.get("adapter_sha256") != CONTROL_ADAPTER_SHA256
        or eval_receipt.get("metrics_sha256")
        != CONTROL_EVAL_METRICS_SHA256
        or eval_receipt.get("model_updates") != 0
        or retention_false.get("items") != 21
        or retention_false.get("correct") != 16
        or retention_false.get("exact_accuracy") != 16 / 21
    ):
        raise KnowledgeTrainError("paired objective trigger result changed")


def validate_protocol(
    protocol: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
]:
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("experiment_id") != "delta_v2"
        or protocol.get("condition_id") != CONDITION_ID
        or protocol.get("state") != "preregistered-for-training"
    ):
        raise KnowledgeTrainError("paired objective identity changed")
    trigger = _mapping(protocol.get("decision_trigger"), "decision_trigger")
    if dict(trigger) != {
        "stage_1_train_run_id": CONTROL_TRAIN_RUN_ID,
        "stage_1_train_receipt_sha256": CONTROL_TRAIN_RECEIPT_SHA256,
        "stage_1_train_metrics_sha256": CONTROL_TRAIN_METRICS_SHA256,
        "stage_1_eval_run_id": CONTROL_EVAL_RUN_ID,
        "stage_1_eval_receipt_sha256": CONTROL_EVAL_RECEIPT_SHA256,
        "stage_1_eval_metrics_sha256": CONTROL_EVAL_METRICS_SHA256,
        "stage_1_full_eval_result": "evaluation-gates-fail",
        "failed_gate": "retention_stable.boolean_false",
        "failed_gate_observed": "16/21",
        "failed_gate_minimum": 0.9,
    }:
        raise KnowledgeTrainError("paired objective trigger changed")
    _validate_control_evidence(repo_root)

    role = _mapping(protocol.get("causal_role"), "causal_role")
    source_binding = _mapping(
        role.get("source_protocol"), "causal_role.source_protocol"
    )
    source_path = repo_root / _relative_path(
        source_binding.get("path"), "causal_role.source_protocol.path"
    )
    if (
        role.get("stage") != 2
        or role.get("name")
        != "paired-boolean-objective-only-comparison"
        or role.get("only_changed_factor") != "learning-objective"
        or role.get("unchanged_factors")
        != [
            "dataset",
            "fresh-base-initialization",
            "lora-topology",
            "optimization",
            "training-unit-order",
            "development-bank",
            "evaluation-bank",
        ]
        or dict(source_binding)
        != {
            "path": str(CONTROL_PROTOCOL_PATH),
            "sha256": (
                "7c97f2581636510403f7af8792e044fd7543c8be66500f5d923f73ef0feb72cd"
            ),
        }
        or source_path != repo_root / CONTROL_PROTOCOL_PATH
        or _sha256(source_path) != source_binding["sha256"]
    ):
        raise KnowledgeTrainError("paired objective causal role changed")
    control = _load_yaml(source_path)
    train_rows, pairs, dev_rows = validate_control_protocol(
        control, repo_root=repo_root
    )
    for field in (
        "dataset",
        "model",
        "lora",
        "training_units",
        "optimization",
    ):
        if protocol.get(field) != control.get(field):
            raise KnowledgeTrainError(
                f"paired objective changed frozen field: {field}"
            )
    implementation = _mapping(
        protocol.get("implementation"), "implementation"
    )
    engine_path = repo_root / _relative_path(
        implementation.get("engine_path"), "implementation.engine_path"
    )
    validator_path = repo_root / _relative_path(
        implementation.get("validator_path"),
        "implementation.validator_path",
    )
    if (
        engine_path != repo_root / ENGINE_PATH
        or _sha256(engine_path) != implementation.get("engine_sha256")
        or validator_path != repo_root / VALIDATOR_PATH
        or _sha256(validator_path)
        != implementation.get("validator_sha256")
    ):
        raise KnowledgeTrainError("paired objective implementation changed")
    if dict(_mapping(protocol.get("objective"), "objective")) != OBJECTIVE:
        raise KnowledgeTrainError("paired objective loss changed")
    if dict(
        _mapping(protocol.get("verification"), "verification")
    ) != {
        "first_gate": "repaired-pair-boolean-margin-v1",
        "verify_true_margin_accuracy_minimum": 0.8,
        "verify_false_margin_accuracy_minimum": 0.8,
        "truth_conditioned_pair_rate_minimum": 0.8,
        "per_corruption_family_false_accuracy_minimum": 0.8,
        "full_frozen_eval_only_after_first_gate": True,
        "full_eval_protocol": "delta-v2-knowledge-eval-v1",
        "acquisition_choice_accuracy_minimum": 0.5,
        "acquisition_boolean_true_accuracy_minimum": 0.8,
        "acquisition_boolean_false_accuracy_minimum": 0.8,
        "retention_boolean_false_accuracy_minimum": 0.9,
        "recall": "advisory",
        "feature": "report-separately",
        "pooled_overall_score": "forbidden",
        "result": "candidate-only-no-promotion",
    }:
        raise KnowledgeTrainError("paired objective verification changed")
    if dict(
        _mapping(protocol.get("safety_gates"), "safety_gates")
    ) != {
        "fresh_base_required": True,
        "trainable_scope": "lora-adapter-only",
        "base_weights_frozen": True,
        "vision_modules_trainable": False,
        "quantization": "none",
        "optimizer_steps_maximum": 60,
        "gradient_norm_clipping": 1.0,
        "exact_input_hashes_required": True,
        "deterministic_execution_required": True,
    }:
        raise KnowledgeTrainError("paired objective safety gates changed")
    if dict(
        _mapping(protocol.get("execution_boundary"), "execution_boundary")
    ) != {
        "modal_training_authorized": True,
        "b2_adapter_write_authorized": True,
        "repaired_pair_diagnostic_authorized": True,
        "full_eval_conditional": True,
        "qlora_authorized": False,
        "full_weight_authorized": False,
        "reinforcement_authorized": False,
        "promotion_authorized": False,
    }:
        raise KnowledgeTrainError("paired objective boundary changed")
    return train_rows, pairs, dev_rows


def _expected_inputs() -> dict[str, Any]:
    paths = _evidence_paths(Path())
    return {
        "run_artifacts": [
            {
                "run_id": CONTROL_TRAIN_RUN_ID,
                "path": str(paths["train_receipt"]),
                "sha256": CONTROL_TRAIN_RECEIPT_SHA256,
            },
            {
                "run_id": CONTROL_TRAIN_RUN_ID,
                "path": str(paths["train_metrics"]),
                "sha256": CONTROL_TRAIN_METRICS_SHA256,
            },
            {
                "run_id": CONTROL_EVAL_RUN_ID,
                "path": str(paths["eval_receipt"]),
                "sha256": CONTROL_EVAL_RECEIPT_SHA256,
            },
            {
                "run_id": CONTROL_EVAL_RUN_ID,
                "path": str(paths["eval_metrics"]),
                "sha256": CONTROL_EVAL_METRICS_SHA256,
            },
        ]
    }


def validate_run_config(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[
    Mapping[str, Any],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
]:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("run_id") != RUN_ID
        or config.get("experiment_id") != "delta_v2"
        or config.get("condition_id") != CONDITION_ID
        or config.get("eval_environment_id") != DATASET_ID
        or config.get("task") != "train"
    ):
        raise KnowledgeTrainError("paired objective config identity changed")
    binding = _mapping(config.get("protocol"), "protocol")
    protocol_path = repo_root / _relative_path(
        binding.get("path"), "protocol.path"
    )
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or protocol_path != repo_root / PROTOCOL_PATH
        or _sha256(protocol_path) != binding.get("sha256")
    ):
        raise KnowledgeTrainError(
            "paired objective protocol binding changed"
        )
    protocol = _load_yaml(protocol_path)
    train_rows, pairs, dev_rows = validate_protocol(
        protocol, repo_root=repo_root
    )
    if dict(_mapping(config.get("model"), "model")) != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "adapter": "new-lora",
        "quantization": "none",
    }:
        raise KnowledgeTrainError("paired objective config model changed")
    runtime = _mapping(config.get("runtime"), "runtime")
    if (
        runtime.get("profile") != "delta-v2-knowledge-modal-v1"
        or runtime.get("python") != "3.11.9"
        or runtime.get("packages") != RUNTIME_PACKAGES
        or runtime.get("device") != "cuda"
        or runtime.get("dtype") != "bfloat16"
        or runtime.get("deterministic_algorithms") is not True
    ):
        raise KnowledgeTrainError("paired objective runtime changed")
    if dict(_mapping(config.get("inputs"), "inputs")) != _expected_inputs():
        raise KnowledgeTrainError("paired objective inputs changed")
    if dict(_mapping(config.get("data"), "data")) != {
        "train_path": str(TRAIN_PATH),
        "train_sha256": protocol["dataset"]["train_sha256"],
        "pairs_path": str(PAIRS_PATH),
        "pairs_sha256": protocol["dataset"]["pairs_sha256"],
        "eval_path": str(DEV_PATH),
        "eval_sha256": protocol["dataset"]["dev_sha256"],
    }:
        raise KnowledgeTrainError("paired objective config data changed")
    if _mapping(config.get("lora"), "lora") != protocol["lora"]:
        raise KnowledgeTrainError("paired objective config LoRA changed")
    if dict(_mapping(config.get("objective"), "objective")) != OBJECTIVE:
        raise KnowledgeTrainError("paired objective config loss changed")
    if dict(_mapping(config.get("training"), "training")) != {
        "module": "experiments.delta_v2.train_knowledge_paired_objective",
        **OPTIMIZATION,
    }:
        raise KnowledgeTrainError("paired objective training changed")
    if _relative_path(config["output"]["dir"], "output.dir") != (
        Path("runs") / RUN_ID
    ):
        raise KnowledgeTrainError("paired objective output changed")
    if dict(_mapping(config.get("modal"), "modal")) != {
        "app_path": MODAL_APP_PATH,
        "gpu": "H100",
        "timeout_seconds": 7200,
    }:
        raise KnowledgeTrainError("paired objective Modal binding changed")
    return protocol, train_rows, pairs, dev_rows


def train(config_path: Path, repo_root: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    protocol, train_rows, pairs, dev_rows = validate_run_config(
        config, repo_root=repo_root
    )
    return train_paired_candidate(
        config_path=config_path,
        repo_root=repo_root,
        config=config,
        protocol=protocol,
        train_rows=train_rows,
        pairs=pairs,
        dev_rows=dev_rows,
        run_id=RUN_ID,
        condition_id=CONDITION_ID,
        protocol_id=PROTOCOL_ID,
        protocol_path=PROTOCOL_PATH,
        dataset_id=DATASET_ID,
        train_path=TRAIN_PATH,
        pairs_path=PAIRS_PATH,
        dev_path=DEV_PATH,
    )


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
    if args.validate_only:
        config = _load_yaml(config_path)
        protocol, train_rows, pairs, dev_rows = validate_run_config(
            config, repo_root=repo_root
        )
        result = {
            "run_id": RUN_ID,
            "protocol_id": PROTOCOL_ID,
            "objective_id": protocol["objective"]["objective_id"],
            "train_rows": len(train_rows),
            "pairs": len(pairs),
            "dev_rows": len(dev_rows),
            "training_units": len(
                build_training_units(train_rows, pairs)
            ),
            "model_invocations_completed": 0,
            "optimizer_steps": 0,
            "status": "valid-unexecuted",
        }
    else:
        result = train(config_path, repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
