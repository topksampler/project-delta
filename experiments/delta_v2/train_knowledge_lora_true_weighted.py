from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2.run_knowledge_eval import (
    DATASET_ID,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODAL_APP_PATH,
    RUNTIME_PACKAGES,
)
from experiments.delta_v2.train_knowledge_lora import (
    DEV_PATH,
    FREEZE_PATH,
    RUN_CONFIG_SCHEMA,
    TARGET_MODULES_REGEX,
    TRAIN_PATH,
    KnowledgeTrainError,
    _load_jsonl,
    _load_yaml,
    _mapping,
    _relative_path,
    _sha256,
    train_validated_candidate,
    validate_rows,
)


PROTOCOL_ID = "delta-v2-knowledge-lora-true-weighted-v1"
PROTOCOL_PATH = Path(
    "experiments/delta_v2/knowledge_lora_true_weighted_protocol.yaml"
)
RUN_ID = "delta-v2-c6-knowledge-lora-true-weighted-qwen35-08b-modal-v1"
CONDITION_ID = "c6_knowledge_lora_true_weighted"
SURFACE_WEIGHTS = {
    "exact_recall": 1,
    "verify_true": 4,
    "verify_false": 1,
}
OPTIMIZATION = {
    "optimizer": "adamw",
    "max_steps": 60,
    "learning_rate": 0.0001,
    "warmup_steps": 6,
    "weight_decay": 0.0,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_sequence_length": 768,
    "max_grad_norm": 1.0,
    "dtype": "bfloat16",
    "seed": 20260730,
    "assistant_only_loss": True,
    "deterministic_algorithms": "required",
}


def validate_protocol(
    protocol: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if (
        protocol.get("schema") != "delta.knowledge_lora_protocol.v1"
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("experiment_id") != "delta_v2"
        or protocol.get("condition_id") != CONDITION_ID
        or protocol.get("state") != "preregistered-for-training"
    ):
        raise KnowledgeTrainError("weighted LoRA protocol identity changed")
    if dict(_mapping(protocol.get("decision_trigger"), "decision_trigger")) != {
        "c5_eval_run_id": (
            "delta-v2-c5-knowledge-lora-eval-qwen35-08b-modal-v1"
        ),
        "c5_eval_metrics_sha256": (
            "592915e674e874e4e5874ed835ed4169127260c2faf88e7e8b92490322670a8b"
        ),
        "training_surface_run_id": (
            "delta-v2-c5-knowledge-lora-train-surface-modal-v1"
        ),
        "training_surface_metrics_sha256": (
            "eac1482e1244933caafc3dd70d70c6b2a3f2adbde091340b1bc530a5216e1d48"
        ),
        "observed_training_verify_true": "0/21",
        "observed_training_verify_false": "21/21",
        "diagnosis": "negative-answer-prior-not-overcome",
    }:
        raise KnowledgeTrainError("weighted LoRA trigger changed")
    dataset = _mapping(protocol.get("dataset"), "dataset")
    freeze_path = repo_root / _relative_path(
        dataset.get("freeze_path"), "dataset.freeze_path"
    )
    train_path = repo_root / _relative_path(
        dataset.get("train_path"), "dataset.train_path"
    )
    dev_path = repo_root / _relative_path(
        dataset.get("dev_path"), "dataset.dev_path"
    )
    if (
        dataset.get("dataset_id") != DATASET_ID
        or freeze_path != repo_root / FREEZE_PATH
        or train_path != repo_root / TRAIN_PATH
        or dev_path != repo_root / DEV_PATH
        or _sha256(freeze_path) != dataset.get("freeze_sha256")
        or _sha256(train_path) != dataset.get("train_sha256")
        or _sha256(dev_path) != dataset.get("dev_sha256")
        or dataset.get("train_rows") != 63
        or dataset.get("dev_rows") != 21
        or dataset.get("evaluation_wording_added_to_training") is not False
    ):
        raise KnowledgeTrainError("weighted LoRA dataset changed")
    if dict(_mapping(protocol.get("model"), "model")) != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "initialization": "fresh-base",
        "base_weights": "frozen",
        "quantization": "none",
        "full_weight_training": "forbidden",
    }:
        raise KnowledgeTrainError("weighted LoRA model changed")
    lora = _mapping(protocol.get("lora"), "lora")
    if dict(lora) != {
        "implementation": "peft-0.19.0",
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "bias": "none",
        "target_modules_regex": TARGET_MODULES_REGEX,
        "vision_modules_trainable": False,
    }:
        raise KnowledgeTrainError("weighted LoRA topology changed")
    if dict(_mapping(protocol.get("sampling"), "sampling")) != {
        "method": "deterministic-virtual-repetition",
        "surface_weights": SURFACE_WEIGHTS,
        "source_rows_mutated": False,
        "seed": 20260730,
    }:
        raise KnowledgeTrainError("weighted LoRA sampling changed")
    if dict(_mapping(protocol.get("optimization"), "optimization")) != (
        OPTIMIZATION
    ):
        raise KnowledgeTrainError("weighted LoRA optimization changed")
    verification = _mapping(protocol.get("verification"), "verification")
    if (
        verification.get("first_gate")
        != "exact-training-surface-diagnostic"
        or verification.get("training_verify_true_accuracy_minimum") != 0.8
        or verification.get("training_verify_false_accuracy_minimum") != 0.8
        or verification.get("full_frozen_eval_if_first_gate_passes") is not True
        or verification.get("frozen_eval_gates_unchanged") is not True
        or verification.get("pooled_overall_score") != "forbidden"
        or verification.get("result") != "candidate-only-no-promotion"
    ):
        raise KnowledgeTrainError("weighted LoRA verification changed")
    if dict(
        _mapping(protocol.get("execution_boundary"), "execution_boundary")
    ) != {
        "modal_training_authorized": True,
        "b2_adapter_write_authorized": True,
        "diagnostic_authorized": True,
        "full_eval_conditional": True,
        "promotion_authorized": False,
    }:
        raise KnowledgeTrainError("weighted LoRA boundary changed")
    train_rows = _load_jsonl(train_path)
    dev_rows = _load_jsonl(dev_path)
    validate_rows(train_rows, expected_rows=63, expected_surfaces={
        "exact_recall": 21,
        "verify_true": 21,
        "verify_false": 21,
    })
    validate_rows(
        dev_rows,
        expected_rows=21,
        expected_surfaces={"dev_recall": 21},
    )
    return train_rows, dev_rows


def validate_run_config(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[
    Mapping[str, Any],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
]:
    if (
        config.get("schema") != RUN_CONFIG_SCHEMA
        or config.get("run_id") != RUN_ID
        or config.get("experiment_id") != "delta_v2"
        or config.get("condition_id") != CONDITION_ID
        or config.get("eval_environment_id") != DATASET_ID
        or config.get("task") != "train"
    ):
        raise KnowledgeTrainError("weighted training identity changed")
    binding = _mapping(config.get("protocol"), "protocol")
    protocol_path = repo_root / _relative_path(
        binding.get("path"), "protocol.path"
    )
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or protocol_path != repo_root / PROTOCOL_PATH
        or _sha256(protocol_path) != binding.get("sha256")
    ):
        raise KnowledgeTrainError("weighted protocol binding changed")
    protocol = _load_yaml(protocol_path)
    train_rows, dev_rows = validate_protocol(
        protocol, repo_root=repo_root
    )
    if dict(_mapping(config.get("model"), "model")) != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "adapter": "new-lora",
        "quantization": "none",
    }:
        raise KnowledgeTrainError("weighted model config changed")
    runtime = _mapping(config.get("runtime"), "runtime")
    if (
        runtime.get("profile") != "delta-v2-knowledge-modal-v1"
        or runtime.get("python") != "3.11.9"
        or runtime.get("packages") != RUNTIME_PACKAGES
        or runtime.get("device") != "cuda"
        or runtime.get("dtype") != "bfloat16"
        or runtime.get("deterministic_algorithms") is not True
    ):
        raise KnowledgeTrainError("weighted runtime changed")
    if dict(_mapping(config.get("data"), "data")) != {
        "train_path": str(TRAIN_PATH),
        "train_sha256": protocol["dataset"]["train_sha256"],
        "eval_path": str(DEV_PATH),
        "eval_sha256": protocol["dataset"]["dev_sha256"],
    }:
        raise KnowledgeTrainError("weighted data config changed")
    if _mapping(config.get("lora"), "lora") != protocol["lora"]:
        raise KnowledgeTrainError("weighted LoRA config changed")
    if dict(_mapping(config.get("sampling"), "sampling")) != (
        protocol["sampling"]
    ):
        raise KnowledgeTrainError("weighted sampling config changed")
    if dict(_mapping(config.get("training"), "training")) != {
        "module": (
            "experiments.delta_v2.train_knowledge_lora_true_weighted"
        ),
        **OPTIMIZATION,
    }:
        raise KnowledgeTrainError("weighted optimization config changed")
    if _relative_path(config["output"]["dir"], "output.dir") != (
        Path("runs") / RUN_ID
    ):
        raise KnowledgeTrainError("weighted output changed")
    if dict(_mapping(config.get("modal"), "modal")) != {
        "app_path": MODAL_APP_PATH,
        "gpu": "H100",
        "timeout_seconds": 7200,
    }:
        raise KnowledgeTrainError("weighted Modal binding changed")
    return protocol, train_rows, dev_rows


def train(config_path: Path, repo_root: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    protocol, train_rows, dev_rows = validate_run_config(
        config, repo_root=repo_root
    )
    return train_validated_candidate(
        config_path=config_path,
        repo_root=repo_root,
        config=config,
        protocol=protocol,
        train_rows=train_rows,
        dev_rows=dev_rows,
        run_id=RUN_ID,
        protocol_id=PROTOCOL_ID,
        protocol_path=PROTOCOL_PATH,
        surface_weights=SURFACE_WEIGHTS,
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
        protocol, train_rows, dev_rows = validate_run_config(
            config, repo_root=repo_root
        )
        result = {
            "run_id": RUN_ID,
            "protocol_id": PROTOCOL_ID,
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "surface_weights": protocol["sampling"]["surface_weights"],
            "model_invocations_completed": 0,
            "status": "valid-unexecuted",
        }
    else:
        result = train(config_path, repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
