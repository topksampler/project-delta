from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2.knowledge_paired_data import (
    FREEZE_PATH,
    validate_freeze,
)
from experiments.delta_v2.knowledge_paired_lora import (
    OBJECTIVE_GENERATIVE,
    build_training_units,
    train_paired_candidate,
    validate_paired_rows,
)
from experiments.delta_v2.run_knowledge_eval import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODAL_APP_PATH,
    RUNTIME_PACKAGES,
)
from experiments.delta_v2.train_knowledge_lora import (
    RUN_CONFIG_SCHEMA,
    TARGET_MODULES_REGEX,
    KnowledgeTrainError,
    _load_jsonl,
    _load_yaml,
    _mapping,
    _relative_path,
    _sha256,
    validate_rows,
)


PROTOCOL_SCHEMA = "delta.knowledge_paired_lora_protocol.v1"
CONFIG_SCHEMA = "delta.knowledge_paired_lora_train_config.v1"
PROTOCOL_ID = "delta-v2-knowledge-paired-sft-control-v1"
CONDITION_ID = "d2_paired_data_sft_control"
RUN_ID = (
    "delta-v2-d2-knowledge-paired-data-sft-control-"
    "qwen35-08b-modal-v1"
)
DATASET_ID = "delta-v2-knowledge-paired-v2"
PROTOCOL_PATH = Path(
    "experiments/delta_v2/knowledge_paired_sft_control_protocol.yaml"
)
TRAIN_PATH = Path(
    "data/experiments/delta_v2/knowledge_paired_v2/train.jsonl"
)
PAIRS_PATH = Path(
    "data/experiments/delta_v2/knowledge_paired_v2/pairs.jsonl"
)
DEV_PATH = Path(
    "data/experiments/delta_v2/knowledge_adaptation_v1/dev.jsonl"
)
EVAL_PATH = Path(
    "data/experiments/delta_v2/knowledge_adaptation_v1/eval.jsonl"
)
ENGINE_PATH = Path(
    "experiments/delta_v2/knowledge_paired_lora.py"
)
OBJECTIVE = {
    "objective_id": OBJECTIVE_GENERATIVE,
    "recall_loss": "teacher-forced-assistant-token-cross-entropy",
    "boolean_pair_loss": (
        "mean-of-positive-and-negative-gold-cross-entropy"
    ),
    "pair_normalization": "one-pair-equals-one-training-unit",
    "truth_margin_term": "absent",
    "pair-ranking_term": "absent",
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
    "deterministic_algorithms": "required",
}


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
        raise KnowledgeTrainError("paired SFT control identity changed")
    trigger = _mapping(protocol.get("decision_trigger"), "decision_trigger")
    if dict(trigger) != {
        "paired_dataset_id": DATASET_ID,
        "paired_dataset_freeze_commit": "ea8e01a",
        "diagnostic_result": (
            "global-label-intercept-not-truth-conditioned"
        ),
        "data_warning_repaired": "annotation-only-negative-bank",
    }:
        raise KnowledgeTrainError("paired SFT control trigger changed")
    role = _mapping(protocol.get("causal_role"), "causal_role")
    if dict(role) != {
        "stage": 1,
        "name": "repaired-data-original-objective-control",
        "establishes_baseline_for": "paired-boolean-objective",
        "interpretable_as_objective_effect_vs_c7": False,
        "stage_2_only_change": "learning-objective",
    }:
        raise KnowledgeTrainError("paired SFT causal role changed")

    dataset = _mapping(protocol.get("dataset"), "dataset")
    freeze_path = repo_root / _relative_path(
        dataset.get("freeze_path"), "dataset.freeze_path"
    )
    train_path = repo_root / _relative_path(
        dataset.get("train_path"), "dataset.train_path"
    )
    pairs_path = repo_root / _relative_path(
        dataset.get("pairs_path"), "dataset.pairs_path"
    )
    dev_path = repo_root / _relative_path(
        dataset.get("dev_path"), "dataset.dev_path"
    )
    eval_path = repo_root / _relative_path(
        dataset.get("frozen_eval_path"), "dataset.frozen_eval_path"
    )
    if (
        dataset.get("dataset_id") != DATASET_ID
        or freeze_path != repo_root / FREEZE_PATH
        or train_path != repo_root / TRAIN_PATH
        or pairs_path != repo_root / PAIRS_PATH
        or dev_path != repo_root / DEV_PATH
        or eval_path != repo_root / EVAL_PATH
        or _sha256(freeze_path) != dataset.get("freeze_sha256")
        or _sha256(train_path) != dataset.get("train_sha256")
        or _sha256(pairs_path) != dataset.get("pairs_sha256")
        or _sha256(dev_path) != dataset.get("dev_sha256")
        or _sha256(eval_path) != dataset.get("frozen_eval_sha256")
        or dataset.get("train_rows") != 147
        or dataset.get("pair_rows") != 63
        or dataset.get("dev_rows") != 21
        or dataset.get("b2_prefix")
        != "datasets/experiments/delta_v2/knowledge_paired_v2/"
        or dataset.get("evaluation_wording_added_to_training") is not False
    ):
        raise KnowledgeTrainError("paired SFT dataset changed")
    freeze = _load_yaml(freeze_path)
    validate_freeze(freeze, repo_root=repo_root)

    implementation = _mapping(
        protocol.get("implementation"), "implementation"
    )
    engine_path = repo_root / _relative_path(
        implementation.get("engine_path"), "implementation.engine_path"
    )
    if (
        engine_path != repo_root / ENGINE_PATH
        or _sha256(engine_path) != implementation.get("engine_sha256")
    ):
        raise KnowledgeTrainError("paired SFT implementation changed")
    if dict(_mapping(protocol.get("model"), "model")) != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "initialization": "fresh-base",
        "base_weights": "frozen",
        "quantization": "none",
        "full_weight_training": "forbidden",
    }:
        raise KnowledgeTrainError("paired SFT model changed")
    if dict(_mapping(protocol.get("lora"), "lora")) != {
        "implementation": "peft-0.19.0",
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "bias": "none",
        "target_modules_regex": TARGET_MODULES_REGEX,
        "vision_modules_trainable": False,
    }:
        raise KnowledgeTrainError("paired SFT LoRA changed")
    if dict(
        _mapping(protocol.get("training_units"), "training_units")
    ) != {
        "construction": "fixed-recall-and-boolean-pair-units",
        "recall_units": 21,
        "boolean_pair_units": 63,
        "units_per_cycle": 84,
        "pair_contains": ["verify_true", "verify_false"],
        "pair_members_share_source_and-corruption-family": True,
        "deterministic_unit_shuffle": True,
    }:
        raise KnowledgeTrainError("paired SFT units changed")
    if dict(_mapping(protocol.get("objective"), "objective")) != OBJECTIVE:
        raise KnowledgeTrainError("paired SFT objective changed")
    if dict(
        _mapping(protocol.get("optimization"), "optimization")
    ) != OPTIMIZATION:
        raise KnowledgeTrainError("paired SFT optimization changed")
    if dict(
        _mapping(protocol.get("verification"), "verification")
    ) != {
        "first_gate": "repaired-pair-boolean-margin-v1",
        "verify_true_margin_accuracy_minimum": 0.8,
        "verify_false_margin_accuracy_minimum": 0.8,
        "truth_conditioned_pair_rate_minimum": 0.8,
        "per_corruption_family_false_accuracy_minimum": 0.8,
        "pooled_overall_score": "forbidden",
        "full_frozen_eval_only_after_first_gate": True,
        "stage_2_requires_separate_protocol_after_stage_1_receipt": True,
        "result": "candidate-only-no-promotion",
    }:
        raise KnowledgeTrainError("paired SFT verification changed")
    if dict(
        _mapping(protocol.get("execution_boundary"), "execution_boundary")
    ) != {
        "modal_training_authorized": True,
        "b2_adapter_write_authorized": True,
        "repaired_pair_diagnostic_authorized": True,
        "full_eval_conditional": True,
        "stage_2_training_authorized": False,
        "promotion_authorized": False,
    }:
        raise KnowledgeTrainError("paired SFT boundary changed")

    train_rows = _load_jsonl(train_path)
    pairs = _load_jsonl(pairs_path)
    dev_rows = _load_jsonl(dev_path)
    validate_paired_rows(train_rows, pairs)
    build_training_units(train_rows, pairs)
    validate_rows(
        dev_rows,
        expected_rows=21,
        expected_surfaces={"dev_recall": 21},
    )
    return train_rows, pairs, dev_rows


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
        raise KnowledgeTrainError("paired SFT config identity changed")
    binding = _mapping(config.get("protocol"), "protocol")
    protocol_path = repo_root / _relative_path(
        binding.get("path"), "protocol.path"
    )
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or protocol_path != repo_root / PROTOCOL_PATH
        or _sha256(protocol_path) != binding.get("sha256")
    ):
        raise KnowledgeTrainError("paired SFT protocol binding changed")
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
        raise KnowledgeTrainError("paired SFT config model changed")
    runtime = _mapping(config.get("runtime"), "runtime")
    if (
        runtime.get("profile") != "delta-v2-knowledge-modal-v1"
        or runtime.get("python") != "3.11.9"
        or runtime.get("packages") != RUNTIME_PACKAGES
        or runtime.get("device") != "cuda"
        or runtime.get("dtype") != "bfloat16"
        or runtime.get("deterministic_algorithms") is not True
    ):
        raise KnowledgeTrainError("paired SFT runtime changed")
    if dict(_mapping(config.get("data"), "data")) != {
        "train_path": str(TRAIN_PATH),
        "train_sha256": protocol["dataset"]["train_sha256"],
        "pairs_path": str(PAIRS_PATH),
        "pairs_sha256": protocol["dataset"]["pairs_sha256"],
        "eval_path": str(DEV_PATH),
        "eval_sha256": protocol["dataset"]["dev_sha256"],
    }:
        raise KnowledgeTrainError("paired SFT config data changed")
    if _mapping(config.get("lora"), "lora") != protocol["lora"]:
        raise KnowledgeTrainError("paired SFT config LoRA changed")
    if dict(_mapping(config.get("objective"), "objective")) != OBJECTIVE:
        raise KnowledgeTrainError("paired SFT config objective changed")
    if dict(_mapping(config.get("training"), "training")) != {
        "module": (
            "experiments.delta_v2.train_knowledge_paired_sft_control"
        ),
        **OPTIMIZATION,
    }:
        raise KnowledgeTrainError("paired SFT config training changed")
    if _relative_path(config["output"]["dir"], "output.dir") != (
        Path("runs") / RUN_ID
    ):
        raise KnowledgeTrainError("paired SFT config output changed")
    if dict(_mapping(config.get("modal"), "modal")) != {
        "app_path": MODAL_APP_PATH,
        "gpu": "H100",
        "timeout_seconds": 7200,
    }:
        raise KnowledgeTrainError("paired SFT Modal binding changed")
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
