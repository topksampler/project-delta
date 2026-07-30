from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2.knowledge_stability_replay_lora import (
    train_stability_candidate,
)
from experiments.delta_v2.run_knowledge_eval import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODAL_APP_PATH,
    RUNTIME_PACKAGES,
)
from experiments.delta_v2.train_knowledge_lora import (
    KnowledgeTrainError,
    _load_yaml,
    _mapping,
    _relative_path,
    _sha256,
)
from experiments.delta_v2.train_knowledge_stability_replay import (
    ACQUISITION_PAIRS_PATH,
    ACQUISITION_TRAIN_PATH,
    DEV_PATH,
    REPLAY_PAIRS_PATH,
    REPLAY_TRAIN_PATH,
    _normalize_replay_rows,
)
from experiments.delta_v2.train_knowledge_stability_replay50 import (
    OPTIMIZATION,
    build_replay50_schedule,
    validate_protocol as validate_replay50_protocol,
)


PROTOCOL_SCHEMA = "delta.knowledge_stability_false2_lora_protocol.v1"
CONFIG_SCHEMA = "delta.knowledge_stability_false2_lora_config.v1"
PROTOCOL_ID = "delta-v2-knowledge-stability-false2-lora-v1"
CONDITION_ID = "d6_stability_false2_sft_control"
RUN_ID = "delta-v2-d6-stability-false2-lora-qwen35-08b-modal-v1"
DATASET_ID = "delta-v2-knowledge-stability-replay-v1"
PROTOCOL_PATH = Path(
    "experiments/delta_v2/knowledge_stability_false2_lora_protocol.yaml"
)
PARENT_PATH = Path(
    "experiments/delta_v2/knowledge_stability_replay50_lora_protocol.yaml"
)
D5_TRAIN_RUN = "delta-v2-d5-stability-replay50-lora-qwen35-08b-modal-v1"
D5_EVAL_RUN = "delta-v2-d5-stability-replay50-eval-modal-v1"
EXPECTED_D5 = {
    (D5_TRAIN_RUN, "run_receipt.json"):
        "448adf5dcd5b049d4cbf3edb2832c2b951ff11eaf059accb262305aa024f986c",
    (D5_TRAIN_RUN, "train_metrics.json"):
        "2971fb93488aeafcc88979327b00a951378b3b77bf4e0c56c95340d769191c7e",
    (D5_EVAL_RUN, "run_receipt.json"):
        "30fd04620b5b6194392e63279aaf73a8d2a0147aaab1abbff807f8f5b9a99dcc",
    (D5_EVAL_RUN, "metrics.json"):
        "eb4681387cf599dbc7babab55f36cc844270cb85c3acce6ff19f609e94a0be29",
}


def build_false2_schedule(
    acquisition_rows: Sequence[Mapping[str, Any]],
    acquisition_pairs: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    replay_pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    schedule = [
        dict(unit)
        for unit in build_replay50_schedule(
            acquisition_rows,
            acquisition_pairs,
            replay_rows,
            replay_pairs,
        )
    ]
    converted: Counter[str] = Counter()
    for unit in schedule:
        source = str(unit["source_stream"])
        if (
            unit["kind"] == "boolean_pair"
            and source in {"acquisition_pair", "replay_pair"}
            and converted[source] < 30
        ):
            unit["positive_row_id"] = unit["negative_row_id"]
            unit["loss_role"] = "negative-only"
            converted[source] += 1
        elif unit["kind"] == "boolean_pair":
            unit["loss_role"] = "balanced-pair"
        else:
            unit["loss_role"] = "recall"
    roles = Counter(str(unit["loss_role"]) for unit in schedule)
    if converted != {"acquisition_pair": 30, "replay_pair": 30} or roles != {
        "balanced-pair": 120,
        "negative-only": 60,
        "recall": 60,
    }:
        raise KnowledgeTrainError("false2 loss exposure schedule changed")
    return schedule


def validate_protocol(
    protocol: Mapping[str, Any], *, repo_root: Path
) -> tuple[Mapping[str, Any], tuple[list[Mapping[str, Any]], ...]]:
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("condition_id") != CONDITION_ID
        or protocol.get("state") != "preregistered-for-training"
    ):
        raise KnowledgeTrainError("false2 identity changed")
    trigger = _mapping(protocol.get("decision_trigger"), "trigger")
    if (
        trigger.get("parent_candidate")
        != "d5_stability_replay50_sft_control"
        or trigger.get("decision")
        != "double-negative-target-loss-exposure"
        or [
            trigger.get("d5_training_receipt_sha256"),
            trigger.get("d5_training_metrics_sha256"),
            trigger.get("d5_stability_receipt_sha256"),
            trigger.get("d5_stability_metrics_sha256"),
        ]
        != list(EXPECTED_D5.values())
    ):
        raise KnowledgeTrainError("false2 trigger changed")
    for (run_id, name), expected in EXPECTED_D5.items():
        if _sha256(repo_root / "runs" / run_id / name) != expected:
            raise KnowledgeTrainError("false2 evidence changed")
    parent_binding = _mapping(protocol.get("parent_protocol"), "parent")
    if (
        parent_binding.get("path") != str(PARENT_PATH)
        or parent_binding.get("sha256")
        != "50d3ce50b03ca863059aec955a4ab53da0549e52b64e4417824d957595c8c05f"
        or parent_binding.get("only_intervention_factor")
        != "boolean-negative-target-loss-exposure"
        or _sha256(repo_root / PARENT_PATH)
        != parent_binding.get("sha256")
    ):
        raise KnowledgeTrainError("false2 parent changed")
    parent = _load_yaml(repo_root / PARENT_PATH)
    base_protocol, datasets = validate_replay50_protocol(
        parent, repo_root=repo_root
    )
    objective = _mapping(protocol.get("objective"), "objective")
    boolean_loss = _mapping(objective.get("boolean_loss"), "boolean_loss")
    if (
        objective.get("objective_id")
        != "assistant-only-generative-sft-false2-v1"
        or objective.get("recall_loss")
        != "teacher-forced-assistant-token-cross-entropy"
        or dict(boolean_loss)
        != {
            "positive_target_exposures": 120,
            "negative_target_exposures": 240,
            "negative_to_positive_ratio": 2.0,
            "implementation":
                "120-balanced-pair-units-plus-60-negative-only-pair-units",
        }
        or objective.get("pair_normalization")
        != "one-schedule-unit-equals-one-training-unit"
        or objective.get("truth_margin_term") != "absent"
        or objective.get("pair-ranking_term") != "absent"
    ):
        raise KnowledgeTrainError("false2 objective changed")
    unchanged = _mapping(protocol.get("unchanged"), "unchanged")
    if dict(unchanged) != {
        "initialization": "fresh-base",
        "base_revision": MODEL_REVISION,
        "replay_share": 0.50,
        "total_training_units": 240,
        "acquisition_units": 120,
        "replay_units": 120,
        "recall_units": 60,
        "boolean_pair_units": 180,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "learning_rate": 0.0001,
        "optimizer_steps": 60,
        "gradient_accumulation_steps": 4,
        "seed": 20260730,
    }:
        raise KnowledgeTrainError("false2 fixed factors changed")
    boundary = _mapping(protocol.get("execution_boundary"), "boundary")
    if dict(boundary) != {
        "modal_training_authorized": True,
        "b2_adapter_write_authorized": True,
        "pre_gate_evaluation_authorized": True,
        "full_eval_authorized": False,
        "verify_v2_authorized": False,
        "qlora_authorized": False,
        "full_weight_authorized": False,
        "reinforcement_authorized": False,
        "promotion_authorized": False,
    }:
        raise KnowledgeTrainError("false2 boundary changed")
    build_false2_schedule(*datasets[:4])
    return base_protocol, datasets


def validate_config(
    config: Mapping[str, Any], *, repo_root: Path
) -> tuple[Mapping[str, Any], tuple[list[Mapping[str, Any]], ...]]:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("run_id") != RUN_ID
        or config.get("condition_id") != CONDITION_ID
        or config.get("task") != "train"
    ):
        raise KnowledgeTrainError("false2 config identity changed")
    binding = _mapping(config.get("protocol"), "protocol")
    path = repo_root / _relative_path(binding.get("path"), "protocol.path")
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or path != repo_root / PROTOCOL_PATH
        or _sha256(path) != binding.get("sha256")
    ):
        raise KnowledgeTrainError("false2 protocol binding changed")
    protocol = _load_yaml(path)
    base_protocol, datasets = validate_protocol(
        protocol, repo_root=repo_root
    )
    if config.get("model") != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "adapter": "new-lora",
        "quantization": "none",
    }:
        raise KnowledgeTrainError("false2 model changed")
    runtime = _mapping(config.get("runtime"), "runtime")
    if (
        runtime.get("packages") != RUNTIME_PACKAGES
        or runtime.get("device") != "cuda"
        or runtime.get("dtype") != "bfloat16"
        or runtime.get("deterministic_algorithms") is not True
    ):
        raise KnowledgeTrainError("false2 runtime changed")
    if config.get("lora") != base_protocol["lora"]:
        raise KnowledgeTrainError("false2 LoRA changed")
    if config.get("objective") != protocol["objective"]:
        raise KnowledgeTrainError("false2 objective binding changed")
    if config.get("training") != {
        "module": "experiments.delta_v2.train_knowledge_stability_false2",
        **OPTIMIZATION,
    }:
        raise KnowledgeTrainError("false2 optimization changed")
    if _relative_path(config["output"]["dir"], "output.dir") != Path("runs") / RUN_ID:
        raise KnowledgeTrainError("false2 output changed")
    if config.get("modal") != {
        "app_path": MODAL_APP_PATH,
        "gpu": "H100",
        "timeout_seconds": 7200,
    }:
        raise KnowledgeTrainError("false2 Modal changed")
    return base_protocol, datasets


def train(config_path: Path, repo_root: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    base_protocol, datasets = validate_config(config, repo_root=repo_root)
    acquisition_rows, acquisition_pairs, replay_rows, replay_pairs, dev_rows = datasets
    combined_rows = [
        *acquisition_rows,
        *_normalize_replay_rows(replay_rows),
    ]
    schedule = build_false2_schedule(*datasets[:4])
    result = train_stability_candidate(
        config_path=config_path,
        repo_root=repo_root,
        config=config,
        protocol=base_protocol,
        train_rows=combined_rows,
        dev_rows=dev_rows,
        schedule=schedule,
        run_id=RUN_ID,
        condition_id=CONDITION_ID,
        protocol_id=PROTOCOL_ID,
        protocol_path=PROTOCOL_PATH,
        dataset_id=DATASET_ID,
        acquisition_train_path=ACQUISITION_TRAIN_PATH,
        acquisition_pairs_path=ACQUISITION_PAIRS_PATH,
        replay_train_path=REPLAY_TRAIN_PATH,
        replay_pairs_path=REPLAY_PAIRS_PATH,
        dev_path=DEV_PATH,
    )
    source_counts = Counter(str(unit["source_stream"]) for unit in schedule)
    role_counts = Counter(str(unit["loss_role"]) for unit in schedule)
    metrics = result["metrics"]
    metrics["source_stream_counts"] = dict(sorted(source_counts.items()))
    metrics["actual_replay_share"] = 0.5
    metrics["loss_role_counts"] = dict(sorted(role_counts.items()))
    metrics["boolean_target_exposures"] = {
        "positive": 120,
        "negative": 240,
        "negative_to_positive_ratio": 2.0,
    }
    output_dir = repo_root / str(config["output"]["dir"])
    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = result["receipt"]
    receipt["objective_id"] = config["objective"]["objective_id"]
    receipt["metrics_sha256"] = _sha256(metrics_path)
    receipt["actual_replay_share"] = 0.5
    receipt["loss_role_counts"] = metrics["loss_role_counts"]
    receipt["boolean_target_exposures"] = metrics[
        "boolean_target_exposures"
    ]
    receipt_path = output_dir / "run_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


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
        _base, datasets = validate_config(config, repo_root=repo_root)
        schedule = build_false2_schedule(*datasets[:4])
        result = {
            "run_id": RUN_ID,
            "schedule_units": len(schedule),
            "schedule_sha256": hashlib.sha256(
                json.dumps(
                    schedule, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "source_stream_counts": dict(
                sorted(
                    Counter(
                        str(unit["source_stream"]) for unit in schedule
                    ).items()
                )
            ),
            "loss_role_counts": dict(
                sorted(
                    Counter(
                        str(unit["loss_role"]) for unit in schedule
                    ).items()
                )
            ),
            "model_invocations_completed": 0,
            "status": "valid-unexecuted",
        }
    else:
        result = train(config_path, repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
