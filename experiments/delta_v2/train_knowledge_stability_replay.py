from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2.knowledge_paired_lora import (
    OBJECTIVE_GENERATIVE,
    build_training_units,
    validate_paired_rows,
)
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
    TARGET_MODULES_REGEX,
    KnowledgeTrainError,
    _load_jsonl,
    _load_yaml,
    _mapping,
    _relative_path,
    _sha256,
    validate_rows,
)


PROTOCOL_SCHEMA = "delta.knowledge_stability_replay_lora_protocol.v1"
CONFIG_SCHEMA = "delta.knowledge_stability_replay_lora_config.v1"
PROTOCOL_ID = "delta-v2-knowledge-stability-replay-lora-v1"
CONDITION_ID = "d4_stability_replay_sft_control"
RUN_ID = "delta-v2-d4-stability-replay-lora-qwen35-08b-modal-v1"
DATASET_ID = "delta-v2-knowledge-stability-replay-v1"
PROTOCOL_PATH = Path(
    "experiments/delta_v2/knowledge_stability_replay_lora_protocol.yaml"
)
ACQUISITION_TRAIN_PATH = Path(
    "data/experiments/delta_v2/knowledge_paired_v2/train.jsonl"
)
ACQUISITION_PAIRS_PATH = Path(
    "data/experiments/delta_v2/knowledge_paired_v2/pairs.jsonl"
)
REPLAY_TRAIN_PATH = Path(
    "data/experiments/delta_v2/knowledge_stability_replay_v1/"
    "replay_train.jsonl"
)
REPLAY_PAIRS_PATH = Path(
    "data/experiments/delta_v2/knowledge_stability_replay_v1/"
    "replay_pairs.jsonl"
)
DEV_PATH = Path(
    "data/experiments/delta_v2/knowledge_adaptation_v1/dev.jsonl"
)
ENGINE_PATH = Path(
    "experiments/delta_v2/knowledge_stability_replay_lora.py"
)
BASE_RUN_ID = "delta-v2-d4-stability-base-qwen35-08b-modal-v1"
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


def _binding_path(
    block: Mapping[str, Any],
    *,
    repo_root: Path,
    expected: Path,
    rows: int | None = None,
) -> Path:
    path = repo_root / _relative_path(block.get("path"), "data.path")
    if (
        path != repo_root / expected
        or _sha256(path) != block.get("sha256")
        or (rows is not None and block.get("rows") != rows)
    ):
        raise KnowledgeTrainError(f"stability binding changed: {expected}")
    return path


def _normalize_replay_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    surfaces = {
        "replay_recall": "exact_recall",
        "replay_verify_true": "verify_true",
        "replay_verify_false": "verify_false",
    }
    normalized = []
    for row in rows:
        item = dict(row)
        try:
            item["surface"] = surfaces[str(row["surface"])]
        except KeyError as exc:
            raise KnowledgeTrainError(
                "unexpected replay training surface"
            ) from exc
        item["schema"] = "delta.knowledge_paired_sft_row.v1"
        normalized.append(item)
    return normalized


def _normalize_replay_pairs(
    pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized = []
    for pair in pairs:
        item = dict(pair)
        item["schema"] = "delta.knowledge_boolean_pair.v1"
        normalized.append(item)
    return normalized


def _draw_stream(
    pool: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    order = [dict(item) for item in pool]
    rng.shuffle(order)
    result = []
    cursor = 0
    while len(result) < count:
        if cursor == len(order):
            rng.shuffle(order)
            cursor = 0
        result.append(dict(order[cursor]))
        cursor += 1
    return result


def build_stability_schedule(
    acquisition_rows: Sequence[Mapping[str, Any]],
    acquisition_pairs: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    replay_pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    acquisition_units = build_training_units(
        acquisition_rows, acquisition_pairs
    )
    acquisition_recall = [
        unit for unit in acquisition_units if unit["kind"] == "recall"
    ]
    acquisition_boolean = [
        unit
        for unit in acquisition_units
        if unit["kind"] == "boolean_pair"
    ]
    normalized_replay_rows = _normalize_replay_rows(replay_rows)
    normalized_replay_pairs = _normalize_replay_pairs(replay_pairs)
    combined_rows = [*acquisition_rows, *normalized_replay_rows]
    combined_pairs = [*acquisition_pairs, *normalized_replay_pairs]
    rows_by_id = {str(row["row_id"]): row for row in combined_rows}
    replay_recall = [
        {
            "unit_id": "replay-recall:" + str(row["row_id"]),
            "kind": "recall",
            "recall_row_id": str(row["row_id"]),
        }
        for row in normalized_replay_rows
        if row["surface"] == "exact_recall"
    ]
    replay_boolean = [
        {
            "unit_id": "replay-pair:" + str(pair["pair_id"]),
            "kind": "boolean_pair",
            "pair_id": str(pair["pair_id"]),
            "positive_row_id": str(pair["positive_row_id"]),
            "negative_row_id": str(pair["negative_row_id"]),
            "corruption_family": "replay:"
            + str(pair["corruption_family"]),
        }
        for pair in normalized_replay_pairs
    ]
    if (
        len(replay_recall) != 15
        or len(replay_boolean) != 45
        or len(rows_by_id) != 252
    ):
        raise KnowledgeTrainError("replay unit pool changed")
    streams = {
        "acquisition_pair": _draw_stream(
            acquisition_boolean, count=135, seed=20260731
        ),
        "acquisition_recall": _draw_stream(
            acquisition_recall, count=45, seed=20260732
        ),
        "replay_pair": _draw_stream(
            replay_boolean, count=45, seed=20260733
        ),
        "replay_recall": _draw_stream(
            replay_recall, count=15, seed=20260734
        ),
    }
    step_types = [
        [
            "acquisition_pair",
            "acquisition_pair",
            "acquisition_recall",
            "replay_pair",
        ]
        for _ in range(45)
    ]
    step_types.extend(
        [
            "acquisition_pair",
            "acquisition_pair",
            "acquisition_pair",
            "replay_recall",
        ]
        for _ in range(15)
    )
    random.Random(20260730).shuffle(step_types)
    cursors = Counter()
    schedule = []
    for step in step_types:
        for stream in step:
            unit = dict(streams[stream][cursors[stream]])
            cursors[stream] += 1
            unit["stream"] = stream
            schedule.append(unit)
    if (
        len(schedule) != 240
        or cursors
        != {
            "acquisition_pair": 135,
            "acquisition_recall": 45,
            "replay_pair": 45,
            "replay_recall": 15,
        }
        or any(
            row_id not in rows_by_id
            for unit in schedule
            for row_id in (
                [unit["recall_row_id"]]
                if unit["kind"] == "recall"
                else [unit["positive_row_id"], unit["negative_row_id"]]
            )
        )
    ):
        raise KnowledgeTrainError("stability schedule changed")
    return schedule


def validate_protocol(
    protocol: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
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
        raise KnowledgeTrainError("stability LoRA identity changed")
    trigger = _mapping(protocol.get("decision_trigger"), "decision_trigger")
    freeze = _mapping(trigger.get("data_freeze"), "data_freeze")
    freeze_path = repo_root / _relative_path(
        freeze.get("path"), "data_freeze.path"
    )
    if (
        trigger.get("design_id")
        != "delta-v2-knowledge-stability-replay-v1"
        or trigger.get("base_run_id") != BASE_RUN_ID
        or trigger.get("intervention")
        != "add-source-disjoint-stability-replay-at-fixed-budget"
        or freeze_path
        != repo_root
        / "experiments/delta_v2/"
        "knowledge_stability_replay_data_freeze.yaml"
        or freeze.get("sha256")
        != "e0956600529a056dc5d13328b2a13b28ff4294e2fa7eafcfddb4ed6bbccb8101"
        or _sha256(freeze_path) != freeze.get("sha256")
        or trigger.get("baseline")
        != {
            "choice": "0/15",
            "boolean_true": "1/15",
            "boolean_false": "14/15",
            "recall": "advisory-0/15",
        }
    ):
        raise KnowledgeTrainError("stability decision trigger changed")
    for name, expected_hash in (
        ("base_receipt", "e91cd432da6d524bfcbb05626386335336a540e14f4d99f017cc11ade50fec97"),
        ("base_metrics", "e6382bca092a87ff5587f7aaf5f1d113f46f00ee9f07f288efa3725be3858158"),
    ):
        block = _mapping(trigger.get(name), name)
        path = repo_root / _relative_path(block.get("path"), f"{name}.path")
        if _sha256(path) != expected_hash or block.get("sha256") != expected_hash:
            raise KnowledgeTrainError("stability baseline receipt changed")
    parent = _mapping(protocol.get("parent_control"), "parent_control")
    parent_path = repo_root / _relative_path(
        parent.get("path"), "parent_control.path"
    )
    if (
        parent_path
        != repo_root
        / "experiments/delta_v2/"
        "knowledge_paired_sft_control_protocol.yaml"
        or parent.get("sha256")
        != "7c97f2581636510403f7af8792e044fd7543c8be66500f5d923f73ef0feb72cd"
        or _sha256(parent_path) != parent.get("sha256")
    ):
        raise KnowledgeTrainError("stability parent control changed")
    data = _mapping(protocol.get("data"), "data")
    paths = [
        _binding_path(
            _mapping(data.get(name), name),
            repo_root=repo_root,
            expected=path,
            rows=rows,
        )
        for name, path, rows in (
            ("acquisition_train", ACQUISITION_TRAIN_PATH, 147),
            ("acquisition_pairs", ACQUISITION_PAIRS_PATH, 63),
            ("replay_train", REPLAY_TRAIN_PATH, 105),
            ("replay_pairs", REPLAY_PAIRS_PATH, 45),
            ("acquisition_dev", DEV_PATH, 21),
        )
    ]
    implementation = _mapping(
        protocol.get("implementation"), "implementation"
    )
    if (
        implementation.get("engine_path") != str(ENGINE_PATH)
        or _sha256(repo_root / ENGINE_PATH)
        != implementation.get("engine_sha256")
    ):
        raise KnowledgeTrainError("stability training engine changed")
    if dict(_mapping(protocol.get("model"), "model")) != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "initialization": "fresh-base",
        "base_weights": "frozen",
        "quantization": "none",
        "full_weight_training": "forbidden",
    }:
        raise KnowledgeTrainError("stability LoRA model changed")
    if dict(_mapping(protocol.get("lora"), "lora")) != {
        "implementation": "peft-0.19.0",
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "bias": "none",
        "target_modules_regex": TARGET_MODULES_REGEX,
        "vision_modules_trainable": False,
    }:
        raise KnowledgeTrainError("stability LoRA topology changed")
    if dict(_mapping(protocol.get("objective"), "objective")) != OBJECTIVE:
        raise KnowledgeTrainError("stability objective changed")
    if dict(_mapping(protocol.get("optimization"), "optimization")) != OPTIMIZATION:
        raise KnowledgeTrainError("stability optimization changed")
    schedule = _mapping(protocol.get("schedule"), "schedule")
    if (
        schedule.get("total_units") != 240
        or schedule.get("acquisition_units") != 180
        or schedule.get("replay_units") != 60
        or schedule.get("replay_share") != 0.25
        or schedule.get("acquisition_boolean_pairs") != 135
        or schedule.get("acquisition_recall") != 45
        or schedule.get("replay_boolean_pairs") != 45
        or schedule.get("replay_recall") != 15
        or schedule.get("step_types")
        != [
            {
                "count": 45,
                "units": [
                    "acquisition_pair",
                    "acquisition_pair",
                    "acquisition_recall",
                    "replay_pair",
                ],
            },
            {
                "count": 15,
                "units": [
                    "acquisition_pair",
                    "acquisition_pair",
                    "acquisition_pair",
                    "replay_recall",
                ],
            },
        ]
        or schedule.get("step_type_order")
        != "deterministic-seeded-shuffle"
        or schedule.get("within_stream_order")
        != "deterministic-seeded-shuffle"
    ):
        raise KnowledgeTrainError("stability schedule contract changed")
    verification = _mapping(
        protocol.get("verification"), "verification"
    )
    if (
        verification.get("full_eval_only_after_first_gate") is not True
        or verification.get("verify_v2_only_after_full_eval") is not True
        or verification.get("first_gate")
        != {
            "acquisition_pair_margin": {
                "verified_true_minimum": 0.80,
                "verified_false_minimum": 0.80,
                "truth_conditioned_pairs_minimum": 0.80,
                "every_corruption_family_false_minimum": 0.80,
            },
            "held_out_stability_dev": {
                "choice_parseable_minimum": 0.90,
                "boolean_parseable_minimum": 1.0,
                "boolean_true_accuracy_minimum": 0.80,
                "boolean_false_accuracy_minimum": 0.90,
                "recall": "advisory",
                "pooled_score": "forbidden",
            },
        }
    ):
        raise KnowledgeTrainError("stability verification changed")
    boundary = _mapping(
        protocol.get("execution_boundary"), "execution_boundary"
    )
    if (
        boundary.get("modal_training_authorized") is not True
        or boundary.get("b2_adapter_write_authorized") is not True
        or boundary.get("pre_gate_evaluation_authorized") is not True
        or boundary.get("full_eval_authorized") is not False
        or boundary.get("verify_v2_authorized") is not False
        or boundary.get("qlora_authorized") is not False
        or boundary.get("full_weight_authorized") is not False
        or boundary.get("reinforcement_authorized") is not False
        or boundary.get("promotion_authorized") is not False
    ):
        raise KnowledgeTrainError("stability execution boundary changed")
    acquisition_rows = _load_jsonl(paths[0])
    acquisition_pairs = _load_jsonl(paths[1])
    replay_rows = _load_jsonl(paths[2])
    replay_pairs = _load_jsonl(paths[3])
    dev_rows = _load_jsonl(paths[4])
    validate_paired_rows(acquisition_rows, acquisition_pairs)
    validate_rows(
        dev_rows,
        expected_rows=21,
        expected_surfaces={"dev_recall": 21},
    )
    build_stability_schedule(
        acquisition_rows, acquisition_pairs, replay_rows, replay_pairs
    )
    return (
        acquisition_rows,
        acquisition_pairs,
        replay_rows,
        replay_pairs,
        dev_rows,
    )


def validate_run_config(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[Mapping[str, Any], tuple[list[Mapping[str, Any]], ...]]:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("run_id") != RUN_ID
        or config.get("experiment_id") != "delta_v2"
        or config.get("condition_id") != CONDITION_ID
        or config.get("task") != "train"
    ):
        raise KnowledgeTrainError("stability run config changed")
    binding = _mapping(config.get("protocol"), "protocol")
    protocol_path = repo_root / _relative_path(
        binding.get("path"), "protocol.path"
    )
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or protocol_path != repo_root / PROTOCOL_PATH
        or _sha256(protocol_path) != binding.get("sha256")
    ):
        raise KnowledgeTrainError("stability protocol binding changed")
    protocol = _load_yaml(protocol_path)
    datasets = validate_protocol(protocol, repo_root=repo_root)
    if dict(_mapping(config.get("model"), "model")) != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "adapter": "new-lora",
        "quantization": "none",
    }:
        raise KnowledgeTrainError("stability config model changed")
    runtime = _mapping(config.get("runtime"), "runtime")
    if (
        runtime.get("profile") != "delta-v2-knowledge-modal-v1"
        or runtime.get("python") != "3.11.9"
        or runtime.get("packages") != RUNTIME_PACKAGES
        or runtime.get("device") != "cuda"
        or runtime.get("dtype") != "bfloat16"
        or runtime.get("deterministic_algorithms") is not True
    ):
        raise KnowledgeTrainError("stability runtime changed")
    if dict(_mapping(config.get("data"), "data")) != {
        "train_path": str(ACQUISITION_TRAIN_PATH),
        "train_sha256": protocol["data"]["acquisition_train"]["sha256"],
        "pairs_path": str(ACQUISITION_PAIRS_PATH),
        "pairs_sha256": protocol["data"]["acquisition_pairs"]["sha256"],
        "replay_train_path": str(REPLAY_TRAIN_PATH),
        "replay_train_sha256": protocol["data"]["replay_train"]["sha256"],
        "replay_pairs_path": str(REPLAY_PAIRS_PATH),
        "replay_pairs_sha256": protocol["data"]["replay_pairs"]["sha256"],
        "eval_path": str(DEV_PATH),
        "eval_sha256": protocol["data"]["acquisition_dev"]["sha256"],
    }:
        raise KnowledgeTrainError("stability config data changed")
    if config.get("inputs") != {
        "run_artifacts": [
            {
                "run_id": BASE_RUN_ID,
                "path": (
                    f"runs/{BASE_RUN_ID}/run_receipt.json"
                ),
                "sha256": protocol["decision_trigger"][
                    "base_receipt"
                ]["sha256"],
            },
            {
                "run_id": BASE_RUN_ID,
                "path": f"runs/{BASE_RUN_ID}/metrics.json",
                "sha256": protocol["decision_trigger"][
                    "base_metrics"
                ]["sha256"],
            },
        ]
    }:
        raise KnowledgeTrainError("stability baseline inputs changed")
    if _mapping(config.get("lora"), "lora") != protocol["lora"]:
        raise KnowledgeTrainError("stability config LoRA changed")
    if dict(_mapping(config.get("objective"), "objective")) != OBJECTIVE:
        raise KnowledgeTrainError("stability config objective changed")
    if dict(_mapping(config.get("training"), "training")) != {
        "module": "experiments.delta_v2.train_knowledge_stability_replay",
        **OPTIMIZATION,
    }:
        raise KnowledgeTrainError("stability config training changed")
    if _relative_path(config["output"]["dir"], "output.dir") != Path("runs") / RUN_ID:
        raise KnowledgeTrainError("stability output changed")
    if dict(_mapping(config.get("modal"), "modal")) != {
        "app_path": MODAL_APP_PATH,
        "gpu": "H100",
        "timeout_seconds": 7200,
    }:
        raise KnowledgeTrainError("stability Modal binding changed")
    return protocol, datasets


def train(config_path: Path, repo_root: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    protocol, datasets = validate_run_config(config, repo_root=repo_root)
    (
        acquisition_rows,
        acquisition_pairs,
        replay_rows,
        replay_pairs,
        dev_rows,
    ) = datasets
    normalized_replay_rows = _normalize_replay_rows(replay_rows)
    combined_rows = [*acquisition_rows, *normalized_replay_rows]
    schedule = build_stability_schedule(
        acquisition_rows, acquisition_pairs, replay_rows, replay_pairs
    )
    return train_stability_candidate(
        config_path=config_path,
        repo_root=repo_root,
        config=config,
        protocol=protocol,
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
        protocol, datasets = validate_run_config(
            config, repo_root=repo_root
        )
        schedule = build_stability_schedule(*datasets[:4])
        counts = Counter(str(unit["stream"]) for unit in schedule)
        result = {
            "run_id": RUN_ID,
            "protocol_id": PROTOCOL_ID,
            "objective_id": protocol["objective"]["objective_id"],
            "schedule_units": len(schedule),
            "stream_counts": dict(sorted(counts.items())),
            "schedule_sha256": hashlib.sha256(
                json.dumps(
                    schedule, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
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
