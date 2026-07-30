from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2.knowledge_paired_lora import (
    build_training_units,
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
    OBJECTIVE,
    OPTIMIZATION,
    REPLAY_PAIRS_PATH,
    REPLAY_TRAIN_PATH,
    _draw_stream,
    _normalize_replay_pairs,
    _normalize_replay_rows,
    validate_protocol as validate_parent_protocol,
)


PROTOCOL_SCHEMA = "delta.knowledge_stability_replay50_lora_protocol.v1"
CONFIG_SCHEMA = "delta.knowledge_stability_replay50_lora_config.v1"
PROTOCOL_ID = "delta-v2-knowledge-stability-replay50-lora-v1"
CONDITION_ID = "d5_stability_replay50_sft_control"
RUN_ID = "delta-v2-d5-stability-replay50-lora-qwen35-08b-modal-v1"
DATASET_ID = "delta-v2-knowledge-stability-replay-v1"
PROTOCOL_PATH = Path(
    "experiments/delta_v2/"
    "knowledge_stability_replay50_lora_protocol.yaml"
)
PARENT_PATH = Path(
    "experiments/delta_v2/knowledge_stability_replay_lora_protocol.yaml"
)
D4_TRAIN_RUN = "delta-v2-d4-stability-replay-lora-qwen35-08b-modal-v1"
D4_MARGIN_RUN = "delta-v2-d4-stability-replay-margin-modal-v1"
D4_STABILITY_RUN = "delta-v2-d4-stability-replay-eval-modal-v1"


def build_replay50_schedule(
    acquisition_rows: Sequence[Mapping[str, Any]],
    acquisition_pairs: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    replay_pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    acquisition = build_training_units(
        acquisition_rows, acquisition_pairs
    )
    acquisition_recall = [
        unit for unit in acquisition if unit["kind"] == "recall"
    ]
    acquisition_pair = [
        unit for unit in acquisition if unit["kind"] == "boolean_pair"
    ]
    normalized_rows = _normalize_replay_rows(replay_rows)
    normalized_pairs = _normalize_replay_pairs(replay_pairs)
    replay_recall = [
        {
            "unit_id": "replay50-recall:" + str(row["row_id"]),
            "kind": "recall",
            "recall_row_id": str(row["row_id"]),
        }
        for row in normalized_rows
        if row["surface"] == "exact_recall"
    ]
    replay_pair = [
        {
            "unit_id": "replay50-pair:" + str(pair["pair_id"]),
            "kind": "boolean_pair",
            "pair_id": str(pair["pair_id"]),
            "positive_row_id": str(pair["positive_row_id"]),
            "negative_row_id": str(pair["negative_row_id"]),
            "corruption_family": "replay:"
            + str(pair["corruption_family"]),
        }
        for pair in normalized_pairs
    ]
    streams = {
        "acquisition_pair": _draw_stream(
            acquisition_pair, count=90, seed=20260801
        ),
        "acquisition_recall": _draw_stream(
            acquisition_recall, count=30, seed=20260802
        ),
        "replay_pair": _draw_stream(
            replay_pair, count=90, seed=20260803
        ),
        "replay_recall": _draw_stream(
            replay_recall, count=30, seed=20260804
        ),
    }
    actual_steps = [
        [
            "acquisition_pair",
            "acquisition_pair",
            "replay_pair",
            "replay_recall",
        ]
        for _ in range(30)
    ]
    actual_steps.extend(
        [
            "acquisition_pair",
            "acquisition_recall",
            "replay_pair",
            "replay_pair",
        ]
        for _ in range(30)
    )
    random.Random(20260730).shuffle(actual_steps)
    compatibility_steps = [
        [
            "acquisition_pair",
            "acquisition_pair",
            "acquisition_recall",
            "replay_pair",
        ]
        for _ in range(45)
    ]
    compatibility_steps.extend(
        [
            "acquisition_pair",
            "acquisition_pair",
            "acquisition_pair",
            "replay_recall",
        ]
        for _ in range(15)
    )
    random.Random(20260805).shuffle(compatibility_steps)
    cursors = Counter()
    schedule = []
    for actual_step, engine_step in zip(
        actual_steps, compatibility_steps, strict=True
    ):
        for source_stream, engine_slot in zip(
            actual_step, engine_step, strict=True
        ):
            unit = dict(streams[source_stream][cursors[source_stream]])
            cursors[source_stream] += 1
            unit["source_stream"] = source_stream
            unit["stream"] = engine_slot
            schedule.append(unit)
    if (
        len(schedule) != 240
        or Counter(unit["source_stream"] for unit in schedule)
        != {
            "acquisition_pair": 90,
            "acquisition_recall": 30,
            "replay_pair": 90,
            "replay_recall": 30,
        }
    ):
        raise KnowledgeTrainError("replay50 source schedule changed")
    return schedule


def validate_protocol(
    protocol: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[Mapping[str, Any], tuple[list[Mapping[str, Any]], ...]]:
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("condition_id") != CONDITION_ID
        or protocol.get("state") != "preregistered-for-training"
    ):
        raise KnowledgeTrainError("replay50 identity changed")
    trigger = _mapping(protocol.get("decision_trigger"), "trigger")
    expected_hashes = {
        (D4_TRAIN_RUN, "run_receipt.json"): (
            "bf4b665b8274012ba6ad1e0e6d1417a3708756c24c0e5369a30f1bb89b1d488b"
        ),
        (D4_MARGIN_RUN, "run_receipt.json"): (
            "f0a12cfebf9a54bf82cd1b42b700b1a3a3a28c8319ba201d4092e2524d360c22"
        ),
        (D4_MARGIN_RUN, "metrics.json"): (
            "b2d38bd87f1a37841c195047ae6fa3d7617d602e1aa7ef48b91a059dbad25e9d"
        ),
        (D4_STABILITY_RUN, "run_receipt.json"): (
            "c26f326f5cbd6f77076bab4b607d2dfc8f78e148c952cdd70d367123da1c91c6"
        ),
        (D4_STABILITY_RUN, "metrics.json"): (
            "7c97c52a0603bf7b9b49d4d36679cad852536f79cfb0e128e6eeef3649ba4d83"
        ),
    }
    declared = [
        trigger.get("d4_training_receipt_sha256"),
        trigger.get("d4_margin_receipt_sha256"),
        trigger.get("d4_margin_metrics_sha256"),
        trigger.get("d4_stability_receipt_sha256"),
        trigger.get("d4_stability_metrics_sha256"),
    ]
    if (
        declared != list(expected_hashes.values())
        or trigger.get("decision")
        != "increase-replay-share-at-fixed-total-budget"
    ):
        raise KnowledgeTrainError("replay50 trigger changed")
    for (run_id, name), expected in expected_hashes.items():
        if _sha256(repo_root / "runs" / run_id / name) != expected:
            raise KnowledgeTrainError("replay50 evidence changed")
    parent_binding = _mapping(protocol.get("parent_protocol"), "parent")
    if (
        parent_binding.get("path") != str(PARENT_PATH)
        or parent_binding.get("sha256")
        != "34e798b2766eafbf5e799ba50cfd9945cdb4003b9edd3779cc705db41ebf18c6"
        or parent_binding.get("only_intervention_factor")
        != "replay-share"
        or _sha256(repo_root / PARENT_PATH)
        != parent_binding.get("sha256")
    ):
        raise KnowledgeTrainError("replay50 parent changed")
    parent = _load_yaml(repo_root / PARENT_PATH)
    datasets = validate_parent_protocol(parent, repo_root=repo_root)
    if protocol.get("unchanged") != {
        "objective": "assistant-only-generative-sft-v1",
        "initialization": "fresh-base",
        "base_revision": MODEL_REVISION,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "learning_rate": 0.0001,
        "optimizer_steps": 60,
        "gradient_accumulation_steps": 4,
        "seed": 20260730,
    }:
        raise KnowledgeTrainError("replay50 fixed factors changed")
    schedule = _mapping(protocol.get("schedule"), "schedule")
    if (
        schedule.get("total_units") != 240
        or schedule.get("acquisition_units") != 120
        or schedule.get("replay_units") != 120
        or schedule.get("replay_share") != 0.50
        or schedule.get("acquisition_boolean_pairs") != 90
        or schedule.get("acquisition_recall") != 30
        or schedule.get("replay_boolean_pairs") != 90
        or schedule.get("replay_recall") != 30
        or schedule.get("total_optimizer_steps_unchanged") is not True
    ):
        raise KnowledgeTrainError("replay50 schedule contract changed")
    boundary = _mapping(
        protocol.get("execution_boundary"), "boundary"
    )
    if (
        boundary.get("modal_training_authorized") is not True
        or boundary.get("b2_adapter_write_authorized") is not True
        or boundary.get("full_eval_authorized") is not False
        or boundary.get("verify_v2_authorized") is not False
        or boundary.get("qlora_authorized") is not False
        or boundary.get("full_weight_authorized") is not False
        or boundary.get("reinforcement_authorized") is not False
        or boundary.get("promotion_authorized") is not False
    ):
        raise KnowledgeTrainError("replay50 boundary changed")
    build_replay50_schedule(*datasets[:4])
    return parent, datasets


def validate_config(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    tuple[list[Mapping[str, Any]], ...],
]:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("run_id") != RUN_ID
        or config.get("condition_id") != CONDITION_ID
        or config.get("task") != "train"
    ):
        raise KnowledgeTrainError("replay50 config identity changed")
    binding = _mapping(config.get("protocol"), "protocol")
    path = repo_root / _relative_path(binding.get("path"), "protocol.path")
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or path != repo_root / PROTOCOL_PATH
        or _sha256(path) != binding.get("sha256")
    ):
        raise KnowledgeTrainError("replay50 protocol binding changed")
    protocol = _load_yaml(path)
    parent, datasets = validate_protocol(protocol, repo_root=repo_root)
    if config.get("model") != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "adapter": "new-lora",
        "quantization": "none",
    }:
        raise KnowledgeTrainError("replay50 model changed")
    runtime = _mapping(config.get("runtime"), "runtime")
    if (
        runtime.get("packages") != RUNTIME_PACKAGES
        or runtime.get("device") != "cuda"
        or runtime.get("dtype") != "bfloat16"
        or runtime.get("deterministic_algorithms") is not True
    ):
        raise KnowledgeTrainError("replay50 runtime changed")
    if config.get("lora") != parent["lora"] or config.get("objective") != OBJECTIVE:
        raise KnowledgeTrainError("replay50 training factors changed")
    if config.get("training") != {
        "module": "experiments.delta_v2.train_knowledge_stability_replay50",
        **OPTIMIZATION,
    }:
        raise KnowledgeTrainError("replay50 optimization changed")
    if _relative_path(config["output"]["dir"], "output.dir") != Path("runs") / RUN_ID:
        raise KnowledgeTrainError("replay50 output changed")
    if config.get("modal") != {
        "app_path": MODAL_APP_PATH,
        "gpu": "H100",
        "timeout_seconds": 7200,
    }:
        raise KnowledgeTrainError("replay50 Modal changed")
    return protocol, parent, datasets


def train(config_path: Path, repo_root: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    protocol, parent, datasets = validate_config(
        config, repo_root=repo_root
    )
    acquisition_rows, acquisition_pairs, replay_rows, replay_pairs, dev_rows = datasets
    combined_rows = [
        *acquisition_rows,
        *_normalize_replay_rows(replay_rows),
    ]
    schedule = build_replay50_schedule(*datasets[:4])
    result = train_stability_candidate(
        config_path=config_path,
        repo_root=repo_root,
        config=config,
        protocol=parent,
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
    source_counts = Counter(
        str(unit["source_stream"]) for unit in schedule
    )
    metrics = result["metrics"]
    metrics["source_stream_counts"] = dict(sorted(source_counts.items()))
    metrics["actual_replay_share"] = (
        source_counts["replay_pair"] + source_counts["replay_recall"]
    ) / len(schedule)
    output_dir = repo_root / str(config["output"]["dir"])
    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = result["receipt"]
    receipt["metrics_sha256"] = _sha256(metrics_path)
    receipt["actual_replay_share"] = metrics["actual_replay_share"]
    receipt["source_stream_counts"] = metrics["source_stream_counts"]
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
        _protocol, _parent, datasets = validate_config(
            config, repo_root=repo_root
        )
        schedule = build_replay50_schedule(*datasets[:4])
        result = {
            "run_id": RUN_ID,
            "schedule_units": len(schedule),
            "source_stream_counts": dict(
                sorted(
                    Counter(
                        unit["source_stream"] for unit in schedule
                    ).items()
                )
            ),
            "schedule_sha256": hashlib.sha256(
                json.dumps(
                    schedule, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "model_invocations_completed": 0,
            "status": "valid-unexecuted",
        }
    else:
        result = train(config_path, repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
