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
    _relative_path,
    _sha256,
)
from experiments.delta_v2.train_knowledge_stability_replay import (
    ACQUISITION_PAIRS_PATH,
    ACQUISITION_TRAIN_PATH,
    DEV_PATH,
    OBJECTIVE,
    OPTIMIZATION,
    _normalize_replay_rows,
)
from experiments.delta_v2.train_knowledge_stability_replay50 import (
    build_replay50_schedule,
)


PROTOCOL_SCHEMA = "delta.knowledge_stability_claim_covered_lora_protocol.v1"
CONFIG_SCHEMA = "delta.knowledge_stability_claim_covered_lora_config.v1"
PROTOCOL_ID = "delta-v2-knowledge-stability-claim-covered-lora-v1"
CONDITION_ID = "d8_stability_claim_covered_sft_control"
RUN_ID = "delta-v2-d8-stability-claim-covered-lora-qwen35-08b-modal-v1"
DATASET_ID = "delta-v2-knowledge-stability-claim-covered-v1"
PROTOCOL_PATH = Path(
    "experiments/delta_v2/"
    "knowledge_stability_claim_covered_lora_protocol.yaml"
)
STABILITY_TRAIN_PATH = Path(
    "data/experiments/delta_v2/knowledge_stability_claim_covered_v1/"
    "train.jsonl"
)
STABILITY_PAIRS_PATH = Path(
    "data/experiments/delta_v2/knowledge_stability_claim_covered_v1/"
    "pairs.jsonl"
)
STABILITY_EVAL_PATH = Path(
    "data/experiments/delta_v2/knowledge_stability_claim_covered_v1/"
    "eval.jsonl"
)
D7_TRAIN_RUN = "delta-v2-d7-stability-lrhalf-lora-qwen35-08b-modal-v1"
D7_EVAL_RUN = "delta-v2-d7-stability-lrhalf-eval-modal-v1"
EVIDENCE = {
    (D7_TRAIN_RUN, "run_receipt.json"):
        "4248f3917f217a03f970322df02af9c9392cb8a24c1e8ef38dd81966b701832b",
    (D7_TRAIN_RUN, "train_metrics.json"):
        "9205fc73c1d9c5e06112c246072ac9ae140b329d78e55c84611000ac8e59f2c5",
    (D7_EVAL_RUN, "run_receipt.json"):
        "1fd15c8642c3feaa32ac76833bef241a0f6988ef4c7fccac4fa9cd6560752c24",
    (D7_EVAL_RUN, "metrics.json"):
        "1bd8b43cf8e363ce6d6ec0a35db9610f1429820a7307b7ebcc1e7b3ebc43ebb5",
}


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _engine_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    surfaces = {
        "claim_covered_recall": "replay_recall",
        "claim_covered_verify_true": "replay_verify_true",
        "claim_covered_verify_false": "replay_verify_false",
    }
    result = []
    for row in rows:
        item = dict(row)
        item["surface"] = surfaces[str(row["surface"])]
        result.append(item)
    return result


def validate_protocol(
    protocol: Mapping[str, Any], *, repo_root: Path
) -> tuple[list[Mapping[str, Any]], ...]:
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("condition_id") != CONDITION_ID
        or protocol.get("state") != "preregistered-for-training"
    ):
        raise KnowledgeTrainError("claim-covered identity changed")
    trigger = protocol["decision_trigger"]
    if (
        [
            trigger.get("d7_training_receipt_sha256"),
            trigger.get("d7_training_metrics_sha256"),
            trigger.get("d7_stability_receipt_sha256"),
            trigger.get("d7_stability_metrics_sha256"),
        ]
        != list(EVIDENCE.values())
        or trigger.get("decision")
        != "train-required-stable-claims-evaluate-held-out-wording"
    ):
        raise KnowledgeTrainError("claim-covered trigger changed")
    for (run_id, name), expected in EVIDENCE.items():
        if _sha256(repo_root / "runs" / run_id / name) != expected:
            raise KnowledgeTrainError("claim-covered evidence changed")
    data = protocol["data"]
    bindings = [
        ("acquisition_train", ACQUISITION_TRAIN_PATH, 147),
        ("acquisition_pairs", ACQUISITION_PAIRS_PATH, 63),
        ("acquisition_dev", DEV_PATH, 21),
        ("stability_train", STABILITY_TRAIN_PATH, 105),
        ("stability_pairs", STABILITY_PAIRS_PATH, 45),
        ("stability_eval", STABILITY_EVAL_PATH, 60),
    ]
    loaded = []
    for name, expected_path, rows in bindings:
        block = data[name]
        path = repo_root / _relative_path(block["path"], f"{name}.path")
        if (
            path != repo_root / expected_path
            or _sha256(path) != block["sha256"]
        ):
            raise KnowledgeTrainError(f"claim-covered {name} changed")
        value = _load_jsonl(path)
        if len(value) != rows:
            raise KnowledgeTrainError(f"claim-covered {name} rows changed")
        loaded.append(value)
    if protocol.get("unchanged") != {
        "objective": "assistant-only-generative-sft-v1",
        "initialization": "fresh-base",
        "base_revision": MODEL_REVISION,
        "replay_share": 0.50,
        "total_training_units": 240,
        "acquisition_units": 120,
        "stability_units": 120,
        "boolean_pair_units": 180,
        "recall_units": 60,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "learning_rate": 0.0001,
        "optimizer_steps": 60,
        "warmup_steps": 6,
        "gradient_accumulation_steps": 4,
        "seed": 20260730,
    }:
        raise KnowledgeTrainError("claim-covered fixed factors changed")
    boundary = protocol["execution_boundary"]
    if (
        boundary.get("modal_training_authorized") is not True
        or boundary.get("stability_pre_gate_authorized") is not True
        or boundary.get("full_eval_authorized") is not False
        or boundary.get("verify_v2_authorized") is not False
        or boundary.get("qlora_authorized") is not False
        or boundary.get("full_weight_authorized") is not False
        or boundary.get("reinforcement_authorized") is not False
        or boundary.get("promotion_authorized") is not False
    ):
        raise KnowledgeTrainError("claim-covered boundary changed")
    return tuple(loaded)


def validate_config(
    config: Mapping[str, Any], *, repo_root: Path
) -> tuple[Mapping[str, Any], tuple[list[Mapping[str, Any]], ...]]:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("run_id") != RUN_ID
        or config.get("condition_id") != CONDITION_ID
        or config.get("task") != "train"
    ):
        raise KnowledgeTrainError("claim-covered config changed")
    binding = config["protocol"]
    path = repo_root / _relative_path(binding["path"], "protocol.path")
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or path != repo_root / PROTOCOL_PATH
        or _sha256(path) != binding.get("sha256")
    ):
        raise KnowledgeTrainError("claim-covered protocol binding changed")
    protocol = _load_yaml(path)
    datasets = validate_protocol(protocol, repo_root=repo_root)
    if config.get("model") != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "adapter": "new-lora",
        "quantization": "none",
    }:
        raise KnowledgeTrainError("claim-covered model changed")
    runtime = config["runtime"]
    if (
        runtime.get("packages") != RUNTIME_PACKAGES
        or runtime.get("device") != "cuda"
        or runtime.get("dtype") != "bfloat16"
        or runtime.get("deterministic_algorithms") is not True
    ):
        raise KnowledgeTrainError("claim-covered runtime changed")
    if config.get("objective") != OBJECTIVE:
        raise KnowledgeTrainError("claim-covered objective changed")
    if config.get("training") != {
        "module":
            "experiments.delta_v2.train_knowledge_stability_claim_covered",
        **OPTIMIZATION,
    }:
        raise KnowledgeTrainError("claim-covered optimization changed")
    if _relative_path(config["output"]["dir"], "output.dir") != Path("runs") / RUN_ID:
        raise KnowledgeTrainError("claim-covered output changed")
    if config.get("modal") != {
        "app_path": MODAL_APP_PATH,
        "gpu": "H100",
        "timeout_seconds": 7200,
    }:
        raise KnowledgeTrainError("claim-covered Modal changed")
    return protocol, datasets


def train(config_path: Path, repo_root: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    protocol, datasets = validate_config(config, repo_root=repo_root)
    acq_rows, acq_pairs, dev_rows, stable_rows, stable_pairs, _eval = datasets
    transformed = _engine_rows(stable_rows)
    schedule = build_replay50_schedule(
        acq_rows, acq_pairs, transformed, stable_pairs
    )
    combined_rows = [
        *acq_rows,
        *_normalize_replay_rows(transformed),
    ]
    engine_protocol = {
        "objective": OBJECTIVE,
        "optimization": OPTIMIZATION,
        "lora": config["lora"],
    }
    result = train_stability_candidate(
        config_path=config_path,
        repo_root=repo_root,
        config=config,
        protocol=engine_protocol,
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
        replay_train_path=STABILITY_TRAIN_PATH,
        replay_pairs_path=STABILITY_PAIRS_PATH,
        dev_path=DEV_PATH,
    )
    source_counts = Counter(str(unit["source_stream"]) for unit in schedule)
    metrics = result["metrics"]
    metrics["source_stream_counts"] = dict(sorted(source_counts.items()))
    metrics["actual_stability_share"] = 0.5
    metrics["stability_training_coverage"] = "same-claims-new-wording"
    output_dir = repo_root / str(config["output"]["dir"])
    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = result["receipt"]
    receipt["metrics_sha256"] = _sha256(metrics_path)
    receipt["actual_stability_share"] = 0.5
    receipt["source_stream_counts"] = metrics["source_stream_counts"]
    receipt["stability_eval_sha256"] = _sha256(
        repo_root / STABILITY_EVAL_PATH
    )
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
        _protocol, datasets = validate_config(config, repo_root=repo_root)
        transformed = _engine_rows(datasets[3])
        schedule = build_replay50_schedule(
            datasets[0], datasets[1], transformed, datasets[4]
        )
        result = {
            "run_id": RUN_ID,
            "schedule_units": len(schedule),
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
