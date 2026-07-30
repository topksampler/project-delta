from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from experiments.delta_v2.knowledge_paired_lora import (
    validate_paired_rows,
)
from experiments.delta_v2.run_knowledge_boolean_margin import (
    TransformersMarginBackend,
)
from experiments.delta_v2.run_knowledge_eval import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODAL_APP_PATH,
    RUNTIME_PACKAGES,
)
from experiments.delta_v2.run_knowledge_paired_margin import (
    summarize_samples,
)
from experiments.delta_v2.train_knowledge_lora import (
    KnowledgeTrainError,
    _load_jsonl,
    _load_yaml,
    _mapping,
    _relative_path,
    _sha256,
)


CONFIG_SCHEMA = "delta.knowledge_stability_replay_margin_config.v1"
PROTOCOL_SCHEMA = "delta.knowledge_stability_replay_margin_protocol.v1"
PROTOCOL_ID = "delta-v2-knowledge-stability-replay-margin-v1"
CONDITION_ID = "d4_stability_replay_margin"
RUN_ID = "delta-v2-d4-stability-replay-margin-modal-v1"
TRAINING_RUN_ID = (
    "delta-v2-d4-stability-replay-lora-qwen35-08b-modal-v1"
)
PROTOCOL_PATH = Path(
    "experiments/delta_v2/"
    "knowledge_stability_replay_margin_protocol.yaml"
)
TRAIN_PATH = Path(
    "data/experiments/delta_v2/knowledge_paired_v2/train.jsonl"
)
PAIRS_PATH = Path(
    "data/experiments/delta_v2/knowledge_paired_v2/pairs.jsonl"
)
FREEZE_PATH = Path(
    "experiments/delta_v2/knowledge_paired_data_freeze.yaml"
)
ADAPTER_PATH = Path("runs") / TRAINING_RUN_ID / "adapter"


class StabilityMarginError(KnowledgeTrainError):
    """The d4 acquisition margin pre-gate cannot run outside its receipt."""


class MarginBackend(Protocol):
    def score_row(self, row: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def runtime_receipt(self) -> Mapping[str, Any]: ...


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_config(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[
    Mapping[str, Any],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
]:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("run_id") != RUN_ID
        or config.get("experiment_id") != "delta_v2"
        or config.get("condition_id") != CONDITION_ID
        or config.get("task") != "eval"
    ):
        raise StabilityMarginError("d4 margin config identity changed")
    binding = _mapping(config.get("protocol"), "protocol")
    protocol_path = repo_root / _relative_path(
        binding.get("path"), "protocol.path"
    )
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or protocol_path != repo_root / PROTOCOL_PATH
        or _sha256(protocol_path) != binding.get("sha256")
    ):
        raise StabilityMarginError("d4 margin protocol binding changed")
    protocol = _load_yaml(protocol_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("condition_id") != CONDITION_ID
        or protocol.get("state") != "preregistered-for-pre-gate"
    ):
        raise StabilityMarginError("d4 margin protocol identity changed")
    trigger = _mapping(protocol.get("decision_trigger"), "decision_trigger")
    expected_trigger = {
        "training_run_id": TRAINING_RUN_ID,
        "training_launch_commit": "922f93b",
        "training_receipt_sha256": (
            "bf4b665b8274012ba6ad1e0e6d1417a3708756c24c0e5369a30f1bb89b1d488b"
        ),
        "training_metrics_sha256": (
            "6544aff7b1d426e2ed3bcef51537941e071ea7fa56ce7fcf8ebf1da449c0e1d1"
        ),
        "adapter_sha256": (
            "0260638f253090c6ab89e38e10ff025fbdc57ef9b736f5457faa26a339fed4c7"
        ),
        "objective_id": "assistant-only-generative-sft-v1",
        "schedule_sha256": (
            "7af9409e76c5c6242c6c1e949dce597c71668ccef75b98a7b4c5d0c1685a577a"
        ),
    }
    if dict(trigger) != expected_trigger:
        raise StabilityMarginError("d4 margin trigger changed")
    training_root = repo_root / "runs" / TRAINING_RUN_ID
    if (
        _sha256(training_root / "run_receipt.json")
        != trigger["training_receipt_sha256"]
        or _sha256(training_root / "train_metrics.json")
        != trigger["training_metrics_sha256"]
        or _sha256(ADAPTER_PATH / "adapter_model.safetensors")
        != trigger["adapter_sha256"]
    ):
        raise StabilityMarginError("d4 margin evidence changed")
    dataset = _mapping(protocol.get("dataset"), "dataset")
    if dict(dataset) != {
        "freeze_path": str(FREEZE_PATH),
        "freeze_sha256": "705eeff619fba976c395e3e8cd5fb53ab3e38d4badb1715dbf1ef995f5b8c490",
        "train_path": str(TRAIN_PATH),
        "train_sha256": "5d1fced8285c18684464a84c12d659a6f68fd50cacabe6c2ef598230a8be0db9",
        "pairs_path": str(PAIRS_PATH),
        "pairs_sha256": "0fc176cbb17e533591b7ac980285d3665a1cb062f26a091070bfbfebc9646819",
        "boolean_rows": 126,
        "pairs": 63,
    }:
        raise StabilityMarginError("d4 margin dataset changed")
    for path, expected_hash in (
        (FREEZE_PATH, dataset["freeze_sha256"]),
        (TRAIN_PATH, dataset["train_sha256"]),
        (PAIRS_PATH, dataset["pairs_sha256"]),
    ):
        if _sha256(repo_root / path) != expected_hash:
            raise StabilityMarginError("d4 margin data bytes changed")
    rows = _load_jsonl(repo_root / TRAIN_PATH)
    pairs = _load_jsonl(repo_root / PAIRS_PATH)
    validate_paired_rows(rows, pairs)
    if dict(_mapping(protocol.get("gates"), "gates")) != {
        "verify_true_margin_accuracy_minimum": 0.8,
        "verify_false_margin_accuracy_minimum": 0.8,
        "truth_conditioned_pair_rate_minimum": 0.8,
        "per_corruption_family_false_accuracy_minimum": 0.8,
        "all_gates_required": True,
        "pooled_overall_score": "forbidden",
    }:
        raise StabilityMarginError("d4 margin gates changed")
    boundary = _mapping(
        protocol.get("execution_boundary"), "execution_boundary"
    )
    if dict(boundary) != {
        "modal_diagnostic_authorized": True,
        "model_updates": 0,
        "adapter_mutation": "forbidden",
        "stability_dev_eval_authorized": True,
        "full_eval_authorized": False,
        "verify_v2_authorized": False,
        "promotion_authorized": False,
    }:
        raise StabilityMarginError("d4 margin boundary changed")
    if dict(_mapping(config.get("model"), "model")) != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "adapter": {
            "kind": "lora",
            "run_id": TRAINING_RUN_ID,
            "path": str(ADAPTER_PATH),
            "sha256": trigger["adapter_sha256"],
        },
        "quantization": "none",
    }:
        raise StabilityMarginError("d4 margin model changed")
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
        raise StabilityMarginError("d4 margin runtime changed")
    if dict(_mapping(config.get("data"), "data")) != {
        "train_path": str(TRAIN_PATH),
        "pairs_path": str(PAIRS_PATH),
    }:
        raise StabilityMarginError("d4 margin config data changed")
    expected_inputs = {
        "run_artifacts": [
            {
                "run_id": TRAINING_RUN_ID,
                "path": f"runs/{TRAINING_RUN_ID}/run_receipt.json",
                "sha256": trigger["training_receipt_sha256"],
            },
            {
                "run_id": TRAINING_RUN_ID,
                "path": f"runs/{TRAINING_RUN_ID}/train_metrics.json",
                "sha256": trigger["training_metrics_sha256"],
            },
        ]
    }
    if config.get("inputs") != expected_inputs:
        raise StabilityMarginError("d4 margin inputs changed")
    if config.get("eval") != {
        "module": (
            "experiments.delta_v2."
            "run_knowledge_stability_replay_margin"
        )
    }:
        raise StabilityMarginError("d4 margin module changed")
    if _relative_path(config["output"]["dir"], "output.dir") != Path("runs") / RUN_ID:
        raise StabilityMarginError("d4 margin output changed")
    if dict(_mapping(config.get("modal"), "modal")) != {
        "app_path": MODAL_APP_PATH,
        "gpu": "A10G",
        "timeout_seconds": 3600,
    }:
        raise StabilityMarginError("d4 margin Modal changed")
    boolean_rows = [
        row
        for row in rows
        if row["surface"] in {"verify_true", "verify_false"}
    ]
    return protocol, boolean_rows, pairs


def execute(
    *,
    config_path: Path,
    repo_root: Path,
    backend: MarginBackend | None = None,
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    protocol, rows, pairs = validate_config(config, repo_root=repo_root)
    pair_by_row = {
        str(row_id): pair
        for pair in pairs
        for row_id in (
            pair["positive_row_id"],
            pair["negative_row_id"],
        )
    }
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    active = backend or TransformersMarginBackend(
        config=config, repo_root=repo_root
    )
    samples = []
    for row in rows:
        sample = dict(active.score_row(row))
        pair = pair_by_row[str(row["row_id"])]
        sample.update(
            {
                "schema": "delta.knowledge_stability_margin_sample.v1",
                "pair_id": pair["pair_id"],
                "corruption_family": pair["corruption_family"],
                "changed_field": pair["changed_field"],
            }
        )
        if any(
            not math.isfinite(float(sample[field]))
            for field in (
                "teacher_forced_gold_nll",
                "sequence_margin_yes_minus_no",
                "gold_signed_margin",
            )
        ):
            raise StabilityMarginError("d4 margin is not finite")
        samples.append(sample)
    metrics = summarize_samples(samples)
    metrics["run_id"] = RUN_ID
    metrics["condition_id"] = CONDITION_ID
    metrics["schema"] = "delta.knowledge_stability_margin_metrics.v1"
    output_dir = repo_root / str(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    samples_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in samples
        ),
        encoding="utf-8",
    )
    metrics_path = output_dir / "metrics.json"
    _write_json(metrics_path, metrics)
    receipt = {
        "schema": "delta.knowledge_stability_margin_receipt.v1",
        "run_id": RUN_ID,
        "condition_id": CONDITION_ID,
        "protocol_id": PROTOCOL_ID,
        "training_run_id": TRAINING_RUN_ID,
        "config_sha256": _sha256(config_path),
        "protocol_sha256": _sha256(repo_root / PROTOCOL_PATH),
        "adapter_sha256": protocol["decision_trigger"]["adapter_sha256"],
        "samples_sha256": _sha256(samples_path),
        "metrics_sha256": _sha256(metrics_path),
        "runtime": dict(active.runtime_receipt()),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "model_updates": 0,
        "status": "pass",
        "first_gate_passed": metrics["first_gate_passed"],
        "promotion_authorized": False,
    }
    _write_json(output_dir / "run_receipt.json", receipt)
    return {"metrics": metrics, "receipt": receipt}


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
        _protocol, rows, pairs = validate_config(
            config, repo_root=repo_root
        )
        result = {
            "run_id": RUN_ID,
            "boolean_rows": len(rows),
            "pairs": len(pairs),
            "model_invocations_completed": 0,
            "status": "valid-unexecuted",
        }
    else:
        result = execute(config_path=config_path, repo_root=repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
