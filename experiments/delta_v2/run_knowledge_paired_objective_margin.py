from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2.knowledge_paired_data import (
    FREEZE_PATH,
    validate_freeze,
)
from experiments.delta_v2.knowledge_paired_lora import validate_paired_rows
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
    METRICS_SCHEMA,
    RECEIPT_SCHEMA,
    SAMPLE_SCHEMA,
    PairedMarginBackend,
    PairedMarginError,
    _write_json,
    summarize_samples,
)
from experiments.delta_v2.train_knowledge_lora import (
    _load_jsonl,
    _load_yaml,
    _mapping,
    _relative_path,
    _sha256,
)


CONFIG_SCHEMA = "delta.knowledge_paired_margin_config.v1"
PROTOCOL_SCHEMA = "delta.knowledge_paired_margin_protocol.v1"
PROTOCOL_ID = "delta-v2-knowledge-paired-objective-margin-v1"
CONDITION_ID = "d3_paired_objective_lora_margin"
RUN_ID = (
    "delta-v2-d3-knowledge-paired-objective-margin-modal-v1"
)
TRAINING_RUN_ID = (
    "delta-v2-d3-knowledge-paired-objective-qwen35-08b-modal-v1"
)
TRAINING_PROTOCOL_PATH = Path(
    "experiments/delta_v2/knowledge_paired_objective_protocol.yaml"
)
SOURCE_MARGIN_PROTOCOL_PATH = Path(
    "experiments/delta_v2/knowledge_paired_margin_protocol.yaml"
)
PROTOCOL_PATH = Path(
    "experiments/delta_v2/"
    "knowledge_paired_objective_margin_protocol.yaml"
)
RUNNER_PATH = Path(
    "experiments/delta_v2/run_knowledge_paired_objective_margin.py"
)
TRAIN_PATH = Path(
    "data/experiments/delta_v2/knowledge_paired_v2/train.jsonl"
)
PAIRS_PATH = Path(
    "data/experiments/delta_v2/knowledge_paired_v2/pairs.jsonl"
)
ADAPTER_PATH = Path("runs") / TRAINING_RUN_ID / "adapter"
TRAINING_RECEIPT_SHA256 = (
    "9b78887be36662b2087a988eb49cb34606f57acf7eae38cda40cdf84f653b171"
)
TRAINING_METRICS_SHA256 = (
    "4ae603a3d0d7219b183e366d28a92716dac21fca14ecca2c885fb77810461bd0"
)
ADAPTER_SHA256 = (
    "3de6466516530533e0a060224df99e05966e1639f30fc71cb52aed2fb3ef21ae"
)
GATES = {
    "verify_true_margin_accuracy_minimum": 0.8,
    "verify_false_margin_accuracy_minimum": 0.8,
    "truth_conditioned_pair_rate_minimum": 0.8,
    "per_corruption_family_false_accuracy_minimum": 0.8,
}


def _expected_scoring() -> dict[str, Any]:
    return {
        "teacher_forced_gold_nll": "assistant-completion-sequence",
        "sequence_margin": (
            "log_probability_yes-minus-log_probability_no"
        ),
        "predicted_yes_if": "sequence-margin-strictly-positive",
        "pair_separation": (
            "true-sequence-margin-minus-false-sequence-margin"
        ),
        "truth_conditioned_pair": (
            "true-margin-positive-and-false-margin-negative"
        ),
        "deterministic_repeats": (
            "not-applicable-exact-forward-pass"
        ),
        "pooled_overall_score": "forbidden",
    }


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
        raise PairedMarginError(
            "paired objective margin config identity changed"
        )
    binding = _mapping(config.get("protocol"), "protocol")
    protocol_path = repo_root / _relative_path(
        binding.get("path"), "protocol.path"
    )
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or protocol_path != repo_root / PROTOCOL_PATH
        or _sha256(protocol_path) != binding.get("sha256")
    ):
        raise PairedMarginError(
            "paired objective margin protocol binding changed"
        )
    protocol = _load_yaml(protocol_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("experiment_id") != "delta_v2"
        or protocol.get("condition_id") != CONDITION_ID
        or protocol.get("state") != "preregistered-for-diagnostic"
    ):
        raise PairedMarginError(
            "paired objective margin protocol identity changed"
        )
    trigger = _mapping(protocol.get("decision_trigger"), "decision_trigger")
    training_binding = _mapping(
        trigger.get("training_protocol"),
        "decision_trigger.training_protocol",
    )
    training_protocol_path = repo_root / _relative_path(
        training_binding.get("path"),
        "decision_trigger.training_protocol.path",
    )
    if (
        dict(trigger)
        != {
            "training_protocol": {
                "path": str(TRAINING_PROTOCOL_PATH),
                "sha256": (
                    "610cf70ccc0678f2bb1d21e43f6142ea21a4ec0fc2da6a4a252af0fae8daaf67"
                ),
            },
            "training_run_id": TRAINING_RUN_ID,
            "training_launch_commit": "2086374",
            "training_receipt_sha256": TRAINING_RECEIPT_SHA256,
            "training_metrics_sha256": TRAINING_METRICS_SHA256,
            "adapter_sha256": ADAPTER_SHA256,
            "objective_id": "paired-boolean-logistic-ranking-v1",
        }
        or training_protocol_path != repo_root / TRAINING_PROTOCOL_PATH
        or _sha256(training_protocol_path)
        != training_binding["sha256"]
    ):
        raise PairedMarginError(
            "paired objective margin trigger changed"
        )
    training_root = repo_root / "runs" / TRAINING_RUN_ID
    if (
        _sha256(training_root / "run_receipt.json")
        != TRAINING_RECEIPT_SHA256
        or _sha256(training_root / "train_metrics.json")
        != TRAINING_METRICS_SHA256
        or _sha256(
            repo_root / ADAPTER_PATH / "adapter_model.safetensors"
        )
        != ADAPTER_SHA256
    ):
        raise PairedMarginError(
            "paired objective margin training evidence changed"
        )

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
    if (
        dict(dataset)
        != {
            "dataset_id": "delta-v2-knowledge-paired-v2",
            "freeze_path": str(FREEZE_PATH),
            "freeze_sha256": (
                "705eeff619fba976c395e3e8cd5fb53ab3e38d4badb1715dbf1ef995f5b8c490"
            ),
            "train_path": str(TRAIN_PATH),
            "train_sha256": (
                "5d1fced8285c18684464a84c12d659a6f68fd50cacabe6c2ef598230a8be0db9"
            ),
            "train_rows": 147,
            "pairs_path": str(PAIRS_PATH),
            "pairs_sha256": (
                "0fc176cbb17e533591b7ac980285d3665a1cb062f26a091070bfbfebc9646819"
            ),
            "pair_rows": 63,
            "included_surfaces": ["verify_true", "verify_false"],
            "included_rows": 126,
            "mutation": "forbidden",
        }
        or freeze_path != repo_root / FREEZE_PATH
        or train_path != repo_root / TRAIN_PATH
        or pairs_path != repo_root / PAIRS_PATH
        or _sha256(freeze_path) != dataset["freeze_sha256"]
        or _sha256(train_path) != dataset["train_sha256"]
        or _sha256(pairs_path) != dataset["pairs_sha256"]
    ):
        raise PairedMarginError(
            "paired objective margin dataset changed"
        )
    validate_freeze(_load_yaml(freeze_path), repo_root=repo_root)
    rows = _load_jsonl(train_path)
    pairs = _load_jsonl(pairs_path)
    validate_paired_rows(rows, pairs)

    scoring = _mapping(protocol.get("scoring"), "scoring")
    source_binding = _mapping(
        scoring.get("source_protocol"), "scoring.source_protocol"
    )
    source_path = repo_root / _relative_path(
        source_binding.get("path"), "scoring.source_protocol.path"
    )
    source_scoring = dict(scoring)
    source_scoring.pop("source_protocol")
    source_protocol = _load_yaml(source_path)
    if (
        dict(source_binding)
        != {
            "path": str(SOURCE_MARGIN_PROTOCOL_PATH),
            "sha256": (
                "4ba79e2fecb720cdec5d7a8a36b648b68669c1d14b38da72cd216a33930ba149"
            ),
        }
        or source_path != repo_root / SOURCE_MARGIN_PROTOCOL_PATH
        or _sha256(source_path) != source_binding["sha256"]
        or source_scoring != _expected_scoring()
        or source_scoring != source_protocol.get("scoring")
    ):
        raise PairedMarginError(
            "paired objective margin scoring changed"
        )
    if dict(_mapping(protocol.get("gates"), "gates")) != {
        **GATES,
        "all_gates_required": True,
        "full_frozen_eval_only_if_pass": True,
        "additional_training_requires_separate_protocol": True,
        "promotion_authorized": False,
    }:
        raise PairedMarginError("paired objective margin gates changed")
    implementation = _mapping(
        protocol.get("implementation"), "implementation"
    )
    runner_path = repo_root / _relative_path(
        implementation.get("path"), "implementation.path"
    )
    if (
        runner_path != repo_root / RUNNER_PATH
        or _sha256(runner_path) != implementation.get("sha256")
    ):
        raise PairedMarginError(
            "paired objective margin implementation changed"
        )
    if dict(
        _mapping(protocol.get("execution_boundary"), "execution_boundary")
    ) != {
        "modal_diagnostic_authorized": True,
        "model_updates": 0,
        "optimizer_steps": 0,
        "adapter_mutation": "forbidden",
        "full_eval_conditional": True,
        "additional_training_authorized": False,
        "promotion_authorized": False,
    }:
        raise PairedMarginError(
            "paired objective margin boundary changed"
        )

    if dict(_mapping(config.get("model"), "model")) != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "adapter": {
            "kind": "lora",
            "run_id": TRAINING_RUN_ID,
            "path": str(ADAPTER_PATH),
            "sha256": ADAPTER_SHA256,
        },
        "quantization": "none",
    }:
        raise PairedMarginError(
            "paired objective margin model changed"
        )
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
        raise PairedMarginError(
            "paired objective margin runtime changed"
        )
    if dict(_mapping(config.get("data"), "data")) != {
        "train_path": str(TRAIN_PATH),
        "pairs_path": str(PAIRS_PATH),
    }:
        raise PairedMarginError(
            "paired objective margin config data changed"
        )
    if dict(_mapping(config.get("inputs"), "inputs")) != {
        "run_artifacts": [
            {
                "run_id": TRAINING_RUN_ID,
                "path": str(
                    Path("runs") / TRAINING_RUN_ID / "run_receipt.json"
                ),
                "sha256": TRAINING_RECEIPT_SHA256,
            },
            {
                "run_id": TRAINING_RUN_ID,
                "path": str(
                    Path("runs") / TRAINING_RUN_ID / "train_metrics.json"
                ),
                "sha256": TRAINING_METRICS_SHA256,
            },
        ]
    }:
        raise PairedMarginError(
            "paired objective margin inputs changed"
        )
    if dict(_mapping(config.get("eval"), "eval")) != {
        "module": (
            "experiments.delta_v2."
            "run_knowledge_paired_objective_margin"
        )
    }:
        raise PairedMarginError(
            "paired objective margin module changed"
        )
    if _relative_path(config["output"]["dir"], "output.dir") != (
        Path("runs") / RUN_ID
    ):
        raise PairedMarginError(
            "paired objective margin output changed"
        )
    if dict(_mapping(config.get("modal"), "modal")) != {
        "app_path": MODAL_APP_PATH,
        "gpu": "A10G",
        "timeout_seconds": 3600,
    }:
        raise PairedMarginError(
            "paired objective margin Modal binding changed"
        )
    boolean_rows = [
        row
        for row in rows
        if row["surface"] in {"verify_true", "verify_false"}
    ]
    if len(boolean_rows) != 126:
        raise PairedMarginError(
            "paired objective margin Boolean rows changed"
        )
    return protocol, boolean_rows, pairs


def summarize_candidate_samples(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = summarize_samples(samples)
    metrics["run_id"] = RUN_ID
    metrics["condition_id"] = CONDITION_ID
    return metrics


def execute(
    *,
    config_path: Path,
    repo_root: Path,
    backend: PairedMarginBackend | None = None,
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    protocol, rows, pairs = validate_config(config, repo_root=repo_root)
    pair_by_row = {}
    for pair in pairs:
        pair_by_row[str(pair["positive_row_id"])] = pair
        pair_by_row[str(pair["negative_row_id"])] = pair
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    active = backend or TransformersMarginBackend(
        config=config,
        repo_root=repo_root,
    )
    samples: list[dict[str, Any]] = []
    for row in rows:
        sample = dict(active.score_row(row))
        pair = pair_by_row[str(row["row_id"])]
        sample.update(
            {
                "schema": SAMPLE_SCHEMA,
                "pair_id": pair["pair_id"],
                "corruption_family": pair["corruption_family"],
                "changed_field": pair["changed_field"],
            }
        )
        for field in (
            "teacher_forced_gold_nll",
            "sequence_margin_yes_minus_no",
            "gold_signed_margin",
        ):
            if not math.isfinite(float(sample[field])):
                raise PairedMarginError(
                    "paired objective margin is not finite"
                )
        samples.append(sample)
    metrics = summarize_candidate_samples(samples)
    output_dir = repo_root / str(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    metrics_path = output_dir / "metrics.json"
    samples_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in samples
        ),
        encoding="utf-8",
    )
    _write_json(metrics_path, metrics)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "run_id": RUN_ID,
        "condition_id": CONDITION_ID,
        "protocol_id": PROTOCOL_ID,
        "training_run_id": TRAINING_RUN_ID,
        "config_sha256": _sha256(config_path),
        "protocol_sha256": _sha256(repo_root / PROTOCOL_PATH),
        "train_sha256": _sha256(repo_root / TRAIN_PATH),
        "pairs_sha256": _sha256(repo_root / PAIRS_PATH),
        "adapter_sha256": protocol["decision_trigger"]["adapter_sha256"],
        "samples_sha256": _sha256(samples_path),
        "metrics_sha256": _sha256(metrics_path),
        "runtime": dict(active.runtime_receipt()),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "model_updates": 0,
        "optimizer_steps": 0,
        "status": "pass",
        "first_gate_passed": metrics["first_gate_passed"],
        "promotion_authorized": False,
    }
    receipt_path = output_dir / "run_receipt.json"
    _write_json(receipt_path, receipt)
    return {"metrics": metrics, "receipt": receipt}


def validate_only(config_path: Path, repo_root: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    _protocol, rows, pairs = validate_config(
        config, repo_root=repo_root
    )
    return {
        "run_id": RUN_ID,
        "protocol_id": PROTOCOL_ID,
        "boolean_rows": len(rows),
        "pairs": len(pairs),
        "model_invocations_completed": 0,
        "optimizer_steps": 0,
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
