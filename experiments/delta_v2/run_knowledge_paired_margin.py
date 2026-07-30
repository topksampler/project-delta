from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Protocol, Sequence

from experiments.delta_v2.knowledge_paired_data import (
    FREEZE_PATH,
    validate_freeze,
)
from experiments.delta_v2.knowledge_paired_lora import (
    EXPECTED_CORRUPTIONS,
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
from experiments.delta_v2.train_knowledge_lora import (
    KnowledgeTrainError,
    _load_jsonl,
    _load_yaml,
    _mapping,
    _relative_path,
    _sha256,
)


CONFIG_SCHEMA = "delta.knowledge_paired_margin_config.v1"
PROTOCOL_SCHEMA = "delta.knowledge_paired_margin_protocol.v1"
PROTOCOL_ID = "delta-v2-knowledge-paired-margin-v1"
CONDITION_ID = "d2_paired_data_sft_control_margin"
RUN_ID = (
    "delta-v2-d2-knowledge-paired-data-sft-control-"
    "margin-modal-v1"
)
TRAINING_RUN_ID = (
    "delta-v2-d2-knowledge-paired-data-sft-control-"
    "qwen35-08b-modal-v1"
)
PROTOCOL_PATH = Path(
    "experiments/delta_v2/knowledge_paired_margin_protocol.yaml"
)
TRAIN_PATH = Path(
    "data/experiments/delta_v2/knowledge_paired_v2/train.jsonl"
)
PAIRS_PATH = Path(
    "data/experiments/delta_v2/knowledge_paired_v2/pairs.jsonl"
)
ADAPTER_PATH = Path("runs") / TRAINING_RUN_ID / "adapter"
SAMPLE_SCHEMA = "delta.knowledge_paired_margin_sample.v1"
METRICS_SCHEMA = "delta.knowledge_paired_margin_metrics.v1"
RECEIPT_SCHEMA = "delta.knowledge_paired_margin_receipt.v1"
GATES = {
    "verify_true_margin_accuracy_minimum": 0.8,
    "verify_false_margin_accuracy_minimum": 0.8,
    "truth_conditioned_pair_rate_minimum": 0.8,
    "per_corruption_family_false_accuracy_minimum": 0.8,
}


class PairedMarginError(KnowledgeTrainError):
    """The repaired-pair margin gate cannot run safely."""


class PairedMarginBackend(Protocol):
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
        raise PairedMarginError("paired margin config identity changed")
    binding = _mapping(config.get("protocol"), "protocol")
    protocol_path = repo_root / _relative_path(
        binding.get("path"), "protocol.path"
    )
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or protocol_path != repo_root / PROTOCOL_PATH
        or _sha256(protocol_path) != binding.get("sha256")
    ):
        raise PairedMarginError("paired margin protocol binding changed")
    protocol = _load_yaml(protocol_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("experiment_id") != "delta_v2"
        or protocol.get("condition_id") != CONDITION_ID
        or protocol.get("state") != "preregistered-for-diagnostic"
    ):
        raise PairedMarginError("paired margin protocol identity changed")
    trigger = _mapping(protocol.get("decision_trigger"), "decision_trigger")
    if dict(trigger) != {
        "training_run_id": TRAINING_RUN_ID,
        "training_launch_commit": "fd3f8f3",
        "training_receipt_sha256": (
            "cff9d721dadbf48d8974504728ef0fba3b14e9b498a0a893f4dbdba8c32c8f07"
        ),
        "training_metrics_sha256": (
            "4cc7035c23ff0b949810e7b7e2ad599243cbb574bb3d111a4a234d9ccfea27b2"
        ),
        "adapter_sha256": (
            "9ef3caaf1f1d83a26df485ee0abcd365d1f932e0752cb9a99d9c5ac3510565a5"
        ),
        "objective_id": "assistant-only-generative-sft-v1",
    }:
        raise PairedMarginError("paired margin trigger changed")
    training_root = repo_root / "runs" / TRAINING_RUN_ID
    if (
        _sha256(training_root / "run_receipt.json")
        != trigger["training_receipt_sha256"]
        or _sha256(training_root / "train_metrics.json")
        != trigger["training_metrics_sha256"]
        or _sha256(
            repo_root / ADAPTER_PATH / "adapter_model.safetensors"
        )
        != trigger["adapter_sha256"]
    ):
        raise PairedMarginError("paired margin training evidence changed")

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
        dataset.get("dataset_id") != "delta-v2-knowledge-paired-v2"
        or freeze_path != repo_root / FREEZE_PATH
        or train_path != repo_root / TRAIN_PATH
        or pairs_path != repo_root / PAIRS_PATH
        or _sha256(freeze_path) != dataset.get("freeze_sha256")
        or _sha256(train_path) != dataset.get("train_sha256")
        or _sha256(pairs_path) != dataset.get("pairs_sha256")
        or dataset.get("train_rows") != 147
        or dataset.get("pair_rows") != 63
        or dataset.get("included_surfaces")
        != ["verify_true", "verify_false"]
        or dataset.get("included_rows") != 126
        or dataset.get("mutation") != "forbidden"
    ):
        raise PairedMarginError("paired margin dataset changed")
    validate_freeze(_load_yaml(freeze_path), repo_root=repo_root)
    rows = _load_jsonl(train_path)
    pairs = _load_jsonl(pairs_path)
    validate_paired_rows(rows, pairs)

    scoring = _mapping(protocol.get("scoring"), "scoring")
    if dict(scoring) != {
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
    }:
        raise PairedMarginError("paired margin scoring changed")
    gates = _mapping(protocol.get("gates"), "gates")
    if dict(gates) != {
        **GATES,
        "all_gates_required": True,
        "full_frozen_eval_only_if_pass": True,
        "paired_objective_training_requires_separate_protocol": True,
        "promotion_authorized": False,
    }:
        raise PairedMarginError("paired margin gates changed")
    boundary = _mapping(
        protocol.get("execution_boundary"), "execution_boundary"
    )
    if dict(boundary) != {
        "modal_diagnostic_authorized": True,
        "model_updates": 0,
        "optimizer_steps": 0,
        "adapter_mutation": "forbidden",
        "full_eval_conditional": True,
        "paired_objective_training_authorized": False,
        "promotion_authorized": False,
    }:
        raise PairedMarginError("paired margin boundary changed")

    model = _mapping(config.get("model"), "model")
    if dict(model) != {
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
        raise PairedMarginError("paired margin model changed")
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
        raise PairedMarginError("paired margin runtime changed")
    if dict(_mapping(config.get("data"), "data")) != {
        "train_path": str(TRAIN_PATH),
        "pairs_path": str(PAIRS_PATH),
    }:
        raise PairedMarginError("paired margin config data changed")
    if dict(_mapping(config.get("eval"), "eval")) != {
        "module": "experiments.delta_v2.run_knowledge_paired_margin"
    }:
        raise PairedMarginError("paired margin module changed")
    if _relative_path(config["output"]["dir"], "output.dir") != (
        Path("runs") / RUN_ID
    ):
        raise PairedMarginError("paired margin output changed")
    if dict(_mapping(config.get("modal"), "modal")) != {
        "app_path": MODAL_APP_PATH,
        "gpu": "A10G",
        "timeout_seconds": 3600,
    }:
        raise PairedMarginError("paired margin Modal binding changed")
    boolean_rows = [
        row
        for row in rows
        if row["surface"] in {"verify_true", "verify_false"}
    ]
    if len(boolean_rows) != 126:
        raise PairedMarginError("paired margin boolean rows changed")
    return protocol, boolean_rows, pairs


def summarize_samples(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(samples) != 126:
        raise PairedMarginError("paired margin sample count changed")
    by_surface: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_pair: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    by_family_false: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_source_true: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        surface = str(sample["surface"])
        by_surface[surface].append(sample)
        by_pair[str(sample["pair_id"])][surface] = sample
        if surface == "verify_false":
            by_family_false[str(sample["corruption_family"])].append(sample)
        else:
            by_source_true[str(sample["source_id"])].append(sample)
    if (
        set(by_surface) != {"verify_true", "verify_false"}
        or set(by_family_false) != set(EXPECTED_CORRUPTIONS)
        or len(by_pair) != 63
        or len(by_source_true) != 21
    ):
        raise PairedMarginError("paired margin grouping changed")

    per_surface: dict[str, Any] = {}
    for surface in ("verify_true", "verify_false"):
        rows = by_surface[surface]
        correct = sum(
            str(row["predicted"]) == str(row["gold"]) for row in rows
        )
        per_surface[surface] = {
            "items": len(rows),
            "margin_correct": correct,
            "margin_accuracy": correct / len(rows),
            "mean_teacher_forced_gold_nll": mean(
                float(row["teacher_forced_gold_nll"]) for row in rows
            ),
            "mean_sequence_margin_yes_minus_no": mean(
                float(row["sequence_margin_yes_minus_no"]) for row in rows
            ),
            "mean_gold_signed_margin": mean(
                float(row["gold_signed_margin"]) for row in rows
            ),
        }
    per_family: dict[str, Any] = {}
    for family, rows in sorted(by_family_false.items()):
        correct = sum(str(row["predicted"]) == "no" for row in rows)
        per_family[family] = {
            "items": len(rows),
            "false_margin_correct": correct,
            "false_margin_accuracy": correct / len(rows),
            "mean_false_margin_yes_minus_no": mean(
                float(row["sequence_margin_yes_minus_no"]) for row in rows
            ),
        }

    separations: list[float] = []
    truth_conditioned = 0
    for pair in by_pair.values():
        if set(pair) != {"verify_true", "verify_false"}:
            raise PairedMarginError("paired margin pair coverage changed")
        true_margin = float(
            pair["verify_true"]["sequence_margin_yes_minus_no"]
        )
        false_margin = float(
            pair["verify_false"]["sequence_margin_yes_minus_no"]
        )
        separations.append(true_margin - false_margin)
        truth_conditioned += true_margin > 0 and false_margin < 0
    unique_true_correct = 0
    for rows in by_source_true.values():
        predictions = {str(row["predicted"]) for row in rows}
        margins = {
            float(row["sequence_margin_yes_minus_no"]) for row in rows
        }
        if len(predictions) != 1 or len(margins) != 1:
            raise PairedMarginError("duplicate true prompts disagree")
        unique_true_correct += predictions == {"yes"}

    gate_cells = {
        "verify_true": (
            per_surface["verify_true"]["margin_accuracy"]
            >= GATES["verify_true_margin_accuracy_minimum"]
        ),
        "verify_false": (
            per_surface["verify_false"]["margin_accuracy"]
            >= GATES["verify_false_margin_accuracy_minimum"]
        ),
        "truth_conditioned_pairs": (
            truth_conditioned / len(separations)
            >= GATES["truth_conditioned_pair_rate_minimum"]
        ),
        "every_corruption_family_false": all(
            cell["false_margin_accuracy"]
            >= GATES["per_corruption_family_false_accuracy_minimum"]
            for cell in per_family.values()
        ),
    }
    passed = all(gate_cells.values())
    return {
        "schema": METRICS_SCHEMA,
        "run_id": RUN_ID,
        "condition_id": CONDITION_ID,
        "status": "pass" if passed else "fail",
        "per_surface": per_surface,
        "per_corruption_family_false": per_family,
        "unique_true_sources": {
            "items": 21,
            "margin_correct": unique_true_correct,
            "margin_accuracy": unique_true_correct / 21,
        },
        "paired": {
            "pairs": len(separations),
            "mean_pair_separation": mean(separations),
            "truth_conditioned_pairs": truth_conditioned,
            "truth_conditioned_pair_rate": (
                truth_conditioned / len(separations)
            ),
        },
        "gates": gate_cells,
        "first_gate_passed": passed,
        "full_frozen_eval_eligible": passed,
        "pooled_overall_score": None,
        "diagnostic_only": True,
        "optimizer_steps": 0,
        "promotion_authorized": False,
    }


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
                raise PairedMarginError("paired margin is not finite")
        samples.append(sample)
    metrics = summarize_samples(samples)
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
