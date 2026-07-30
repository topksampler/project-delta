from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from experiments.delta_v2.run_knowledge_eval import (
    ADAPTER_RUN_ID,
    BASE_RUN_ID,
    DATASET_ID,
    METRICS_SCHEMA,
    PROTOCOL_ID,
    RECEIPT_SCHEMA,
)


COMPARISON_SCHEMA = "delta.knowledge_condition_comparison.v1"
LORA_PROTOCOL_PATH = Path(
    "experiments/delta_v2/knowledge_lora_protocol_v2.yaml"
)
EXPECTED_CELLS = {
    "acquisition_added.choice": 21,
    "acquisition_added.boolean_true": 21,
    "acquisition_added.boolean_false": 21,
    "acquisition_added.recall": 21,
    "retention_stable.choice": 21,
    "retention_stable.boolean_true": 21,
    "retention_stable.boolean_false": 21,
    "retention_stable.recall": 21,
    "feature_retention.verified_behavior_json": 1,
}


class KnowledgeComparisonError(ValueError):
    """The frozen base and LoRA results cannot be compared safely."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KnowledgeComparisonError(f"{field} must be a mapping")
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeComparisonError(f"cannot read JSON: {path}") from exc
    return _mapping(payload, str(path))


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise KnowledgeComparisonError(f"cannot read YAML: {path}") from exc
    return _mapping(payload, str(path))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_result(
    *,
    run_dir: Path,
    run_id: str,
    condition_id: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    metrics_path = run_dir / "metrics.json"
    receipt_path = run_dir / "run_receipt.json"
    metrics = _load_json(metrics_path)
    receipt = _load_json(receipt_path)
    if (
        metrics.get("schema") != METRICS_SCHEMA
        or metrics.get("protocol_id") != PROTOCOL_ID
        or metrics.get("status") != "pass"
        or metrics.get("items") != 169
        or metrics.get("model_outputs") != 338
        or metrics.get("deterministic_repeats") is not True
        or metrics.get("pooled_overall_accuracy") is not None
    ):
        raise KnowledgeComparisonError("knowledge metrics boundary changed")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("run_id") != run_id
        or receipt.get("condition_id") != condition_id
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("dataset_id") != DATASET_ID
        or receipt.get("status") != "pass"
        or receipt.get("eval_items") != 169
        or receipt.get("model_outputs") != 338
        or receipt.get("metrics_sha256") != _sha256(metrics_path)
    ):
        raise KnowledgeComparisonError("knowledge run receipt changed")
    cells = _mapping(metrics.get("cells"), "metrics.cells")
    if set(cells) != set(EXPECTED_CELLS):
        raise KnowledgeComparisonError("knowledge comparison cells changed")
    for cell_id, expected_items in EXPECTED_CELLS.items():
        cell = _mapping(cells[cell_id], cell_id)
        items = cell.get("items")
        correct = cell.get("correct")
        parseable = cell.get("parseable")
        if (
            items != expected_items
            or not isinstance(correct, int)
            or not 0 <= correct <= items
            or not isinstance(parseable, int)
            or not 0 <= parseable <= items
            or cell.get("exact_accuracy") != correct / items
            or cell.get("parseable_rate") != parseable / items
        ):
            raise KnowledgeComparisonError(
                f"invalid knowledge comparison cell: {cell_id}"
            )
    return metrics, receipt


def compare(
    *,
    repo_root: Path,
    base_dir: Path,
    adapter_dir: Path,
) -> dict[str, Any]:
    base, base_receipt = _validate_result(
        run_dir=base_dir,
        run_id=BASE_RUN_ID,
        condition_id="c4_knowledge_base",
    )
    adapter, adapter_receipt = _validate_result(
        run_dir=adapter_dir,
        run_id=ADAPTER_RUN_ID,
        condition_id="c5_knowledge_lora",
    )
    for field in ("protocol_sha256", "eval_sha256"):
        if base_receipt.get(field) != adapter_receipt.get(field):
            raise KnowledgeComparisonError(
                f"base and LoRA {field} do not match"
            )

    protocol_path = repo_root / LORA_PROTOCOL_PATH
    protocol = _load_yaml(protocol_path)
    if (
        protocol.get("protocol_id") != "delta-v2-knowledge-lora-v2"
        or protocol.get("state") != "preregistered-for-training"
    ):
        raise KnowledgeComparisonError("LoRA gate protocol changed")
    gates = _mapping(
        protocol.get("verification_gates"),
        "verification_gates",
    )
    acquisition_gates = _mapping(
        gates.get("acquisition"),
        "verification_gates.acquisition",
    )
    retention_gates = _mapping(
        gates.get("retention"),
        "verification_gates.retention",
    )

    cells: dict[str, dict[str, Any]] = {}
    for cell_id in sorted(EXPECTED_CELLS):
        base_cell = base["cells"][cell_id]
        adapter_cell = adapter["cells"][cell_id]
        cells[cell_id] = {
            "items": adapter_cell["items"],
            "base_correct": base_cell["correct"],
            "adapter_correct": adapter_cell["correct"],
            "correct_delta": (
                adapter_cell["correct"] - base_cell["correct"]
            ),
            "base_exact_accuracy": base_cell["exact_accuracy"],
            "adapter_exact_accuracy": adapter_cell["exact_accuracy"],
            "exact_accuracy_delta": (
                adapter_cell["exact_accuracy"]
                - base_cell["exact_accuracy"]
            ),
            "base_parseable": base_cell["parseable"],
            "adapter_parseable": adapter_cell["parseable"],
        }

    gate_specs = {
        "acquisition_choice": (
            "acquisition_added.choice",
            acquisition_gates["choice_accuracy_minimum"],
        ),
        "acquisition_boolean_true": (
            "acquisition_added.boolean_true",
            acquisition_gates["boolean_true_accuracy_minimum"],
        ),
        "acquisition_boolean_false": (
            "acquisition_added.boolean_false",
            acquisition_gates["boolean_false_accuracy_minimum"],
        ),
        "retention_boolean_false": (
            "retention_stable.boolean_false",
            retention_gates["boolean_false_accuracy_minimum"],
        ),
    }
    gate_results: dict[str, dict[str, Any]] = {}
    for gate_id, (cell_id, minimum) in gate_specs.items():
        observed = cells[cell_id]["adapter_exact_accuracy"]
        gate_results[gate_id] = {
            "cell": cell_id,
            "minimum": minimum,
            "observed": observed,
            "pass": observed >= minimum,
        }
    candidate_gate_pass = all(
        gate["pass"] for gate in gate_results.values()
    )
    return {
        "schema": COMPARISON_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "lora_protocol_id": protocol["protocol_id"],
        "lora_protocol_sha256": _sha256(protocol_path),
        "dataset_id": DATASET_ID,
        "base_run_id": BASE_RUN_ID,
        "adapter_run_id": ADAPTER_RUN_ID,
        "identical_protocol_and_eval_bank": True,
        "pooled_overall_accuracy": None,
        "cells": cells,
        "preregistered_gates": gate_results,
        "candidate_gate_pass": candidate_gate_pass,
        "candidate_state": (
            "evaluation-gates-pass"
            if candidate_gate_pass
            else "evaluation-gates-fail"
        ),
        "promotion_authorized": False,
        "claim_boundary": {
            "acquisition": "same-claim-new-surface",
            "held_out_fact_generalization": False,
            "recall": "advisory",
            "feature": "separate",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare frozen delta_v2 base and LoRA knowledge results."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare(
        repo_root=args.repo_root.resolve(),
        base_dir=args.base_dir.resolve(),
        adapter_dir=args.adapter_dir.resolve(),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
