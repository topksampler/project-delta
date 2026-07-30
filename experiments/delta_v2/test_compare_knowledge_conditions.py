from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.delta_v2.compare_knowledge_conditions import (
    EXPECTED_CELLS,
    KnowledgeComparisonError,
    compare,
)
from experiments.delta_v2.run_knowledge_eval import (
    ADAPTER_RUN_ID,
    BASE_RUN_ID,
    DATASET_ID,
    METRICS_SCHEMA,
    PROTOCOL_ID,
    RECEIPT_SCHEMA,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def write_result(
    run_dir: Path,
    *,
    run_id: str,
    condition_id: str,
    correct: dict[str, int],
) -> None:
    cells = {}
    for cell_id, items in EXPECTED_CELLS.items():
        cell_correct = correct.get(cell_id, 0)
        cells[cell_id] = {
            "stratum": cell_id.split(".", 1)[0],
            "probe_kind": cell_id.split(".", 1)[1],
            "items": items,
            "correct": cell_correct,
            "exact_accuracy": cell_correct / items,
            "parseable": items,
            "parseable_rate": 1.0,
        }
    metrics = {
        "schema": METRICS_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "pass",
        "items": 169,
        "model_outputs": 338,
        "deterministic_repeats": True,
        "pooled_overall_accuracy": None,
        "cells": cells,
    }
    run_dir.mkdir()
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_sha256 = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "run_id": run_id,
        "condition_id": condition_id,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "status": "pass",
        "eval_items": 169,
        "model_outputs": 338,
        "metrics_sha256": metrics_sha256,
        "protocol_sha256": "protocol",
        "eval_sha256": "eval",
    }
    (run_dir / "run_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class KnowledgeComparisonTests(unittest.TestCase):
    def test_preregistered_gates_pass_without_pooled_score(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_dir = root / "base"
            adapter_dir = root / "adapter"
            write_result(
                base_dir,
                run_id=BASE_RUN_ID,
                condition_id="c4_knowledge_base",
                correct={},
            )
            write_result(
                adapter_dir,
                run_id=ADAPTER_RUN_ID,
                condition_id="c5_knowledge_lora",
                correct={
                    "acquisition_added.choice": 11,
                    "acquisition_added.boolean_true": 17,
                    "acquisition_added.boolean_false": 17,
                    "retention_stable.boolean_false": 19,
                },
            )
            result = compare(
                repo_root=REPO_ROOT,
                base_dir=base_dir,
                adapter_dir=adapter_dir,
            )

        self.assertTrue(result["candidate_gate_pass"])
        self.assertIsNone(result["pooled_overall_accuracy"])
        self.assertFalse(result["promotion_authorized"])

    def test_one_failed_gate_fails_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_dir = root / "base"
            adapter_dir = root / "adapter"
            correct = {
                "acquisition_added.choice": 11,
                "acquisition_added.boolean_true": 17,
                "acquisition_added.boolean_false": 17,
                "retention_stable.boolean_false": 18,
            }
            write_result(
                base_dir,
                run_id=BASE_RUN_ID,
                condition_id="c4_knowledge_base",
                correct={},
            )
            write_result(
                adapter_dir,
                run_id=ADAPTER_RUN_ID,
                condition_id="c5_knowledge_lora",
                correct=correct,
            )
            metrics_path = adapter_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["cells"]["acquisition_added.choice"]["correct"] = 10
            metrics["cells"]["acquisition_added.choice"][
                "exact_accuracy"
            ] = 10 / 21
            metrics_path.write_text(
                json.dumps(metrics, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            receipt_path = adapter_dir / "run_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["metrics_sha256"] = hashlib.sha256(
                metrics_path.read_bytes()
            ).hexdigest()
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            result = compare(
                repo_root=REPO_ROOT,
                base_dir=base_dir,
                adapter_dir=adapter_dir,
            )

        self.assertFalse(result["candidate_gate_pass"])
        self.assertFalse(
            result["preregistered_gates"]["acquisition_choice"]["pass"]
        )

    def test_mismatched_eval_bank_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_dir = root / "base"
            adapter_dir = root / "adapter"
            write_result(
                base_dir,
                run_id=BASE_RUN_ID,
                condition_id="c4_knowledge_base",
                correct={},
            )
            write_result(
                adapter_dir,
                run_id=ADAPTER_RUN_ID,
                condition_id="c5_knowledge_lora",
                correct={},
            )
            receipt_path = adapter_dir / "run_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["eval_sha256"] = "different"
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                KnowledgeComparisonError,
                "eval_sha256",
            ):
                compare(
                    repo_root=REPO_ROOT,
                    base_dir=base_dir,
                    adapter_dir=adapter_dir,
                )


if __name__ == "__main__":
    unittest.main()
