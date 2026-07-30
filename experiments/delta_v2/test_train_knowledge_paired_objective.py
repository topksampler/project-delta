from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from experiments.delta_v2.knowledge_paired_lora import OBJECTIVE_PAIRED
from experiments.delta_v2.train_knowledge_lora import KnowledgeTrainError
from experiments.delta_v2.train_knowledge_paired_objective import (
    validate_run_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/delta_v2/"
    "d3_paired_objective_lora_qwen35_08b_modal_v1.yaml"
)


class KnowledgePairedObjectiveTests(unittest.TestCase):
    def test_only_objective_changes_from_stage_1(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        protocol, rows, pairs, dev = validate_run_config(
            config, repo_root=REPO_ROOT
        )
        control = yaml.safe_load(
            (
                REPO_ROOT
                / "experiments/delta_v2/"
                "knowledge_paired_sft_control_protocol.yaml"
            ).read_text(encoding="utf-8")
        )
        for field in (
            "dataset",
            "model",
            "lora",
            "training_units",
            "optimization",
        ):
            self.assertEqual(protocol[field], control[field])
        self.assertNotEqual(protocol["objective"], control["objective"])
        self.assertEqual(
            protocol["objective"]["objective_id"], OBJECTIVE_PAIRED
        )
        self.assertEqual((len(rows), len(pairs), len(dev)), (147, 63, 21))

    def test_safety_boundary_is_frozen(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        protocol, _rows, _pairs, _dev = validate_run_config(
            config, repo_root=REPO_ROOT
        )
        self.assertEqual(
            protocol["safety_gates"]["trainable_scope"],
            "lora-adapter-only",
        )
        self.assertEqual(
            protocol["safety_gates"]["optimizer_steps_maximum"], 60
        )
        self.assertFalse(
            protocol["execution_boundary"]["qlora_authorized"]
        )
        self.assertFalse(
            protocol["execution_boundary"]["full_weight_authorized"]
        )
        self.assertFalse(
            protocol["execution_boundary"]["reinforcement_authorized"]
        )

    def test_objective_mutation_fails_closed(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        config["objective"]["ranking_weight"] = 2.0
        with self.assertRaisesRegex(
            KnowledgeTrainError, "config loss changed"
        ):
            validate_run_config(config, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
