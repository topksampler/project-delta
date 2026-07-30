from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

import torch
import yaml

from experiments.delta_v2.knowledge_paired_lora import (
    OBJECTIVE_GENERATIVE,
    build_training_units,
    paired_boolean_objective,
)
from experiments.delta_v2.train_knowledge_lora import (
    KnowledgeTrainError,
)
from experiments.delta_v2.train_knowledge_paired_sft_control import (
    validate_run_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/delta_v2/"
    "d2_paired_data_sft_control_qwen35_08b_modal_v1.yaml"
)


class KnowledgePairedLoraTests(unittest.TestCase):
    def test_control_config_is_valid_without_model_work(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        protocol, rows, pairs, dev = validate_run_config(
            config, repo_root=REPO_ROOT
        )
        units = build_training_units(rows, pairs)
        self.assertEqual(len(rows), 147)
        self.assertEqual(len(pairs), 63)
        self.assertEqual(len(dev), 21)
        self.assertEqual(len(units), 84)
        self.assertEqual(
            Counter(unit["kind"] for unit in units),
            {"recall": 21, "boolean_pair": 63},
        )
        self.assertEqual(
            protocol["objective"]["objective_id"], OBJECTIVE_GENERATIVE
        )
        self.assertEqual(protocol["optimization"]["max_steps"], 60)
        self.assertEqual(protocol["lora"]["rank"], 8)

    def test_paired_objective_rewards_correct_signs_and_separation(self) -> None:
        bad, _ = paired_boolean_objective(
            true_yes_log_probability=torch.tensor(-2.0),
            true_no_log_probability=torch.tensor(-1.0),
            false_yes_log_probability=torch.tensor(-1.0),
            false_no_log_probability=torch.tensor(-2.0),
            ranking_margin=1.0,
            ranking_weight=1.0,
            torch=torch,
        )
        good, components = paired_boolean_objective(
            true_yes_log_probability=torch.tensor(-0.5),
            true_no_log_probability=torch.tensor(-2.0),
            false_yes_log_probability=torch.tensor(-2.0),
            false_no_log_probability=torch.tensor(-0.5),
            ranking_margin=1.0,
            ranking_weight=1.0,
            torch=torch,
        )
        self.assertLess(float(good), float(bad))
        self.assertGreater(float(components["true_margin"]), 0)
        self.assertLess(float(components["false_margin"]), 0)
        self.assertGreater(float(components["separation"]), 1)

    def test_objective_mutation_fails_closed(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        config["objective"]["truth_margin_term"] = "enabled"
        with self.assertRaisesRegex(
            KnowledgeTrainError, "config objective changed"
        ):
            validate_run_config(config, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
