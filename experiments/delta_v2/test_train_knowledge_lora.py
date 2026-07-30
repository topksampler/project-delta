from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from experiments.delta_v2.train_knowledge_lora import (
    KnowledgeTrainError,
    TARGET_MODULES_REGEX,
    assistant_only_labels,
    target_module_allowed,
    validate_run_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/delta_v2/"
    "c5_knowledge_lora_qwen35_08b_modal_v2.yaml"
)


class KnowledgeLoraTests(unittest.TestCase):
    def test_config_is_frozen_and_truth_balanced(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        protocol, train_rows, dev_rows = validate_run_config(
            config,
            repo_root=REPO_ROOT,
        )
        self.assertEqual(len(train_rows), 63)
        self.assertEqual(len(dev_rows), 21)
        self.assertEqual(protocol["optimization"]["max_steps"], 60)
        answers = [
            row["messages"][1]["content"]
            for row in train_rows
        ]
        self.assertEqual(answers.count("yes"), 21)
        self.assertEqual(answers.count("no"), 21)

    def test_assistant_only_mask(self) -> None:
        labels = assistant_only_labels(
            [10, 11, 12],
            [10, 11, 12, 20, 21],
            max_sequence_length=8,
        )
        self.assertEqual(labels, [-100, -100, -100, 20, 21])

    def test_assistant_mask_fails_on_nonprefix(self) -> None:
        with self.assertRaisesRegex(
            KnowledgeTrainError,
            "assistant-only prefix",
        ):
            assistant_only_labels(
                [10, 99],
                [10, 11, 20],
                max_sequence_length=8,
            )

    def test_assistant_mask_never_truncates(self) -> None:
        with self.assertRaisesRegex(
            KnowledgeTrainError,
            "exceeds max sequence length",
        ):
            assistant_only_labels(
                [10, 11],
                [10, 11, 20, 21],
                max_sequence_length=3,
            )

    def test_target_regex_selects_language_only(self) -> None:
        self.assertIn("language_model", TARGET_MODULES_REGEX)
        self.assertTrue(
            target_module_allowed(
                "model.language_model.layers.3.self_attn.q_proj"
            )
        )
        self.assertTrue(
            target_module_allowed(
                "model.language_model.layers.0.linear_attn.in_proj_qkv"
            )
        )
        self.assertTrue(
            target_module_allowed(
                "model.language_model.layers.4.mlp.down_proj"
            )
        )
        self.assertFalse(
            target_module_allowed(
                "model.visual.blocks.0.attn.q_proj"
            )
        )

    def test_changed_steps_fail_closed(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        config["training"]["max_steps"] = 61
        with self.assertRaisesRegex(
            KnowledgeTrainError,
            "optimization changed",
        ):
            validate_run_config(config, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
