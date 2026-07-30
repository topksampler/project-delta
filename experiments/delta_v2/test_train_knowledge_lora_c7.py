from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

import yaml

from experiments.delta_v2.train_knowledge_lora import (
    KnowledgeTrainError,
    build_weighted_order,
)
from experiments.delta_v2.train_knowledge_lora_c7 import (
    SURFACE_WEIGHTS,
    validate_run_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/delta_v2/"
    "c7_knowledge_lora_2to1_qwen35_08b_modal_v1.yaml"
)
C6_PROTOCOL_PATH = (
    REPO_ROOT
    / "experiments/delta_v2/"
    "knowledge_lora_true_weighted_protocol.yaml"
)
C7_PROTOCOL_PATH = (
    REPO_ROOT
    / "experiments/delta_v2/knowledge_lora_c7_protocol.yaml"
)


class C7KnowledgeLoraTests(unittest.TestCase):
    def test_config_changes_only_the_preregistered_sampler_weight(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        protocol, train_rows, dev_rows = validate_run_config(
            config, repo_root=REPO_ROOT
        )
        order = build_weighted_order(train_rows, SURFACE_WEIGHTS)
        exposure = Counter(train_rows[index]["surface"] for index in order)
        self.assertEqual(len(train_rows), 63)
        self.assertEqual(len(dev_rows), 21)
        self.assertEqual(len(order), 84)
        self.assertEqual(
            exposure,
            {
                "exact_recall": 21,
                "verify_true": 42,
                "verify_false": 21,
            },
        )
        self.assertEqual(protocol["model"]["initialization"], "fresh-base")
        self.assertFalse(
            protocol["dataset"]["evaluation_wording_added_to_training"]
        )

        c6 = yaml.safe_load(C6_PROTOCOL_PATH.read_text(encoding="utf-8"))
        c7 = yaml.safe_load(C7_PROTOCOL_PATH.read_text(encoding="utf-8"))
        for section in (
            "dataset",
            "model",
            "lora",
            "optimization",
            "verification",
            "execution_boundary",
        ):
            self.assertEqual(c7[section], c6[section], section)
        self.assertEqual(
            {
                key: value
                for key, value in c7["sampling"].items()
                if key != "surface_weights"
            },
            {
                key: value
                for key, value in c6["sampling"].items()
                if key != "surface_weights"
            },
        )
        self.assertEqual(
            c7["sampling"]["surface_weights"],
            {
                **c6["sampling"]["surface_weights"],
                "verify_true": 2,
            },
        )

    def test_changed_c7_true_weight_fails_closed(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        config["sampling"]["surface_weights"]["verify_true"] = 3
        with self.assertRaisesRegex(
            KnowledgeTrainError,
            "sampling config changed",
        ):
            validate_run_config(config, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
