from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from experiments.delta_v2.run_knowledge_lora_c7_train_surface_diagnostic import (
    ADAPTER_SHA256,
    PROTOCOL_ID,
    RUN_ID,
    TRAINING_METRICS_SHA256,
    TRAIN_RUN_ID,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
C6_PROTOCOL_PATH = (
    REPO_ROOT
    / "experiments/delta_v2/"
    "knowledge_true_weighted_train_surface_diagnostic_protocol.yaml"
)
C7_PROTOCOL_PATH = (
    REPO_ROOT
    / "experiments/delta_v2/"
    "knowledge_lora_c7_train_surface_diagnostic_protocol.yaml"
)
C7_CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/delta_v2/"
    "c7_knowledge_lora_2to1_train_surface_modal_v1.yaml"
)


class C7DiagnosticProtocolTests(unittest.TestCase):
    def test_gate_is_unchanged_and_binds_completed_training(self) -> None:
        c6 = yaml.safe_load(C6_PROTOCOL_PATH.read_text(encoding="utf-8"))
        c7 = yaml.safe_load(C7_PROTOCOL_PATH.read_text(encoding="utf-8"))
        for section in ("dataset", "generation", "scoring", "decision"):
            self.assertEqual(c7[section], c6[section], section)
        self.assertEqual(
            c7["trigger"],
            {
                "training_run_id": TRAIN_RUN_ID,
                "training_metrics_sha256": TRAINING_METRICS_SHA256,
                "adapter_sha256": ADAPTER_SHA256,
            },
        )

        config = yaml.safe_load(C7_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["run_id"], RUN_ID)
        self.assertEqual(config["protocol"]["protocol_id"], PROTOCOL_ID)
        self.assertEqual(config["model"]["adapter"]["run_id"], TRAIN_RUN_ID)
        self.assertEqual(
            config["model"]["adapter"]["sha256"],
            ADAPTER_SHA256,
        )
        self.assertEqual(config["modal"]["gpu"], "A10G")


if __name__ == "__main__":
    unittest.main()
