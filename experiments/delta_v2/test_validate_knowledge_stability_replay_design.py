from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from experiments.delta_v2.validate_knowledge_stability_replay_design import (
    StabilityReplayDesignError,
    validate_design,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DESIGN_PATH = (
    REPO_ROOT
    / "experiments/delta_v2/"
    "knowledge_stability_replay_design.yaml"
)


class StabilityReplayDesignTests(unittest.TestCase):
    def test_partition_is_disjoint_and_training_stays_blocked(self) -> None:
        design = yaml.safe_load(DESIGN_PATH.read_text(encoding="utf-8"))
        result = validate_design(design, repo_root=REPO_ROOT)
        self.assertEqual(result["eligible_sources"], 711)
        self.assertEqual(result["blocked_current_eval_sources"], 42)
        self.assertEqual(
            {
                name: block["sources"]
                for name, block in result["partitions"].items()
            },
            {
                "replay_train": 15,
                "stability_dev": 15,
                "verify_v2": 21,
            },
        )
        self.assertFalse(result["training_authorized"])
        self.assertEqual(result["model_invocations_completed"], 0)

    def test_family_quotas_and_config_path_diversity_are_exact(self) -> None:
        design = yaml.safe_load(DESIGN_PATH.read_text(encoding="utf-8"))
        result = validate_design(design, repo_root=REPO_ROOT)
        self.assertEqual(
            result["partitions"]["replay_train"]["families"],
            {
                "python.config_field.v1": 6,
                "python.environment_variable.v1": 9,
            },
        )
        self.assertEqual(
            result["partitions"]["stability_dev"]["config_paths"], 6
        )
        self.assertEqual(
            result["partitions"]["verify_v2"]["config_paths"], 9
        )

    def test_partition_hash_mutation_fails_closed(self) -> None:
        design = yaml.safe_load(DESIGN_PATH.read_text(encoding="utf-8"))
        design["partitions"]["derived_partition_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            StabilityReplayDesignError,
            "derived stability partition changed",
        ):
            validate_design(design, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
