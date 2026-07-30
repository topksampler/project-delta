from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from experiments.delta_v2.run_knowledge_paired_margin import (
    PairedMarginError,
)
from experiments.delta_v2.run_knowledge_paired_objective_margin import (
    CONDITION_ID,
    RUN_ID,
    summarize_candidate_samples,
    validate_config,
)
from experiments.delta_v2.test_run_knowledge_paired_margin import _sample


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/delta_v2/"
    "d3_paired_objective_margin_modal_v1.yaml"
)


class KnowledgePairedObjectiveMarginTests(unittest.TestCase):
    def test_config_validates_without_model_work(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        _protocol, rows, pairs = validate_config(
            config, repo_root=REPO_ROOT
        )
        self.assertEqual((len(rows), len(pairs)), (126, 63))

    def test_candidate_summary_keeps_separate_gates(self) -> None:
        families = (
            ["annotation_swap"] * 21
            + ["config_default_swap"] * 9
            + ["env_getter_swap"] * 12
            + ["full_card_swap"] * 21
        )
        samples = []
        for pair, family in enumerate(families):
            samples.append(
                _sample(
                    pair,
                    surface="verify_true",
                    family=family,
                    correct=True,
                )
            )
            samples.append(
                _sample(
                    pair,
                    surface="verify_false",
                    family=family,
                    correct=True,
                )
            )
        result = summarize_candidate_samples(samples)
        self.assertEqual(result["run_id"], RUN_ID)
        self.assertEqual(result["condition_id"], CONDITION_ID)
        self.assertTrue(result["first_gate_passed"])
        self.assertIsNone(result["pooled_overall_score"])

    def test_adapter_mutation_fails_closed(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        config["model"]["adapter"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            PairedMarginError, "model changed"
        ):
            validate_config(config, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
