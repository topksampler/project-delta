from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from experiments.delta_v2.run_knowledge_paired_margin import (
    PairedMarginError,
    summarize_samples,
    validate_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/delta_v2/"
    "d2_paired_data_sft_control_margin_modal_v1.yaml"
)


def _sample(
    pair: int,
    *,
    surface: str,
    family: str,
    correct: bool,
) -> dict:
    gold = "yes" if surface == "verify_true" else "no"
    predicted = gold if correct else ("no" if gold == "yes" else "yes")
    margin = 1.0 if predicted == "yes" else -1.0
    return {
        "pair_id": f"pair:{pair}",
        "source_id": f"source:{pair // 3}",
        "surface": surface,
        "corruption_family": family,
        "gold": gold,
        "predicted": predicted,
        "teacher_forced_gold_nll": 0.1,
        "sequence_margin_yes_minus_no": margin,
        "gold_signed_margin": margin if gold == "yes" else -margin,
    }


class KnowledgePairedMarginTests(unittest.TestCase):
    def test_config_validates_without_model_work(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        _protocol, rows, pairs = validate_config(
            config, repo_root=REPO_ROOT
        )
        self.assertEqual(len(rows), 126)
        self.assertEqual(len(pairs), 63)

    def test_perfect_samples_pass_every_separate_gate(self) -> None:
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
        result = summarize_samples(samples)
        self.assertTrue(result["first_gate_passed"])
        self.assertIsNone(result["pooled_overall_score"])
        self.assertEqual(result["paired"]["truth_conditioned_pairs"], 63)

    def test_one_failed_family_blocks_gate(self) -> None:
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
                    correct=family != "config_default_swap",
                )
            )
        result = summarize_samples(samples)
        self.assertFalse(result["first_gate_passed"])
        self.assertFalse(
            result["gates"]["every_corruption_family_false"]
        )

    def test_adapter_hash_mutation_fails_closed(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        config["model"]["adapter"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            PairedMarginError, "model changed"
        ):
            validate_config(config, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
