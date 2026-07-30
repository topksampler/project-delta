from __future__ import annotations

import unittest
from pathlib import Path

from experiments.delta_v2.run_knowledge_boolean_margin import (
    MarginError,
    audit_pairs,
    summarize_samples,
    validate_only,
)
from experiments.delta_v2.train_knowledge_lora import _load_jsonl


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = (
    REPO_ROOT
    / "data/experiments/delta_v2/knowledge_adaptation_v1/train.jsonl"
)
BASE_CONFIG = (
    REPO_ROOT
    / "configs/experiments/delta_v2/"
    "d1_margin_base_qwen35_08b_modal_v1.yaml"
)


def sample(
    source_id: str,
    surface: str,
    margin: float,
) -> dict:
    gold = "yes" if surface == "verify_true" else "no"
    return {
        "source_id": source_id,
        "surface": surface,
        "gold": gold,
        "predicted": "yes" if margin > 0 else "no",
        "teacher_forced_gold_nll": 1.0,
        "first_token_margin_yes_minus_no": margin / 2,
        "sequence_margin_yes_minus_no": margin,
        "gold_signed_margin": margin if gold == "yes" else -margin,
    }


class KnowledgeBooleanMarginTests(unittest.TestCase):
    def test_frozen_pairs_are_format_matched_but_annotation_only(self) -> None:
        audit = audit_pairs(_load_jsonl(TRAIN_PATH))
        self.assertEqual(audit["source_pairs"], 21)
        self.assertTrue(audit["outer_prompt_template_identical"])
        self.assertTrue(audit["contract_key_order_identical"])
        self.assertEqual(audit["changed_fields"], ["annotation"])
        self.assertEqual(
            audit["false_annotations"],
            {"int": 6, "str": 15},
        )
        self.assertEqual(
            sum(audit["true_annotations"].values()),
            21,
        )
        self.assertEqual(
            audit["semantic_negative_diversity"],
            "annotation-only",
        )

    def test_base_config_validates_without_model_work(self) -> None:
        result = validate_only(BASE_CONFIG, REPO_ROOT)
        self.assertEqual(result["boolean_rows"], 42)
        self.assertEqual(result["source_pairs"], 21)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertEqual(result["status"], "valid-unexecuted")

    def test_summary_keeps_surfaces_and_pairs_separate(self) -> None:
        samples = [
            sample("a", "verify_true", 2.0),
            sample("a", "verify_false", -1.0),
            sample("b", "verify_true", -0.5),
            sample("b", "verify_false", 0.5),
        ]
        metrics = summarize_samples(
            samples,
            run_id="run",
            condition_id="condition",
        )
        self.assertEqual(
            metrics["per_surface"]["verify_true"]["margin_accuracy"],
            0.5,
        )
        self.assertEqual(
            metrics["per_surface"]["verify_false"]["margin_accuracy"],
            0.5,
        )
        self.assertEqual(metrics["paired"]["mean_pair_separation"], 1.0)
        self.assertEqual(metrics["paired"]["truth_conditioned_pairs"], 1)
        self.assertIsNone(metrics["pooled_overall_score"])

    def test_incomplete_pair_fails_closed(self) -> None:
        with self.assertRaisesRegex(MarginError, "pair coverage"):
            summarize_samples(
                [
                    sample("a", "verify_true", 1.0),
                    sample("b", "verify_false", -1.0),
                ],
                run_id="run",
                condition_id="condition",
            )


if __name__ == "__main__":
    unittest.main()
