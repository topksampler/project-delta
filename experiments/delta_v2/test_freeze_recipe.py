from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from experiments.delta_v2.freeze_recipe import (
    RecipeFreezeError,
    validate_audit,
    validate_policy,
    verify_file_records,
)


def policy() -> dict:
    return {
        "schema": "delta.build_recipe_freeze.v1",
        "recipe_id": "recipe:test",
        "experiment_id": "delta_v2",
        "repository": "vllm-project/vllm",
        "frozen_transition": "development",
        "code_commit": "a" * 40,
        "protocol": {
            "audit_actor": {
                "kind": "evidence-grounded-llm",
                "amendment_timing": (
                    "before-development-freeze-and-before-acceptance-unseal"
                ),
                "truth_authority": (
                    "deterministic-sources-and-executable-probes"
                ),
            },
            "acceptance_feature_policy": {
                "state": "excluded-from-v1-eval",
                "reason": "not-generalized",
                "development_feature_evidence_retained": True,
            },
        },
        "acceptance_unlock": {
            "initial_state": "sealed",
            "condition": "this-manifest-validates",
            "forbidden_operations": [
                "target-model-run",
                "training",
                "fine-tuning",
                "modal-job",
                "lambda-job",
                "b2-write",
            ],
        },
        "test_gate": {
            "command": ["python", "-m", "unittest"],
            "expected_tests": 111,
        },
        "freeze_state": "frozen",
    }


class FreezePolicyTest(unittest.TestCase):
    def test_accepts_explicit_protocol_amendment_and_unlock_gate(self) -> None:
        validate_policy(policy())

    def test_rejects_llm_as_unqualified_truth_authority(self) -> None:
        altered = copy.deepcopy(policy())
        altered["protocol"]["audit_actor"]["truth_authority"] = "llm"

        with self.assertRaisesRegex(RecipeFreezeError, "amendment"):
            validate_policy(altered)

    def test_rejects_training_unlock(self) -> None:
        altered = copy.deepcopy(policy())
        altered["acceptance_unlock"]["forbidden_operations"].remove("training")

        with self.assertRaisesRegex(RecipeFreezeError, "operations"):
            validate_policy(altered)


class FileBindingTest(unittest.TestCase):
    def test_exact_hash_passes_and_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "bound.txt"
            path.write_text("one\n", encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            record = [{"path": "bound.txt", "sha256": digest}]

            verified = verify_file_records(
                record,
                repo_root=root,
                field="files",
            )
            self.assertEqual(verified[0]["sha256"], digest)

            path.write_text("two\n", encoding="utf-8")
            with self.assertRaisesRegex(RecipeFreezeError, "hash mismatch"):
                verify_file_records(
                    record,
                    repo_root=root,
                    field="files",
                )

    def test_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RecipeFreezeError, "within"):
                verify_file_records(
                    [{"path": "../escape", "sha256": "a" * 64}],
                    repo_root=Path(temp_dir),
                    field="files",
                )


class AuditGateTest(unittest.TestCase):
    def test_accepts_passing_eval_and_llm_judge(self) -> None:
        validate_audit(
            "eval_items",
            {
                "schema": "delta.eval_item_build_audit.v1",
                "status": "pass",
                "acceptance_accessed": False,
                "eval_items": 0,
                "unique_eval_ids": True,
                "unique_source_ids": True,
                "control_balance_matches": True,
            },
        )
        validate_audit(
            "llm_judge",
            {
                "schema": "delta.llm_judge_audit.v1",
                "status": "pass",
                "acceptance_accessed": False,
                "root_checks": "pass",
                "dimension_pass_rates": {
                    "truth": 1.0,
                    "version_status": 1.0,
                    "answerability": 1.0,
                },
            },
        )

    def test_rejects_acceptance_access_before_freeze(self) -> None:
        with self.assertRaisesRegex(RecipeFreezeError, "accessed acceptance"):
            validate_audit(
                "source_split",
                {
                    "schema": "delta.source_split_audit.v1",
                    "status": "pass",
                    "transition": "development",
                    "acceptance_accessed": True,
                    "eval_sources": 0,
                    "cross_split_units": [],
                },
            )

    def test_rejects_judge_below_any_dimension_threshold(self) -> None:
        with self.assertRaisesRegex(RecipeFreezeError, "LLM audit"):
            validate_audit(
                "llm_judge",
                {
                    "schema": "delta.llm_judge_audit.v1",
                    "status": "pass",
                    "acceptance_accessed": False,
                    "root_checks": "pass",
                    "dimension_pass_rates": {
                        "truth": 1.0,
                        "version_status": 0.89,
                        "answerability": 1.0,
                    },
                },
            )


if __name__ == "__main__":
    unittest.main()
