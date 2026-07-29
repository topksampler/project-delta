from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.delta_v2.freeze_environment import (
    EnvironmentFreezeError,
    acceptance_source_ids,
    development_source_ids,
    validate_eval_summary,
    validate_items,
    validate_judge_summary,
    validate_policy,
    verify_bindings,
)


def policy(environment_revision: int = 1) -> dict:
    acceptance_feature_status = (
        "partial" if environment_revision == 1 else "implemented"
    )
    acceptance_items = 42 if environment_revision == 1 else 43
    acceptance_feature_items = 0 if environment_revision == 1 else 1
    limitations = (
        [
            "acceptance-facts-only",
            "acceptance-additions-and-stable-controls-only",
            "acceptance-feature-eval-not-generalized",
            "acceptance-selection-amended-after-unseal",
            "acceptance-endpoint-seen-in-prior-broad-e1-work",
            "no-target-model-result",
        ]
        if environment_revision == 1
        else [
            "acceptance-additions-and-stable-controls-only",
            "acceptance-selection-amended-after-unseal",
            "acceptance-feature-selected-after-unseal",
            "acceptance-endpoint-seen-in-prior-broad-e1-work",
            "endpoint-plugin-probe-isolated-no-full-server",
            "no-target-model-result",
        ]
    )
    return {
        "schema": "delta.eval_environment.v1",
        "environment_id": (
            f"delta-v2-vllm-source-build-v{environment_revision}"
        ),
        "experiment_id": "delta_v2",
        "repository": "vllm-project/vllm",
        "code_commit": "a" * 40,
        "freeze_state": "frozen",
        "publication_state": "local-reproducible-not-published",
        "module_status": {
            "sense_source_change": "implemented",
            "sense_target_model_drift": "deferred",
            "build_fact_environment": "implemented",
            "build_development_feature_proof": "implemented",
            "build_acceptance_feature_eval": acceptance_feature_status,
            "decide": "deferred",
            "adapt": "deferred",
            "verify": "deferred",
            "memory": "deferred",
        },
        "truth_policy": {
            "gold_authority": (
                "deterministic-source-and-executable-evidence"
            ),
            "llm_role": "review-only",
            "llm_may_change_gold": False,
            "conflict_action": "fail-build",
        },
        "dataset_contract": {
            "development": {
                "transition": "v0.22.0-to-v0.23.0",
                "train_items": 65,
                "dev_items": 22,
                "eval_items": 0,
                "verified_feature_items": 1,
            },
            "acceptance": {
                "transition": "v0.25.1-to-v0.26.0",
                "train_items": 0,
                "dev_items": 0,
                "eval_items": acceptance_items,
                "added_items": 21,
                "stable_control_items": 21,
                "verified_feature_items": acceptance_feature_items,
            },
            "scorers": [
                "exact-enum-v1",
                "exact-structured-json-v1",
            ],
            "cross_transition_source_overlap": 0,
        },
        "execution": {
            "target_model_runs": 0,
            "training_runs": 0,
            "fine_tuning_runs": 0,
            "forbidden_without_separate_authorization": [
                "target-model-run",
                "training",
                "fine-tuning",
                "modal-job",
                "lambda-job",
                "b2-write",
            ],
        },
        "limitations": limitations,
        "test_gate": {
            "command": ["python", "-m", "unittest"],
            "expected_tests": 1,
        },
    }


def eval_summary(transition: str, environment_revision: int = 1) -> dict:
    common = {
        "schema": "delta.eval_item_build_audit.v1",
        "status": "pass",
        "unique_eval_ids": True,
        "unique_source_ids": True,
        "control_balance_matches": True,
        "deterministic_order": True,
    }
    if transition == "development":
        return {
            **common,
            "items": 87,
            "split_counts": {"dev": 22, "train": 65},
            "eval_items": 0,
            "acceptance_accessed": False,
        }
    acceptance_items = 42 if environment_revision == 1 else 43
    task_counts = {"atomic_fact_change_status": 42}
    if environment_revision == 2:
        task_counts["feature_behavior_matrix"] = 1
    return {
        **common,
        "contract_id": (
            "delta-v2-acceptance-eval-items-v2"
            if environment_revision == 1
            else "delta-v2-acceptance-eval-items-v3"
        ),
        "items": acceptance_items,
        "split_counts": {"eval": acceptance_items},
        "task_counts": task_counts,
        "fact_status_counts": {"added": 21, "stable": 21},
        "cross_transition_source_overlap": [],
        "acceptance_accessed": True,
        "outputs": {"cross_transition_exclusions": {"rows": 995}},
    }


def judge_summary(transition: str) -> dict:
    return {
        "schema": "delta.llm_judge_audit.v1",
        "status": "pass",
        "root_checks": "pass",
        "items": 32,
        "acceptance_accessed": transition == "acceptance",
        "dimension_pass_rates": {
            "truth": 1.0,
            "version_status": 1.0,
            "answerability": 1.0,
        },
    }


def item(source_id: str, split: str, scorer: str = "exact-enum-v1") -> dict:
    return {
        "schema": "delta.eval_item.v1",
        "source_id": source_id,
        "eval_id": f"eval:{source_id}",
        "split": split,
        "scorer": scorer,
    }


class PolicyTest(unittest.TestCase):
    def test_policy_preserves_partial_loop_and_truth_boundary(self) -> None:
        validate_policy(policy())

    def test_policy_rejects_llm_gold_authority(self) -> None:
        altered = copy.deepcopy(policy())
        altered["truth_policy"]["llm_may_change_gold"] = True

        with self.assertRaisesRegex(EnvironmentFreezeError, "truth policy"):
            validate_policy(altered)

    def test_policy_rejects_completed_acceptance_feature_claim(self) -> None:
        altered = copy.deepcopy(policy())
        altered["module_status"]["build_acceptance_feature_eval"] = "implemented"

        with self.assertRaisesRegex(EnvironmentFreezeError, "overclaims"):
            validate_policy(altered)

    def test_v2_policy_requires_completed_acceptance_feature_and_limit(self) -> None:
        validate_policy(policy(2))
        altered = copy.deepcopy(policy(2))
        altered["module_status"]["build_acceptance_feature_eval"] = "partial"

        with self.assertRaisesRegex(EnvironmentFreezeError, "overclaims"):
            validate_policy(altered)


class BindingTest(unittest.TestCase):
    def test_binding_hash_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "bound.json"
            path.write_text("{}\n", encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            records = [{"id": "one", "path": "bound.json", "sha256": digest}]

            self.assertEqual(verify_bindings(records, repo_root=root)["one"], path)
            path.write_text('{"changed":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(EnvironmentFreezeError, "hash mismatch"):
                verify_bindings(records, repo_root=root)


class AuditTest(unittest.TestCase):
    def test_exact_development_and_acceptance_counts_pass(self) -> None:
        validate_eval_summary(eval_summary("development"), transition="development")
        validate_eval_summary(eval_summary("acceptance"), transition="acceptance")
        validate_eval_summary(
            eval_summary("acceptance", 2),
            transition="acceptance",
        )

    def test_acceptance_overlap_claim_fails(self) -> None:
        altered = eval_summary("acceptance")
        altered["cross_transition_source_overlap"] = ["fact:seen"]

        with self.assertRaisesRegex(EnvironmentFreezeError, "counts changed"):
            validate_eval_summary(altered, transition="acceptance")

    def test_judge_below_threshold_fails(self) -> None:
        altered = judge_summary("acceptance")
        altered["dimension_pass_rates"]["truth"] = 0.89

        with self.assertRaisesRegex(EnvironmentFreezeError, "LLM audit"):
            validate_judge_summary(altered, transition="acceptance")


class DatasetTest(unittest.TestCase):
    def test_eval_item_identity_and_scorer_are_fail_closed(self) -> None:
        source_ids = validate_items(
            [item("fact:one", "eval")],
            allowed_splits={"eval"},
            expected_count=1,
        )
        self.assertEqual(source_ids, {"fact:one"})

        with self.assertRaisesRegex(EnvironmentFreezeError, "scorer"):
            validate_items(
                [item("fact:one", "eval", "llm-rubric")],
                allowed_splits={"eval"},
                expected_count=1,
            )

    def test_development_source_split_rejects_eval(self) -> None:
        rows = [
            {
                "schema": "delta.source_split_assignment.v1",
                "contract_id": "test",
                "source_id": "fact:one",
                "split": "eval",
            }
        ]
        with self.assertRaisesRegex(
            EnvironmentFreezeError,
            "development source split",
        ):
            development_source_ids(rows)

    def test_acceptance_source_split_requires_eval_and_transition(self) -> None:
        row = {
            "schema": "delta.source_split_assignment.v1",
            "contract_id": "test",
            "source_id": "feature:one",
            "split": "eval",
            "transition": "acceptance",
        }

        self.assertEqual(acceptance_source_ids([row]), {"feature:one"})
        row["transition"] = "development"
        with self.assertRaisesRegex(
            EnvironmentFreezeError,
            "acceptance source split",
        ):
            acceptance_source_ids([row])


if __name__ == "__main__":
    unittest.main()
