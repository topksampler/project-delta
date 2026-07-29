from __future__ import annotations

import copy
import json
import unittest
from collections import Counter
from pathlib import Path

from experiments.delta_v2.build_eval import (
    FACT_STATUSES,
    EvalBuildError,
    audit_items,
    build_fact_item,
    build_human_audit_packet,
    build_items,
    index_splits,
    score_response,
    select_fact_records,
    validate_contract,
)


CONTRACT_PATH = Path(__file__).with_name("eval_item_contract.yaml")


def contract_record() -> dict:
    import yaml

    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


def fact_record(
    fact_id: str,
    *,
    status: str,
    family: str = "python.environment_variable.v1",
) -> dict:
    return {
        "schema": "delta.atomic_fact_delta.v1",
        "fact_id": fact_id,
        "family": family,
        "status": status,
        "semantic_key": f"python-environment-variable:{fact_id}",
        "value_before": None,
        "value_after": None,
        "evidence_before": {"path": "before.py"} if status != "added" else None,
        "evidence_after": {"path": "after.py"} if status != "removed" else None,
        "verifier": {"extractor_version": "test-v1"},
    }


def split_record(source_id: str, split: str) -> dict:
    return {
        "schema": "delta.source_split_assignment.v1",
        "contract_id": "delta-v2-development-source-split-v1",
        "source_id": source_id,
        "split": split,
    }


def feature_record() -> dict:
    return {
        "schema": "delta.feature_delta.v1",
        "feature_id": "feature:vllm-prefix-cache-retention",
        "source_candidate_id": "feature-candidate:example",
        "repository": "owner/repo",
        "transition": "development",
        "summary": "Feature",
        "status": "added",
        "evidence_ids": [
            "fact:added",
            "file:one",
            "pull-request:owner/repo#1",
        ],
        "behavior_probe_ids": [
            "behavior-probe:vllm-prefix-cache-retention-mechanics-v1"
        ],
        "verification": {
            "old_revision": "unavailable",
            "new_revision": "pass",
            "recipe": "test-v1",
            "probe_results": {
                "behavior-probe:vllm-prefix-cache-retention-mechanics-v1": {
                    "sha256": "a" * 64,
                    "claim_scope": "complete-candidate",
                }
            },
        },
        "environment_status": "development-only-unfrozen",
    }


def mechanics_result() -> dict:
    all_indices = list(range(16))
    cases = [
        ("dense_default", {"kind": "cache_state", "cached_indices": all_indices}),
        (
            "interval_64",
            {"kind": "cache_state", "cached_indices": [3, 7, 11, 14, 15]},
        ),
        ("latest_only", {"kind": "cache_state", "cached_indices": [14]}),
        (
            "negative_rejected",
            {"kind": "construction_error", "category": "non_negative"},
        ),
        (
            "misaligned_rejected",
            {
                "kind": "construction_error",
                "category": "scheduler_block_size_multiple",
            },
        ),
    ]
    return {
        "schema": "delta.behavior_probe_result.v1",
        "probe_id": "behavior-probe:vllm-prefix-cache-retention-mechanics-v1",
        "claim_scope": "complete-candidate",
        "status": "pass",
        "cases": [
            {
                "case_id": case_id,
                "status": "pass",
                "observed_after": observed,
            }
            for case_id, observed in cases
        ],
    }


class ContractTest(unittest.TestCase):
    def test_real_contract_is_valid_and_unfrozen(self) -> None:
        contract = contract_record()

        validate_contract(contract)
        self.assertEqual(contract["freeze_state"], "unfrozen")

    def test_rejects_development_eval_items(self) -> None:
        contract = copy.deepcopy(contract_record())
        contract["split_policy"]["development_eval_items"] = "allowed"

        with self.assertRaisesRegex(EvalBuildError, "split policy"):
            validate_contract(contract)

    def test_rejects_teacher_truth(self) -> None:
        contract = copy.deepcopy(contract_record())
        contract["generator"]["teacher_role"] = "gold"

        with self.assertRaisesRegex(EvalBuildError, "teacher models"):
            validate_contract(contract)


class EligibilityTest(unittest.TestCase):
    def test_includes_all_nonstable_and_balanced_stable_controls(self) -> None:
        contract = contract_record()
        records = [
            fact_record("fact:added", status="added"),
            fact_record("fact:changed", status="changed"),
            fact_record("fact:stable-a", status="stable"),
            fact_record("fact:stable-b", status="stable"),
            fact_record("fact:stable-c", status="stable"),
        ]
        splits = {str(record["fact_id"]): "train" for record in records}

        selected = select_fact_records(records, splits, contract)
        reasons = Counter(reason for _, reason in selected)

        self.assertEqual(reasons["nonstable"], 2)
        self.assertEqual(reasons["stable_control"], 2)

    def test_stable_selection_is_deterministic(self) -> None:
        contract = contract_record()
        records = [
            fact_record("fact:added", status="added"),
            fact_record("fact:stable-a", status="stable"),
            fact_record("fact:stable-b", status="stable"),
        ]
        splits = {str(record["fact_id"]): "train" for record in records}

        first = select_fact_records(records, splits, contract)
        second = select_fact_records(reversed(records), splits, contract)

        self.assertEqual(
            [(record["fact_id"], reason) for record, reason in first],
            [(record["fact_id"], reason) for record, reason in second],
        )

    def test_split_index_rejects_eval(self) -> None:
        with self.assertRaisesRegex(EvalBuildError, "cannot be assigned to eval"):
            index_splits(
                [split_record("fact:one", "eval")],
                "delta-v2-development-source-split-v1",
            )


class ItemGenerationTest(unittest.TestCase):
    def inputs(self) -> tuple[list[dict], list[dict], list[dict], dict]:
        facts = [
            fact_record("fact:added", status="added"),
            fact_record("fact:stable", status="stable"),
        ]
        feature = feature_record()
        splits = [
            split_record("fact:added", "train"),
            split_record("fact:stable", "train"),
            split_record(feature["feature_id"], "train"),
        ]
        probes = {
            "behavior-probe:vllm-prefix-cache-retention-mechanics-v1": (
                mechanics_result()
            )
        }
        return facts, [feature], splits, probes

    def test_items_inherit_source_split_and_feature_gold(self) -> None:
        facts, features, splits, probes = self.inputs()

        items = build_items(
            contract=contract_record(),
            fact_records=facts,
            feature_records=features,
            split_records=splits,
            probe_results=probes,
        )

        self.assertEqual(len(items), 3)
        self.assertEqual({item["split"] for item in items}, {"train"})
        feature_item = next(
            item for item in items if item["task_kind"] == "feature_behavior_matrix"
        )
        self.assertEqual(feature_item["gold"]["unset"], "dense")
        self.assertEqual(
            feature_item["gold"]["positive_aligned_interval"],
            "sparse_interval_plus_latest_boundary",
        )

    def test_fact_scorer_accepts_only_exact_label(self) -> None:
        record = fact_record("fact:added", status="added")
        item = build_fact_item(
            record,
            split="train",
            eligibility_reason="nonstable",
            contract=contract_record(),
        )

        self.assertTrue(score_response(item, "added"))
        self.assertTrue(score_response(item, " Added\n"))
        self.assertFalse(score_response(item, "changed"))
        self.assertFalse(score_response(item, "It was added."))

    def test_feature_scorer_requires_exact_json_structure(self) -> None:
        facts, features, splits, probes = self.inputs()
        items = build_items(
            contract=contract_record(),
            fact_records=facts,
            feature_records=features,
            split_records=splits,
            probe_results=probes,
        )
        item = next(
            item for item in items if item["task_kind"] == "feature_behavior_matrix"
        )

        self.assertTrue(score_response(item, json.dumps(item["gold"])))
        self.assertFalse(score_response(item, "{}"))
        self.assertFalse(score_response(item, "not json"))

    def test_audit_detects_split_rewrite(self) -> None:
        facts, features, splits, probes = self.inputs()
        items = build_items(
            contract=contract_record(),
            fact_records=facts,
            feature_records=features,
            split_records=splits,
            probe_results=probes,
        )
        altered = [dict(item) for item in items]
        altered[0]["split"] = "dev"

        audit = audit_items(altered, splits, contract_record())

        self.assertEqual(audit["status"], "fail")
        self.assertTrue(audit["split_mismatches"])

    def test_build_and_audit_are_deterministic(self) -> None:
        facts, features, splits, probes = self.inputs()
        first = build_items(
            contract=contract_record(),
            fact_records=facts,
            feature_records=features,
            split_records=splits,
            probe_results=probes,
        )
        second = build_items(
            contract=contract_record(),
            fact_records=reversed(facts),
            feature_records=features,
            split_records=reversed(splits),
            probe_results=probes,
        )

        self.assertEqual(first, second)
        self.assertEqual(
            audit_items(first, splits, contract_record())["status"],
            "pass",
        )


class HumanAuditPacketTest(unittest.TestCase):
    def test_sample_is_deterministic_stratified_and_includes_feature(self) -> None:
        items: list[dict] = []
        contract = contract_record()
        families = [
            "python.cli_option.v1",
            "python.config_field.v1",
            "python.environment_variable.v1",
            "python.literal_domain.v1",
        ]
        for index in range(40):
            family = families[index % len(families)]
            status = FACT_STATUSES[index % len(FACT_STATUSES)]
            item = build_fact_item(
                fact_record(
                    f"fact:{index:02d}",
                    status=status,
                    family=family,
                ),
                split="train" if index % 2 == 0 else "dev",
                eligibility_reason=(
                    "stable_control" if status == "stable" else "nonstable"
                ),
                contract=contract,
            )
            items.append(item)
        feature = {
            "schema": "delta.eval_item.v1",
            "eval_id": "eval:feature",
            "source_id": "feature:one",
            "source_kind": "feature_delta",
            "split": "train",
            "task_kind": "feature_behavior_matrix",
            "prompt": "feature",
            "gold": {},
            "provenance": {},
        }
        items.append(feature)

        first = build_human_audit_packet(items, contract)
        second = build_human_audit_packet(reversed(items), contract)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)
        self.assertIn("feature:one", {row["source_id"] for row in first})
        self.assertTrue(
            all(
                row["review"]
                == {
                    "truth": None,
                    "version_status": None,
                    "answerability": None,
                    "notes": None,
                }
                for row in first
            )
        )


if __name__ == "__main__":
    unittest.main()
