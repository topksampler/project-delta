from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.delta_v2.source_split import (
    Source,
    SourceSplitError,
    assignment_for_anchor,
    audit_assignments,
    build_assignments,
    build_units,
    collect_sources,
    load_yaml,
    serialize_rows,
    validate_contract,
)


CONTRACT_PATH = Path(__file__).with_name("source_split_contract.yaml")


def fact_record(fact_id: str, *, status: str = "added") -> dict:
    before = None if status == "added" else {"path": "before.py"}
    after = None if status == "removed" else {"path": "after.py"}
    return {
        "schema": "delta.atomic_fact_delta.v1",
        "fact_id": fact_id,
        "family": "python.environment_variable.v1",
        "status": status,
        "value_before": None,
        "value_after": {"key": fact_id},
        "evidence_before": before,
        "evidence_after": after,
        "verifier": {"extractor_version": "test-v1"},
    }


def feature_record(
    feature_id: str,
    fact_ids: list[str],
    *,
    transition: str = "development",
) -> dict:
    record = {
        "schema": "delta.feature_delta.v1",
        "feature_id": feature_id,
        "source_candidate_id": "feature-candidate:example",
        "repository": "owner/repo",
        "transition": transition,
        "summary": "Example feature",
        "status": "added",
        "evidence_ids": [
            *fact_ids,
            "file:one",
            "pull-request:owner/repo#1",
        ],
        "behavior_probe_ids": ["behavior-probe:complete"],
        "verification": {
            "old_revision": "unavailable",
            "new_revision": "pass",
            "recipe": "test-v1",
            "probe_results": {
                "behavior-probe:complete": {
                    "sha256": "a" * 64,
                    "claim_scope": "complete-candidate",
                }
            },
        },
        "environment_status": (
            "development-only-unfrozen"
            if transition == "development"
            else "acceptance-feature-unfrozen"
        ),
    }
    if transition == "acceptance":
        record["fact_policy"] = "no-post-unseal-fact-family-expansion"
    return record


class ContractTest(unittest.TestCase):
    def test_real_contract_reserves_eval_for_sealed_acceptance(self) -> None:
        contract = load_yaml(CONTRACT_PATH)

        validate_contract(contract)
        self.assertEqual(
            contract["acceptance_assignment"],
            {
                "state": "sealed",
                "unlock_condition": "validated-development-recipe-freeze",
                "algorithm": "all-sources-eval-v1",
                "future_split": "eval",
                "development_builder_behavior": "reject",
            },
        )

    def test_rejects_eval_in_development(self) -> None:
        contract = copy.deepcopy(load_yaml(CONTRACT_PATH))
        contract["development_assignment"]["eval_sources"] = "allowed"

        with self.assertRaisesRegex(SourceSplitError, "must be forbidden"):
            validate_contract(contract)

    def test_rejects_claimed_freeze(self) -> None:
        contract = copy.deepcopy(load_yaml(CONTRACT_PATH))
        contract["freeze_state"] = "frozen"

        with self.assertRaisesRegex(SourceSplitError, "must not claim"):
            validate_contract(contract)


class SourceEligibilityTest(unittest.TestCase):
    def test_accepts_verified_facts_and_promoted_features(self) -> None:
        sources = collect_sources(
            [fact_record("fact:one")],
            [feature_record("feature:one", ["fact:one"])],
        )

        self.assertEqual(set(sources), {"fact:one", "feature:one"})

    def test_rejects_candidate_in_feature_input(self) -> None:
        candidate = {
            "schema": "delta.feature_candidate.v1",
            "candidate_id": "feature-candidate:one",
        }

        with self.assertRaisesRegex(SourceSplitError, "FeatureDelta schema"):
            collect_sources([fact_record("fact:one")], [candidate])

    def test_rejects_unverified_fact_shape(self) -> None:
        fact = fact_record("fact:one")
        fact["verifier"] = None

        with self.assertRaisesRegex(SourceSplitError, "verifier must be a mapping"):
            collect_sources([fact], [])

    def test_acceptance_feature_can_preserve_frozen_fact_boundary(self) -> None:
        sources = collect_sources(
            [fact_record("fact:one")],
            [
                feature_record(
                    "feature:acceptance",
                    [],
                    transition="acceptance",
                )
            ],
        )

        self.assertEqual(
            sources["feature:acceptance"].referenced_fact_ids,
            (),
        )

    def test_development_feature_still_requires_atomic_fact(self) -> None:
        feature = feature_record("feature:development", [])

        with self.assertRaisesRegex(
            SourceSplitError,
            "requires atomic-fact evidence",
        ):
            collect_sources([fact_record("fact:one")], [feature])


class LeakageUnitTest(unittest.TestCase):
    def test_feature_and_referenced_fact_share_one_unit(self) -> None:
        sources = collect_sources(
            [fact_record("fact:one"), fact_record("fact:two")],
            [feature_record("feature:one", ["fact:one"])],
        )

        units = build_units(sources)

        self.assertEqual(
            units,
            {
                "fact:one": ("fact:one", "feature:one"),
                "fact:two": ("fact:two",),
            },
        )

    def test_transitive_feature_overlap_is_indivisible(self) -> None:
        sources = collect_sources(
            [fact_record("fact:a"), fact_record("fact:b")],
            [
                feature_record("feature:a", ["fact:a", "fact:b"]),
                feature_record("feature:b", ["fact:b"]),
            ],
        )

        units = build_units(sources)

        self.assertEqual(len(units), 1)
        self.assertEqual(
            next(iter(units.values())),
            ("fact:a", "fact:b", "feature:a", "feature:b"),
        )

    def test_missing_referenced_fact_is_rejected(self) -> None:
        sources = collect_sources(
            [fact_record("fact:one")],
            [feature_record("feature:one", ["fact:missing"])],
        )

        with self.assertRaisesRegex(SourceSplitError, "references missing fact"):
            build_units(sources)

    def test_fact_frozen_acceptance_feature_is_its_own_unit(self) -> None:
        sources = collect_sources(
            [fact_record("fact:one")],
            [
                feature_record(
                    "feature:acceptance",
                    [],
                    transition="acceptance",
                )
            ],
        )

        self.assertEqual(
            build_units(sources),
            {
                "fact:one": ("fact:one",),
                "feature:acceptance": ("feature:acceptance",),
            },
        )


class AssignmentTest(unittest.TestCase):
    def test_assignment_is_stable_and_content_blind(self) -> None:
        first = assignment_for_anchor("fact:one", "salt-v1")
        second = assignment_for_anchor("fact:one", "salt-v1")
        changed = assignment_for_anchor("fact:one", "salt-v2")

        self.assertEqual(first, second)
        self.assertNotEqual(first["assignment_hash"], changed["assignment_hash"])
        self.assertIn(first["split"], {"train", "dev"})

    def test_every_member_in_unit_has_same_split(self) -> None:
        contract = load_yaml(CONTRACT_PATH)
        sources = collect_sources(
            [fact_record("fact:one"), fact_record("fact:two")],
            [feature_record("feature:one", ["fact:one"])],
        )

        rows = build_assignments(contract, sources)
        grouped = [
            row for row in rows if row["unit_anchor_id"] == "fact:one"
        ]

        self.assertEqual(len(grouped), 2)
        self.assertEqual(len({row["split"] for row in grouped}), 1)
        self.assertEqual(len({row["split_unit_id"] for row in grouped}), 1)

    def test_output_and_audit_are_deterministic(self) -> None:
        contract = load_yaml(CONTRACT_PATH)
        sources = collect_sources(
            [fact_record("fact:two"), fact_record("fact:one")],
            [feature_record("feature:one", ["fact:one"])],
        )

        first = build_assignments(contract, sources)
        second = build_assignments(contract, sources)
        first_audit = audit_assignments(contract, sources, first)
        second_audit = audit_assignments(contract, sources, second)

        self.assertEqual(serialize_rows(first), serialize_rows(second))
        self.assertEqual(first_audit, second_audit)
        self.assertEqual(first_audit["status"], "pass")
        self.assertEqual(first_audit["eval_sources"], 0)
        self.assertFalse(first_audit["acceptance_accessed"])

    def test_acceptance_is_sealed_then_assigns_every_source_to_eval(
        self,
    ) -> None:
        contract = load_yaml(CONTRACT_PATH)
        sources = collect_sources(
            [
                fact_record("fact:one", status="added"),
                fact_record("fact:two", status="stable"),
            ],
            [],
        )

        with self.assertRaisesRegex(SourceSplitError, "sealed"):
            build_assignments(
                contract,
                sources,
                transition="acceptance",
            )

        rows = build_assignments(
            contract,
            sources,
            transition="acceptance",
            allow_acceptance=True,
        )
        audit = audit_assignments(
            contract,
            sources,
            rows,
            transition="acceptance",
            allow_acceptance=True,
        )

        self.assertEqual({row["split"] for row in rows}, {"eval"})
        self.assertEqual(
            {row["transition"] for row in rows},
            {"acceptance"},
        )
        self.assertEqual(audit["status"], "pass")
        self.assertTrue(audit["acceptance_accessed"])

    def test_audit_detects_cross_split_leakage(self) -> None:
        contract = load_yaml(CONTRACT_PATH)
        sources = {
            "fact:one": Source(
                source_id="fact:one",
                source_kind="atomic_fact_delta",
                status="added",
                family="family",
            ),
            "feature:one": Source(
                source_id="feature:one",
                source_kind="feature_delta",
                status="added",
                family=None,
                referenced_fact_ids=("fact:one",),
            ),
        }
        rows = build_assignments(contract, sources)
        altered = [dict(row) for row in rows]
        altered[1]["split"] = (
            "dev" if altered[0]["split"] == "train" else "train"
        )

        audit = audit_assignments(contract, sources, altered)

        self.assertEqual(audit["status"], "fail")
        self.assertEqual(len(audit["cross_split_units"]), 1)


if __name__ == "__main__":
    unittest.main()
