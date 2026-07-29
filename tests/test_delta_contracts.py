from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError

from lab.delta.contracts import (
    ActiveState,
    CandidateState,
    ContractValidationError,
    DriftEventV2,
    GateResult,
    InterventionOutcome,
    InterventionPlan,
    PromotionDecision,
    ReconcileRun,
    canonical_json,
)


class DeltaContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.drift = DriftEventV2(
            source_before={"repo": "example/repo", "revision": "v1"},
            source_after={"repo": "example/repo", "revision": "v2"},
            structural_census={"changed": 2, "removed": 1},
            behavioral_failures={"changed": {"failed": 3, "total": 8}},
            affected_claim_ids=("claim-1", "claim-2"),
            drift_types=("changed", "removed"),
            drift_score=0.375,
            paths={"evidence": "reports/drift.json"},
        )
        self.plan = InterventionPlan(
            drift_event_id=self.drift.drift_event_id,
            policy_version="least-cost-v1",
            selected_level="L1",
            recipe="hybrid_diff",
            config_inputs={"top_k": 4},
            expected_gates={"recovery": 0.8, "stable_regression": 0.02},
            cost_bound=3.5,
            rationale_codes=("CHANGED_FAILURE_MASS",),
        )
        self.candidate = CandidateState(
            plan_id=self.plan.plan_id,
            drift_event_id=self.drift.drift_event_id,
            base_model="model/base",
            source_revision="v2",
            eval_environment_id="eval-env-1",
            status="succeeded",
            index_revision="index-v2",
            run_ids=("run-1",),
            artifact_hashes={"metrics": "abc123"},
        )
        self.outcome = InterventionOutcome(
            candidate_id=self.candidate.candidate_id,
            status="succeeded",
            metrics={"affected_accuracy": 0.875, "stable_accuracy": 0.98},
            per_drift_metrics={"changed": {"accuracy": 0.875}},
            timing={"total_seconds": 12.5},
            cost=1.25,
        )
        self.active = ActiveState(
            base_model="model/base",
            source_revision="v1",
            eval_environment_id="eval-env-0",
            artifact_hashes={"manifest": "def456"},
        )
        self.gates = (
            GateResult(
                name="recovery",
                passed=True,
                actual=0.875,
                threshold=0.8,
                evidence_path="reports/metrics.json",
            ),
            GateResult(
                name="regression",
                passed=True,
                actual=0.01,
                threshold=0.02,
                evidence_path="reports/metrics.json",
            ),
        )

    def test_ids_are_deterministic_and_content_derived(self) -> None:
        reordered = DriftEventV2(
            source_before={"revision": "v1", "repo": "example/repo"},
            source_after={"revision": "v2", "repo": "example/repo"},
            structural_census={"removed": 1, "changed": 2},
            behavioral_failures={"changed": {"total": 8, "failed": 3}},
            affected_claim_ids=("claim-1", "claim-2"),
            drift_types=("changed", "removed"),
            drift_score=0.375,
            paths={"evidence": "reports/drift.json"},
        )
        self.assertEqual(self.drift.drift_event_id, reordered.drift_event_id)

        changed = DriftEventV2.from_dict(
            {
                **self.drift.to_dict(),
                "drift_score": 0.5,
                "drift_event_id": None,
            }
        )
        self.assertNotEqual(self.drift.drift_event_id, changed.drift_event_id)
        self.assertEqual(len(self.drift.drift_event_id), 64)

    def test_all_contracts_round_trip(self) -> None:
        run = ReconcileRun(
            repo="example/repo",
            source_before="v1",
            source_after="v2",
            experiment_plugin="e1_vllm",
            status="running",
            stage_status={"sense": "completed", "build": "running"},
            child_run_ids=("run-1",),
            artifact_hashes={"drift": self.drift.content_hash},
        )
        decision = PromotionDecision(
            candidate_id=self.candidate.candidate_id,
            outcome_id=self.outcome.outcome_id,
            expected_active_state_id=self.active.state_id,
            status="pending_approval",
            gates=self.gates,
            rationale_codes=("ALL_HARD_GATES_PASS",),
        )
        contracts = (
            run,
            self.drift,
            self.plan,
            self.candidate,
            self.outcome,
            decision,
            self.active,
        )
        for contract in contracts:
            with self.subTest(contract=type(contract).__name__):
                restored = type(contract).from_dict(contract.to_dict())
                self.assertEqual(restored.to_dict(), contract.to_dict())
                self.assertEqual(json.loads(contract.to_json()), contract.to_dict())

    def test_canonical_json_is_sorted_and_compact(self) -> None:
        self.assertEqual(canonical_json({"z": 1, "a": [2, 3]}), '{"a":[2,3],"z":1}')

    def test_wrong_schema_and_tampered_id_are_rejected(self) -> None:
        serialized = self.plan.to_dict()
        with self.assertRaises(ContractValidationError):
            InterventionPlan.from_dict({**serialized, "schema_id": "delta.plan.v0"})
        with self.assertRaises(ContractValidationError):
            InterventionPlan.from_dict({**serialized, "plan_id": "0" * 64})

    def test_invalid_status_level_and_gate_consistency_are_rejected(self) -> None:
        with self.assertRaises(ContractValidationError):
            ReconcileRun(
                repo="example/repo",
                source_before="v1",
                source_after="v2",
                experiment_plugin="e1_vllm",
                status="cancelled",
            )
        with self.assertRaises(ContractValidationError):
            InterventionPlan(
                drift_event_id=self.drift.drift_event_id,
                policy_version="v1",
                selected_level="L9",
                recipe="unknown",
                config_inputs={},
                expected_gates={},
                cost_bound=0,
                rationale_codes=(),
            )
        with self.assertRaises(ContractValidationError):
            InterventionPlan(
                drift_event_id=self.drift.drift_event_id,
                policy_version="v1",
                selected_level="L3",
                recipe="lora",
                config_inputs={},
                expected_gates={},
                cost_bound=10,
                rationale_codes=("FT",),
                ft_eligible=False,
            )
        failed_gate = GateResult(
            name="recovery",
            passed=False,
            actual=0.5,
            threshold=0.8,
            evidence_path="reports/metrics.json",
        )
        with self.assertRaises(ContractValidationError):
            PromotionDecision(
                candidate_id=self.candidate.candidate_id,
                outcome_id=self.outcome.outcome_id,
                expected_active_state_id=self.active.state_id,
                status="pending_approval",
                gates=(failed_gate,),
            )

    def test_nested_data_is_mutation_resistant(self) -> None:
        metadata = {"nested": {"values": [1, 2]}}
        state = ActiveState(
            base_model="model/base",
            source_revision="v1",
            eval_environment_id="eval-1",
            metadata=metadata,
        )
        original_id = state.state_id
        metadata["nested"]["values"].append(3)
        self.assertEqual(state.to_dict()["metadata"]["nested"]["values"], [1, 2])
        self.assertEqual(state.state_id, original_id)
        with self.assertRaises(TypeError):
            state.metadata["new"] = "value"  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            state.source_revision = "v2"  # type: ignore[misc]

    def test_non_json_metadata_is_rejected(self) -> None:
        with self.assertRaises(ContractValidationError):
            ActiveState(
                base_model="model/base",
                source_revision="v1",
                eval_environment_id="eval-1",
                metadata={"bad": object()},
            )


if __name__ == "__main__":
    unittest.main()
