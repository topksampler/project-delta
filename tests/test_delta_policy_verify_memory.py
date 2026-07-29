from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lab.delta.contracts import (
    ActiveState,
    CandidateState,
    CandidateStatus,
    DriftEventV2,
    InterventionOutcome,
    OutcomeStatus,
    PromotionStatus,
)
from lab.delta.memory import (
    ImmutableCollisionError,
    InvalidStateError,
    LocalStateRegistry,
    PointerRaceError,
)
from lab.delta.policy import choose_plan
from lab.delta.verify import VerificationThresholds, verify_candidate


def active(*, revision: str = "old", candidate_id: str | None = None) -> ActiveState:
    return ActiveState(
        base_model="model",
        source_revision=revision,
        eval_environment_id="eval",
        candidate_id=candidate_id,
        metadata={"metrics": {"stable": 0.90}},
    )


def drift(changed: float | None, removed: float | None) -> DriftEventV2:
    failures = {}
    if changed is not None:
        failures["changed"] = changed
    if removed is not None:
        failures["removed"] = removed
    return DriftEventV2(
        source_before={"revision": "old"},
        source_after={"revision": "new"},
        structural_census={"changed": 1},
        behavioral_failures=failures,
        affected_claim_ids=("claim",),
    )


def candidate() -> CandidateState:
    return CandidateState(
        plan_id="plan",
        drift_event_id="drift",
        base_model="model",
        source_revision="new",
        eval_environment_id="eval",
        status=CandidateStatus.SUCCEEDED,
        run_ids=("run",),
        artifact_hashes={"metrics": "sha256"},
        paths={"candidate": "runs/run"},
    )


def outcome(
    cand: CandidateState,
    *,
    changed: float = 0.85,
    removed: float = 0.85,
    stable: float = 0.89,
    coverage: float = 0.99,
    cost: float = 2.0,
    provenance: bool = True,
    environment_sealed: bool | None = None,
) -> InterventionOutcome:
    metadata = (
        {}
        if environment_sealed is None
        else {"environment_sealed": environment_sealed}
    )
    return InterventionOutcome(
        candidate_id=cand.candidate_id,
        status=OutcomeStatus.SUCCEEDED,
        metrics={"stable": stable, "coverage": coverage, "pooled_exact": 1.0},
        per_drift_metrics={"changed": changed, "removed": removed},
        timing={"seconds": 1},
        cost=cost,
        paths={"metrics": "runs/run/metrics.json"} if provenance else {},
        metadata=metadata,
    )


def prepared_registry(
    root: Path,
) -> tuple[LocalStateRegistry, ActiveState, ActiveState, object]:
    registry = LocalStateRegistry(root)
    old = active()
    cand = candidate()
    new = active(revision="new", candidate_id=cand.candidate_id)
    registry.initialize(old)
    registry.register_state(new)
    decision = verify_candidate(cand, outcome(cand), old)
    return registry, old, new, decision


class PolicyVerifyMemoryTest(unittest.TestCase):
    def test_policy_covers_noop_hybrid_fallback_oracle_and_l3(self) -> None:
        current = active()
        self.assertEqual(choose_plan(drift(0.90, 0.80), current).recipe, "no_op")
        hybrid = choose_plan(drift(0.40, 0.90), current)
        self.assertEqual(hybrid.selected_level.value, "L1")
        self.assertEqual(hybrid.recipe, "hybrid_diff")
        fallback = choose_plan(
            drift(0.40, 0.90),
            current,
            history=({"recipe": "hybrid_diff", "regressed": True},),
        )
        self.assertEqual(fallback.recipe, "index_refresh")
        oracle = choose_plan(drift(0.0, 0.0), current, oracle_debug=True)
        self.assertTrue(oracle.metadata["debug_only"])
        blocked = choose_plan(
            drift(0.0, 0.0),
            current,
            budget={"allowed_levels": ("L3",)},
            ft_reopened=True,
            l3_approved=False,
        )
        self.assertEqual(blocked.selected_level.value, "L0")
        l3 = choose_plan(
            drift(0.0, 0.0),
            current,
            budget={"allowed_levels": ("L3",), "max_cost": 20},
            ft_reopened=True,
            l3_approved=True,
        )
        self.assertEqual(l3.selected_level.value, "L3")
        self.assertTrue(l3.ft_eligible)

    def test_verify_requires_every_non_pooled_gate(self) -> None:
        cand = candidate()
        decision = verify_candidate(cand, outcome(cand), active())
        self.assertIs(decision.status, PromotionStatus.PENDING_APPROVAL)
        self.assertEqual(
            {gate.name for gate in decision.gates},
            {"recovery", "stable_regression", "coverage", "cost", "provenance"},
        )
        self.assertTrue(all(gate.passed for gate in decision.gates))
        pooled_only = InterventionOutcome(
            candidate_id=cand.candidate_id,
            status=OutcomeStatus.SUCCEEDED,
            metrics={"pooled_exact": 1.0, "stable": 0.9, "coverage": 1.0},
            per_drift_metrics={},
            timing={},
            cost=1.0,
            paths={"metrics": "metrics.json"},
        )
        rejected = verify_candidate(cand, pooled_only, active())
        self.assertIs(rejected.status, PromotionStatus.REJECTED)

    def test_verify_rejects_unsealed_eval_environment(self) -> None:
        cand = candidate()
        checked = verify_candidate(
            cand,
            outcome(cand, environment_sealed=False),
            active(),
        )
        self.assertIs(checked.status, PromotionStatus.REJECTED)
        gate = next(
            item for item in checked.gates if item.name == "environment_sealed"
        )
        self.assertFalse(gate.passed)
        self.assertEqual(
            gate.details["rationale_code"], "ENVIRONMENT_UNSEALED"
        )

    def test_verify_rejects_each_hard_gate(self) -> None:
        cases = [
            ({"changed": 0.79}, "recovery"),
            ({"stable": 0.80}, "stable_regression"),
            ({"coverage": 0.94}, "coverage"),
            ({"cost": 3.0}, "cost"),
            ({"provenance": False}, "provenance"),
        ]
        for changes, failed_gate in cases:
            with self.subTest(gate=failed_gate):
                cand = candidate()
                checked = verify_candidate(
                    cand,
                    outcome(cand, **changes),
                    active(),
                    thresholds=VerificationThresholds(max_cost=2.0),
                )
                self.assertIs(checked.status, PromotionStatus.REJECTED)
                gate = next(item for item in checked.gates if item.name == failed_gate)
                self.assertFalse(gate.passed)

    def test_verify_rejects_costlier_candidate_after_cheaper_pass(self) -> None:
        cand = candidate()
        first = verify_candidate(cand, outcome(cand, cost=1.0), active())
        second = verify_candidate(
            cand, outcome(cand, cost=2.0), active(), history=(first,)
        )
        self.assertIs(second.status, PromotionStatus.REJECTED)
        cost_gate = next(gate for gate in second.gates if gate.name == "cost")
        self.assertEqual(
            cost_gate.details["rationale_code"], "CHEAPER_PASSING_CANDIDATE_EXISTS"
        )

    def test_registry_approves_with_cas_and_keeps_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry, old, new, decision = prepared_registry(root)
            self.assertEqual(registry.approve(decision), new.state_id)
            self.assertEqual(registry.active_state_id(), new.state_id)
            self.assertEqual(registry.read_state(old.state_id)["source_revision"], "old")
            decisions = list((root / "decisions").glob("*.json"))
            transitions = list((root / "transitions").glob("*.json"))
            self.assertEqual(len(decisions), 1)
            self.assertEqual(len(transitions), 1)
            self.assertEqual(json.loads(decisions[0].read_text())["status"], "approved")

    def test_registry_rejects_cas_race_and_non_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry, old, new, decision = prepared_registry(Path(tmp))
            registry.rollback(new.state_id)
            with self.assertRaises(PointerRaceError):
                registry.approve(decision)
            cand = candidate()
            rejected = verify_candidate(cand, outcome(cand, changed=0.1), old)
            with self.assertRaises(InvalidStateError):
                registry.approve(rejected)

    def test_registry_rollback_retains_history_and_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry, old, new, decision = prepared_registry(root)
            registry.approve(decision)
            before = len(list((root / "decisions").glob("*.json")))
            rollback_id = registry.rollback(old.state_id)
            self.assertTrue(rollback_id.startswith("rollback:"))
            self.assertEqual(registry.active_state_id(), old.state_id)
            self.assertEqual(len(list((root / "decisions").glob("*.json"))), before + 1)
            self.assertTrue(registry.has_state(new.state_id))
            with self.assertRaises(ImmutableCollisionError):
                registry.register_state(old)
            with self.assertRaises(InvalidStateError):
                registry.rollback("missing")


if __name__ == "__main__":
    unittest.main()
