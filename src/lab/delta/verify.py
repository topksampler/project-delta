"""Hard-gated candidate verification for DELTA."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from .contracts import GateResult, PromotionDecision


@dataclass(frozen=True)
class VerificationThresholds:
    recovery: float = 0.80
    minimum_improvement: float = 0.05
    stable_regression_budget: float = 0.02
    coverage: float = 0.95
    max_cost: float = 1.0e18


def _value(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _metric(value: Any, name: str, default: float | None = None) -> float | None:
    direct = _value(value, name, f"{name}_score", f"{name}_accuracy")
    if direct is not None and not isinstance(direct, Mapping):
        return float(direct)
    metrics = _value(value, "metrics", "scores", "summary", default={})
    if not metrics:
        metadata = _value(value, "metadata", default={})
        metrics = _value(metadata, "metrics", "scores", "summary", default={})
    if isinstance(metrics, Mapping) and name in metrics:
        item = metrics[name]
        raw = _value(item, "score", "accuracy", "value", "rate", default=item)
        return float(raw)
    return default


def _gate(
    name: str,
    passed: bool,
    observed: Any,
    threshold: Any,
    reason_code: str,
    evidence_path: str,
) -> GateResult:
    return GateResult(
        name=name,
        passed=passed,
        actual=observed,
        threshold=threshold,
        evidence_path=evidence_path,
        hard=True,
        details={"rationale_code": reason_code},
    )


def _prior_passed_cheaper(history: Iterable[Any], cost: float) -> bool:
    for decision in history:
        raw_status = _value(decision, "status", "decision", default="")
        status = str(raw_status.value if isinstance(raw_status, Enum) else raw_status).lower()
        passed = status in {"pending_approval", "approved", "promoted"} or bool(
            _value(decision, "passed", "hard_gates_passed", default=False)
        )
        prior_cost = _value(decision, "cost", "measured_cost", "estimated_cost")
        if prior_cost is None:
            prior_cost = _value(_value(decision, "metadata", default={}), "measured_cost")
        if passed and prior_cost is not None and float(prior_cost) < cost:
            return True
    return False


def verify_candidate(
    candidate: Any,
    outcome: Any,
    active_state: Any,
    *,
    thresholds: VerificationThresholds = VerificationThresholds(),
    history: Iterable[Any] = (),
) -> PromotionDecision:
    """Evaluate non-pooled hard gates and emit an approval-pending decision."""

    candidate_id = str(
        _value(candidate, "candidate_id", "state_id", "candidate_state_id", "id", default="")
    )
    active_id = str(_value(active_state, "state_id", "id", default=""))
    per_drift = _value(outcome, "per_drift_metrics", default={})
    changed = _metric(per_drift, "changed")
    removed = _metric(per_drift, "removed")
    affected_scores = [score for score in (changed, removed) if score is not None]
    recovery_observed = min(affected_scores) if affected_scores else None
    active_changed = _metric(active_state, "changed")
    active_removed = _metric(active_state, "removed")
    improvements = [
        candidate_score - active_score
        for candidate_score, active_score in (
            (changed, active_changed),
            (removed, active_removed),
        )
        if candidate_score is not None and active_score is not None
    ]
    recovery_pass = (
        recovery_observed is not None
        and recovery_observed >= thresholds.recovery
        and all(delta >= thresholds.minimum_improvement for delta in improvements)
    )

    candidate_stable = _metric(outcome, "stable")
    active_stable = _metric(active_state, "stable")
    regression = (
        active_stable - candidate_stable
        if active_stable is not None and candidate_stable is not None
        else None
    )
    stable_pass = (
        regression is not None and regression <= thresholds.stable_regression_budget
    )

    coverage = _metric(outcome, "coverage")
    coverage_pass = coverage is not None and coverage >= thresholds.coverage

    cost = float(_value(outcome, "cost", "measured_cost", "estimated_cost", default=1.0e18))
    cost_pass = cost <= thresholds.max_cost and not _prior_passed_cheaper(history, cost)

    artifact_hashes = _value(candidate, "artifact_hashes", default={})
    run_ids = _value(candidate, "run_ids", default=())
    outcome_paths = _value(outcome, "paths", default={})
    provenance_pass = (
        isinstance(artifact_hashes, Mapping)
        and bool(artifact_hashes)
        and bool(run_ids)
        and isinstance(outcome_paths, Mapping)
        and bool(outcome_paths)
    )
    outcome_metadata = _value(outcome, "metadata", default={})
    environment_sealed = _value(
        outcome_metadata, "environment_sealed", default=None
    )
    evidence_path = str(
        _value(outcome_paths, "metrics", "evidence", "report", default="inline:outcome")
    )

    gates = [
        _gate(
            "recovery",
            recovery_pass,
            recovery_observed,
            {
                "floor": thresholds.recovery,
                "minimum_improvement": thresholds.minimum_improvement,
            },
            "RECOVERY_PASS" if recovery_pass else "RECOVERY_FAILED",
            evidence_path,
        ),
        _gate(
            "stable_regression",
            stable_pass,
            regression,
            thresholds.stable_regression_budget,
            "STABLE_REGRESSION_PASS" if stable_pass else "STABLE_REGRESSION_FAILED",
            evidence_path,
        ),
        _gate(
            "coverage",
            coverage_pass,
            coverage,
            thresholds.coverage,
            "COVERAGE_PASS" if coverage_pass else "COVERAGE_FAILED",
            evidence_path,
        ),
        _gate(
            "cost",
            cost_pass,
            cost,
            thresholds.max_cost,
            "COST_PASS"
            if cost_pass
            else (
                "CHEAPER_PASSING_CANDIDATE_EXISTS"
                if _prior_passed_cheaper(history, cost)
                else "COST_FAILED"
            ),
            evidence_path,
        ),
        _gate(
            "provenance",
            provenance_pass,
            provenance_pass,
            True,
            "PROVENANCE_PASS" if provenance_pass else "PROVENANCE_FAILED",
            evidence_path,
        ),
    ]
    if environment_sealed is not None:
        sealed_pass = environment_sealed is True
        gates.append(
            _gate(
                "environment_sealed",
                sealed_pass,
                environment_sealed,
                True,
                (
                    "ENVIRONMENT_SEALED_PASS"
                    if sealed_pass
                    else "ENVIRONMENT_UNSEALED"
                ),
                evidence_path,
            )
        )
    gates_tuple = tuple(gates)
    passed = all(bool(_value(gate, "passed", default=False)) for gate in gates_tuple)
    status = "pending_approval" if passed else "rejected"
    failed_reasons = tuple(
        str(_value(_value(gate, "details", default={}), "rationale_code"))
        for gate in gates_tuple
        if not bool(_value(gate, "passed", default=False))
    )
    return PromotionDecision(
        candidate_id=candidate_id,
        outcome_id=str(_value(outcome, "outcome_id", "id", default="")),
        expected_active_state_id=active_id,
        status=status,
        gates=gates_tuple,
        rationale_codes=failed_reasons or ("ALL_HARD_GATES_PASS",),
        paths={"evidence": evidence_path},
        metadata={"measured_cost": cost},
    )
