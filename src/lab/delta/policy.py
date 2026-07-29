"""Deterministic, auditable intervention selection for DELTA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .contracts import InterventionPlan


@dataclass(frozen=True)
class PolicyThresholds:
    """Explicit cutoffs used by :func:`choose_plan`."""

    affected_pass: float = 0.80
    transition_failure_mass: float = 0.20
    hybrid_cost: float = 1.0
    index_refresh_cost: float = 2.0
    l3_cost: float = 10.0


def _value(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _metric(drift: Any, name: str) -> float | None:
    direct = _value(drift, name, f"{name}_score", f"{name}_accuracy")
    if direct is not None and not isinstance(direct, (Mapping, list, tuple)):
        return float(direct)

    summaries = _value(
        drift,
        "behavioral_failures",
        "summary",
        "summaries",
        "metrics",
        "slices",
        default=(),
    )
    if isinstance(summaries, Mapping):
        item = summaries.get(name)
        if item is None and isinstance(summaries.get("scores"), Mapping):
            item = summaries["scores"].get(name)
        if item is not None:
            raw = _value(item, "score", "accuracy", "value", "rate", default=item)
            return float(raw)
    for item in summaries or ():
        label = _value(item, "name", "metric", "slice", "drift_type", "kind")
        if label == name:
            raw = _value(item, "score", "accuracy", "value", "rate")
            if raw is not None:
                return float(raw)
    return None


def _budget_allows(budget: Any, level: str, cost: float) -> bool:
    if budget is None:
        return True
    allowed = _value(budget, "allowed_levels", "levels")
    if allowed is not None and level not in set(allowed):
        return False
    max_level = _value(budget, "max_level")
    if max_level is not None:
        order = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}
        if order[level] > order[str(max_level)]:
            return False
    remaining = _value(budget, "remaining", "remaining_cost", "max_cost")
    return remaining is None or cost <= float(remaining)


def _known_regression(history: Iterable[Any], action: str, profile: str | None) -> bool:
    for outcome in history:
        metadata = _value(outcome, "metadata", default={})
        recorded_action = _value(outcome, "action", "intervention", "recipe")
        if recorded_action is None:
            recorded_action = _value(metadata, "action", "intervention", "recipe")
        if recorded_action != action:
            continue
        outcome_profile = _value(outcome, "profile", "profile_id", "knowledge_profile")
        if outcome_profile is None:
            outcome_profile = _value(metadata, "profile", "profile_id", "knowledge_profile")
        regressed = bool(
            _value(outcome, "regressed", "stable_regression", "profile_regression", default=False)
        )
        if not regressed:
            regressed = bool(
                _value(
                    metadata,
                    "regressed",
                    "stable_regression",
                    "profile_regression",
                    default=False,
                )
            )
        if regressed and (profile is None or outcome_profile in (None, profile)):
            return True
    return False


def choose_plan(
    drift: Any,
    active_state: Any,
    budget: Any = None,
    history: Iterable[Any] = (),
    *,
    thresholds: PolicyThresholds = PolicyThresholds(),
    oracle_debug: bool = False,
    ft_reopened: bool = False,
    l3_approved: bool = False,
) -> InterventionPlan:
    """Choose the least-cost eligible intervention from summary-like drift data."""

    event_id = str(
        _value(drift, "drift_event_id", "event_id", "drift_id", "id", default="drift")
    )
    active_id = str(_value(active_state, "state_id", "id", default=""))
    profile = _value(drift, "profile_id", "profile", "knowledge_profile")
    if profile is None:
        profile = _value(
            _value(drift, "metadata", default={}),
            "profile_id",
            "profile",
            "knowledge_profile",
        )
    changed = _metric(drift, "changed")
    removed = _metric(drift, "removed")
    affected = [score for score in (changed, removed) if score is not None]

    def result(
        level: str,
        action: str,
        reasons: tuple[str, ...],
        cost: float,
        *,
        debug_only: bool = False,
        requires_approval: bool = False,
    ) -> InterventionPlan:
        return InterventionPlan(
            drift_event_id=event_id,
            policy_version="deterministic-v1",
            selected_level=level,
            recipe=action,
            config_inputs={
                "expected_active_state_id": active_id,
                "affected_pass": thresholds.affected_pass,
                "transition_failure_mass": thresholds.transition_failure_mass,
            },
            expected_gates={
                "recovery": thresholds.affected_pass,
                "stable_regression": "non_regression",
                "coverage": "required",
                "cost": cost,
                "provenance": "required",
            },
            cost_bound=cost,
            rationale_codes=reasons,
            ft_eligible=level == "L3" and ft_reopened and l3_approved,
            metadata={
                "debug_only": debug_only,
                "requires_approval": requires_approval,
            },
        )

    if oracle_debug:
        return result(
            "L1",
            "oracle_context",
            ("ORACLE_DEBUG_ONLY",),
            thresholds.hybrid_cost,
            debug_only=True,
        )

    if affected and all(score >= thresholds.affected_pass for score in affected):
        return result("L0", "no_op", ("AFFECTED_METRICS_PASS",), 0.0)

    transition_failure = any(
        (1.0 - score) >= thresholds.transition_failure_mass for score in affected
    )
    if (
        transition_failure
        and _budget_allows(budget, "L1", thresholds.hybrid_cost)
        and not _known_regression(history, "hybrid_diff", profile)
    ):
        return result(
            "L1",
            "hybrid_diff",
            ("CHANGED_OR_REMOVED_FAILURE_MASS", "FT_FROZEN"),
            thresholds.hybrid_cost,
        )

    if (
        _budget_allows(budget, "L2", thresholds.index_refresh_cost)
        and not _known_regression(history, "index_refresh", profile)
    ):
        reasons = ["INDEX_REFRESH_FALLBACK", "FT_FROZEN"]
        if transition_failure:
            reasons.append("L1_BLOCKED_BY_BUDGET_OR_HISTORY")
        else:
            reasons.append("AFFECTED_METRICS_MISSING_OR_INCONCLUSIVE")
        return result("L2", "index_refresh", tuple(reasons), thresholds.index_refresh_cost)

    if (
        ft_reopened
        and l3_approved
        and _budget_allows(budget, "L3", thresholds.l3_cost)
        and not _known_regression(history, "lora", profile)
    ):
        return result(
            "L3",
            "lora",
            ("FT_EXPLICITLY_REOPENED", "L3_EXPLICITLY_APPROVED"),
            thresholds.l3_cost,
            requires_approval=True,
        )

    reasons = ["NO_ELIGIBLE_INTERVENTION"]
    if ft_reopened != l3_approved:
        reasons.append("L3_REQUIRES_REOPEN_AND_APPROVAL")
    return result("L0", "no_op", tuple(reasons), 0.0)
