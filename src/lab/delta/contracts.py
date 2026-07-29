"""Versioned data contracts for the deterministic DELTA control plane.

The contracts deliberately use only the Python standard library.  They are
frozen dataclasses and recursively freeze JSON containers on construction, so
callers cannot accidentally mutate data after its content-derived identifier
has been calculated.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, ClassVar, Iterator, Mapping, TypeVar


class ContractValidationError(ValueError):
    """Raised when serialized contract data violates its schema."""


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class ReconcileStatus(_StringEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"


class DriftType(_StringEnum):
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    MOVED = "moved"


class InterventionLevel(_StringEnum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


class CandidateStatus(_StringEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class OutcomeStatus(_StringEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class PromotionStatus(_StringEnum):
    PENDING_APPROVAL = "pending_approval"
    REJECTED = "rejected"
    APPROVED = "approved"


def _fail(path: str, message: str) -> ContractValidationError:
    return ContractValidationError(f"{path}: {message}")


def _require_string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise _fail(path, "must be a string")
    if not allow_empty and not value:
        raise _fail(path, "must not be empty")
    return value


def _require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise _fail(path, "must be a boolean")
    return value


def _require_number(
    value: Any, path: str, *, minimum: float | None = None
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(path, "must be a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise _fail(path, "must be finite")
    if minimum is not None and value < minimum:
        raise _fail(path, f"must be at least {minimum}")
    return value


E = TypeVar("E", bound=Enum)


def _coerce_enum(value: Any, enum_type: type[E], path: str) -> E:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(repr(item.value) for item in enum_type)
        raise _fail(path, f"must be one of {allowed}") from exc


class FrozenMapping(Mapping[str, Any]):
    """A small immutable mapping that remains compatible with dataclasses.asdict."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = dict(data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenMapping({self._data!r})"

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenMapping:
        return self


def _freeze_json(value: Any, path: str = "value") -> Any:
    """Validate and recursively freeze a JSON-compatible value."""

    if isinstance(value, Enum):
        value = value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _fail(path, "JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise _fail(path, "JSON object keys must be strings")
            frozen[key] = _freeze_json(item, f"{path}.{key}")
        return FrozenMapping(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{path}[{index}]") for index, item in enumerate(value))
    raise _fail(path, f"{type(value).__name__} is not JSON-compatible")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Return compact, key-sorted JSON suitable for hashing and storage."""

    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return json.dumps(
        _thaw_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    """Return a stable SHA-256 hex digest for JSON-compatible content."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _freeze_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(path, "must be a mapping")
    return _freeze_json(value, path)


def _freeze_strings(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise _fail(path, "must be a list or tuple")
    result = tuple(_require_string(item, f"{path}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise _fail(path, "must not contain duplicates")
    return result


def _contract_input(
    data: Mapping[str, Any],
    *,
    schema_id: str,
    id_field: str,
    allowed_fields: set[str],
) -> tuple[dict[str, Any], str | None]:
    if not isinstance(data, Mapping):
        raise _fail("contract", "must be a mapping")
    actual_schema = data.get("schema_id")
    if actual_schema != schema_id:
        raise _fail("schema_id", f"expected {schema_id!r}, got {actual_schema!r}")
    unknown = set(data) - allowed_fields - {"schema_id", id_field}
    if unknown:
        raise _fail("contract", f"unknown fields: {', '.join(sorted(unknown))}")
    supplied_id = data.get(id_field)
    if supplied_id is not None:
        _require_string(supplied_id, id_field)
    return {key: value for key, value in data.items() if key not in {"schema_id", id_field}}, supplied_id


@dataclass(frozen=True)
class _Contract:
    SCHEMA_ID: ClassVar[str]
    ID_FIELD: ClassVar[str]

    @property
    def schema_id(self) -> str:
        return self.SCHEMA_ID

    def _content(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"schema_id": self.SCHEMA_ID}
        for field in fields(self):
            payload[field.name] = _thaw_json(getattr(self, field.name))
        return payload

    @property
    def stable_id(self) -> str:
        return content_hash(self._content())

    @property
    def content_hash(self) -> str:
        return self.stable_id

    def to_dict(self) -> dict[str, Any]:
        payload = self._content()
        payload[self.ID_FIELD] = self.stable_id
        return payload

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def _allowed_fields(cls) -> set[str]:
        return {field.name for field in fields(cls)}

    @classmethod
    def _parse(cls, data: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
        return _contract_input(
            data,
            schema_id=cls.SCHEMA_ID,
            id_field=cls.ID_FIELD,
            allowed_fields=cls._allowed_fields(),
        )

    @classmethod
    def _check_id(cls, instance: _Contract, supplied_id: str | None) -> None:
        if supplied_id is not None and supplied_id != instance.stable_id:
            raise _fail(cls.ID_FIELD, "does not match canonical contract content")


@dataclass(frozen=True, kw_only=True)
class ReconcileRun(_Contract):
    SCHEMA_ID: ClassVar[str] = "delta.reconcile_run.v1"
    ID_FIELD: ClassVar[str] = "reconcile_id"

    repo: str
    source_before: str
    source_after: str
    experiment_plugin: str
    status: ReconcileStatus = ReconcileStatus.PENDING
    stage_status: Mapping[str, Any] = field(default_factory=dict)
    child_run_ids: tuple[str, ...] = ()
    artifact_hashes: Mapping[str, Any] = field(default_factory=dict)
    paths: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("repo", "source_before", "source_after", "experiment_plugin"):
            object.__setattr__(self, name, _require_string(getattr(self, name), name))
        object.__setattr__(self, "status", _coerce_enum(self.status, ReconcileStatus, "status"))
        stage_status = _freeze_mapping(self.stage_status, "stage_status")
        for stage, status in stage_status.items():
            _require_string(stage, "stage_status key")
            _coerce_enum(status, ReconcileStatus, f"stage_status.{stage}")
        object.__setattr__(self, "stage_status", stage_status)
        object.__setattr__(self, "child_run_ids", _freeze_strings(self.child_run_ids, "child_run_ids"))
        for name in ("artifact_hashes", "paths", "metadata"):
            object.__setattr__(self, name, _freeze_mapping(getattr(self, name), name))

    @property
    def reconcile_id(self) -> str:
        return self.stable_id

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReconcileRun:
        values, supplied_id = cls._parse(data)
        instance = cls(**values)
        cls._check_id(instance, supplied_id)
        return instance


@dataclass(frozen=True, kw_only=True)
class DriftEventV2(_Contract):
    SCHEMA_ID: ClassVar[str] = "delta.drift_event.v2"
    ID_FIELD: ClassVar[str] = "drift_event_id"

    source_before: Mapping[str, Any]
    source_after: Mapping[str, Any]
    structural_census: Mapping[str, Any]
    behavioral_failures: Mapping[str, Any]
    affected_claim_ids: tuple[str, ...]
    drift_types: tuple[DriftType, ...] = ()
    drift_score: float = 0.0
    paths: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "source_before",
            "source_after",
            "structural_census",
            "behavioral_failures",
            "paths",
            "metadata",
        ):
            object.__setattr__(self, name, _freeze_mapping(getattr(self, name), name))
        object.__setattr__(
            self, "affected_claim_ids", _freeze_strings(self.affected_claim_ids, "affected_claim_ids")
        )
        if not isinstance(self.drift_types, (list, tuple)):
            raise _fail("drift_types", "must be a list or tuple")
        drift_types = tuple(
            _coerce_enum(value, DriftType, f"drift_types[{index}]")
            for index, value in enumerate(self.drift_types)
        )
        if len(set(drift_types)) != len(drift_types):
            raise _fail("drift_types", "must not contain duplicates")
        object.__setattr__(self, "drift_types", drift_types)
        object.__setattr__(
            self, "drift_score", _require_number(self.drift_score, "drift_score", minimum=0.0)
        )

    @property
    def drift_event_id(self) -> str:
        return self.stable_id

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DriftEventV2:
        values, supplied_id = cls._parse(data)
        instance = cls(**values)
        cls._check_id(instance, supplied_id)
        return instance


@dataclass(frozen=True, kw_only=True)
class InterventionPlan(_Contract):
    SCHEMA_ID: ClassVar[str] = "delta.intervention_plan.v1"
    ID_FIELD: ClassVar[str] = "plan_id"

    drift_event_id: str
    policy_version: str
    selected_level: InterventionLevel
    recipe: str
    config_inputs: Mapping[str, Any]
    expected_gates: Mapping[str, Any]
    cost_bound: float
    rationale_codes: tuple[str, ...]
    ft_eligible: bool = False
    paths: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("drift_event_id", "policy_version", "recipe"):
            object.__setattr__(self, name, _require_string(getattr(self, name), name))
        level = _coerce_enum(self.selected_level, InterventionLevel, "selected_level")
        object.__setattr__(self, "selected_level", level)
        object.__setattr__(self, "ft_eligible", _require_bool(self.ft_eligible, "ft_eligible"))
        if level is InterventionLevel.L3 and not self.ft_eligible:
            raise _fail("selected_level", "L3 requires ft_eligible=true")
        for name in ("config_inputs", "expected_gates", "paths", "metadata"):
            object.__setattr__(self, name, _freeze_mapping(getattr(self, name), name))
        object.__setattr__(
            self, "cost_bound", _require_number(self.cost_bound, "cost_bound", minimum=0.0)
        )
        object.__setattr__(
            self, "rationale_codes", _freeze_strings(self.rationale_codes, "rationale_codes")
        )

    @property
    def plan_id(self) -> str:
        return self.stable_id

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InterventionPlan:
        values, supplied_id = cls._parse(data)
        instance = cls(**values)
        cls._check_id(instance, supplied_id)
        return instance


@dataclass(frozen=True, kw_only=True)
class CandidateState(_Contract):
    SCHEMA_ID: ClassVar[str] = "delta.candidate_state.v1"
    ID_FIELD: ClassVar[str] = "candidate_id"

    plan_id: str
    drift_event_id: str
    base_model: str
    source_revision: str
    eval_environment_id: str
    status: CandidateStatus = CandidateStatus.PENDING
    adapter_revision: str | None = None
    index_revision: str | None = None
    run_ids: tuple[str, ...] = ()
    artifact_hashes: Mapping[str, Any] = field(default_factory=dict)
    paths: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("plan_id", "drift_event_id", "base_model", "source_revision", "eval_environment_id"):
            object.__setattr__(self, name, _require_string(getattr(self, name), name))
        object.__setattr__(self, "status", _coerce_enum(self.status, CandidateStatus, "status"))
        for name in ("adapter_revision", "index_revision"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _require_string(value, name))
        object.__setattr__(self, "run_ids", _freeze_strings(self.run_ids, "run_ids"))
        for name in ("artifact_hashes", "paths", "metadata"):
            object.__setattr__(self, name, _freeze_mapping(getattr(self, name), name))

    @property
    def candidate_id(self) -> str:
        return self.stable_id

    @property
    def state_id(self) -> str:
        """Alias used by state registries for candidate states."""

        return self.candidate_id

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateState:
        values, supplied_id = cls._parse(data)
        instance = cls(**values)
        cls._check_id(instance, supplied_id)
        return instance


@dataclass(frozen=True, kw_only=True)
class InterventionOutcome(_Contract):
    SCHEMA_ID: ClassVar[str] = "delta.intervention_outcome.v1"
    ID_FIELD: ClassVar[str] = "outcome_id"

    candidate_id: str
    status: OutcomeStatus
    metrics: Mapping[str, Any]
    per_drift_metrics: Mapping[str, Any]
    timing: Mapping[str, Any]
    cost: float
    paths: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _require_string(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "status", _coerce_enum(self.status, OutcomeStatus, "status"))
        for name in ("metrics", "per_drift_metrics", "timing", "paths", "metadata"):
            object.__setattr__(self, name, _freeze_mapping(getattr(self, name), name))
        object.__setattr__(self, "cost", _require_number(self.cost, "cost", minimum=0.0))

    @property
    def outcome_id(self) -> str:
        return self.stable_id

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InterventionOutcome:
        values, supplied_id = cls._parse(data)
        instance = cls(**values)
        cls._check_id(instance, supplied_id)
        return instance


@dataclass(frozen=True, kw_only=True)
class GateResult:
    """Evidence for one VERIFY gate, embedded in a promotion decision."""

    name: str
    passed: bool
    actual: Any
    threshold: Any
    evidence_path: str
    hard: bool = True
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_string(self.name, "name"))
        object.__setattr__(self, "passed", _require_bool(self.passed, "passed"))
        object.__setattr__(self, "actual", _freeze_json(self.actual, "actual"))
        object.__setattr__(self, "threshold", _freeze_json(self.threshold, "threshold"))
        object.__setattr__(
            self, "evidence_path", _require_string(self.evidence_path, "evidence_path")
        )
        object.__setattr__(self, "hard", _require_bool(self.hard, "hard"))
        object.__setattr__(self, "details", _freeze_mapping(self.details, "details"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "actual": _thaw_json(self.actual),
            "threshold": _thaw_json(self.threshold),
            "evidence_path": self.evidence_path,
            "hard": self.hard,
            "details": _thaw_json(self.details),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GateResult:
        if not isinstance(data, Mapping):
            raise _fail("gate", "must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise _fail("gate", f"unknown fields: {', '.join(sorted(unknown))}")
        try:
            return cls(**data)
        except TypeError as exc:
            raise _fail("gate", str(exc)) from exc


@dataclass(frozen=True, kw_only=True)
class PromotionDecision(_Contract):
    SCHEMA_ID: ClassVar[str] = "delta.promotion_decision.v1"
    ID_FIELD: ClassVar[str] = "decision_id"

    candidate_id: str
    outcome_id: str
    expected_active_state_id: str
    status: PromotionStatus
    gates: tuple[GateResult, ...]
    rationale_codes: tuple[str, ...] = ()
    paths: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("candidate_id", "outcome_id", "expected_active_state_id"):
            object.__setattr__(self, name, _require_string(getattr(self, name), name))
        status = _coerce_enum(self.status, PromotionStatus, "status")
        object.__setattr__(self, "status", status)
        if not isinstance(self.gates, (list, tuple)):
            raise _fail("gates", "must be a list or tuple")
        parsed_gates = tuple(
            gate if isinstance(gate, GateResult) else GateResult.from_dict(gate)
            for gate in self.gates
        )
        if not parsed_gates:
            raise _fail("gates", "must not be empty")
        names = [gate.name for gate in parsed_gates]
        if len(set(names)) != len(names):
            raise _fail("gates", "gate names must be unique")
        hard_passed = all(gate.passed for gate in parsed_gates if gate.hard)
        if status in (PromotionStatus.PENDING_APPROVAL, PromotionStatus.APPROVED) and not hard_passed:
            raise _fail("status", f"{status.value} requires every hard gate to pass")
        object.__setattr__(self, "gates", parsed_gates)
        object.__setattr__(
            self, "rationale_codes", _freeze_strings(self.rationale_codes, "rationale_codes")
        )
        for name in ("paths", "metadata"):
            object.__setattr__(self, name, _freeze_mapping(getattr(self, name), name))

    def _content(self) -> dict[str, Any]:
        return {
            "schema_id": self.SCHEMA_ID,
            "candidate_id": self.candidate_id,
            "outcome_id": self.outcome_id,
            "expected_active_state_id": self.expected_active_state_id,
            "status": self.status.value,
            "gates": [gate.to_dict() for gate in self.gates],
            "rationale_codes": list(self.rationale_codes),
            "paths": _thaw_json(self.paths),
            "metadata": _thaw_json(self.metadata),
        }

    @property
    def decision_id(self) -> str:
        return self.stable_id

    @property
    def candidate_state_id(self) -> str:
        """Alias retained for MEMORY implementations that use state terminology."""

        return self.candidate_id

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PromotionDecision:
        values, supplied_id = cls._parse(data)
        instance = cls(**values)
        cls._check_id(instance, supplied_id)
        return instance


@dataclass(frozen=True, kw_only=True)
class ActiveState(_Contract):
    SCHEMA_ID: ClassVar[str] = "delta.active_state.v1"
    ID_FIELD: ClassVar[str] = "state_id"

    base_model: str
    source_revision: str
    eval_environment_id: str
    candidate_id: str | None = None
    adapter_revision: str | None = None
    index_revision: str | None = None
    artifact_hashes: Mapping[str, Any] = field(default_factory=dict)
    paths: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("base_model", "source_revision", "eval_environment_id"):
            object.__setattr__(self, name, _require_string(getattr(self, name), name))
        for name in ("candidate_id", "adapter_revision", "index_revision"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _require_string(value, name))
        for name in ("artifact_hashes", "paths", "metadata"):
            object.__setattr__(self, name, _freeze_mapping(getattr(self, name), name))

    @property
    def state_id(self) -> str:
        return self.stable_id

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ActiveState:
        values, supplied_id = cls._parse(data)
        instance = cls(**values)
        cls._check_id(instance, supplied_id)
        return instance


__all__ = [
    "ActiveState",
    "CandidateState",
    "CandidateStatus",
    "ContractValidationError",
    "DriftEventV2",
    "DriftType",
    "GateResult",
    "InterventionLevel",
    "InterventionOutcome",
    "InterventionPlan",
    "OutcomeStatus",
    "PromotionDecision",
    "PromotionStatus",
    "ReconcileRun",
    "ReconcileStatus",
    "canonical_json",
    "content_hash",
]
