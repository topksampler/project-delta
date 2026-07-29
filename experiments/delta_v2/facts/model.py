from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXTRACTOR_VERSION = "delta-v2-python-ast-v1"
OBSERVATION_SCHEMA = "delta.atomic_fact_observation.v1"
REJECTION_SCHEMA = "delta.atomic_fact_rejection.v1"
DELTA_SCHEMA = "delta.atomic_fact_delta.v1"
DELTA_STATUSES = ("stable", "added", "removed", "changed")


class FactExtractionError(ValueError):
    """Extraction cannot continue without weakening the registered contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"kind": "float", "value": repr(value)}
    if isinstance(value, bytes):
        return {"kind": "bytes", "hex": value.hex()}
    if isinstance(value, complex):
        return {"kind": "complex", "value": repr(value)}
    if value is Ellipsis:
        return {"kind": "ellipsis"}
    raise FactExtractionError(
        f"cannot normalize scalar of type {type(value).__name__}"
    )


def normalize_literal(value: Any) -> Any:
    if value is None or isinstance(
        value,
        (bool, int, float, str, bytes, complex),
    ) or value is Ellipsis:
        return _json_scalar(value)
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [normalize_literal(item) for item in value],
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [normalize_literal(item) for item in value],
        }
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": [
                [normalize_literal(key), normalize_literal(item)]
                for key, item in value.items()
            ],
        }
    if isinstance(value, (set, frozenset)):
        items = [normalize_literal(item) for item in value]
        items.sort(key=canonical_json_bytes)
        return {
            "kind": type(value).__name__,
            "items": items,
        }
    raise FactExtractionError(
        f"cannot normalize literal of type {type(value).__name__}"
    )


def try_literal(node: ast.AST) -> tuple[bool, Any]:
    try:
        value = ast.literal_eval(node)
        return True, normalize_literal(value)
    except (FactExtractionError, ValueError, TypeError, SyntaxError, MemoryError):
        return False, None


def canonical_ast(value: Any) -> Any:
    """Convert an AST into location-free, deterministic JSON data."""

    if isinstance(value, ast.AST):
        return {
            "node": type(value).__name__,
            "fields": {
                field: canonical_ast(child)
                for field, child in ast.iter_fields(value)
            },
        }
    if isinstance(value, list):
        return [canonical_ast(item) for item in value]
    return _json_scalar(value)


def stable_fact_id(
    repository_id: str,
    family_id: str,
    semantic_key: str,
) -> str:
    identity = "\0".join(
        (
            DELTA_SCHEMA,
            repository_id,
            family_id,
            semantic_key,
        )
    ).encode("utf-8")
    return f"fact:{hashlib.sha256(identity).hexdigest()}"


@dataclass(frozen=True)
class Evidence:
    snapshot_role: str
    path: str
    line_start: int
    line_end: int
    git_blob_sha1: str
    source_sha256: str
    extractor_version: str = EXTRACTOR_VERSION

    def __post_init__(self) -> None:
        if self.line_start < 1 or self.line_end < self.line_start:
            raise FactExtractionError(
                f"invalid evidence lines: {self.line_start}-{self.line_end}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_role": self.snapshot_role,
            "path": self.path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "git_blob_sha1": self.git_blob_sha1,
            "source_sha256": self.source_sha256,
            "extractor_version": self.extractor_version,
        }


@dataclass(frozen=True)
class FactObservation:
    fact_id: str
    family_id: str
    semantic_key: str
    value: Any
    evidence: Evidence

    @classmethod
    def create(
        cls,
        *,
        repository_id: str,
        family_id: str,
        semantic_key: str,
        value: Any,
        evidence: Evidence,
    ) -> "FactObservation":
        canonical_json_bytes(value)
        return cls(
            fact_id=stable_fact_id(
                repository_id,
                family_id,
                semantic_key,
            ),
            family_id=family_id,
            semantic_key=semantic_key,
            value=value,
            evidence=evidence,
        )

    @property
    def sort_key(self) -> tuple[str, str]:
        return self.family_id, self.semantic_key

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": OBSERVATION_SCHEMA,
            "fact_id": self.fact_id,
            "family_id": self.family_id,
            "semantic_key": self.semantic_key,
            "value": self.value,
            "evidence": self.evidence.as_dict(),
        }


@dataclass(frozen=True)
class RejectedFactCandidate:
    rejection_id: str
    family_id: str
    reason: str
    evidence: Evidence

    @classmethod
    def create(
        cls,
        *,
        repository_id: str,
        family_id: str,
        reason: str,
        evidence: Evidence,
    ) -> "RejectedFactCandidate":
        identity = canonical_json_bytes(
            {
                "schema": REJECTION_SCHEMA,
                "repository_id": repository_id,
                "family_id": family_id,
                "reason": reason,
                "evidence": evidence.as_dict(),
            }
        )
        return cls(
            rejection_id=f"rejection:{hashlib.sha256(identity).hexdigest()}",
            family_id=family_id,
            reason=reason,
            evidence=evidence,
        )

    @property
    def sort_key(self) -> tuple[str, str, int, str]:
        return (
            self.family_id,
            self.evidence.path,
            self.evidence.line_start,
            self.reason,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": REJECTION_SCHEMA,
            "rejection_id": self.rejection_id,
            "family_id": self.family_id,
            "reason": self.reason,
            "evidence": self.evidence.as_dict(),
        }


class ObservationStore:
    def __init__(self) -> None:
        self._observations: dict[
            tuple[str, str],
            FactObservation,
        ] = {}
        self._rejections: list[RejectedFactCandidate] = []

    def add(self, observation: FactObservation) -> None:
        key = observation.sort_key
        if key in self._observations:
            raise FactExtractionError(
                "duplicate semantic key in one family snapshot: "
                f"{observation.family_id} {observation.semantic_key}"
            )
        self._observations[key] = observation

    def reject(self, rejection: RejectedFactCandidate) -> None:
        self._rejections.append(rejection)

    @property
    def observations(self) -> list[FactObservation]:
        return sorted(self._observations.values(), key=lambda row: row.sort_key)

    @property
    def rejections(self) -> list[RejectedFactCandidate]:
        return sorted(self._rejections, key=lambda row: row.sort_key)


@dataclass(frozen=True)
class AtomicFactDelta:
    fact_id: str
    family: str
    semantic_key: str
    status: str
    value_before: Any | None
    value_after: Any | None
    evidence_before: Evidence | None
    evidence_after: Evidence | None
    verifier: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.status not in DELTA_STATUSES:
            raise FactExtractionError(
                f"invalid AtomicFactDelta status: {self.status}"
            )

    @property
    def sort_key(self) -> tuple[str, str]:
        return self.family, self.semantic_key

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": DELTA_SCHEMA,
            "fact_id": self.fact_id,
            "family": self.family,
            "semantic_key": self.semantic_key,
            "status": self.status,
            "value_before": self.value_before,
            "value_after": self.value_after,
            "evidence_before": (
                self.evidence_before.as_dict()
                if self.evidence_before is not None
                else None
            ),
            "evidence_after": (
                self.evidence_after.as_dict()
                if self.evidence_after is not None
                else None
            ),
            "verifier": dict(self.verifier),
        }


def join_observations(
    before: Sequence[FactObservation],
    after: Sequence[FactObservation],
    *,
    verifier: Mapping[str, Any],
) -> list[AtomicFactDelta]:
    before_by_key = {row.sort_key: row for row in before}
    after_by_key = {row.sort_key: row for row in after}
    if len(before_by_key) != len(before):
        raise FactExtractionError("duplicate observations on before side")
    if len(after_by_key) != len(after):
        raise FactExtractionError("duplicate observations on after side")

    rows: list[AtomicFactDelta] = []
    for family, semantic_key in sorted(set(before_by_key) | set(after_by_key)):
        old = before_by_key.get((family, semantic_key))
        new = after_by_key.get((family, semantic_key))
        if old is None:
            status = "added"
            fact_id = new.fact_id
        elif new is None:
            status = "removed"
            fact_id = old.fact_id
        else:
            if old.fact_id != new.fact_id:
                raise FactExtractionError(
                    f"fact ID changed across snapshots: {family} {semantic_key}"
                )
            status = (
                "stable"
                if canonical_json_bytes(old.value)
                == canonical_json_bytes(new.value)
                else "changed"
            )
            fact_id = old.fact_id
        rows.append(
            AtomicFactDelta(
                fact_id=fact_id,
                family=family,
                semantic_key=semantic_key,
                status=status,
                value_before=old.value if old is not None else None,
                value_after=new.value if new is not None else None,
                evidence_before=old.evidence if old is not None else None,
                evidence_after=new.evidence if new is not None else None,
                verifier=verifier,
            )
        )
    return rows


def _observation_projection(
    rows: Iterable[FactObservation],
) -> dict[tuple[str, str], tuple[str, Any, dict[str, Any]]]:
    return {
        row.sort_key: (
            row.fact_id,
            row.value,
            row.evidence.as_dict(),
        )
        for row in rows
    }


def audit_fact_deltas(
    rows: Sequence[AtomicFactDelta],
    before: Sequence[FactObservation],
    after: Sequence[FactObservation],
) -> dict[str, Any]:
    projected_before: list[FactObservation] = []
    projected_after: list[FactObservation] = []
    status_counts = {status: 0 for status in DELTA_STATUSES}
    family_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        status_counts[row.status] += 1
        family_counts.setdefault(
            row.family,
            {status: 0 for status in DELTA_STATUSES},
        )[row.status] += 1
        if row.evidence_before is not None:
            projected_before.append(
                FactObservation(
                    fact_id=row.fact_id,
                    family_id=row.family,
                    semantic_key=row.semantic_key,
                    value=row.value_before,
                    evidence=row.evidence_before,
                )
            )
        if row.evidence_after is not None:
            projected_after.append(
                FactObservation(
                    fact_id=row.fact_id,
                    family_id=row.family,
                    semantic_key=row.semantic_key,
                    value=row.value_after,
                    evidence=row.evidence_after,
                )
            )
    return {
        "rows": len(rows),
        "observations_before": len(before),
        "observations_after": len(after),
        "status_counts": status_counts,
        "family_status_counts": family_counts,
        "reconstructs_before": (
            _observation_projection(projected_before)
            == _observation_projection(before)
        ),
        "reconstructs_after": (
            _observation_projection(projected_after)
            == _observation_projection(after)
        ),
        "unique_fact_ids": len({row.fact_id for row in rows}) == len(rows),
        "deterministic_order": list(rows)
        == sorted(rows, key=lambda row: row.sort_key),
    }


def write_jsonl(rows: Iterable[Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for row in rows:
            line = canonical_json_bytes(row.as_dict()) + b"\n"
            handle.write(line)
            digest.update(line)
    return digest.hexdigest()


def write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
