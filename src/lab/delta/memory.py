"""Local immutable state registry with an atomic active-state pointer."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping


class RegistryError(RuntimeError):
    pass


class InvalidStateError(RegistryError):
    pass


class PointerRaceError(RegistryError):
    pass


class ImmutableCollisionError(RegistryError):
    pass


def _value(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


class LocalStateRegistry:
    """Filesystem implementation whose object layout can be mirrored to B2."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.states = self.root / "states"
        self.decisions = self.root / "decisions"
        self.transitions = self.root / "transitions"
        self.active_path = self.states / "active.json"
        self.lock_path = self.root / ".active.lock"
        self.states.mkdir(parents=True, exist_ok=True)
        self.decisions.mkdir(parents=True, exist_ok=True)
        self.transitions.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        deadline = time.monotonic() + 5.0
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(fd)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise PointerRaceError("active-state lock is held")
                time.sleep(0.01)
        try:
            yield
        finally:
            self.lock_path.unlink(missing_ok=True)

    @staticmethod
    def _write_immutable(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
        except FileExistsError as exc:
            raise ImmutableCollisionError(f"immutable object already exists: {path}") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    def _write_active(self, state_id: str) -> None:
        payload = json.dumps({"state_id": state_id}, indent=2, sort_keys=True) + "\n"
        temp = self.states / f".active.{os.getpid()}.{time.time_ns()}.tmp"
        try:
            with open(temp, "x", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.active_path)
            directory_fd = os.open(self.states, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temp.unlink(missing_ok=True)

    def register_state(self, state: Any) -> str:
        state_id = str(_value(state, "state_id", "id", default=""))
        if not state_id:
            raise InvalidStateError("state_id is required")
        self._write_immutable(self.states / state_id / "state.json", state)
        return state_id

    def initialize(self, state: Any) -> str:
        """Register and activate the first state; never overwrite an active pointer."""

        state_id = self.register_state(state)
        with self._locked():
            if self.active_path.exists():
                raise PointerRaceError("active state is already initialized")
            self._write_active(state_id)
        return state_id

    def active_state_id(self) -> str:
        if not self.active_path.exists():
            raise InvalidStateError("active state is not initialized")
        try:
            payload = json.loads(self.active_path.read_text(encoding="utf-8"))
            state_id = str(payload["state_id"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise InvalidStateError("active pointer is invalid") from exc
        if not self.has_state(state_id):
            raise InvalidStateError(f"active pointer references missing state: {state_id}")
        return state_id

    def has_state(self, state_id: str) -> bool:
        return (self.states / state_id / "state.json").is_file()

    def read_state(self, state_id: str) -> dict[str, Any]:
        path = self.states / state_id / "state.json"
        if not path.is_file():
            raise InvalidStateError(f"unknown state: {state_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def record_decision(self, decision: Any) -> str:
        """Persist a pending or rejected decision before any pointer mutation."""

        decision_id = str(_value(decision, "decision_id", "id", default=""))
        if not decision_id:
            raise InvalidStateError("decision_id is required")
        self._write_immutable(self.decisions / f"{decision_id}.json", decision)
        return decision_id

    def _candidate_state_id(self, candidate_id: str) -> str:
        if self.has_state(candidate_id):
            return candidate_id
        for path in self.states.glob("*/state.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("candidate_id") == candidate_id:
                return str(payload["state_id"])
        raise InvalidStateError(f"candidate state is not registered: {candidate_id}")

    def approve(self, decision: Any) -> str:
        raw_status = _value(decision, "status", "decision", default="")
        status = str(raw_status.value if isinstance(raw_status, Enum) else raw_status).lower()
        if status != "pending_approval":
            raise InvalidStateError("only pending_approval decisions can be approved")
        expected = str(
            _value(decision, "expected_active_state_id", "active_state_id", default="")
        )
        candidate_id = str(
            _value(decision, "candidate_id", "candidate_state_id", "state_id", default="")
        )
        decision_id = str(_value(decision, "decision_id", "id", default=""))
        if not expected or not candidate_id or not decision_id:
            raise InvalidStateError("decision identifiers are incomplete")
        candidate = self._candidate_state_id(candidate_id)

        with self._locked():
            actual = self.active_state_id()
            if actual != expected:
                raise PointerRaceError(
                    f"active state changed: expected {expected}, found {actual}"
                )
            approved_payload = _jsonable(decision)
            approved_payload.pop("decision_id", None)
            approved_payload["status"] = "approved"
            try:
                from .contracts import PromotionDecision

                approved = PromotionDecision.from_dict(approved_payload)
                approved_id = approved.decision_id
            except (ImportError, AttributeError):
                approved = approved_payload
                approved_id = decision_id
            self._write_immutable(self.decisions / f"{approved_id}.json", approved)
            self._write_immutable(
                self.transitions / f"{approved_id}.json",
                {
                    "transition_id": approved_id,
                    "kind": "promotion",
                    "from_state_id": actual,
                    "to_state_id": candidate,
                    "decision_id": approved_id,
                },
            )
            if self.active_state_id() != expected:
                raise PointerRaceError("active pointer changed during approval")
            self._write_active(candidate)
        return candidate

    def rollback(self, target_state_id: str) -> str:
        if not self.has_state(target_state_id):
            raise InvalidStateError(f"unknown rollback target: {target_state_id}")
        with self._locked():
            current = self.active_state_id()
            if current == target_state_id:
                raise InvalidStateError("rollback target is already active")
            transition_id = (
                f"rollback:{current}:{target_state_id}:{time.time_ns()}"
            )
            record = {
                "decision_id": transition_id,
                "status": "approved",
                "kind": "rollback",
                "expected_active_state_id": current,
                "candidate_state_id": target_state_id,
                "reason_codes": ["EXPLICIT_ROLLBACK"],
            }
            self._write_immutable(self.decisions / f"{transition_id}.json", record)
            self._write_immutable(
                self.transitions / f"{transition_id}.json",
                {
                    "transition_id": transition_id,
                    "kind": "rollback",
                    "from_state_id": current,
                    "to_state_id": target_state_id,
                    "decision_id": transition_id,
                },
            )
            if self.active_state_id() != current:
                raise PointerRaceError("active pointer changed during rollback")
            self._write_active(target_state_id)
        return transition_id


StateRegistry = LocalStateRegistry
