"""Bounded small-model worker envelopes and deterministic validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

SURFACE_TASK = "surface_paraphrase_v1"
EXPLANATION_TASK = "decision_explanation_v1"
ALLOWED_TASKS = {SURFACE_TASK, EXPLANATION_TASK}


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class WorkerRequest:
    """A credential-free request whose immutable inputs define its ID."""

    task: str
    payload: Mapping[str, Any]
    constraints: Mapping[str, Any]
    prompt: str

    def __post_init__(self) -> None:
        if self.task not in ALLOWED_TASKS:
            raise ValueError(f"unsupported worker task: {self.task}")
        if not self.prompt.strip():
            raise ValueError("worker prompt must not be empty")

    @property
    def request_id(self) -> str:
        raw = _canonical(
            {
                "task": self.task,
                "payload": self.payload,
                "constraints": self.constraints,
                "prompt": self.prompt,
            }
        )
        return f"worker-{hashlib.sha256(raw.encode()).hexdigest()[:20]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "delta.worker_request.v1",
            "request_id": self.request_id,
            "task": self.task,
            "payload": dict(self.payload),
            "constraints": dict(self.constraints),
            "prompt": self.prompt,
        }


@dataclass(frozen=True)
class WorkerResult:
    """Raw proposal plus the controller-owned validation result."""

    request_id: str
    task: str
    model_id: str
    prompt_sha256: str
    decoding: Mapping[str, Any]
    raw_output: str
    accepted: bool
    reject_reason: str | None = None
    fallback_used: str | None = None

    def __post_init__(self) -> None:
        if self.task not in ALLOWED_TASKS:
            raise ValueError(f"unsupported worker task: {self.task}")
        if self.accepted and self.reject_reason:
            raise ValueError("accepted worker result cannot have a reject reason")
        if not self.accepted and not self.reject_reason:
            raise ValueError("rejected worker result requires a reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "delta.worker_result.v1",
            "request_id": self.request_id,
            "task": self.task,
            "model_id": self.model_id,
            "prompt_sha256": self.prompt_sha256,
            "decoding": dict(self.decoding),
            "raw_output": self.raw_output,
            "accepted": self.accepted,
            "reject_reason": self.reject_reason,
            "fallback_used": self.fallback_used,
        }


def validate_proposal(
    *,
    request: WorkerRequest,
    model_id: str,
    raw_output: str,
    validator: Callable[[str], str | None],
    decoding: Mapping[str, Any] | None = None,
    fallback_used: str | None = None,
) -> WorkerResult:
    """Record a proposal; the supplied deterministic validator owns acceptance."""

    reason = validator(raw_output)
    return WorkerResult(
        request_id=request.request_id,
        task=request.task,
        model_id=model_id,
        prompt_sha256=hashlib.sha256(request.prompt.encode()).hexdigest(),
        decoding=dict(decoding or {"do_sample": False}),
        raw_output=raw_output,
        accepted=reason is None,
        reject_reason=reason,
        fallback_used=fallback_used,
    )
