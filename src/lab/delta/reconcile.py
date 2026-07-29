"""Durable, resumable DELTA reconciliation stage machine."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

STAGES = ("resolve", "pin", "build", "sense", "decide", "adapt", "verify", "record")
RECEIPT_SCHEMA = "delta.reconcile.receipt.v1"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class ReconcileRequest:
    repo: str
    source_from: str
    source_to: str
    target: str = "modal"
    dry_run: bool = False

    @property
    def run_id(self) -> str:
        digest = hashlib.sha256(_canonical(asdict(self)).encode()).hexdigest()[:16]
        return f"reconcile-{digest}"


@dataclass(frozen=True)
class StageContext:
    request: ReconcileRequest
    run_id: str
    run_dir: Path
    config_dir: Path
    outputs: Mapping[str, Any]
    approved: bool
    runner: "CommandRunner"


class ReconcilePlugin(Protocol):
    experiment_id: str

    def run_stage(self, stage: str, context: StageContext) -> Mapping[str, Any] | None:
        ...


class CommandRunner:
    """Injectable subprocess boundary used by real plugins and fake tests."""

    def __init__(
        self,
        invoke: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self._invoke = invoke

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        completed = self._invoke(
            list(command),
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env else None,
            check=True,
            capture_output=True,
            text=True,
        )
        return {
            "command": list(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }


class ReconcileError(RuntimeError):
    pass


class ApprovalRequired(ReconcileError):
    pass


class Reconciler:
    def __init__(
        self,
        *,
        artifact_root: Path,
        plugin: ReconcilePlugin,
        runner: CommandRunner | None = None,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.plugin = plugin
        self.runner = runner or CommandRunner()

    def receipt_path(self, run_id: str) -> Path:
        return self.artifact_root / "reconciles" / run_id / "receipt.json"

    def load(self, run_id: str) -> dict[str, Any]:
        path = self.receipt_path(run_id)
        if not path.is_file():
            raise ReconcileError(f"unknown reconcile run: {run_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _new_receipt(self, request: ReconcileRequest) -> dict[str, Any]:
        return {
            "schema": RECEIPT_SCHEMA,
            "run_id": request.run_id,
            "experiment_id": self.plugin.experiment_id,
            "request": asdict(request),
            "status": "running",
            "approved": False,
            "created_at": _now(),
            "updated_at": _now(),
            "stages": {
                stage: {"status": "pending", "attempts": 0}
                for stage in STAGES
            },
        }

    def _save(self, receipt: dict[str, Any]) -> None:
        receipt["updated_at"] = _now()
        _atomic_json(self.receipt_path(receipt["run_id"]), receipt)

    def approve(self, run_id: str) -> dict[str, Any]:
        receipt = self.load(run_id)
        if receipt["status"] != "pending_approval":
            raise ReconcileError("only a pending promotion decision can be approved")
        hook = getattr(self.plugin, "approve", None)
        if hook is not None:
            approval_output = hook(receipt)
        else:
            record = receipt["stages"]["record"].get("output") or {}
            decision = record.get("decision")
            if not isinstance(decision, Mapping):
                raise ReconcileError("record stage did not emit a typed decision")
            from .contracts import PromotionDecision
            from .memory import LocalStateRegistry

            state_id = LocalStateRegistry(self.artifact_root).approve(
                PromotionDecision.from_dict(decision)
            )
            approval_output = {"active_state_id": state_id}
        receipt["approved"] = True
        receipt["approved_at"] = _now()
        receipt["status"] = "approved"
        if approval_output is not None:
            receipt["approval"] = dict(approval_output)
        self._save(receipt)
        return receipt

    def reconcile(
        self,
        request: ReconcileRequest,
        *,
        resume: bool = False,
        through: str | None = None,
    ) -> dict[str, Any]:
        if through is not None and through not in STAGES:
            raise ReconcileError(
                f"unknown stage {through!r}; expected one of {', '.join(STAGES)}"
            )
        path = self.receipt_path(request.run_id)
        if path.exists():
            if not resume:
                raise ReconcileError(
                    f"run {request.run_id} already exists; pass --resume"
                )
            receipt = self.load(request.run_id)
            if receipt["request"] != asdict(request):
                raise ReconcileError("resume request does not match receipt")
        else:
            receipt = self._new_receipt(request)
            self._save(receipt)

        run_dir = path.parent
        stop_index = STAGES.index(through) if through else len(STAGES) - 1
        for index, stage in enumerate(STAGES):
            if index > stop_index:
                break
            checkpoint = receipt["stages"][stage]
            if checkpoint["status"] == "completed":
                continue
            checkpoint["status"] = "running"
            checkpoint["attempts"] += 1
            checkpoint["started_at"] = _now()
            checkpoint.pop("error", None)
            receipt["status"] = "running"
            self._save(receipt)

            outputs = {
                name: state.get("output")
                for name, state in receipt["stages"].items()
                if state["status"] == "completed"
            }
            context = StageContext(
                request=request,
                run_id=request.run_id,
                run_dir=run_dir,
                config_dir=run_dir / "configs",
                outputs=outputs,
                approved=bool(receipt["approved"]),
                runner=self.runner,
            )
            try:
                output = self.plugin.run_stage(stage, context)
            except Exception as exc:
                checkpoint["status"] = "failed"
                checkpoint["failed_at"] = _now()
                checkpoint["error"] = f"{type(exc).__name__}: {exc}"
                receipt["status"] = "failed"
                self._save(receipt)
                raise

            checkpoint["status"] = "completed"
            checkpoint["completed_at"] = _now()
            checkpoint["output"] = dict(output or {})
            self._save(receipt)

        last_completed = all(
            receipt["stages"][stage]["status"] == "completed"
            for stage in STAGES
        )
        record = receipt["stages"]["record"].get("output") or {}
        decision = record.get("decision")
        if isinstance(decision, Mapping):
            decision_status = decision.get("status")
        else:
            decision_status = record.get("status", decision)
        if last_completed and decision_status in {"pending_approval", "rejected"}:
            receipt["status"] = decision_status
        else:
            receipt["status"] = "completed" if last_completed else "paused"
        self._save(receipt)
        return receipt
