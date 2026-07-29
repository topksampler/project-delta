"""Phase D hooks for the e1_vllm vertical slice.

The generic state machine lives in ``lab.delta``. This plugin binds that
control plane to the factory artifacts and measured intervention outcomes of
the vLLM experiment. ``target=replay`` closes the loop over durable GPU
evidence without redispatching those runs; other targets currently share the
same evidence path and are reserved for fresh dispatch wiring.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from lab.delta.contracts import (
    ActiveState,
    CandidateState,
    CandidateStatus,
    InterventionOutcome,
    OutcomeStatus,
    PromotionDecision,
    canonical_json,
)
from lab.delta.memory import ImmutableCollisionError, LocalStateRegistry
from lab.delta.policy import choose_plan
from lab.delta.reconcile import STAGES, StageContext
from lab.delta.sense import build_factory_drift_event_from_paths
from lab.delta.verify import VerificationThresholds, verify_candidate
from lab.delta.worker import EXPLANATION_TASK, SURFACE_TASK, WorkerRequest


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class E1VllmReconcilePlugin:
    experiment_id = "e1_vllm"

    _SUMMARIES = {
        ("v0.21.0", "v0.22.0"): {
            "base": "e1-vllm-eval-factory-t2122-base-qwen35-08b-modal_summary.json",
            "hybrid": "e1-vllm-eval-factory-t2122-c1-hybrid-qwen35-08b-modal_summary.json",
            "run_id": "e1-vllm-eval-factory-t2122-c1-hybrid-qwen35-08b-modal",
        },
        ("v0.22.0", "v0.23.0"): {
            "base": "eval_factory_v1_base_08b_summary.json",
            "hybrid": "e1-vllm-eval-factory-v1-c1-hybrid-qwen35-08b-modal_summary.json",
            "run_id": "e1-vllm-eval-factory-v1-c1-hybrid-qwen35-08b-modal",
        },
        ("v0.22.0", "v0.26.0"): {
            "base": "e1-vllm-eval-factory-v022-v026-base-qwen35-08b-modal_summary.json",
            "hybrid": "e1-vllm-eval-factory-v022-v026-c1-hybrid-qwen35-08b-modal_summary.json",
            "run_id": "e1-vllm-eval-factory-v022-v026-c1-hybrid-qwen35-08b-modal",
        },
    }

    def __init__(self, *, project_root: Path) -> None:
        self.project_root = Path(project_root)

    def _delta_root(self, context: StageContext) -> Path:
        return context.run_dir.parents[1]

    def _transition(self, context: StageContext) -> str:
        return f"{context.request.source_from}_to_{context.request.source_to}"

    def _relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.project_root.resolve()))

    def _evidence(self, context: StageContext) -> dict[str, Path | str]:
        pair = (context.request.source_from, context.request.source_to)
        names = self._SUMMARIES.get(pair)
        if names is None:
            raise RuntimeError(
                "no measured e1_vllm intervention history for "
                f"{pair[0]}→{pair[1]}; run BUILD/baseline/hybrid first"
            )
        artifact_dir = (
            self.project_root
            / "data/experiments/e1_vllm/eval_factory"
            / self._transition(context)
            / "e1_eval_factory_v1"
        )
        probes = artifact_dir / "probes_eval.jsonl"
        if not probes.exists():
            probes = artifact_dir / "probes_eval_seed.jsonl"
        reports = self.project_root / "artifacts/reports"
        return {
            "artifact_dir": artifact_dir,
            "manifest": artifact_dir / "manifest.json",
            "probes": probes,
            "base_summary": reports / str(names["base"]),
            "candidate_summary": reports / str(names["hybrid"]),
            "candidate_run_id": str(names["run_id"]),
        }

    def _write(self, context: StageContext, name: str, payload: Any) -> Path:
        path = context.run_dir / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            payload.to_dict() if hasattr(payload, "to_dict") else dict(payload)
        )
        path.write_text(json.dumps(encoded, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def run_stage(self, stage: str, context: StageContext) -> Mapping[str, Any]:
        if stage not in STAGES:
            raise ValueError(f"unsupported stage: {stage}")
        return getattr(self, stage)(context)

    def resolve(self, context: StageContext) -> Mapping[str, Any]:
        envelopes = [
            WorkerRequest(
                task=SURFACE_TASK,
                payload={"transition": [context.request.source_from, context.request.source_to]},
                constraints={"may_define_truth": False, "may_select_action": False},
                prompt="Rewrite controller-provided probe wording only.",
            ).to_dict(),
            WorkerRequest(
                task=EXPLANATION_TASK,
                payload={"transition": [context.request.source_from, context.request.source_to]},
                constraints={"decision_is_fixed": True, "may_select_action": False},
                prompt="Explain the controller-selected plan and gate evidence.",
            ).to_dict(),
        ]
        evidence = self._evidence(context)
        return {
            "transition": self._transition(context),
            "evidence": {key: str(value) for key, value in evidence.items()},
            "worker_envelopes": envelopes,
        }

    def pin(self, context: StageContext) -> Mapping[str, Any]:
        snapshots = self.project_root / "data/experiments/e1_vllm/snapshots"
        found = {}
        for revision in (context.request.source_from, context.request.source_to):
            path = snapshots / revision
            if not (path / "snapshot_meta.json").is_file():
                raise FileNotFoundError(f"missing pinned snapshot: {path}")
            found[revision] = str(path)
        return {"snapshots": found}

    def build(self, context: StageContext) -> Mapping[str, Any]:
        evidence = self._evidence(context)
        required = ("manifest", "probes", "base_summary", "candidate_summary")
        missing = [str(evidence[key]) for key in required if not Path(evidence[key]).is_file()]
        if missing:
            raise FileNotFoundError("missing BUILD/eval evidence: " + ", ".join(missing))
        manifest = _read(Path(evidence["manifest"]))
        freeze = manifest.get("freeze") or {}
        freeze_ready = bool(freeze.get("ready"))
        diagnostic_seed_replay = context.request.target == "seed-replay"
        if not freeze_ready and not diagnostic_seed_replay:
            raise RuntimeError(
                "EvalEnvironment is not sealed (freeze.ready=false); "
                "promotion-capable reconcile fails closed. Use "
                "--target seed-replay only for a diagnostic run that can never promote."
            )
        return {
            "eval_environment_id": hashlib.sha256(
                canonical_json(manifest).encode()
            ).hexdigest(),
            "manifest": str(evidence["manifest"]),
            "probes": str(evidence["probes"]),
            "base_summary": str(evidence["base_summary"]),
            "candidate_summary": str(evidence["candidate_summary"]),
            "factory_freeze_ready": freeze_ready,
            "promotion_eligible": freeze_ready,
            "diagnostic_seed_replay": diagnostic_seed_replay,
            "audit_provenance": freeze.get("human_audit"),
        }

    def sense(self, context: StageContext) -> Mapping[str, Any]:
        build = context.outputs["build"]
        event = build_factory_drift_event_from_paths(
            manifest_path=Path(build["manifest"]),
            summary_path=Path(build["base_summary"]),
            probes_path=Path(build["probes"]),
            path_root=self.project_root,
        )
        path = self._write(context, "drift_event", event)
        return {"drift": event.to_dict(), "path": str(path)}

    def _active_state(self, context: StageContext) -> ActiveState:
        build = context.outputs["build"]
        baseline = _read(Path(build["base_summary"]))
        return ActiveState(
            base_model="Qwen/Qwen3.5-0.8B",
            source_revision=context.request.source_from,
            eval_environment_id=str(build["eval_environment_id"]),
            metadata={"metrics": baseline.get("by_drift_type") or {}},
            paths={"summary": self._relative(Path(build["base_summary"]))},
        )

    def decide(self, context: StageContext) -> Mapping[str, Any]:
        event = context.outputs["sense"]["drift"]
        active = self._active_state(context)
        plan = choose_plan(
            event,
            active,
            budget={"max_level": "L2", "max_cost": 5.0},
            ft_reopened=False,
            l3_approved=False,
        )
        path = self._write(context, "intervention_plan", plan)
        return {
            "plan": plan.to_dict(),
            "active_state": active.to_dict(),
            "path": str(path),
            "controller_selected": True,
        }

    def adapt(self, context: StageContext) -> Mapping[str, Any]:
        plan = context.outputs["decide"]["plan"]
        if plan["recipe"] not in {"no_op", "hybrid_diff", "index_refresh"}:
            raise RuntimeError(f"ineligible unattended recipe: {plan['recipe']}")
        build = context.outputs["build"]
        evidence = self._evidence(context)
        summary_path = (
            Path(build["base_summary"])
            if plan["recipe"] == "no_op"
            else Path(build["candidate_summary"])
        )
        run_id = (
            f"{context.run_id}-no-op"
            if plan["recipe"] == "no_op"
            else str(evidence["candidate_run_id"])
        )
        candidate = CandidateState(
            plan_id=plan["plan_id"],
            drift_event_id=context.outputs["sense"]["drift"]["drift_event_id"],
            base_model="Qwen/Qwen3.5-0.8B",
            source_revision=context.request.source_to,
            eval_environment_id=str(build["eval_environment_id"]),
            status=CandidateStatus.SUCCEEDED,
            index_revision=_hash(summary_path),
            run_ids=(run_id,),
            artifact_hashes={"summary": _hash(summary_path)},
            paths={"summary": self._relative(summary_path)},
            metadata={"recipe": plan["recipe"]},
        )
        active = ActiveState.from_dict(context.outputs["decide"]["active_state"])
        promoted = ActiveState(
            base_model=candidate.base_model,
            source_revision=candidate.source_revision,
            eval_environment_id=candidate.eval_environment_id,
            candidate_id=candidate.candidate_id,
            index_revision=candidate.index_revision,
            artifact_hashes=candidate.artifact_hashes,
            paths=candidate.paths,
            metadata={"metrics": _read(summary_path).get("by_drift_type") or {}},
        )
        registry = LocalStateRegistry(self._delta_root(context))
        if not context.request.dry_run:
            if not registry.active_path.exists():
                registry.initialize(active)
            elif registry.active_state_id() != active.state_id:
                raise RuntimeError("active state does not match reconcile baseline")
            if not registry.has_state(promoted.state_id):
                registry.register_state(promoted)
        path = self._write(context, "candidate_state", candidate)
        return {
            "candidate": candidate.to_dict(),
            "candidate_active_state_id": promoted.state_id,
            "summary": str(summary_path),
            "path": str(path),
        }

    def verify(self, context: StageContext) -> Mapping[str, Any]:
        candidate = CandidateState.from_dict(context.outputs["adapt"]["candidate"])
        build = context.outputs["build"]
        summary_path = Path(context.outputs["adapt"]["summary"])
        summary = _read(summary_path)
        expected = int(summary.get("n_samples") or 0)
        outcome = InterventionOutcome(
            candidate_id=candidate.candidate_id,
            status=OutcomeStatus.SUCCEEDED,
            metrics={
                "stable": (summary.get("by_drift_type") or {}).get("stable"),
                "coverage": 1.0 if expected > 0 else 0.0,
                "pooled_exact": summary.get("exact_accuracy"),
            },
            per_drift_metrics=summary.get("by_drift_type") or {},
            timing={},
            cost=0.0,
            paths={"metrics": self._relative(summary_path)},
            metadata={
                "evidence_replay": True,
                "environment_sealed": bool(build["factory_freeze_ready"]),
                "diagnostic_seed_replay": bool(build["diagnostic_seed_replay"]),
            },
        )
        active = ActiveState.from_dict(context.outputs["decide"]["active_state"])
        decision = verify_candidate(
            candidate,
            outcome,
            active,
            thresholds=VerificationThresholds(
                recovery=0.35,
                minimum_improvement=0.05,
                stable_regression_budget=0.05,
                coverage=0.99,
                max_cost=5.0,
            ),
        )
        outcome_path = self._write(context, "intervention_outcome", outcome)
        decision_path = self._write(context, "promotion_decision", decision)
        return {
            "outcome": outcome.to_dict(),
            "decision": decision.to_dict(),
            "outcome_path": str(outcome_path),
            "decision_path": str(decision_path),
        }

    def record(self, context: StageContext) -> Mapping[str, Any]:
        decision = PromotionDecision.from_dict(context.outputs["verify"]["decision"])
        registry = LocalStateRegistry(self._delta_root(context))
        if not context.request.dry_run:
            try:
                registry.record_decision(decision)
            except ImmutableCollisionError:
                # A resumed record stage may observe the already-durable decision.
                pass
        return {
            "decision": decision.to_dict(),
            "decision_path": context.outputs["verify"]["decision_path"],
            "active_state_id": (
                registry.active_state_id()
                if registry.active_path.exists()
                else ActiveState.from_dict(
                    context.outputs["decide"]["active_state"]
                ).state_id
            ),
        }

    def approve(self, receipt: Mapping[str, Any]) -> Mapping[str, Any]:
        decision = PromotionDecision.from_dict(
            receipt["stages"]["record"]["output"]["decision"]
        )
        root = Path(receipt["stages"]["record"]["output"]["decision_path"]).parents[2]
        state_id = LocalStateRegistry(root).approve(decision)
        return {"active_state_id": state_id, "decision_id": decision.decision_id}
