from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

from experiments.e1_vllm.reconcile_plugin import E1VllmReconcilePlugin
from lab.delta.reconcile import (
    CommandRunner,
    ReconcileRequest,
    Reconciler,
    StageContext,
)
from lab.delta.registry import UnsupportedRepositoryError, plugin_for_repo


class FakePlugin:
    experiment_id = "fake"

    def __init__(self, *, reject: bool = False, fail_stage_once: str | None = None):
        self.reject = reject
        self.fail_stage_once = fail_stage_once
        self.calls: Counter[str] = Counter()

    def run_stage(self, stage, context):
        self.calls[stage] += 1
        if stage == self.fail_stage_once and self.calls[stage] == 1:
            raise RuntimeError("injected stage failure")
        if stage == "decide":
            return {"plan": {"action": "index_refresh"}}
        if stage == "verify":
            return {"accepted": not self.reject}
        if stage == "record":
            return {
                "decision": {
                    "status": "rejected" if self.reject else "pending_approval"
                },
                "verification": context.outputs["verify"],
            }
        return {"stage": stage}

    def approve(self, receipt):
        self.calls["approve"] += 1
        return {"active_state_id": "state-new"}


def request(*, dry_run: bool = False, target: str = "modal") -> ReconcileRequest:
    return ReconcileRequest(
        repo="https://github.com/vllm-project/vllm.git",
        source_from="v0.22.0",
        source_to="v0.23.0",
        target=target,
        dry_run=dry_run,
    )


class ReconcileTest(unittest.TestCase):
    def test_happy_path_stops_pending_explicit_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = FakePlugin()
            reconciler = Reconciler(artifact_root=Path(tmp), plugin=plugin)
            receipt = reconciler.reconcile(request())

            self.assertEqual(receipt["status"], "pending_approval")
            self.assertEqual(receipt["stages"]["decide"]["status"], "completed")
            self.assertEqual(receipt["stages"]["adapt"]["status"], "completed")
            self.assertFalse(receipt["approved"])
            self.assertEqual(plugin.calls["adapt"], 1)

            approved = reconciler.approve(receipt["run_id"])
            self.assertEqual(approved["status"], "approved")
            self.assertTrue(approved["approved"])
            self.assertEqual(plugin.calls["approve"], 1)

    def test_rejected_candidate_is_recorded_without_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = FakePlugin(reject=True)
            reconciler = Reconciler(artifact_root=Path(tmp), plugin=plugin)
            receipt = reconciler.reconcile(request())

            self.assertEqual(receipt["status"], "rejected")
            self.assertFalse(receipt["stages"]["verify"]["output"]["accepted"])
            self.assertEqual(
                receipt["stages"]["record"]["output"]["decision"]["status"], "rejected"
            )

    def test_failure_is_resumable_without_repeating_completed_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = FakePlugin(fail_stage_once="adapt")
            reconciler = Reconciler(artifact_root=Path(tmp), plugin=plugin)

            with self.assertRaisesRegex(RuntimeError, "injected"):
                reconciler.reconcile(request())
            failed = reconciler.load(request().run_id)
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["stages"]["adapt"]["attempts"], 1)

            completed = reconciler.reconcile(request(), resume=True)
            self.assertEqual(completed["status"], "pending_approval")
            for stage in ("resolve", "pin", "build", "sense", "decide"):
                self.assertEqual(plugin.calls[stage], 1)
            self.assertEqual(plugin.calls["adapt"], 2)
            self.assertEqual(plugin.calls["verify"], 1)
            self.assertEqual(plugin.calls["record"], 1)

    def test_dry_run_mutates_only_reconcile_area(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invoke = Mock(side_effect=AssertionError("dry-run dispatched a child"))
            plugin = plugin_for_repo(
                request().repo,
                project_root=Path(__file__).resolve().parents[1],
            )
            receipt = Reconciler(
                artifact_root=root,
                plugin=plugin,
                runner=CommandRunner(invoke),
            ).reconcile(request(dry_run=True))

            files = [path.relative_to(root) for path in root.rglob("*") if path.is_file()]
            expected_prefix = Path("reconciles") / receipt["run_id"]
            self.assertTrue(files)
            self.assertTrue(all(expected_prefix in path.parents for path in files))
            invoke.assert_not_called()

    def test_unsupported_repository_has_clear_error(self) -> None:
        with self.assertRaisesRegex(
            UnsupportedRepositoryError, "unsupported DELTA repository"
        ):
            plugin_for_repo("https://github.com/example/not-supported")

    def test_unsealed_environment_requires_diagnostic_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = (
                root
                / "data/experiments/e1_vllm/eval_factory"
                / "v0.22.0_to_v0.26.0/e1_eval_factory_v1"
            )
            reports = root / "artifacts/reports"
            artifact_dir.mkdir(parents=True)
            reports.mkdir(parents=True)
            (artifact_dir / "manifest.json").write_text(
                json.dumps({"freeze": {"ready": False}}),
                encoding="utf-8",
            )
            (artifact_dir / "probes_eval_seed.jsonl").write_text(
                "{}\n",
                encoding="utf-8",
            )
            for name in (
                "e1-vllm-eval-factory-v022-v026-base-qwen35-08b-modal_summary.json",
                "e1-vllm-eval-factory-v022-v026-c1-hybrid-qwen35-08b-modal_summary.json",
            ):
                (reports / name).write_text("{}\n", encoding="utf-8")

            plugin = E1VllmReconcilePlugin(project_root=root)

            def context(target: str) -> StageContext:
                req = ReconcileRequest(
                    repo=request().repo,
                    source_from="v0.22.0",
                    source_to="v0.26.0",
                    target=target,
                )
                return StageContext(
                    request=req,
                    run_id=req.run_id,
                    run_dir=root / "artifacts/delta/reconciles" / req.run_id,
                    config_dir=root / "artifacts/delta/configs",
                    outputs={},
                    approved=False,
                    runner=CommandRunner(),
                )

            with self.assertRaisesRegex(RuntimeError, "not sealed"):
                plugin.build(context("replay"))
            diagnostic = plugin.build(context("seed-replay"))
            self.assertFalse(diagnostic["promotion_eligible"])
            self.assertTrue(diagnostic["diagnostic_seed_replay"])


if __name__ == "__main__":
    unittest.main()
