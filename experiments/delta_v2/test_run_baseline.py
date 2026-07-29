from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.delta_v2.baseline_protocol import BaselineProtocolError
from experiments.delta_v2.run_baseline import (
    EXPECTED_RUN_ID,
    execute_baseline,
    validate_only,
    validate_run_config,
)
from experiments.delta_v2.baseline_protocol import load_yaml
from lab.dispatch import modal_driver


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/delta_v2/c0_base_qwen35_08b_modal_v3.yaml"
)


def correct_response(item: dict) -> str:
    if item["scorer"] == "exact-enum-v1":
        return str(item["gold"]["status"])
    return json.dumps(item["gold"], sort_keys=True, separators=(",", ":"))


class FixtureBackend:
    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.requests: list[dict] = []

    def generate(self, request: dict) -> tuple[str, str]:
        serialized = json.dumps(request, sort_keys=True)
        for forbidden in (
            '"gold"',
            '"provenance"',
            '"scorer"',
            '"behavior_probe_result"',
            '"judge_evidence"',
        ):
            if forbidden in serialized:
                raise AssertionError(f"model request leaked {forbidden}")
        self.requests.append(copy.deepcopy(request))
        return self.answers[request["eval_id"]], "stop"

    def runtime_receipt(self) -> dict:
        return {
            "backend": "truth-fixture",
            "model_invocations_completed": 0,
        }


class NondeterministicBackend(FixtureBackend):
    def generate(self, request: dict) -> tuple[str, str]:
        response, finish_reason = super().generate(request)
        if request["repeat_index"] == 2:
            response += " "
        return response, finish_reason


class RunConfigTest(unittest.TestCase):
    def test_real_config_binds_one_frozen_baseline(self) -> None:
        config = load_yaml(CONFIG_PATH)

        protocol, items = validate_run_config(
            config,
            repo_root=REPO_ROOT,
        )

        self.assertEqual(config["run_id"], EXPECTED_RUN_ID)
        self.assertEqual(protocol["protocol_id"], config["protocol"]["protocol_id"])
        self.assertEqual(len(items), 43)

    def test_validate_only_builds_86_requests_without_model_runtime(self) -> None:
        result = validate_only(
            config_path=CONFIG_PATH,
            repo_root=REPO_ROOT,
        )

        self.assertEqual(result["status"], "valid-unexecuted")
        self.assertEqual(result["eval_items"], 43)
        self.assertEqual(result["requests"], 86)
        self.assertEqual(result["model_invocations_completed"], 0)
        self.assertFalse(result["target_model_results_exist"])

    def test_config_drift_fails_closed(self) -> None:
        config = copy.deepcopy(dict(load_yaml(CONFIG_PATH)))
        config["model"]["revision"] = "different"

        with self.assertRaisesRegex(
            BaselineProtocolError,
            "model binding changed",
        ):
            validate_run_config(config, repo_root=REPO_ROOT)


class RunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        config = load_yaml(CONFIG_PATH)
        _, items = validate_run_config(config, repo_root=REPO_ROOT)
        self.answers = {
            str(item["eval_id"]): correct_response(dict(item))
            for item in items
        }

    def test_fixture_run_captures_raw_outputs_and_stratified_audit(self) -> None:
        backend = FixtureBackend(self.answers)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"

            result = execute_baseline(
                config_path=CONFIG_PATH,
                repo_root=REPO_ROOT,
                backend=backend,
                output_dir_override=output_dir,
            )

            self.assertEqual(len(backend.requests), 86)
            self.assertEqual(result["receipt"]["model_outputs"], 86)
            self.assertEqual(
                result["audit"]["interpretation"]["state"],
                "no-observed-acceptance-gap",
            )
            self.assertIsNone(result["audit"]["pooled_overall_accuracy"])
            samples = [
                json.loads(line)
                for line in (output_dir / "samples.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(samples), 86)
            self.assertEqual(
                set(samples[0]),
                {
                    "schema",
                    "protocol_id",
                    "request_id",
                    "eval_id",
                    "repeat_index",
                    "model_revision",
                    "raw_response",
                    "finish_reason",
                },
            )
            self.assertTrue((output_dir / "metrics.json").is_file())
            self.assertTrue((output_dir / "run_receipt.json").is_file())

    def test_repeat_disagreement_writes_no_claimed_result(self) -> None:
        backend = NondeterministicBackend(self.answers)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"

            with self.assertRaisesRegex(
                BaselineProtocolError,
                "nondeterministic target response",
            ):
                execute_baseline(
                    config_path=CONFIG_PATH,
                    repo_root=REPO_ROOT,
                    backend=backend,
                    output_dir_override=output_dir,
                )

            self.assertFalse(output_dir.exists())


class ModalRoutingTest(unittest.TestCase):
    def test_dispatch_uses_experiment_owned_modal_app(self) -> None:
        with mock.patch("lab.dispatch.modal_driver.subprocess.run") as run:
            modal_driver.dispatch(
                repo_root=REPO_ROOT,
                task="eval",
                config_path=CONFIG_PATH,
                run_id=EXPECTED_RUN_ID,
                gpu="A10G",
            )

        command = run.call_args.args[0]
        self.assertIn(
            str(REPO_ROOT / "experiments/delta_v2/modal_baseline_app.py"),
            command,
        )
        self.assertEqual(command[command.index("--task") + 1], "eval")
        self.assertEqual(command[command.index("--gpu") + 1], "A10G")


if __name__ == "__main__":
    unittest.main()
