from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from experiments.delta_v2.baseline_protocol import (
    BaselineProtocolError,
    load_yaml,
)
from experiments.delta_v2.run_task_calibration import (
    EXPECTED_RUN_ID,
    execute_task_calibration,
    validate_only,
    validate_run_config,
)
from experiments.delta_v2.task_calibration import (
    FACT_LABEL_ORDER,
    PROTOCOL_ID,
    audit_request_context,
    build_task_calibration_requests,
    validate_task_calibration_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = Path(__file__).with_name("task_calibration_protocol.yaml")
CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/delta_v2/"
    "c1_task_calibration_qwen35_08b_modal_v1.yaml"
)


def correct_response(item: MappingLike) -> str:
    if item["scorer"] == "exact-enum-v1":
        return str(item["gold"]["status"])
    return json.dumps(item["gold"], sort_keys=True, separators=(",", ":"))


MappingLike = dict


class FixtureBackend:
    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.requests: list[dict] = []

    def generate(self, request: dict) -> tuple[str, str]:
        self.requests.append(copy.deepcopy(request))
        return self.answers[request["eval_id"]], "stop"

    def runtime_receipt(self) -> dict:
        return {
            "backend": "truth-fixture",
            "model_invocations_completed": 0,
        }


class TaskCalibrationProtocolTest(unittest.TestCase):
    def test_context_uses_balanced_development_train_examples_only(self) -> None:
        protocol = load_yaml(PROTOCOL_PATH)

        _, _, fact_examples, feature_example = (
            validate_task_calibration_protocol(
                protocol,
                repo_root=REPO_ROOT,
            )
        )

        self.assertTrue(
            all(item["split"] == "train" for item in fact_examples)
        )
        self.assertEqual(feature_example["split"], "train")
        self.assertEqual(
            Counter(item["gold"]["status"] for item in fact_examples),
            Counter({label: 2 for label in FACT_LABEL_ORDER}),
        )

    def test_request_audit_keeps_acceptance_truth_out(self) -> None:
        protocol = load_yaml(PROTOCOL_PATH)
        baseline, items, facts, feature = (
            validate_task_calibration_protocol(
                protocol,
                repo_root=REPO_ROOT,
            )
        )
        requests = build_task_calibration_requests(
            protocol,
            baseline_protocol=baseline,
            acceptance_items=items,
            fact_examples=facts,
            feature_example=feature,
        )

        audit = audit_request_context(
            requests=requests,
            acceptance_items=items,
            fact_examples=facts,
            feature_example=feature,
        )

        self.assertEqual(len(requests), 86)
        self.assertFalse(audit["acceptance_gold_visible"])
        self.assertFalse(audit["acceptance_provenance_visible"])
        self.assertFalse(audit["development_dev_items_visible"])
        self.assertEqual(
            audit["development_fact_label_counts"],
            {label: 2 for label in sorted(FACT_LABEL_ORDER)},
        )
        self.assertTrue(
            all(
                request["model_input"]["messages"][-1]["content"]
                == next(
                    item["prompt"]
                    for item in items
                    if item["eval_id"] == request["eval_id"]
                )
                for request in requests
            )
        )

    def test_protocol_drift_fails_closed(self) -> None:
        protocol = copy.deepcopy(dict(load_yaml(PROTOCOL_PATH)))
        protocol["adaptation"]["weight_updates"] = True

        with self.assertRaisesRegex(
            BaselineProtocolError,
            "method changed",
        ):
            validate_task_calibration_protocol(
                protocol,
                repo_root=REPO_ROOT,
            )


class TaskCalibrationRunTest(unittest.TestCase):
    def setUp(self) -> None:
        config = load_yaml(CONFIG_PATH)
        _, _, items, _, _ = validate_run_config(
            config,
            repo_root=REPO_ROOT,
        )
        self.answers = {
            str(item["eval_id"]): correct_response(dict(item))
            for item in items
        }

    def test_config_and_request_bytes_are_frozen(self) -> None:
        result = validate_only(
            config_path=CONFIG_PATH,
            repo_root=REPO_ROOT,
        )

        self.assertEqual(result["run_id"], EXPECTED_RUN_ID)
        self.assertEqual(result["protocol_id"], PROTOCOL_ID)
        self.assertEqual(result["requests"], 86)
        self.assertEqual(
            result["request_sha256"],
            "96e209a30ca9b723f8e28e2af9c233f2a48b277c913bbb5da5bc4e7019ffbe77",
        )
        self.assertEqual(result["status"], "valid-unexecuted")

    def test_fixture_execution_preserves_original_scoring(self) -> None:
        backend = FixtureBackend(self.answers)
        with tempfile.TemporaryDirectory() as tmp:
            result = execute_task_calibration(
                config_path=CONFIG_PATH,
                repo_root=REPO_ROOT,
                backend=backend,
                output_dir_override=Path(tmp) / "run",
            )

        self.assertEqual(len(backend.requests), 86)
        self.assertEqual(
            result["audit"]["interpretation"]["state"],
            "no-observed-acceptance-gap",
        )
        self.assertTrue(
            result["audit"]["adaptation"]["request_audit"]["status"]
            == "pass"
        )
        self.assertFalse(result["audit"]["adaptation"]["weight_updates"])

    def test_config_cannot_switch_to_acceptance_context(self) -> None:
        config = copy.deepcopy(dict(load_yaml(CONFIG_PATH)))
        config["data"]["path"] = config["eval"]["path"]

        with self.assertRaisesRegex(
            BaselineProtocolError,
            "development data changed",
        ):
            validate_run_config(config, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
