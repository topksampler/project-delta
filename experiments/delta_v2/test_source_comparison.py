from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments.delta_v2.baseline_protocol import (
    BaselineProtocolError,
    load_yaml,
)
from experiments.delta_v2.run_source_comparison import (
    EXPECTED_REQUEST_SHA256,
    EXPECTED_RUN_ID,
    execute_source_comparison,
    validate_only,
    validate_run_config,
)
from experiments.delta_v2.source_comparison import (
    PROTOCOL_ID,
    audit_source_comparison_requests,
    build_source_comparison_requests,
    validate_source_comparison_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = Path(__file__).with_name(
    "source_comparison_protocol.yaml"
)
CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/delta_v2/"
    "c3_source_comparison_qwen35_08b_modal_v1.yaml"
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
        self.requests.append(copy.deepcopy(request))
        return self.answers[request["eval_id"]], "stop"

    def runtime_receipt(self) -> dict:
        return {
            "backend": "truth-fixture",
            "model_invocations_completed": 0,
        }


class SourceComparisonProtocolTest(unittest.TestCase):
    def test_fact_projection_is_three_booleans_only(self) -> None:
        protocol = load_yaml(PROTOCOL_PATH)

        baseline, items, evidence, feature = (
            validate_source_comparison_protocol(
                protocol,
                repo_root=REPO_ROOT,
            )
        )
        requests = build_source_comparison_requests(
            protocol,
            baseline_protocol=baseline,
            items=items,
            evidence_by_eval=evidence,
            feature_evidence=feature,
        )
        audit = audit_source_comparison_requests(requests)

        self.assertEqual(len(evidence), 42)
        self.assertTrue(
            all(
                set(row)
                == {
                    "before_present",
                    "after_present",
                    "canonical_equal",
                }
                for row in evidence.values()
            )
        )
        self.assertEqual(len(requests), 86)
        self.assertFalse(audit["canonical_hashes_visible"])
        self.assertFalse(audit["gold_label_visible"])
        self.assertFalse(audit["status_field_visible"])

    def test_projection_still_determines_frozen_fact_gold(self) -> None:
        protocol = load_yaml(PROTOCOL_PATH)
        _, items, evidence, _ = validate_source_comparison_protocol(
            protocol,
            repo_root=REPO_ROOT,
        )

        for item in items:
            if item["source_kind"] != "atomic_fact_delta":
                continue
            row = evidence[item["eval_id"]]
            if not row["before_present"] and row["after_present"]:
                observed = "added"
            elif row["before_present"] and not row["after_present"]:
                observed = "removed"
            elif row["canonical_equal"]:
                observed = "stable"
            else:
                observed = "changed"
            self.assertEqual(observed, item["gold"]["status"])

    def test_representation_cannot_add_gold(self) -> None:
        protocol = copy.deepcopy(dict(load_yaml(PROTOCOL_PATH)))
        protocol["representation"]["fact_projection"].append("gold")

        with self.assertRaisesRegex(
            BaselineProtocolError,
            "representation changed",
        ):
            validate_source_comparison_protocol(
                protocol,
                repo_root=REPO_ROOT,
            )


class SourceComparisonRunTest(unittest.TestCase):
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
        self.assertEqual(result["request_sha256"], EXPECTED_REQUEST_SHA256)
        self.assertEqual(result["requests"], 86)
        self.assertEqual(result["status"], "valid-unexecuted")

    def test_fixture_execution_preserves_upper_bound_limit(self) -> None:
        backend = FixtureBackend(self.answers)
        with tempfile.TemporaryDirectory() as tmp:
            result = execute_source_comparison(
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
        self.assertFalse(
            result["audit"]["adaptation"]["production_retrieval_claim"]
        )
        self.assertFalse(result["audit"]["adaptation"]["weight_updates"])

    def test_config_cannot_replace_truth_source(self) -> None:
        config = copy.deepcopy(dict(load_yaml(CONFIG_PATH)))
        config["truth"]["feature_probe_path"] = config["eval"]["path"]

        with self.assertRaisesRegex(
            BaselineProtocolError,
            "truth data changed",
        ):
            validate_run_config(config, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
