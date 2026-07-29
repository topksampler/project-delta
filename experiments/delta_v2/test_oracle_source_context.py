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
from experiments.delta_v2.oracle_source_context import (
    PROTOCOL_ID,
    audit_oracle_requests,
    build_oracle_requests,
    validate_oracle_protocol,
)
from experiments.delta_v2.run_oracle_source_context import (
    EXPECTED_REQUEST_SHA256,
    EXPECTED_RUN_ID,
    execute_oracle_source_context,
    validate_only,
    validate_run_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = Path(__file__).with_name(
    "oracle_source_context_protocol.yaml"
)
CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/delta_v2/"
    "c2_oracle_source_context_qwen35_08b_modal_v1.yaml"
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


class OracleProtocolTest(unittest.TestCase):
    def test_all_items_route_to_minimal_frozen_source_evidence(self) -> None:
        protocol = load_yaml(PROTOCOL_PATH)

        baseline, items, evidence, feature = validate_oracle_protocol(
            protocol,
            repo_root=REPO_ROOT,
        )
        requests = build_oracle_requests(
            protocol,
            baseline_protocol=baseline,
            items=items,
            evidence_by_eval=evidence,
            feature_evidence=feature,
        )
        audit = audit_oracle_requests(requests)

        self.assertEqual(len(evidence), 42)
        self.assertEqual(len(feature), 8)
        self.assertEqual(len(requests), 86)
        self.assertFalse(audit["gold_label_visible"])
        self.assertFalse(audit["status_field_visible"])
        self.assertFalse(audit["provenance_visible"])
        self.assertFalse(audit["production_retrieval_claim"])
        serialized = json.dumps(requests, sort_keys=True)
        for forbidden in (
            '"gold"',
            '"status"',
            '"provenance"',
            '"scorer"',
            '"fact_id"',
            '"verifier"',
        ):
            self.assertNotIn(forbidden, serialized)

    def test_fact_evidence_contains_presence_and_hashes_not_values(self) -> None:
        protocol = load_yaml(PROTOCOL_PATH)
        _, items, evidence, _ = validate_oracle_protocol(
            protocol,
            repo_root=REPO_ROOT,
        )

        for item in items:
            if item["source_kind"] != "atomic_fact_delta":
                continue
            projected = evidence[item["eval_id"]]
            self.assertEqual(
                set(projected),
                {
                    "semantic_key",
                    "before_present",
                    "before_canonical_sha256",
                    "after_present",
                    "after_canonical_sha256",
                },
            )
            self.assertNotIn("value_before", projected)
            self.assertNotIn("value_after", projected)

    def test_oracle_role_cannot_claim_production_retrieval(self) -> None:
        protocol = copy.deepcopy(dict(load_yaml(PROTOCOL_PATH)))
        protocol["adaptation"]["production_retrieval_claim"] = "implemented"

        with self.assertRaisesRegex(
            BaselineProtocolError,
            "adaptation method changed",
        ):
            validate_oracle_protocol(protocol, repo_root=REPO_ROOT)


class OracleRunTest(unittest.TestCase):
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

    def test_fixture_execution_keeps_oracle_limit_in_metrics(self) -> None:
        backend = FixtureBackend(self.answers)
        with tempfile.TemporaryDirectory() as tmp:
            result = execute_oracle_source_context(
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
        self.assertEqual(
            result["audit"]["adaptation"]["role"],
            "upper-bound-control",
        )
        self.assertFalse(
            result["audit"]["adaptation"]["production_retrieval_claim"]
        )

    def test_config_cannot_replace_truth_source(self) -> None:
        config = copy.deepcopy(dict(load_yaml(CONFIG_PATH)))
        config["truth"]["fact_deltas_path"] = config["eval"]["path"]

        with self.assertRaisesRegex(
            BaselineProtocolError,
            "truth data changed",
        ):
            validate_run_config(config, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
