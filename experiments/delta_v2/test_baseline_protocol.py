from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.delta_v2.baseline_protocol import (
    OUTPUT_SCHEMA,
    BaselineProtocolError,
    build_requests,
    load_jsonl,
    load_yaml,
    score_outputs,
    serialize_jsonl,
    validate_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = Path(__file__).with_name("target_baseline_protocol.yaml")


def protocol_record() -> dict:
    return copy.deepcopy(dict(load_yaml(PROTOCOL_PATH)))


def correct_response(item: dict) -> str:
    if item["scorer"] == "exact-enum-v1":
        return str(item["gold"]["status"])
    return json.dumps(item["gold"], sort_keys=True, separators=(",", ":"))


def outputs_for(items: list[dict], response_for=correct_response) -> list[dict]:
    protocol = protocol_record()
    rows = []
    for item in items:
        response = response_for(item)
        for repeat_index in (1, 2):
            rows.append(
                {
                    "schema": OUTPUT_SCHEMA,
                    "protocol_id": protocol["protocol_id"],
                    "eval_id": item["eval_id"],
                    "repeat_index": repeat_index,
                    "model_revision": protocol["target_model"]["revision"],
                    "raw_response": response,
                    "finish_reason": "stop",
                }
            )
    return rows


class ProtocolContractTest(unittest.TestCase):
    def test_real_protocol_binds_frozen_environment_and_43_eval_items(self) -> None:
        protocol = protocol_record()

        items = validate_protocol(protocol, repo_root=REPO_ROOT)

        self.assertEqual(len(items), 43)
        self.assertEqual(protocol["execution_boundary"]["model_invocations_completed"], 0)
        self.assertFalse(
            protocol["execution_boundary"]["dispatch_authorized_by_this_file"]
        )

    def test_hash_drift_fails_closed(self) -> None:
        protocol = protocol_record()
        protocol["evaluation_input"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(
            BaselineProtocolError,
            "evaluation-input boundary",
        ):
            validate_protocol(protocol, repo_root=REPO_ROOT)

    def test_runtime_or_sampling_drift_fails_closed(self) -> None:
        protocol = protocol_record()
        protocol["generation"]["do_sample"] = True

        with self.assertRaisesRegex(
            BaselineProtocolError,
            "generation lock",
        ):
            validate_protocol(protocol, repo_root=REPO_ROOT)


class RequestEnvelopeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = protocol_record()
        self.items = list(
            validate_protocol(self.protocol, repo_root=REPO_ROOT)
        )

    def test_requests_are_deterministic_and_repeat_every_item_twice(self) -> None:
        first = build_requests(self.protocol, self.items)
        second = build_requests(self.protocol, list(reversed(self.items)))

        self.assertEqual(len(first), 86)
        self.assertEqual(
            serialize_jsonl(first),
            serialize_jsonl(
                sorted(
                    second,
                    key=lambda row: (row["eval_id"], row["repeat_index"]),
                )
            ),
        )

    def test_model_visible_envelope_contains_prompt_but_no_truth(self) -> None:
        requests = build_requests(self.protocol, self.items)
        serialized = serialize_jsonl(requests)

        self.assertIn(b'"messages"', serialized)
        for forbidden in (
            b'"gold"',
            b'"provenance"',
            b'"scorer"',
            b'"behavior_probe_result"',
            b'"judge_evidence"',
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertTrue(
            all(
                request["model_input"]["chat_template_kwargs"]
                == {"enable_thinking": False}
                for request in requests
            )
        )


class OfflineScoringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = protocol_record()
        self.items = [
            dict(item)
            for item in validate_protocol(
                self.protocol,
                repo_root=REPO_ROOT,
            )
        ]

    def test_all_correct_fixture_preserves_separate_strata(self) -> None:
        summary = score_outputs(
            protocol=self.protocol,
            items=self.items,
            outputs=outputs_for(self.items),
        )

        self.assertEqual(summary["status"], "pass")
        self.assertIsNone(summary["pooled_overall_accuracy"])
        self.assertEqual(
            {
                key: value["items"]
                for key, value in summary["strata"].items()
            },
            {
                "atomic_fact_added": 21,
                "atomic_fact_stable": 21,
                "feature_added": 1,
            },
        )
        self.assertTrue(
            all(
                value["exact_accuracy"] == 1.0
                for value in summary["strata"].values()
            )
        )
        self.assertEqual(
            summary["interpretation"]["state"],
            "no-observed-acceptance-gap",
        )

    def test_stable_failure_prevents_false_drift_localization(self) -> None:
        summary = score_outputs(
            protocol=self.protocol,
            items=self.items,
            outputs=outputs_for(self.items, lambda _: "added"),
        )

        self.assertEqual(
            summary["strata"]["atomic_fact_added"]["exact_accuracy"],
            1.0,
        )
        self.assertEqual(
            summary["strata"]["atomic_fact_stable"]["exact_accuracy"],
            0.0,
        )
        self.assertEqual(
            summary["interpretation"]["state"],
            "baseline-incapable-of-localizing-drift",
        )

    def test_strong_stable_added_gap_localizes_drift(self) -> None:
        def response(item: dict) -> str:
            if item["source_kind"] == "feature_delta":
                return correct_response(item)
            if item["gold"]["status"] == "stable":
                return "stable"
            return "stable"

        summary = score_outputs(
            protocol=self.protocol,
            items=self.items,
            outputs=outputs_for(self.items, response),
        )

        self.assertEqual(
            summary["strata"]["atomic_fact_added"]["exact_accuracy"],
            0.0,
        )
        self.assertEqual(
            summary["strata"]["atomic_fact_stable"]["exact_accuracy"],
            1.0,
        )
        self.assertEqual(
            summary["interpretation"]["state"],
            "localized-added-fact-drift",
        )
        self.assertLess(
            summary["interpretation"]["added_less_than_stable_p_value"],
            0.05,
        )

    def test_repeat_disagreement_fails_instead_of_being_averaged(self) -> None:
        outputs = outputs_for(self.items)
        outputs[1]["raw_response"] = "different"

        with self.assertRaisesRegex(
            BaselineProtocolError,
            "nondeterministic target response",
        ):
            score_outputs(
                protocol=self.protocol,
                items=self.items,
                outputs=outputs,
            )

    def test_missing_output_fails_closed(self) -> None:
        outputs = outputs_for(self.items)

        with self.assertRaisesRegex(
            BaselineProtocolError,
            "coverage is incomplete",
        ):
            score_outputs(
                protocol=self.protocol,
                items=self.items,
                outputs=outputs[:-1],
            )


if __name__ == "__main__":
    unittest.main()
