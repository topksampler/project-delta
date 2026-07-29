from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.delta_v2.feature_promotion import (
    FeaturePromotionError,
    audit_promotion,
    load_probe_result,
    validate_feature_delta,
)


def candidate_record() -> dict:
    return {
        "candidate_id": "feature-candidate:example",
        "evidence_ids": [
            "fact:one",
            "file:one",
            "pull-request:owner/repo#1",
        ],
    }


def feature_record() -> dict:
    return {
        "schema": "delta.feature_delta.v1",
        "feature_id": "feature:example",
        "source_candidate_id": "feature-candidate:example",
        "repository": "owner/repo",
        "transition": "development",
        "summary": "Example",
        "status": "added",
        "evidence_ids": [
            "fact:one",
            "file:one",
            "pull-request:owner/repo#1",
        ],
        "behavior_probe_ids": ["behavior-probe:complete"],
        "verification": {
            "old_revision": "unavailable",
            "new_revision": "pass",
            "recipe": "example-v1",
            "probe_results": {
                "behavior-probe:complete": {
                    "sha256": "0" * 64,
                    "claim_scope": "complete-candidate",
                }
            },
        },
        "environment_status": "development-only-unfrozen",
    }


def probe_result() -> dict:
    return {
        "schema": "delta.behavior_probe_result.v1",
        "probe_id": "behavior-probe:complete",
        "candidate_id": "feature-candidate:example",
        "claim_scope": "complete-candidate",
        "status": "pass",
        "sources": {
            "development_before": {"commit_sha": "a" * 40},
            "development_after": {"commit_sha": "b" * 40},
        },
    }


def serialized_result(result: dict) -> tuple[dict, str]:
    raw = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    return result, hashlib.sha256(raw).hexdigest()


class FeatureDeltaValidationTest(unittest.TestCase):
    def test_accepts_verified_added_feature(self) -> None:
        validate_feature_delta(feature_record())

    def test_requires_complete_candidate_probe(self) -> None:
        feature = feature_record()
        feature["verification"]["probe_results"]["behavior-probe:complete"][
            "claim_scope"
        ] = "configuration-interface-only"

        with self.assertRaisesRegex(FeaturePromotionError, "complete-candidate"):
            validate_feature_delta(feature)

    def test_cannot_claim_eval_environment_freeze(self) -> None:
        feature = feature_record()
        feature["environment_status"] = "frozen"

        with self.assertRaisesRegex(FeaturePromotionError, "must not claim"):
            validate_feature_delta(feature)

    def test_added_feature_requires_old_unavailable_new_pass(self) -> None:
        feature = feature_record()
        feature["verification"]["old_revision"] = "pass"

        with self.assertRaisesRegex(FeaturePromotionError, "unavailable"):
            validate_feature_delta(feature)


class PromotionAuditTest(unittest.TestCase):
    def matching_inputs(self) -> tuple[dict, dict, dict]:
        candidate = candidate_record()
        feature = feature_record()
        result, digest = serialized_result(probe_result())
        feature["verification"]["probe_results"]["behavior-probe:complete"][
            "sha256"
        ] = digest
        return candidate, feature, {
            "behavior-probe:complete": (result, digest)
        }

    def test_exact_evidence_and_probe_result_promote(self) -> None:
        candidate, feature, results = self.matching_inputs()

        audit = audit_promotion(candidate, feature, results)

        self.assertEqual(audit["status"], "pass")
        self.assertFalse(audit["acceptance_accessed"])
        self.assertFalse(audit["eval_environment_frozen"])

    def test_hash_or_status_mismatch_fails(self) -> None:
        candidate, feature, results = self.matching_inputs()
        altered = copy.deepcopy(results)
        altered["behavior-probe:complete"] = (
            {**altered["behavior-probe:complete"][0], "status": "fail"},
            altered["behavior-probe:complete"][1],
        )

        audit = audit_promotion(candidate, feature, altered)

        self.assertEqual(audit["status"], "fail")
        self.assertFalse(
            audit["probe_audits"][0]["checks"]["status_passes"]
        )

    def test_candidate_evidence_cannot_be_added_or_dropped(self) -> None:
        candidate, feature, results = self.matching_inputs()
        feature["evidence_ids"].append("file:two")

        with self.assertRaisesRegex(FeaturePromotionError, "exactly preserve"):
            audit_promotion(candidate, feature, results)

    def test_probe_must_use_exact_development_roles(self) -> None:
        candidate, feature, results = self.matching_inputs()
        result, digest = results["behavior-probe:complete"]
        result = copy.deepcopy(result)
        result["sources"]["acceptance_after"] = {"commit_sha": "c" * 40}

        with self.assertRaisesRegex(FeaturePromotionError, "exactly the development"):
            audit_promotion(
                candidate,
                feature,
                {"behavior-probe:complete": (result, digest)},
            )


class ProbeResultLoadingTest(unittest.TestCase):
    def test_hashes_exact_result_bytes(self) -> None:
        raw = (
            json.dumps(probe_result(), sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "result.json"
            path.write_bytes(raw)

            result, digest = load_probe_result(path)

        self.assertEqual(result["status"], "pass")
        self.assertEqual(digest, hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    unittest.main()
