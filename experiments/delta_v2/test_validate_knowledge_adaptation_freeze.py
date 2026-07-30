from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from experiments.delta_v2.knowledge_adaptation import (
    KnowledgeAdaptationError,
)
from experiments.delta_v2.validate_knowledge_adaptation_freeze import (
    validate_freeze,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    REPO_ROOT
    / "experiments/delta_v2/knowledge_adaptation_freeze.yaml"
)


class KnowledgeAdaptationFreezeTests(unittest.TestCase):
    def test_frozen_dataset_validates(self) -> None:
        manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
        result = validate_freeze(manifest, repo_root=REPO_ROOT)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["bindings_checked"], 8)
        self.assertEqual(result["eval_rows"], 169)

    def test_pooled_score_cannot_be_enabled(self) -> None:
        manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["scoreboards"]["pooled_overall_score"] = "allowed"
        with self.assertRaisesRegex(
            KnowledgeAdaptationError,
            "scoreboard boundary changed",
        ):
            validate_freeze(manifest, repo_root=REPO_ROOT)

    def test_retention_overlap_cannot_be_relaxed(self) -> None:
        manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["leakage_gate"]["retention_source_overlap"] = 1
        with self.assertRaisesRegex(
            KnowledgeAdaptationError,
            "leakage gate changed",
        ):
            validate_freeze(manifest, repo_root=REPO_ROOT)

    def test_bound_file_change_fails_closed(self) -> None:
        manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["bindings"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            KnowledgeAdaptationError,
            "binding bytes changed",
        ):
            validate_freeze(manifest, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
