from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.delta_v2.fact_family_manifest import (
    FactFamilyManifestError,
    load_fact_family_manifest,
    manifest_summary,
)


MANIFEST_PATH = Path(__file__).with_name("fact_families.yaml")
EXPECTED_FAMILY_IDS = [
    "python.config_field.v1",
    "python.cli_option.v1",
    "python.environment_variable.v1",
    "python.literal_domain.v1",
]


def load_payload() -> dict:
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("real manifest must be a mapping")
    return payload


def write_payload(temp_dir: str, payload: dict) -> Path:
    path = Path(temp_dir) / "fact_families.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (path.parent / "snapshots.yaml").write_text(
        (MANIFEST_PATH.parent / "snapshots.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return path


class FactFamilyManifestTest(unittest.TestCase):
    def test_real_manifest_preregisters_exact_family_set(self) -> None:
        manifest = load_fact_family_manifest(MANIFEST_PATH)

        self.assertEqual(
            [family.family_id for family in manifest.families],
            EXPECTED_FAMILY_IDS,
        )
        self.assertEqual(manifest.transition, "development")
        self.assertEqual(manifest.parser_feature_version, "3.11")
        self.assertEqual(
            manifest_summary(manifest)["family_count"],
            4,
        )

    def test_requires_independent_complete_snapshot_extraction(self) -> None:
        payload = load_payload()
        payload["extraction"]["extract_each_snapshot_independently"] = False

        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_payload(temp_dir, payload)
            with self.assertRaisesRegex(
                FactFamilyManifestError,
                "extracted independently",
            ):
                load_fact_family_manifest(path)

    def test_rejects_parser_contract_drift(self) -> None:
        payload = load_payload()
        payload["extraction"]["parser"]["feature_version"] = "3.12"

        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_payload(temp_dir, payload)
            with self.assertRaisesRegex(
                FactFamilyManifestError,
                "feature version must be 3.11",
            ):
                load_fact_family_manifest(path)

        payload = load_payload()
        payload["extraction"]["parser"]["canonical_form"] = "ast.unparse"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_payload(temp_dir, payload)
            with self.assertRaisesRegex(
                FactFamilyManifestError,
                "canonical form",
            ):
                load_fact_family_manifest(path)

        payload = load_payload()
        payload["extraction"]["source_selection"] = "changed-files-only"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_payload(temp_dir, payload)
            with self.assertRaisesRegex(
                FactFamilyManifestError,
                "complete-snapshot-tree",
            ):
                load_fact_family_manifest(path)

    def test_rejects_acceptance_transition(self) -> None:
        payload = load_payload()
        payload["transition"] = "acceptance"

        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_payload(temp_dir, payload)
            with self.assertRaisesRegex(
                FactFamilyManifestError,
                "development transition",
            ):
                load_fact_family_manifest(path)

    def test_requires_existing_relative_snapshot_manifest(self) -> None:
        payload = load_payload()
        payload["source_manifest"] = "missing.yaml"

        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_payload(temp_dir, payload)
            with self.assertRaisesRegex(
                FactFamilyManifestError,
                "does not exist",
            ):
                load_fact_family_manifest(path)

        payload = load_payload()
        payload["source_manifest"] = "/tmp/snapshots.yaml"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_payload(temp_dir, payload)
            with self.assertRaisesRegex(
                FactFamilyManifestError,
                "relative path",
            ):
                load_fact_family_manifest(path)

    def test_rejects_duplicate_family_ids_and_missing_rules(self) -> None:
        payload = load_payload()
        payload["families"].append(dict(payload["families"][0]))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_payload(temp_dir, payload)
            with self.assertRaisesRegex(FactFamilyManifestError, "must be unique"):
                load_fact_family_manifest(path)

        payload = load_payload()
        del payload["families"][0]["inclusion_rules"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_payload(temp_dir, payload)
            with self.assertRaisesRegex(
                FactFamilyManifestError,
                "missing required fields",
            ):
                load_fact_family_manifest(path)

    def test_requires_explicit_rejections_and_evidence(self) -> None:
        payload = load_payload()
        payload["families"][0]["rejection_rules"] = ["Silently skip it."]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_payload(temp_dir, payload)
            with self.assertRaisesRegex(
                FactFamilyManifestError,
                "explicit rejection behavior",
            ):
                load_fact_family_manifest(path)

        payload = load_payload()
        payload["evidence"]["required_fields"].remove("source_sha256")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_payload(temp_dir, payload)
            with self.assertRaisesRegex(
                FactFamilyManifestError,
                "evidence.required_fields",
            ):
                load_fact_family_manifest(path)


if __name__ == "__main__":
    unittest.main()
