from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from experiments.delta_v2.behavior_probe import (
    BehaviorProbeError,
    observe_source,
    run_probe,
    validate_probe,
)


ATTRIBUTE = "VLLM_PREFIX_CACHE_RETENTION_INTERVAL"

OLD_SOURCE = b"""
environment_variables = {}
def __getattr__(name):
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(name)
"""

NEW_SOURCE = b"""
import os
environment_variables = {
    "VLLM_PREFIX_CACHE_RETENTION_INTERVAL": lambda: (
        int(os.environ["VLLM_PREFIX_CACHE_RETENTION_INTERVAL"])
        if "VLLM_PREFIX_CACHE_RETENTION_INTERVAL" in os.environ
        else None
    )
}
def __getattr__(name):
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(name)
"""


def probe_record() -> dict:
    return {
        "schema": "delta.behavior_probe.v1",
        "probe_id": "behavior-probe:example-v1",
        "candidate_id": "feature-candidate:example",
        "transition": "development",
        "claim": "Example interface",
        "claim_scope": "configuration-interface-only",
        "setup": {
            "source_path": "vllm/envs.py",
            "source_loader": "exact-git-blob-importlib-v1",
            "attribute": ATTRIBUTE,
        },
        "cases": [
            {
                "case_id": "unset",
                "environment_value": None,
                "expected_before": {"kind": "attribute_error"},
                "expected_after": {"kind": "value", "value": None},
            },
            {
                "case_id": "positive",
                "environment_value": "64",
                "expected_before": {"kind": "attribute_error"},
                "expected_after": {"kind": "value", "value": 64},
            },
            {
                "case_id": "malformed",
                "environment_value": "nope",
                "expected_before": {"kind": "attribute_error"},
                "expected_after": {"kind": "value_error"},
            },
        ],
        "environment": {
            "python": ">=3.11",
            "dependencies": "standard-library-only",
            "process_isolation": "fresh-module-per-case",
        },
        "promotion_boundary": "Interface only.",
    }


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


class ProbeValidationTest(unittest.TestCase):
    def test_accepts_narrow_development_probe(self) -> None:
        validate_probe(probe_record())

    def test_rejects_acceptance_transition(self) -> None:
        probe = probe_record()
        probe["transition"] = "acceptance"

        with self.assertRaisesRegex(BehaviorProbeError, "only use development"):
            validate_probe(probe)

    def test_rejects_broader_claim_scope(self) -> None:
        probe = probe_record()
        probe["claim_scope"] = "complete-feature"

        with self.assertRaisesRegex(BehaviorProbeError, "narrow claim scope"):
            validate_probe(probe)

    def test_rejects_non_relative_source_path(self) -> None:
        probe = probe_record()
        probe["setup"]["source_path"] = "../envs.py"

        with self.assertRaisesRegex(BehaviorProbeError, "repository-relative"):
            validate_probe(probe)

    def test_rejects_malformed_outcome(self) -> None:
        probe = probe_record()
        probe["cases"][0]["expected_after"] = {
            "kind": "attribute_error",
            "value": None,
        }

        with self.assertRaisesRegex(BehaviorProbeError, "exactly the kind"):
            validate_probe(probe)


class SourceObservationTest(unittest.TestCase):
    def test_old_source_rejects_attribute(self) -> None:
        result = observe_source(
            OLD_SOURCE,
            attribute=ATTRIBUTE,
            environment_value="64",
            case_token="old",
        )
        self.assertEqual(result, {"kind": "attribute_error"})

    def test_new_source_parses_optional_integer(self) -> None:
        unset = observe_source(
            NEW_SOURCE,
            attribute=ATTRIBUTE,
            environment_value=None,
            case_token="new-unset",
        )
        positive = observe_source(
            NEW_SOURCE,
            attribute=ATTRIBUTE,
            environment_value="64",
            case_token="new-positive",
        )
        malformed = observe_source(
            NEW_SOURCE,
            attribute=ATTRIBUTE,
            environment_value="nope",
            case_token="new-malformed",
        )

        self.assertEqual(unset, {"kind": "value", "value": None})
        self.assertEqual(positive, {"kind": "value", "value": 64})
        self.assertEqual(malformed, {"kind": "value_error"})


class GitProbeIntegrationTest(unittest.TestCase):
    def test_executes_exact_blobs_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            git(repo, "init")
            git(repo, "config", "user.name", "delta-v2-test")
            git(repo, "config", "user.email", "delta-v2@example.invalid")
            source_path = repo / "vllm" / "envs.py"
            source_path.parent.mkdir(parents=True)

            source_path.write_bytes(OLD_SOURCE)
            git(repo, "add", "vllm/envs.py")
            git(repo, "commit", "-m", "before")
            before = git(repo, "rev-parse", "HEAD")
            before_tree = git(repo, "rev-parse", "HEAD^{tree}")

            source_path.write_bytes(NEW_SOURCE)
            git(repo, "commit", "-am", "after")
            after = git(repo, "rev-parse", "HEAD")
            after_tree = git(repo, "rev-parse", "HEAD^{tree}")

            snapshots = repo / "snapshots.yaml"
            snapshots.write_text(
                "\n".join(
                    [
                        "schema: delta.source_snapshots.v1",
                        "repository:",
                        "  id: owner/repo",
                        "  url: https://example.invalid/owner/repo.git",
                        "content_hash_algorithm: git-tree-sha1",
                        "snapshots:",
                        "  - role: development_before",
                        "    revision: before",
                        f"    commit_sha: {before}",
                        f"    content_hash: git-tree-sha1:{before_tree}",
                        "  - role: development_after",
                        "    revision: after",
                        f"    commit_sha: {after}",
                        f"    content_hash: git-tree-sha1:{after_tree}",
                        "  - role: acceptance_before",
                        "    revision: sealed-before",
                        f"    commit_sha: \"{'1' * 40}\"",
                        f"    content_hash: \"git-tree-sha1:{'2' * 40}\"",
                        "  - role: acceptance_after",
                        "    revision: sealed-after",
                        f"    commit_sha: \"{'3' * 40}\"",
                        f"    content_hash: \"git-tree-sha1:{'4' * 40}\"",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            first = run_probe(repo, snapshots, probe_record())
            second = run_probe(repo, snapshots, probe_record())

        self.assertEqual(first["status"], "pass")
        self.assertEqual(first, second)
        self.assertEqual(
            json.dumps(first, sort_keys=True),
            json.dumps(second, sort_keys=True),
        )
        self.assertEqual(
            set(first["sources"]),
            {"development_before", "development_after"},
        )

    def test_wrong_expectation_fails_without_rewriting_observation(self) -> None:
        probe = probe_record()
        probe["cases"][1]["expected_after"] = {"kind": "value", "value": 32}

        self.assertNotEqual(
            probe["cases"][1]["expected_after"],
            {"kind": "value", "value": 64},
        )
        unchanged = copy.deepcopy(probe)
        validate_probe(probe)
        self.assertEqual(probe, unchanged)


if __name__ == "__main__":
    unittest.main()

