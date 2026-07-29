from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from experiments.delta_v2.feature_trace import (
    FeatureTraceError,
    audit_evidence_links,
    validate_candidate,
    validate_pr_evidence,
    verify_transition_membership,
)


def candidate_record() -> dict:
    return {
        "schema": "delta.feature_candidate.v1",
        "candidate_id": "feature-candidate:example",
        "repository": "owner/repo",
        "transition": "development",
        "summary": "Example candidate",
        "proposed_status": "added",
        "proposal_method": "human",
        "verification_status": "pending",
        "evidence_ids": [
            "fact:one",
            "file:one",
            "pull-request:owner/repo#7",
        ],
        "behavior_probe_ids": [],
    }


def pr_record() -> dict:
    return {
        "schema": "delta.pull_request_evidence.v1",
        "evidence_id": "pull-request:owner/repo#7",
        "repository": "owner/repo",
        "number": 7,
        "state": "merged",
        "merge_commit_sha": "a" * 40,
        "counts": {"changed_files": 1},
        "changed_paths": ["src/example.py"],
        "development_transition_membership": {
            "development_before_contains_merge": False,
            "development_after_contains_merge": True,
        },
        "provenance_role": "candidate-discovery-and-grouping-only",
    }


class RecordValidationTest(unittest.TestCase):
    def test_accepts_pending_candidate_and_merged_pr(self) -> None:
        validate_candidate(candidate_record())
        validate_pr_evidence(pr_record())

    def test_candidate_cannot_claim_verification(self) -> None:
        candidate = candidate_record()
        candidate["verification_status"] = "verified"

        with self.assertRaisesRegex(FeatureTraceError, "must remain pending"):
            validate_candidate(candidate)

    def test_candidate_cannot_claim_probe_before_promotion(self) -> None:
        candidate = candidate_record()
        candidate["behavior_probe_ids"] = ["probe:one"]

        with self.assertRaisesRegex(FeatureTraceError, "cannot claim BehaviorProbe"):
            validate_candidate(candidate)

    def test_pr_must_be_provenance_only(self) -> None:
        record = pr_record()
        record["provenance_role"] = "ground-truth"

        with self.assertRaisesRegex(FeatureTraceError, "must not be labeled as truth"):
            validate_pr_evidence(record)

    def test_pr_path_count_must_match(self) -> None:
        record = pr_record()
        record["counts"]["changed_files"] = 2

        with self.assertRaisesRegex(FeatureTraceError, "count does not match"):
            validate_pr_evidence(record)


class EvidenceAuditTest(unittest.TestCase):
    def test_resolves_exact_fact_file_and_pr_links(self) -> None:
        result = audit_evidence_links(
            candidate_record(),
            pr_record(),
            [{"fact_id": "fact:one"}],
            [
                {
                    "file_id": "file:one",
                    "path_before": "src/example.py",
                    "path_after": "src/example.py",
                }
            ],
        )

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["linked_fact_count"], 1)
        self.assertEqual(result["linked_file_count"], 1)

    def test_reports_unresolved_ids_and_path_disagreement(self) -> None:
        candidate = candidate_record()
        candidate["evidence_ids"].append("file:missing")
        result = audit_evidence_links(
            candidate,
            pr_record(),
            [{"fact_id": "fact:one"}],
            [
                {
                    "file_id": "file:one",
                    "path_before": "src/different.py",
                    "path_after": "src/different.py",
                }
            ],
        )

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["unresolved_evidence_ids"], ["file:missing"])
        self.assertEqual(result["missing_pr_paths"], ["src/example.py"])
        self.assertEqual(result["extra_candidate_paths"], ["src/different.py"])


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


class TransitionMembershipTest(unittest.TestCase):
    def test_checks_only_declared_development_commits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            git(repo, "init")
            git(repo, "config", "user.name", "delta-v2-test")
            git(repo, "config", "user.email", "delta-v2@example.invalid")

            source = repo / "source.txt"
            source.write_text("before\n", encoding="utf-8")
            git(repo, "add", "source.txt")
            git(repo, "commit", "-m", "before")
            before = git(repo, "rev-parse", "HEAD")
            before_tree = git(repo, "rev-parse", "HEAD^{tree}")

            source.write_text("merge\n", encoding="utf-8")
            git(repo, "commit", "-am", "merged PR")
            merge = git(repo, "rev-parse", "HEAD")

            source.write_text("after\n", encoding="utf-8")
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
            record = pr_record()
            record["merge_commit_sha"] = merge

            result = verify_transition_membership(repo, snapshots, record)

        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["development_before_contains_merge"])
        self.assertTrue(result["development_after_contains_merge"])

    def test_detects_incorrect_membership_declaration(self) -> None:
        record = copy.deepcopy(pr_record())
        record["development_transition_membership"][
            "development_before_contains_merge"
        ] = True

        with self.assertRaisesRegex(FeatureTraceError, "must not be contained"):
            validate_pr_evidence(record)


class SerializationTest(unittest.TestCase):
    def test_audit_is_deterministically_serializable(self) -> None:
        result = audit_evidence_links(
            candidate_record(),
            pr_record(),
            [{"fact_id": "fact:one"}],
            [
                {
                    "file_id": "file:one",
                    "path_before": "src/example.py",
                    "path_after": "src/example.py",
                }
            ],
        )

        first = json.dumps(result, sort_keys=True, separators=(",", ":"))
        second = json.dumps(result, sort_keys=True, separators=(",", ":"))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
