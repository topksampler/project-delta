from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from experiments.delta_v2.kv_retention_probe import (
    KVRetentionProbeError,
    _normalize_construction_error,
    evaluate_observations,
    load_probe,
    validate_probe,
    verify_source_tree,
)


PROBE_PATH = (
    Path(__file__).with_name("probes")
    / "prefix_cache_retention_mechanics.yaml"
)


def cache_state(indices: list[int]) -> dict:
    return {
        "kind": "cache_state",
        "cached_indices": indices,
        "replay_computed_tokens": 240,
    }


def minimal_probe() -> dict:
    return {
        "schema": "delta.behavior_probe.v1",
        "probe_id": "behavior-probe:example",
        "candidate_id": "feature-candidate:example",
        "transition": "development",
        "claim": "Example",
        "claim_scope": "complete-candidate",
        "scenario": {
            "id": "pure-swa-16x16-v1",
            "block_size": 16,
            "sliding_window": 16,
            "prompt_blocks": 16,
            "num_cache_blocks": 100,
            "max_model_len": 8192,
        },
        "cases": [
            {
                "case_id": "dense",
                "environment_value": None,
                "expected_before": cache_state([0, 1]),
                "expected_after": cache_state([0, 1]),
            }
        ],
        "runtime": {
            "target": "local-cpu",
            "environment_status": "development-only-unfrozen",
            "dependency_source": {
                "snapshot_role": "development_after",
                "path": "requirements/common.txt",
            },
            "source_execution": "exact-detached-worktree-v1",
            "process_isolation": "one-process-per-snapshot",
        },
        "promotion_rule": "All cases pass.",
    }


def observations(outcome: dict) -> dict:
    return {
        "runtime": {"python": "3.11"},
        "observations": [{"case_id": "dense", "outcome": outcome}],
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


class ProbeContractTest(unittest.TestCase):
    def test_real_contract_is_valid_and_has_required_cases(self) -> None:
        probe = load_probe(PROBE_PATH)

        self.assertEqual(
            [case["case_id"] for case in probe["cases"]],
            [
                "dense_default",
                "interval_64",
                "latest_only",
                "negative_rejected",
                "misaligned_rejected",
            ],
        )

    def test_rejects_acceptance_transition(self) -> None:
        probe = minimal_probe()
        probe["transition"] = "acceptance"

        with self.assertRaisesRegex(KVRetentionProbeError, "only use development"):
            validate_probe(probe)

    def test_rejects_frozen_development_runtime_claim(self) -> None:
        probe = minimal_probe()
        probe["runtime"]["environment_status"] = "frozen"

        with self.assertRaisesRegex(KVRetentionProbeError, "must not claim a freeze"):
            validate_probe(probe)

    def test_rejects_unsorted_or_duplicate_cached_indices(self) -> None:
        probe = minimal_probe()
        probe["cases"][0]["expected_after"]["cached_indices"] = [1, 0, 1]

        with self.assertRaisesRegex(KVRetentionProbeError, "sorted unique"):
            validate_probe(probe)

    def test_normalizes_only_preregistered_errors(self) -> None:
        self.assertEqual(
            _normalize_construction_error(
                "must be non-negative and a multiple of scheduler_block_size",
                "-32",
                16,
            ),
            "non_negative",
        )
        self.assertEqual(
            _normalize_construction_error(
                "must be non-negative and a multiple of scheduler_block_size",
                "33",
                16,
            ),
            "scheduler_block_size_multiple",
        )
        with self.assertRaisesRegex(KVRetentionProbeError, "unexpected"):
            _normalize_construction_error("different failure", "33", 16)


class ObservationEvaluationTest(unittest.TestCase):
    def test_exact_old_new_matches_pass(self) -> None:
        probe = minimal_probe()
        outcome = cache_state([0, 1])

        result = evaluate_observations(
            probe,
            {
                "development_before": observations(outcome),
                "development_after": observations(outcome),
            },
        )

        self.assertEqual(result[0]["status"], "pass")

    def test_mismatch_fails_without_changing_expected_truth(self) -> None:
        probe = minimal_probe()
        original = copy.deepcopy(probe)

        result = evaluate_observations(
            probe,
            {
                "development_before": observations(cache_state([0, 1])),
                "development_after": observations(cache_state([1])),
            },
        )

        self.assertEqual(result[0]["status"], "fail")
        self.assertEqual(probe, original)

    def test_missing_or_duplicate_case_is_rejected(self) -> None:
        probe = minimal_probe()
        with self.assertRaisesRegex(KVRetentionProbeError, "missing runtime case"):
            evaluate_observations(
                probe,
                {
                    "development_before": {
                        "runtime": {},
                        "observations": [],
                    },
                    "development_after": observations(cache_state([0, 1])),
                },
            )

        duplicated = observations(cache_state([0, 1]))
        duplicated["observations"].append(duplicated["observations"][0])
        with self.assertRaisesRegex(KVRetentionProbeError, "duplicate runtime case"):
            evaluate_observations(
                probe,
                {
                    "development_before": duplicated,
                    "development_after": observations(cache_state([0, 1])),
                },
            )

    def test_results_are_deterministically_serializable(self) -> None:
        probe = minimal_probe()
        outcome = cache_state([0, 1])
        result = evaluate_observations(
            probe,
            {
                "development_before": observations(outcome),
                "development_after": observations(outcome),
            },
        )

        first = json.dumps(result, sort_keys=True, separators=(",", ":"))
        second = json.dumps(result, sort_keys=True, separators=(",", ":"))
        self.assertEqual(first, second)


class SourceTreeVerificationTest(unittest.TestCase):
    def test_checks_both_commit_and_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            git(repo, "init")
            git(repo, "config", "user.name", "delta-v2-test")
            git(repo, "config", "user.email", "delta-v2@example.invalid")
            (repo / "source.txt").write_text("source\n", encoding="utf-8")
            git(repo, "add", "source.txt")
            git(repo, "commit", "-m", "source")
            commit = git(repo, "rev-parse", "HEAD")
            tree = git(repo, "rev-parse", "HEAD^{tree}")

            result = verify_source_tree(
                repo,
                expected_commit=commit,
                expected_tree=f"git-tree-sha1:{tree}",
            )

            self.assertEqual(result["commit_sha"], commit)
            with self.assertRaisesRegex(KVRetentionProbeError, "tree mismatch"):
                verify_source_tree(
                    repo,
                    expected_commit=commit,
                    expected_tree=f"git-tree-sha1:{'0' * 40}",
                )


if __name__ == "__main__":
    unittest.main()
