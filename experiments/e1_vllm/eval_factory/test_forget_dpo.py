"""Unit tests for forget scorer + DPO pair exporter (no GPU)."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from .build_claim_local_edits import build as build_claim_local
from .export_dpo_pairs import (
    ABSENT_REJECTED,
    convert,
    rejected_answer,
    to_dpo_row,
)
from .forget_score import (
    attach_gold_before,
    gold_before_for_probe,
    matches_gold,
    summarize_forget,
    write_gold_before_bank,
)
from .summarize_run import aggregate


class GoldBeforeTest(unittest.TestCase):
    def test_changed_constraint_introduced_is_stable_status(self) -> None:
        probe = {
            "id": "c1:delta",
            "drift_type": "changed",
            "probe_form": "version_delta",
            "display_entity": "CacheConfig.size",
            "gold": {
                "must_contain_any": [
                    ["ge=1", "ge 1", ">= 1", "greater than or equal to 1"]
                ]
            },
            "expected_change": {
                "dimension": "constraints",
                "changes": {"ge": {"before": None, "after": 1}},
            },
        }
        before = gold_before_for_probe(probe)
        self.assertIsNotNone(before)
        assert before is not None
        self.assertEqual(before["must_contain"], ["CacheConfig.size"])
        self.assertIn("unchanged", before["must_contain_any"][0])

    def test_removed_delta_stale_is_stable_status(self) -> None:
        probe = {
            "id": "r1:delta",
            "drift_type": "removed",
            "probe_form": "version_delta",
            "display_entity": "VLLM_RPC_TIMEOUT",
            "gold": {
                "must_contain": ["VLLM_RPC_TIMEOUT"],
                "must_contain_any": [["removed", "no longer", "only"], ["0.22.0"]],
            },
        }
        before = gold_before_for_probe(probe)
        self.assertIsNotNone(before)
        assert before is not None
        self.assertEqual(before["must_contain"], ["VLLM_RPC_TIMEOUT"])
        self.assertTrue(
            matches_gold("VLLM_RPC_TIMEOUT is unchanged across both releases.", before)
        )
        self.assertFalse(
            matches_gold("VLLM_RPC_TIMEOUT was removed; only in 0.22.0.", before)
        )

    def test_removed_existence_after_stale_is_yes(self) -> None:
        probe = {
            "id": "r1:exists:0.23.0",
            "drift_type": "removed",
            "probe_form": "versioned_existence",
            "gold": {"boolean": "no"},
        }
        self.assertEqual(gold_before_for_probe(probe), {"boolean": "yes"})

    def test_stable_and_added_have_no_gold_before(self) -> None:
        for drift in ("stable", "added"):
            probe = {
                "id": f"{drift}:delta",
                "drift_type": drift,
                "probe_form": "version_delta",
                "gold": {"must_contain_any": [["unchanged", "stable", "both"]]},
            }
            self.assertIsNone(gold_before_for_probe(probe))


class ForgetSummaryTest(unittest.TestCase):
    def test_forget_and_remember_rates(self) -> None:
        probes = [
            {
                "id": "changed:1",
                "drift_type": "changed",
                "probe_form": "version_delta",
                "display_entity": "X",
                "gold": {"must_contain_any": [["ge=1", ">= 1"]]},
                "expected_change": {
                    "dimension": "constraints",
                    "changes": {"ge": {"before": None, "after": 1}},
                },
            },
            {
                "id": "removed:1",
                "drift_type": "removed",
                "probe_form": "version_delta",
                "display_entity": "Y",
                "gold": {
                    "must_contain": ["Y"],
                    "must_contain_any": [["removed", "no longer"], ["0.22.0"]],
                },
            },
            {
                "id": "stable:1",
                "drift_type": "stable",
                "probe_form": "version_delta",
                "display_entity": "Z",
                "gold": {
                    "must_contain": ["Z"],
                    "must_contain_any": [["unchanged", "stable", "both"]],
                },
            },
        ]
        samples = [
            # stale on changed
            {"id": "changed:1", "output": "X is unchanged.", "content_score": 0.0},
            # remembers removed
            {
                "id": "removed:1",
                "output": "Y was removed; only in 0.22.0.",
                "content_score": 1.0,
            },
            # stable ignored by forget denominator
            {
                "id": "stable:1",
                "output": "Z is unchanged / stable in both.",
                "content_score": 1.0,
            },
        ]
        summary = summarize_forget(samples=samples, probes=probes)
        self.assertEqual(summary["n_scored"], 2)
        self.assertEqual(summary["forget_rate"], 0.5)
        self.assertEqual(summary["remember_rate"], 0.5)
        self.assertEqual(summary["by_drift_type"]["changed"]["forget_rate"], 1.0)
        self.assertEqual(summary["by_drift_type"]["removed"]["remember_rate"], 1.0)

    def test_summarize_run_embeds_forget_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probes_path = root / "probes.jsonl"
            samples_path = root / "samples.jsonl"
            out_path = root / "summary.json"
            probes = [
                {
                    "id": "removed:1",
                    "drift_type": "removed",
                    "entity_type": "env_var",
                    "probe_form": "version_delta",
                    "display_entity": "Y",
                    "gold": {
                        "must_contain": ["Y"],
                        "must_contain_any": [["removed", "no longer"], ["0.22.0"]],
                    },
                }
            ]
            samples = [
                {
                    "id": "removed:1",
                    "output": "Y is unchanged and stable.",
                    "content_score": 0.0,
                }
            ]
            probes_path.write_text(
                json.dumps(probes[0]) + "\n", encoding="utf-8"
            )
            samples_path.write_text(
                json.dumps(samples[0]) + "\n", encoding="utf-8"
            )
            result = aggregate(
                samples_path=samples_path,
                probes_path=probes_path,
                out_path=out_path,
            )
            self.assertEqual(result["schema"], "delta.eval_factory.run_summary.v2")
            self.assertEqual(result["forget"]["n_scored"], 1)
            self.assertEqual(result["forget"]["forget_rate"], 1.0)
            self.assertEqual(result["forget"]["remember_rate"], 0.0)

    def test_write_gold_before_bank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probes_path = root / "probes.jsonl"
            out_path = root / "bank.jsonl"
            probes_path.write_text(
                json.dumps(
                    {
                        "id": "removed:1",
                        "drift_type": "removed",
                        "probe_form": "version_delta",
                        "display_entity": "Y",
                        "gold": {
                            "must_contain": ["Y"],
                            "must_contain_any": [["removed"], ["0.22.0"]],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            meta = write_gold_before_bank(probes_path=probes_path, out_path=out_path)
            self.assertEqual(meta["n_with_gold_before"], 1)
            bank = attach_gold_before(
                [json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])]
            )
            self.assertIn("gold_before", bank[0])
            self.assertIn("gold_after", bank[0])


class DpoExportTest(unittest.TestCase):
    def test_changed_pair_prefers_after_rejects_before(self) -> None:
        probe = {
            "id": "changed:1",
            "claim_id": "c1",
            "drift_type": "changed",
            "probe_form": "version_delta",
            "question": "What changed for X?",
            "display_entity": "X",
            "gold": {"must_contain_any": [["ge=1", ">= 1"]]},
            "expected_change": {
                "dimension": "constraints",
                "changes": {"ge": {"before": None, "after": 1}},
            },
        }
        row = to_dpo_row(probe, random.Random(0))
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["prompt"][0]["role"], "user")
        self.assertIn("ge=1", row["chosen"][0]["content"])
        self.assertIn("unchanged", row["rejected"][0]["content"].lower())

    def test_added_rejects_absent(self) -> None:
        probe = {
            "id": "added:1",
            "drift_type": "added",
            "probe_form": "version_delta",
            "question": "What happened to FLAG?",
            "gold": {
                "must_contain": ["FLAG"],
                "must_contain_any": [["added", "new"], ["0.23.0"]],
            },
        }
        self.assertEqual(rejected_answer(probe, random.Random(0)), ABSENT_REJECTED)

    def test_stable_rejects_wrong_default(self) -> None:
        probe = {
            "id": "stable:1",
            "drift_type": "stable",
            "probe_form": "versioned_existence",
            "question": "Does FLAG exist?",
            "gold": {"boolean": "yes"},
        }
        rejected = rejected_answer(probe, random.Random(0))
        self.assertEqual(rejected, "No.")

    def test_convert_writes_slim_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probes_path = root / "probes.jsonl"
            out_path = root / "dpo.jsonl"
            rows = [
                {
                    "id": "changed:1",
                    "drift_type": "changed",
                    "probe_form": "version_delta",
                    "question": "What changed for X?",
                    "display_entity": "X",
                    "gold": {"must_contain_any": [["ge=1"]]},
                    "expected_change": {
                        "dimension": "constraints",
                        "changes": {"ge": {"before": None, "after": 1}},
                    },
                },
                {
                    "id": "added:1",
                    "drift_type": "added",
                    "probe_form": "version_delta",
                    "question": "What happened to FLAG?",
                    "gold": {
                        "must_contain": ["FLAG"],
                        "must_contain_any": [["added"], ["0.23.0"]],
                    },
                },
                {
                    "id": "stable:1",
                    "drift_type": "stable",
                    "probe_form": "versioned_existence",
                    "question": "Does Z exist?",
                    "gold": {"boolean": "yes"},
                },
            ]
            probes_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            meta = convert(
                probes_path=probes_path,
                out_path=out_path,
                delta_copies=2,
                stable_cap=1,
                seed=1,
            )
            self.assertGreater(meta["n"], 0)
            slim = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(set(slim[0]), {"prompt", "chosen", "rejected"})
            self.assertIn("changed", meta["by_drift"])


class ClaimLocalEditsTest(unittest.TestCase):
    def test_orders_changed_before_added_and_caps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probes_path = root / "probes.jsonl"
            out_dir = root / "edits"
            rows = []
            for drift, cid in (
                ("added", "claim:added"),
                ("changed", "claim:changed"),
                ("removed", "claim:removed"),
            ):
                for i in range(3):
                    rows.append(
                        {
                            "id": f"{cid}:{i}",
                            "claim_id": cid,
                            "drift_type": drift,
                            "probe_form": "versioned_existence",
                            "question": f"Q {cid} {i}?",
                            "gold": {"boolean": "yes" if drift != "removed" else "no"},
                        }
                    )
            for i in range(40):
                rows.append(
                    {
                        "id": f"stable:{i}",
                        "claim_id": f"stable:{i}",
                        "drift_type": "stable",
                        "probe_form": "versioned_existence",
                        "question": f"Stable {i}?",
                        "gold": {"boolean": "yes"},
                    }
                )
            probes_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            meta = build_claim_local(
                probes_path=probes_path,
                out_dir=out_dir,
                max_claims=2,
                claim_copies=2,
                stable_replay_n=8,
                canary_n=10,
                seed=7,
            )
            self.assertEqual(meta["n_edits"], 2)
            self.assertEqual(meta["edits"][0]["primary_drift"], "changed")
            self.assertEqual(meta["edits"][1]["primary_drift"], "removed")
            self.assertIn("claim:added", meta["deferred_claim_ids"])
            self.assertTrue(Path(meta["canary_path"]).exists())
            train = Path(meta["edits"][0]["train_path"])
            self.assertTrue(train.exists())
            n_pairs = sum(1 for line in train.read_text().splitlines() if line.strip())
            self.assertEqual(n_pairs, meta["edits"][0]["n_pairs"])


if __name__ == "__main__":
    unittest.main()
