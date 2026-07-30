from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import yaml

from experiments.delta_v2.knowledge_stability_replay_data import (
    CONTRACT_PATH,
    FREEZE_PATH,
    StabilityReplayDataError,
    build_replay_data,
    validate_contract,
    validate_freeze,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (
    REPO_ROOT
    / "data/experiments/delta_v2/knowledge_stability_replay_v1"
)


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


class StabilityReplayDataTests(unittest.TestCase):
    def test_rebuild_matches_frozen_outputs_and_passes_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summary = build_replay_data(
                contract_path=REPO_ROOT / CONTRACT_PATH,
                repo_root=REPO_ROOT,
                output_root_override=output,
            )
            for name in (
                "replay_train.jsonl",
                "replay_pairs.jsonl",
                "stability_dev.jsonl",
            ):
                self.assertEqual(
                    (output / name).read_bytes(),
                    (OUTPUT_ROOT / name).read_bytes(),
                )
            audit = summary["shortcut_audit"]
            self.assertEqual(set(audit["source_overlaps"].values()), {0})
            self.assertEqual(set(audit["prompt_overlaps"].values()), {0})
            self.assertLessEqual(
                audit["maximum_empirical_value_only_accuracy"], 0.60
            )
            self.assertEqual(
                audit["empirical_prompt_length_only_accuracy"], 0.50
            )
            self.assertEqual(
                summary["execution_boundary"]["model_invocations_completed"],
                0,
            )

    def test_rows_are_balanced_and_dev_is_source_disjoint(self) -> None:
        rows = _jsonl(OUTPUT_ROOT / "replay_train.jsonl")
        pairs = _jsonl(OUTPUT_ROOT / "replay_pairs.jsonl")
        dev = _jsonl(OUTPUT_ROOT / "stability_dev.jsonl")
        self.assertEqual(
            Counter(row["surface"] for row in rows),
            {
                "replay_recall": 15,
                "replay_verify_true": 45,
                "replay_verify_false": 45,
            },
        )
        self.assertEqual(
            Counter(row["probe_kind"] for row in dev),
            {
                "choice": 15,
                "boolean_true": 15,
                "boolean_false": 15,
                "recall": 15,
            },
        )
        self.assertEqual(len(pairs), 45)
        self.assertFalse(
            {row["source_id"] for row in rows}
            & {row["source_id"] for row in dev}
        )
        by_id = {row["row_id"]: row for row in rows}
        for pair in pairs:
            self.assertEqual(
                len(
                    by_id[pair["positive_row_id"]]["messages"][0]["content"]
                ),
                len(
                    by_id[pair["negative_row_id"]]["messages"][0]["content"]
                ),
            )
            self.assertEqual(
                by_id[pair["positive_row_id"]]["messages"][1]["content"],
                "yes",
            )
            self.assertEqual(
                by_id[pair["negative_row_id"]]["messages"][1]["content"],
                "no",
            )
            self.assertNotEqual(
                pair["source_id"], pair["donor_source_id"]
            )

    def test_contract_gate_mutation_fails_closed(self) -> None:
        contract = yaml.safe_load(
            (REPO_ROOT / CONTRACT_PATH).read_text(encoding="utf-8")
        )
        contract["execution_boundary"]["lora_training_authorized"] = True
        with self.assertRaisesRegex(
            StabilityReplayDataError, "execution boundary changed"
        ):
            validate_contract(contract, repo_root=REPO_ROOT)

    def test_freeze_binds_exact_reproducible_bytes(self) -> None:
        freeze = yaml.safe_load(
            (REPO_ROOT / FREEZE_PATH).read_text(encoding="utf-8")
        )
        validate_freeze(freeze, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
