from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import yaml

from experiments.delta_v2.knowledge_paired_data import (
    CONTRACT_PATH,
    FREEZE_PATH,
    KnowledgePairedDataError,
    build_paired_data,
    validate_contract,
    validate_freeze,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (
    REPO_ROOT / "data/experiments/delta_v2/knowledge_paired_v2"
)


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


class KnowledgePairedDataTests(unittest.TestCase):
    def test_build_matches_frozen_outputs_and_shortcut_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summary = build_paired_data(
                contract_path=REPO_ROOT / CONTRACT_PATH,
                repo_root=REPO_ROOT,
                output_root_override=output,
            )
            self.assertEqual(
                (output / "train.jsonl").read_bytes(),
                (OUTPUT_ROOT / "train.jsonl").read_bytes(),
            )
            self.assertEqual(
                (output / "pairs.jsonl").read_bytes(),
                (OUTPUT_ROOT / "pairs.jsonl").read_bytes(),
            )
            audit = summary["shortcut_audit"]
            self.assertEqual(audit["frozen_eval_prompt_overlap"], 0)
            self.assertEqual(audit["original_eval_prompt_overlap"], 0)
            self.assertEqual(audit["annotation_only_negative_share"], 1 / 3)
            self.assertLessEqual(
                audit["maximum_empirical_value_only_accuracy"], 0.60
            )

    def test_every_pair_is_balanced_grounded_and_changes_contract(self) -> None:
        rows = _jsonl(OUTPUT_ROOT / "train.jsonl")
        pairs = _jsonl(OUTPUT_ROOT / "pairs.jsonl")
        by_id = {row["row_id"]: row for row in rows}
        self.assertEqual(
            Counter(row["surface"] for row in rows),
            {
                "exact_recall": 21,
                "verify_true": 63,
                "verify_false": 63,
            },
        )
        self.assertEqual(len(pairs), 63)
        self.assertEqual(len({pair["pair_id"] for pair in pairs}), 63)
        for pair in pairs:
            positive = by_id[pair["positive_row_id"]]
            negative = by_id[pair["negative_row_id"]]
            self.assertEqual(positive["pair_id"], pair["pair_id"])
            self.assertEqual(negative["pair_id"], pair["pair_id"])
            self.assertEqual(positive["pair_role"], "positive")
            self.assertEqual(negative["pair_role"], "negative")
            self.assertEqual(positive["messages"][1]["content"], "yes")
            self.assertEqual(negative["messages"][1]["content"], "no")
            self.assertNotEqual(
                pair["source_id"], pair["donor_source_id"]
            )

    def test_contract_hash_mutation_fails_closed(self) -> None:
        contract = yaml.safe_load(
            (REPO_ROOT / CONTRACT_PATH).read_text(encoding="utf-8")
        )
        contract["shortcut_gates"][
            "empirical_value_only_accuracy_maximum_per_family"
        ] = 0.75
        with self.assertRaisesRegex(
            KnowledgePairedDataError, "shortcut gates changed"
        ):
            validate_contract(contract, repo_root=REPO_ROOT)

    def test_freeze_binds_exact_reproducible_bytes(self) -> None:
        freeze = yaml.safe_load(
            (REPO_ROOT / FREEZE_PATH).read_text(encoding="utf-8")
        )
        validate_freeze(freeze, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
