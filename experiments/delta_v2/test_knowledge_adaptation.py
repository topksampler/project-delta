from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.delta_v2.knowledge_adaptation import (
    CONTRACT_ID,
    KnowledgeAdaptationError,
    _decode_canonical_ast,
    build_datasets,
    canonical_json,
    distractor_cards,
    knowledge_card,
    validate_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    REPO_ROOT / "experiments/delta_v2/knowledge_adaptation_contract.yaml"
)


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


class KnowledgeAdaptationTests(unittest.TestCase):
    def test_contract_binds_frozen_truth(self) -> None:
        contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
        facts, selection, feature, output = validate_contract(
            contract,
            repo_root=REPO_ROOT,
        )
        self.assertTrue(facts.is_file())
        self.assertTrue(selection.is_file())
        self.assertTrue(feature.is_file())
        self.assertEqual(
            output,
            Path("data/experiments/delta_v2/knowledge_adaptation_v1"),
        )

    def test_canonical_ast_round_trip_renders_python(self) -> None:
        payload = {
            "node": "BinOp",
            "fields": {
                "left": {
                    "node": "Name",
                    "fields": {
                        "id": "int",
                        "ctx": {"node": "Load", "fields": {}},
                    },
                },
                "op": {"node": "BitOr", "fields": {}},
                "right": {
                    "node": "Constant",
                    "fields": {"value": None, "kind": None},
                },
            },
        }
        import ast

        node = _decode_canonical_ast(payload)
        self.assertEqual(ast.unparse(ast.fix_missing_locations(node)), "int | None")

    def test_cards_and_distractors_are_distinct(self) -> None:
        facts_path = (
            REPO_ROOT
            / "data/experiments/delta_v2/acceptance_attempt_2/facts/"
            "atomic_fact_deltas_acceptance.jsonl"
        )
        added = [
            row
            for row in _jsonl(facts_path)
            if row["status"] == "added"
        ]
        self.assertEqual(len(added), 21)
        for record in added:
            card = knowledge_card(record)
            distractors = distractor_cards(record, card)
            encoded = {canonical_json(card)}
            encoded.update(canonical_json(row) for row in distractors)
            self.assertEqual(len(encoded), 4)

    def test_build_is_deterministic_and_leak_closed(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = Path(first_dir)
            second = Path(second_dir)
            summary_first = build_datasets(
                contract_path=CONTRACT_PATH,
                repo_root=REPO_ROOT,
                output_root_override=first,
            )
            summary_second = build_datasets(
                contract_path=CONTRACT_PATH,
                repo_root=REPO_ROOT,
                output_root_override=second,
            )
            self.assertEqual(summary_first, summary_second)
            for filename in ("train.jsonl", "dev.jsonl", "eval.jsonl"):
                self.assertEqual(
                    (first / filename).read_bytes(),
                    (second / filename).read_bytes(),
                )

            train = _jsonl(first / "train.jsonl")
            dev = _jsonl(first / "dev.jsonl")
            probes = _jsonl(first / "eval.jsonl")
            self.assertEqual(len(train), 63)
            self.assertEqual(len(dev), 21)
            self.assertEqual(len(probes), 169)
            self.assertEqual(
                {row["stratum"] for row in probes},
                {
                    "acquisition_added",
                    "retention_stable",
                    "feature_retention",
                },
            )
            self.assertEqual(
                summary_first["evaluation"]["acquisition_source_overlap"],
                21,
            )
            self.assertEqual(
                summary_first["evaluation"]["retention_source_overlap"],
                0,
            )
            self.assertEqual(
                summary_first["evaluation"]["exact_prompt_overlap"],
                0,
            )
            self.assertIsNone(
                summary_first["evaluation"]["pooled_overall_score"]
            )

    def test_changed_contract_fails_closed(self) -> None:
        contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
        contract["contract_id"] = CONTRACT_ID + "-changed"
        with self.assertRaisesRegex(
            KnowledgeAdaptationError,
            "identity changed",
        ):
            validate_contract(contract, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
