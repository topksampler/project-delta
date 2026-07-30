from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

import yaml

from experiments.delta_v2.train_knowledge_lora import (
    KnowledgeTrainError,
    _load_jsonl,
)
from experiments.delta_v2.train_knowledge_stability_replay import (
    ACQUISITION_PAIRS_PATH,
    ACQUISITION_TRAIN_PATH,
    REPLAY_PAIRS_PATH,
    REPLAY_TRAIN_PATH,
    build_stability_schedule,
    validate_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPO_ROOT
    / "experiments/delta_v2/"
    "knowledge_stability_replay_lora_protocol.yaml"
)


class StabilityReplayTrainTests(unittest.TestCase):
    def test_schedule_has_exact_fixed_budget_and_microbatch_shapes(self) -> None:
        schedule = build_stability_schedule(
            _load_jsonl(REPO_ROOT / ACQUISITION_TRAIN_PATH),
            _load_jsonl(REPO_ROOT / ACQUISITION_PAIRS_PATH),
            _load_jsonl(REPO_ROOT / REPLAY_TRAIN_PATH),
            _load_jsonl(REPO_ROOT / REPLAY_PAIRS_PATH),
        )
        self.assertEqual(len(schedule), 240)
        self.assertEqual(
            Counter(unit["stream"] for unit in schedule),
            {
                "acquisition_pair": 135,
                "acquisition_recall": 45,
                "replay_pair": 45,
                "replay_recall": 15,
            },
        )
        steps = [schedule[index : index + 4] for index in range(0, 240, 4)]
        self.assertEqual(
            Counter(
                tuple(unit["stream"] for unit in step)
                for step in steps
            ),
            {
                (
                    "acquisition_pair",
                    "acquisition_pair",
                    "acquisition_recall",
                    "replay_pair",
                ): 45,
                (
                    "acquisition_pair",
                    "acquisition_pair",
                    "acquisition_pair",
                    "replay_recall",
                ): 15,
            },
        )

    def test_full_eval_authorization_mutation_fails_closed(self) -> None:
        protocol = yaml.safe_load(
            PROTOCOL_PATH.read_text(encoding="utf-8")
        )
        protocol["execution_boundary"]["full_eval_authorized"] = True
        with self.assertRaisesRegex(
            KnowledgeTrainError, "execution boundary changed"
        ):
            validate_protocol(protocol, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
