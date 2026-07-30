from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import yaml

from experiments.delta_v2.run_knowledge_stability_base import (
    EVAL_PATH,
    PROTOCOL_PATH,
    StabilityBaseError,
    execute,
    validate_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/delta_v2/"
    "d4_stability_base_qwen35_08b_modal_v1.yaml"
)


class GoldBackend:
    def __init__(self) -> None:
        rows = [
            json.loads(line)
            for line in (REPO_ROOT / EVAL_PATH)
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.gold = {row["probe_id"]: row["gold"] for row in rows}

    def generate(self, request: Mapping[str, Any]) -> tuple[str, str]:
        value = self.gold[str(request["probe_id"])]
        if isinstance(value, dict):
            return json.dumps(
                value,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ), "stop"
        return str(value), "stop"

    def runtime_receipt(self) -> Mapping[str, Any]:
        return {"backend": "gold-test"}


class StabilityBaseTests(unittest.TestCase):
    def test_gold_backend_scores_each_cell_without_pooling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = execute(
                config_path=CONFIG_PATH,
                repo_root=REPO_ROOT,
                backend=GoldBackend(),
                output_dir_override=Path(temporary),
            )
        metrics = result["metrics"]
        self.assertIsNone(metrics["pooled_overall_accuracy"])
        self.assertEqual(set(metrics["cells"]), {
            "choice",
            "boolean_true",
            "boolean_false",
            "recall",
        })
        self.assertTrue(
            all(
                cell["exact_accuracy"] == 1.0
                for cell in metrics["cells"].values()
            )
        )
        self.assertEqual(result["receipt"]["model_updates"], 0)

    def test_training_authorization_mutation_fails_closed(self) -> None:
        protocol = yaml.safe_load(
            (REPO_ROOT / PROTOCOL_PATH).read_text(encoding="utf-8")
        )
        protocol["execution_boundary"]["lora_training_authorized"] = True
        with self.assertRaisesRegex(
            StabilityBaseError, "execution boundary changed"
        ):
            validate_protocol(protocol, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
