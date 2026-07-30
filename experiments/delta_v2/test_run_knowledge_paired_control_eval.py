from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from experiments.delta_v2.run_knowledge_paired_control_eval import (
    PairedControlEvalError,
    validate_config,
)
from experiments.delta_v2.run_knowledge_eval import (
    audit_requests,
    build_requests,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/delta_v2/"
    "d2_paired_data_sft_control_eval_modal_v1.yaml"
)


class KnowledgePairedControlEvalTests(unittest.TestCase):
    def test_extension_uses_unchanged_169_item_recipe(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        extension, parent, probes = validate_config(
            config, repo_root=REPO_ROOT
        )
        requests = build_requests(
            config=config,
            protocol=parent,
            probes=probes,
        )
        audit = audit_requests(requests)
        self.assertEqual(len(probes), 169)
        self.assertEqual(len(requests), 338)
        self.assertEqual(audit["model_visible_fields"], ["prompt"])
        self.assertFalse(audit["gold_visible"])
        self.assertEqual(
            extension["parent_eval"]["only_extension"],
            "target-condition-and-adapter-lineage",
        )
        self.assertTrue(extension["decision_trigger"]["first_gate_passed"])

    def test_generation_recipe_is_exact_parent_protocol(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        _extension, parent, _probes = validate_config(
            config, repo_root=REPO_ROOT
        )
        self.assertEqual(
            parent["generation"],
            {
                "method": "greedy",
                "do_sample": False,
                "max_new_tokens": 256,
                "repeats": 2,
                "seed": 20260730,
                "repeat_invariant": (
                    "byte-identical-raw-continuation"
                ),
            },
        )
        self.assertEqual(parent["reporting"]["pooled_overall_score"], "forbidden")

    def test_adapter_mutation_fails_closed(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        config["model"]["adapter"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            PairedControlEvalError, "model changed"
        ):
            validate_config(config, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
