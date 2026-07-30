from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from experiments.delta_v2.run_knowledge_eval import (
    audit_requests,
    build_requests,
)
from experiments.delta_v2.run_knowledge_paired_objective_eval import (
    PairedObjectiveEvalError,
    validate_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/delta_v2/"
    "d3_paired_objective_eval_modal_v1.yaml"
)


class KnowledgePairedObjectiveEvalTests(unittest.TestCase):
    def test_extension_uses_unchanged_parent_recipe(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        extension, parent, probes = validate_config(
            config, repo_root=REPO_ROOT
        )
        requests = build_requests(
            config=config, protocol=parent, probes=probes
        )
        audit = audit_requests(requests)
        self.assertEqual((len(probes), len(requests)), (169, 338))
        self.assertEqual(audit["model_visible_fields"], ["prompt"])
        self.assertFalse(audit["gold_visible"])
        self.assertEqual(
            extension["parent_eval"]["only_extension"],
            "target-condition-and-adapter-lineage",
        )
        self.assertEqual(
            extension["decision_trigger"]["truth_conditioned_pairs"],
            "59/63",
        )

    def test_original_comparison_gates_are_bound(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        extension, _parent, _probes = validate_config(
            config, repo_root=REPO_ROOT
        )
        gates = extension["comparison_gates"]
        self.assertEqual(
            (
                gates["acquisition_choice_accuracy_minimum"],
                gates["acquisition_boolean_true_accuracy_minimum"],
                gates["acquisition_boolean_false_accuracy_minimum"],
                gates["retention_boolean_false_accuracy_minimum"],
            ),
            (0.5, 0.8, 0.8, 0.9),
        )
        self.assertEqual(gates["pooled_overall_score"], "forbidden")

    def test_adapter_mutation_fails_closed(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        config["model"]["adapter"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            PairedObjectiveEvalError, "model changed"
        ):
            validate_config(config, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
