from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import yaml

from experiments.delta_v2.run_knowledge_eval import (
    BASE_RUN_ID,
    KnowledgeEvalError,
    PROTOCOL_PATH,
    _load_yaml,
    audit_requests,
    build_requests,
    execute,
    score_one,
    validate_protocol,
    validate_run_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = (
    REPO_ROOT
    / "configs/experiments/delta_v2/"
    "c4_knowledge_base_qwen35_08b_modal_v1.yaml"
)


class FakeBackend:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    def generate(self, request: Mapping[str, Any]) -> tuple[str, str]:
        return next(self._responses), "stop"

    def runtime_receipt(self) -> Mapping[str, Any]:
        return {"backend": "fake-deterministic"}


class KnowledgeEvalTests(unittest.TestCase):
    def test_protocol_has_two_separate_scoreboards(self) -> None:
        protocol = _load_yaml(REPO_ROOT / PROTOCOL_PATH)
        rows = validate_protocol(protocol, repo_root=REPO_ROOT)
        self.assertEqual(len(rows), 169)
        self.assertEqual(
            {row["stratum"] for row in rows},
            {
                "acquisition_added",
                "retention_stable",
                "feature_retention",
            },
        )
        self.assertEqual(
            protocol["reporting"]["pooled_overall_score"],
            "forbidden",
        )

    def test_base_config_and_request_boundary(self) -> None:
        config = _load_yaml(BASE_CONFIG)
        protocol, probes = validate_run_config(
            config,
            repo_root=REPO_ROOT,
        )
        requests = build_requests(
            config=config,
            protocol=protocol,
            probes=probes,
        )
        audit = audit_requests(requests)
        self.assertEqual(config["run_id"], BASE_RUN_ID)
        self.assertEqual(len(requests), 338)
        self.assertFalse(audit["gold_visible"])
        self.assertEqual(
            set(requests[0]["model_input"]),
            {
                "messages",
                "add_generation_prompt",
                "chat_template_kwargs",
            },
        )

    def test_exact_scorers_do_not_repair(self) -> None:
        choice = {"scorer": "exact-choice-v1", "gold": "A"}
        boolean = {"scorer": "exact-boolean-v1", "gold": "yes"}
        structured = {"scorer": "exact-json-v1", "gold": {"x": 1}}
        self.assertEqual(score_one(choice, " A\n")[:2], (True, True))
        self.assertEqual(score_one(choice, "A.")[:2], (False, False))
        self.assertEqual(score_one(boolean, "Yes")[:2], (False, False))
        self.assertEqual(
            score_one(structured, '{"x":1}')[:2],
            (True, True),
        )
        self.assertEqual(
            score_one(structured, '```json\n{"x":1}\n```')[:2],
            (False, False),
        )

    def test_fake_perfect_execution_writes_after_scoring(self) -> None:
        config = _load_yaml(BASE_CONFIG)
        protocol, probes = validate_run_config(
            config,
            repo_root=REPO_ROOT,
        )
        responses = [
            (
                json.dumps(probe["gold"], separators=(",", ":"))
                if isinstance(probe["gold"], dict)
                else str(probe["gold"])
            )
            for probe in probes
            for _ in range(int(protocol["generation"]["repeats"]))
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result = execute(
                config_path=BASE_CONFIG,
                repo_root=REPO_ROOT,
                backend=FakeBackend(responses),
                output_dir_override=output_dir,
            )
            self.assertTrue(result["metrics"]["deterministic_repeats"])
            self.assertIsNone(
                result["metrics"]["pooled_overall_accuracy"]
            )
            self.assertTrue((output_dir / "samples.jsonl").is_file())
            self.assertTrue((output_dir / "metrics.json").is_file())
            self.assertTrue((output_dir / "run_receipt.json").is_file())

    def test_nondeterministic_repeats_write_nothing(self) -> None:
        config = _load_yaml(BASE_CONFIG)
        protocol, probes = validate_run_config(
            config,
            repo_root=REPO_ROOT,
        )
        responses: list[str] = []
        for probe in probes:
            gold = (
                json.dumps(probe["gold"], separators=(",", ":"))
                if isinstance(probe["gold"], dict)
                else str(probe["gold"])
            )
            responses.extend([gold, gold])
        responses[1] = "different"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with self.assertRaisesRegex(
                KnowledgeEvalError,
                "nondeterministic",
            ):
                execute(
                    config_path=BASE_CONFIG,
                    repo_root=REPO_ROOT,
                    backend=FakeBackend(responses),
                    output_dir_override=output_dir,
                )
            self.assertEqual(list(output_dir.iterdir()), [])

    def test_changed_eval_hash_fails_closed(self) -> None:
        config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
        config["eval"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            KnowledgeEvalError,
            "eval binding changed",
        ):
            validate_run_config(config, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
