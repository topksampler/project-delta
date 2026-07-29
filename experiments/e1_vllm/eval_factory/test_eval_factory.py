from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .cli_flags import extract_cli_flags
from .entities import extract_config_fields, extract_env_contracts
from .factory import build_claims, build_probes
from .generate_surfaces import _clean, _validate_surface


class CliFlagExtractionTest(unittest.TestCase):
    def test_extracts_static_and_dynamic_contracts_with_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp)
            source = snapshot / "vllm" / "entrypoints" / "cli.py"
            source.parent.mkdir(parents=True)
            source.write_text(
                "\n".join(
                    [
                        "DEFAULT = 8",
                        "def register(parser):",
                        "    parser.add_argument(",
                        "        '--alpha',",
                        "        default=DEFAULT,",
                        "        choices=['x', 'y'],",
                        "        help='Alpha mode',",
                        "    )",
                    ]
                ),
                encoding="utf-8",
            )

            rows = extract_cli_flags(snapshot)

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["entity"], "--alpha")
            self.assertEqual(row["scope"], "register")
            self.assertEqual(row["contract"]["choices"], ["x", "y"])
            self.assertEqual(row["contract"]["default"], {"dynamic": "DEFAULT"})
            self.assertEqual(row["evidence"]["line_start"], 3)
            self.assertEqual(row["verifier"]["result"], "pass")


class ClaimDiffTest(unittest.TestCase):
    @staticmethod
    def _row(entity: str, default: int, path: str) -> dict:
        return {
            "entity_type": "cli_flag",
            "entity": entity,
            "scope": "register",
            "contract": {
                "default": default,
                "choices": None,
                "required": None,
                "action": None,
                "help": "help",
            },
            "evidence": {
                "source_path": path,
                "line_start": 1,
                "line_end": 1,
                "source_sha256": "abc",
            },
            "verifier": {"id": "python_ast_argparse_v1", "result": "pass"},
        }

    def test_diff_classifies_stable_changed_added_and_removed(self) -> None:
        before = [
            self._row("--stable", 1, "old.py"),
            self._row("--changed", 1, "old.py"),
            self._row("--removed", 1, "old.py"),
        ]
        after = [
            self._row("--stable", 1, "new.py"),
            self._row("--changed", 2, "new.py"),
            self._row("--added", 1, "new.py"),
        ]

        claims = build_claims(
            before_rows=before,
            after_rows=after,
            before_spec={"git_tag": "v1"},
            after_spec={"git_tag": "v2"},
        )
        status = {row["entity"]: row["status"] for row in claims}

        self.assertEqual(
            status,
            {
                "--added": "added",
                "--changed": "changed",
                "--removed": "removed",
                "--stable": "stable",
            },
        )
        self.assertEqual(
            {row["split"] for row in claims} - {"train", "dev", "eval"},
            set(),
        )

    def test_delta_stratum_reserves_evaluation_claims(self) -> None:
        after = [
            self._row(f"--added-{index}", 1, "new.py") for index in range(5)
        ]
        claims = build_claims(
            before_rows=[],
            after_rows=after,
            before_spec={"git_tag": "v1"},
            after_spec={"git_tag": "v2"},
        )
        counts = {
            split: sum(row["split"] == split for row in claims)
            for split in ("train", "dev", "eval")
        }
        self.assertEqual(counts, {"train": 2, "dev": 1, "eval": 2})

    def test_changed_constraint_probe_requires_the_actual_delta(self) -> None:
        def config_row(constraints: dict) -> dict:
            return {
                **self._row("vllm.config.CacheConfig.size", 1, "cache.py"),
                "entity_type": "config_field",
                "display_entity": "CacheConfig.size",
                "scope": "vllm.config.CacheConfig",
                "contract": {
                    "annotation": "int",
                    "default": 1,
                    "constraints": constraints,
                    "description": "Size.",
                },
            }

        claims = build_claims(
            before_rows=[config_row({})],
            after_rows=[config_row({"ge": 1})],
            before_spec={"git_tag": "v1"},
            after_spec={"git_tag": "v2"},
        )
        probe = build_probes(
            claims,
            before_version="1.0.0",
            after_version="2.0.0",
        )[2]

        self.assertIn("validation constraint", probe["question"])
        self.assertEqual(probe["expected_change"]["dimension"], "constraints")
        self.assertIn(">= 1", probe["gold"]["must_contain_any"][0])


class SurfaceValidationTest(unittest.TestCase):
    def test_delta_question_must_not_presuppose_change(self) -> None:
        seed = {
            "claim_id": "vllm:cli_flag:--alpha",
            "display_entity": "--alpha",
            "probe_form": "version_delta",
            "question": (
                "Between vLLM 0.22.0 and 0.23.0, what happened to "
                "the CLI flag `--alpha`?"
            ),
        }
        bad = "How did `--alpha` change between vLLM 0.22.0 and 0.23.0?"
        good = (
            "Did `--alpha` differ between vLLM 0.22.0 and 0.23.0, "
            "and if so, how?"
        )

        self.assertEqual(
            _validate_surface(seed, bad, 1.0),
            "delta_question_presupposes_change",
        )
        self.assertIsNone(_validate_surface(seed, good, 1.0))

    def test_cleanup_keeps_only_the_question(self) -> None:
        text = "Does vLLM expose `--alpha`? Please respond with yes or no."
        self.assertEqual(_clean(text), "Does vLLM expose `--alpha`?")


class MultiFamilyExtractionTest(unittest.TestCase):
    def test_env_registry_and_config_constraints_are_static(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp)
            package = snapshot / "vllm"
            config_dir = package / "config"
            config_dir.mkdir(parents=True)
            (package / "envs.py").write_text(
                "\n".join(
                    [
                        "import os",
                        "environment_variables = {",
                        "  'VLLM_ALPHA': lambda: os.getenv('VLLM_ALPHA', 'x'),",
                        "}",
                    ]
                ),
                encoding="utf-8",
            )
            (config_dir / "cache.py").write_text(
                "\n".join(
                    [
                        "from pydantic import Field",
                        "def config(cls): return cls",
                        "@config",
                        "class CacheConfig:",
                        "    size: int = Field(default=1, ge=1)",
                        '    """Number of entries."""',
                    ]
                ),
                encoding="utf-8",
            )

            env_rows, env_rejected = extract_env_contracts(snapshot)
            config_rows, config_rejected = extract_config_fields(snapshot)

            self.assertEqual(env_rejected, [])
            self.assertEqual(env_rows[0]["entity"], "VLLM_ALPHA")
            self.assertEqual(env_rows[0]["contract"]["default"], "x")
            self.assertEqual(config_rejected, [])
            self.assertEqual(config_rows[0]["display_entity"], "CacheConfig.size")
            self.assertEqual(config_rows[0]["contract"]["default"], 1)
            self.assertEqual(config_rows[0]["contract"]["constraints"], {"ge": 1})


if __name__ == "__main__":
    unittest.main()
