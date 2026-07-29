from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.delta_v2.facts.build import (
    build_development_facts,
    write_development_outputs,
)
from experiments.delta_v2.facts.model import (
    Evidence,
    FactExtractionError,
    FactObservation,
    ObservationStore,
    audit_fact_deltas,
    canonical_ast,
    join_observations,
)
from experiments.delta_v2.facts.python_ast import (
    CLI_OPTION_FAMILY,
    CONFIG_FIELD_FAMILY,
    ENVIRONMENT_VARIABLE_FAMILY,
    LITERAL_DOMAIN_FAMILY,
    PythonSource,
    extract_python_source,
)


FACT_FAMILIES_PATH = Path(__file__).parents[1] / "fact_families.yaml"


def git_blob_id(content: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(content)}\0".encode("ascii") + content
    ).hexdigest()


def source(
    text: str,
    *,
    path: str,
    role: str = "development_before",
) -> PythonSource:
    content = text.encode("utf-8")
    return PythonSource(
        snapshot_role=role,
        path=path,
        git_blob_sha1=git_blob_id(content),
        content=content,
    )


def extract(text: str, *, path: str) -> ObservationStore:
    store = ObservationStore()
    extract_python_source(
        repository_id="owner/repo",
        source=source(text, path=path),
        store=store,
    )
    return store


def run_git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


class CanonicalAstTest(unittest.TestCase):
    def test_canonical_ast_ignores_source_locations(self) -> None:
        first = ast.parse("value = call(1)\n").body[0]
        second = ast.parse("\n\nvalue = call(1)\n").body[0]

        self.assertEqual(canonical_ast(first), canonical_ast(second))
        json.dumps(canonical_ast(first), sort_keys=True)


class PythonFamilyExtractionTest(unittest.TestCase):
    def test_extracts_config_fields_and_rejects_class_variables(self) -> None:
        store = extract(
            """
from dataclasses import field
from typing import ClassVar

@config
class DemoConfig:
    VERSION: ClassVar[int] = 1
    required: int
    limit: int = Field(default=4, gt=0, le=8)
    items: list[str] = field(default_factory=list, init=False)
""",
            path="vllm/config/demo.py",
        )

        rows = [
            row
            for row in store.observations
            if row.family_id == CONFIG_FIELD_FAMILY
        ]
        self.assertEqual(len(rows), 3)
        by_key = {row.semantic_key: row for row in rows}
        limit = by_key[
            "python-config-field:vllm.config.demo:DemoConfig.limit"
        ]
        self.assertEqual(limit.value["default_kind"], "pydantic-field")
        self.assertEqual(
            limit.value["literal_default"],
            {"kind": "literal", "value": 4},
        )
        self.assertEqual(
            set(limit.value["static_field_constraints"]),
            {"gt", "le"},
        )
        self.assertEqual(
            [row.reason for row in store.rejections],
            ["ClassVar is not an instance configuration field"],
        )

    def test_extracts_cli_options_and_rejects_ambiguous_calls(self) -> None:
        store = extract(
            """
def register(parser):
    parser.add_argument(
        "-n",
        "--long-name",
        type=int,
        choices=[1, 2],
        default=DEFAULT,
        required=True,
    )
    parser.add_argument("positional")
    parser.add_argument(*FLAGS)
""",
            path="vllm/entrypoints/demo.py",
        )

        rows = [
            row
            for row in store.observations
            if row.family_id == CLI_OPTION_FAMILY
        ]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(
            row.semantic_key,
            "python-cli-option:vllm.entrypoints.demo:"
            "register:--long-name",
        )
        self.assertEqual(
            row.value["destination"],
            {"kind": "derived", "value": "long_name"},
        )
        self.assertEqual(row.value["choices"]["kind"], "literal")
        self.assertEqual(row.value["default_expression"]["kind"], "expression")
        self.assertEqual(
            [rejection.reason for rejection in store.rejections],
            [
                "add_argument option strings must be "
                "hyphen-prefixed literals",
                "add_argument uses expanded positional arguments",
            ],
        )

    def test_cli_primary_option_ties_follow_declaration_order(self) -> None:
        store = extract(
            """
parser.add_argument("--aa", "--bb")
""",
            path="vllm/cli.py",
        )

        self.assertEqual(
            store.observations[0].semantic_key,
            "python-cli-option:vllm.cli:<module>:--aa",
        )

    def test_extracts_environment_registry_with_type_annotation(self) -> None:
        store = extract(
            """
if TYPE_CHECKING:
    VLLM_DEMO: int = 0

environment_variables = {
    "VLLM_DEMO": lambda: int(os.getenv("VLLM_DEMO", "0")),
    VARIABLE_NAME: lambda: None,
    **EXTRA_ENV,
}
""",
            path="vllm/envs.py",
        )

        rows = [
            row
            for row in store.observations
            if row.family_id == ENVIRONMENT_VARIABLE_FAMILY
        ]
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0].semantic_key,
            "python-environment-variable:VLLM_DEMO",
        )
        self.assertEqual(
            rows[0].value["type_checking_annotation_ast"]["node"],
            "Name",
        )
        self.assertEqual(
            [rejection.reason for rejection in store.rejections],
            [
                "environment registry key is not a string literal",
                "environment registry uses dictionary expansion",
            ],
        )

    def test_extracts_direct_literal_domains_and_rejects_wrappers(self) -> None:
        store = extract(
            """
Mode = Literal["fast", -1, None, b"x"]
Wrapped = Literal["x"] | None
Computed = Literal[make_value()]
A = B = Literal["duplicate-target"]
""",
            path="vllm/config/demo.py",
        )

        rows = [
            row
            for row in store.observations
            if row.family_id == LITERAL_DOMAIN_FAMILY
        ]
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0].value["ordered_literal_values"],
            ["fast", -1, None, {"kind": "bytes", "hex": "78"}],
        )
        self.assertEqual(
            [rejection.reason for rejection in store.rejections],
            [
                "Literal is nested inside a union or wrapper",
                "Literal domain contains a computed or unsupported value",
                "Literal alias target is not one simple name",
            ],
        )

    def test_duplicate_semantic_key_fails_family_snapshot(self) -> None:
        with self.assertRaisesRegex(
            FactExtractionError,
            "duplicate semantic key",
        ):
            extract(
                """
def register(parser):
    parser.add_argument("--same")
    parser.add_argument("--same")
""",
                path="vllm/cli.py",
            )


class AtomicFactJoinTest(unittest.TestCase):
    def evidence(self, role: str, line: int) -> Evidence:
        return Evidence(
            snapshot_role=role,
            path="vllm/demo.py",
            line_start=line,
            line_end=line,
            git_blob_sha1="a" * 40,
            source_sha256="b" * 64,
        )

    def observation(
        self,
        semantic_key: str,
        value: object,
        role: str,
        line: int,
    ) -> FactObservation:
        return FactObservation.create(
            repository_id="owner/repo",
            family_id=CLI_OPTION_FAMILY,
            semantic_key=semantic_key,
            value=value,
            evidence=self.evidence(role, line),
        )

    def test_join_assigns_all_statuses_and_reconstructs_both_sides(self) -> None:
        before = [
            self.observation("stable", {"value": 1}, "development_before", 1),
            self.observation("changed", {"value": 1}, "development_before", 2),
            self.observation("removed", {"value": 1}, "development_before", 3),
        ]
        after = [
            self.observation("stable", {"value": 1}, "development_after", 10),
            self.observation("changed", {"value": 2}, "development_after", 11),
            self.observation("added", {"value": 1}, "development_after", 12),
        ]

        rows = join_observations(
            before,
            after,
            verifier={"extractor_version": "test"},
        )
        audit = audit_fact_deltas(rows, before, after)

        self.assertEqual(
            {row.semantic_key: row.status for row in rows},
            {
                "added": "added",
                "changed": "changed",
                "removed": "removed",
                "stable": "stable",
            },
        )
        self.assertTrue(audit["reconstructs_before"])
        self.assertTrue(audit["reconstructs_after"])
        self.assertTrue(audit["unique_fact_ids"])
        self.assertTrue(audit["deterministic_order"])


class DevelopmentBuildIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.repo = self.root / "source"
        self.repo.mkdir()
        run_git(self.repo, "init")
        run_git(self.repo, "config", "user.email", "delta@example.com")
        run_git(self.repo, "config", "user.name", "Project DELTA")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_source(self, *, after: bool) -> None:
        config_dir = self.repo / "vllm" / "config"
        entrypoints_dir = self.repo / "vllm" / "entrypoints"
        config_dir.mkdir(parents=True, exist_ok=True)
        entrypoints_dir.mkdir(parents=True, exist_ok=True)
        mode_values = '"fast", "safe"' if after else '"fast"'
        default = '"safe"' if after else '"fast"'
        (config_dir / "demo.py").write_text(
            "from typing import Literal\n"
            f"Mode = Literal[{mode_values}]\n"
            "@config\n"
            "class DemoConfig:\n"
            f"    mode: Mode = {default}\n",
            encoding="utf-8",
        )
        option = "--new-option" if after else "--old-option"
        (entrypoints_dir / "cli.py").write_text(
            "def register(parser):\n"
            f'    parser.add_argument("{option}", default=1)\n',
            encoding="utf-8",
        )
        env_default = "1" if after else "0"
        (self.repo / "vllm" / "envs.py").write_text(
            "if TYPE_CHECKING:\n"
            "    VLLM_DEMO: int = 0\n"
            "environment_variables = {\n"
            '    "VLLM_DEMO": '
            f'lambda: int(os.getenv("VLLM_DEMO", "{env_default}")),\n'
            "}\n",
            encoding="utf-8",
        )

    def commit(self, message: str) -> tuple[str, str]:
        run_git(self.repo, "add", ".")
        run_git(self.repo, "commit", "-m", message)
        commit_sha = run_git(self.repo, "rev-parse", "HEAD")
        tree_sha = run_git(self.repo, "rev-parse", "HEAD^{tree}")
        return commit_sha, tree_sha

    def write_manifests(
        self,
        before: tuple[str, str],
        after: tuple[str, str],
    ) -> Path:
        manifest_dir = self.root / "experiment"
        manifest_dir.mkdir()
        snapshots = {
            "schema": "delta.source_snapshots.v1",
            "repository": {
                "id": "owner/repo",
                "url": "https://example.invalid/owner/repo.git",
            },
            "content_hash_algorithm": "git-tree-sha1",
            "snapshots": [
                {
                    "role": "development_before",
                    "revision": "before",
                    "commit_sha": before[0],
                    "content_hash": f"git-tree-sha1:{before[1]}",
                },
                {
                    "role": "development_after",
                    "revision": "after",
                    "commit_sha": after[0],
                    "content_hash": f"git-tree-sha1:{after[1]}",
                },
                {
                    "role": "acceptance_before",
                    "revision": "sealed-before",
                    "commit_sha": before[0],
                    "content_hash": f"git-tree-sha1:{before[1]}",
                },
                {
                    "role": "acceptance_after",
                    "revision": "sealed-after",
                    "commit_sha": after[0],
                    "content_hash": f"git-tree-sha1:{after[1]}",
                },
            ],
        }
        (manifest_dir / "snapshots.yaml").write_text(
            yaml.safe_dump(snapshots, sort_keys=False),
            encoding="utf-8",
        )
        fact_payload = yaml.safe_load(
            FACT_FAMILIES_PATH.read_text(encoding="utf-8")
        )
        fact_path = manifest_dir / "fact_families.yaml"
        fact_path.write_text(
            yaml.safe_dump(fact_payload, sort_keys=False),
            encoding="utf-8",
        )
        return fact_path

    def test_real_git_build_is_independent_reconstructable_and_deterministic(
        self,
    ) -> None:
        self.write_source(after=False)
        before = self.commit("before")
        self.write_source(after=True)
        after = self.commit("after")
        fact_manifest = self.write_manifests(before, after)

        first = build_development_facts(
            repo=self.repo,
            fact_manifest_path=fact_manifest,
        )
        second = build_development_facts(
            repo=self.repo,
            fact_manifest_path=fact_manifest,
        )
        first_dir = self.root / "first"
        second_dir = self.root / "second"
        first_summary = write_development_outputs(first, first_dir)
        second_summary = write_development_outputs(second, second_dir)

        self.assertEqual(first_summary, second_summary)
        self.assertTrue(first.summary["reconstructs_before"])
        self.assertTrue(first.summary["reconstructs_after"])
        self.assertEqual(
            first.summary["status_counts"],
            {
                "stable": 0,
                "added": 1,
                "removed": 1,
                "changed": 3,
            },
        )
        for first_path in sorted(first_dir.iterdir()):
            second_path = second_dir / first_path.name
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
        self.assertEqual(
            {
                row.evidence.snapshot_role
                for row in first.before.observations
            },
            {"development_before"},
        )
        self.assertEqual(
            {
                row.evidence.snapshot_role
                for row in first.after.observations
            },
            {"development_after"},
        )


if __name__ == "__main__":
    unittest.main()
