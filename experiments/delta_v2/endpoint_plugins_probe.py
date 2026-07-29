from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping, Sequence
from unittest.mock import patch

import yaml

from experiments.delta_v2.inventory import (
    load_snapshot_manifest,
    select_transition,
    verify_snapshot,
)


PROBE_SCHEMA = "delta.behavior_probe.v1"
RESULT_SCHEMA = "delta.behavior_probe_result.v1"
ROLES = ("acceptance_before", "acceptance_after")
ROLE_EXPECTATIONS = {
    "acceptance_before": "expected_before",
    "acceptance_after": "expected_after",
}
LOADER_PATH = "vllm/plugins/__init__.py"
SERVER_PATH = "vllm/entrypoints/openai/api_server.py"
PROTOCOL_PATH = "vllm/plugins/endpoint_plugins/interface.py"
DOC_PATH = "docs/design/endpoint_plugins.md"
TEST_PATH = "tests/plugins_tests/test_endpoint_plugins.py"


class EndpointPluginsProbeError(ValueError):
    """The endpoint-plugin probe contract, sources, or observations are invalid."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EndpointPluginsProbeError(f"{field} must be a mapping")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise EndpointPluginsProbeError(f"{field} must be a non-empty string")
    return value


def validate_probe(probe: Mapping[str, Any]) -> None:
    if probe.get("schema") != PROBE_SCHEMA:
        raise EndpointPluginsProbeError("unsupported BehaviorProbe schema")
    if probe.get("probe_id") != (
        "behavior-probe:vllm-endpoint-plugins-framework-v1"
    ):
        raise EndpointPluginsProbeError("unexpected endpoint-plugin probe ID")
    if probe.get("candidate_id") != (
        "feature-candidate:vllm-endpoint-plugins"
    ):
        raise EndpointPluginsProbeError("unexpected endpoint-plugin candidate")
    if probe.get("transition") != "acceptance":
        raise EndpointPluginsProbeError("endpoint-plugin probe requires acceptance")
    if probe.get("claim_scope") != "complete-candidate":
        raise EndpointPluginsProbeError("probe must cover the complete candidate")
    sources = _mapping(probe.get("source_contract"), "source_contract")
    if dict(sources) != {
        "loader": LOADER_PATH,
        "server": SERVER_PATH,
        "protocol": PROTOCOL_PATH,
        "documentation": DOC_PATH,
        "tests": TEST_PATH,
    }:
        raise EndpointPluginsProbeError("source contract paths changed")
    cases = probe.get("cases")
    if not isinstance(cases, list) or len(cases) != 1:
        raise EndpointPluginsProbeError("probe requires one complete-framework case")
    case = _mapping(cases[0], "cases[0]")
    if case.get("case_id") != "complete_endpoint_plugin_framework":
        raise EndpointPluginsProbeError("unexpected probe case")
    for field in ("expected_before", "expected_after"):
        outcome = _mapping(case.get(field), f"cases[0].{field}")
        if outcome.get("kind") not in {
            "unavailable",
            "endpoint_plugin_framework",
        }:
            raise EndpointPluginsProbeError(f"{field} has unsupported outcome")
    if case["expected_before"].get("kind") != "unavailable":
        raise EndpointPluginsProbeError("old endpoint-plugin feature must be absent")
    if case["expected_after"].get("kind") != "endpoint_plugin_framework":
        raise EndpointPluginsProbeError("new endpoint-plugin feature must pass")
    runtime = _mapping(probe.get("runtime"), "runtime")
    if (
        runtime.get("target") != "local-cpu"
        or runtime.get("source_execution")
        != "exact-git-blob-isolated-ast-v1"
        or runtime.get("dependency_policy") != "stdlib-and-probe-stubs-only"
        or runtime.get("environment_status")
        != "acceptance-feature-unfrozen"
    ):
        raise EndpointPluginsProbeError("runtime contract changed")


def load_probe(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EndpointPluginsProbeError(f"cannot read probe: {path}") from exc
    probe = _mapping(payload, str(path))
    validate_probe(probe)
    return probe


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise EndpointPluginsProbeError(
            f"git {' '.join(args)} failed: {detail}"
        )
    return completed


def _source_blob(repo: Path, commit: str, path: str) -> dict[str, Any] | None:
    tree_entry = _git(repo, "ls-tree", commit, "--", path).stdout
    if not tree_entry:
        return None
    try:
        metadata, listed_path = tree_entry.decode("utf-8").rstrip("\n").split(
            "\t",
            maxsplit=1,
        )
        _mode, object_type, blob = metadata.split()
    except ValueError as exc:
        raise EndpointPluginsProbeError(
            f"unexpected Git tree entry for {commit}:{path}"
        ) from exc
    if listed_path != path or object_type != "blob":
        raise EndpointPluginsProbeError(f"source path is not a blob: {path}")
    raw = _git(repo, "show", f"{commit}:{path}").stdout
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EndpointPluginsProbeError(f"source is not UTF-8: {path}") from exc
    return {
        "path": path,
        "git_blob_sha1": blob,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source": source,
    }


def _definition_node(
    source: str,
    name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise EndpointPluginsProbeError("pinned source does not parse") from exc
    matches = [
        node
        for node in tree.body
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        )
        and node.name == name
    ]
    if len(matches) > 1:
        raise EndpointPluginsProbeError(f"duplicate source definition: {name}")
    return matches[0] if matches else None


def _compile_definition(
    source: str,
    name: str,
    namespace: dict[str, Any],
) -> Any:
    node = _definition_node(source, name)
    if node is None:
        raise EndpointPluginsProbeError(f"source definition absent: {name}")
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, f"<pinned:{name}>", "exec"), namespace)
    return namespace[name]


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[tuple[Any, ...]] = []
        self.infos: list[tuple[Any, ...]] = []
        self.exceptions: list[tuple[Any, ...]] = []

    def warning(self, *args: Any) -> None:
        self.warnings.append(args)

    def info(self, *args: Any) -> None:
        self.infos.append(args)

    def exception(self, *args: Any) -> None:
        self.exceptions.append(args)


class _Plugin:
    def __init__(
        self,
        *,
        name: str = "endpoint",
        required_tasks: tuple[str, ...] | None = None,
    ) -> None:
        self.name = name
        self.required_tasks = required_tasks
        self.attached_apps: list[Any] = []
        self.init_calls: list[tuple[Any, Any, Any]] = []

    def attach_router(self, app: Any) -> None:
        self.attached_apps.append(app)

    async def init_state(
        self,
        engine_client: Any,
        state: Any,
        args: Any,
    ) -> None:
        self.init_calls.append((engine_client, state, args))


def _exercise_loader(source: str) -> dict[str, Any]:
    logger = _Logger()
    envs = SimpleNamespace(VLLM_PLUGINS=None)
    loader_calls: list[str] = []
    factories: dict[str, Callable[[], _Plugin]] = {}

    def load_plugins_by_group(group: str) -> dict[str, Callable[[], _Plugin]]:
        loader_calls.append(group)
        return factories

    load_endpoint_plugins = _compile_definition(
        source,
        "load_endpoint_plugins",
        {
            "Any": Any,
            "ENDPOINT_PLUGINS_GROUP": "vllm.endpoint_plugins",
            "envs": envs,
            "load_plugins_by_group": load_plugins_by_group,
            "logger": logger,
        },
    )

    discovered = [SimpleNamespace(name="discovered")]
    with patch.object(importlib.metadata, "entry_points", return_value=discovered):
        default_off = load_endpoint_plugins(("generate",))
    default_off_pass = (
        default_off == []
        and loader_calls == []
        and len(logger.warnings) == 1
    )

    matching = _Plugin(name="matching", required_tasks=("generate",))
    factories = {"matching": lambda: matching}
    envs.VLLM_PLUGINS = ["matching"]
    allowed = load_endpoint_plugins(("generate",))
    allowlisted_match_pass = (
        allowed == [matching]
        and loader_calls[-1] == "vllm.endpoint_plugins"
    )

    mismatch = _Plugin(name="mismatch", required_tasks=("embed",))
    factories = {"mismatch": lambda: mismatch}
    envs.VLLM_PLUGINS = ["mismatch"]
    task_miss = load_endpoint_plugins(("generate",))
    task_gate_pass = task_miss == []

    def raising_factory() -> _Plugin:
        raise RuntimeError("expected probe failure")

    survivor = _Plugin(name="survivor")
    factories = {
        "raising": raising_factory,
        "survivor": lambda: survivor,
    }
    envs.VLLM_PLUGINS = ["raising", "survivor"]
    resilient = load_endpoint_plugins(("generate",))
    factory_failure_isolated = (
        resilient == [survivor] and len(logger.exceptions) == 1
    )
    return {
        "default_off": default_off_pass,
        "allowlisted_task_match_loads": allowlisted_match_pass,
        "required_task_miss_skips": task_gate_pass,
        "factory_failure_isolated": factory_failure_isolated,
    }


@contextmanager
def _stub_vllm_plugins(
    loader: Callable[[tuple[str, ...]], list[_Plugin]],
) -> Iterator[None]:
    previous_vllm = sys.modules.get("vllm")
    previous_plugins = sys.modules.get("vllm.plugins")
    vllm_module = types.ModuleType("vllm")
    vllm_module.__path__ = []  # type: ignore[attr-defined]
    plugins_module = types.ModuleType("vllm.plugins")
    plugins_module.load_endpoint_plugins = loader  # type: ignore[attr-defined]
    sys.modules["vllm"] = vllm_module
    sys.modules["vllm.plugins"] = plugins_module
    try:
        yield
    finally:
        if previous_vllm is None:
            sys.modules.pop("vllm", None)
        else:
            sys.modules["vllm"] = previous_vllm
        if previous_plugins is None:
            sys.modules.pop("vllm.plugins", None)
        else:
            sys.modules["vllm.plugins"] = previous_plugins


def _exercise_lifecycle(source: str) -> dict[str, Any]:
    plugin = _Plugin()
    loader_calls: list[tuple[str, ...]] = []

    def loader(tasks: tuple[str, ...]) -> list[_Plugin]:
        loader_calls.append(tasks)
        return [plugin]

    namespace = {"Any": Any}
    attach = _compile_definition(
        source,
        "_attach_endpoint_plugins",
        namespace,
    )
    initialize = _compile_definition(
        source,
        "_init_endpoint_plugins_state",
        namespace,
    )
    app = SimpleNamespace(state=SimpleNamespace())
    engine = object()
    args = SimpleNamespace()
    with _stub_vllm_plugins(loader):
        attach(app, ("generate",))
    asyncio.run(initialize(engine, app.state, args))
    return {
        "route_phase_attaches": (
            loader_calls == [("generate",)]
            and plugin.attached_apps == [app]
            and app.state.endpoint_plugins == [plugin]
        ),
        "state_phase_initializes": plugin.init_calls == [
            (engine, app.state, args)
        ],
    }


def _protocol_contract(source: str) -> dict[str, Any]:
    node = _definition_node(source, "EndpointPlugin")
    if not isinstance(node, ast.ClassDef):
        return {
            "runtime_checkable_protocol": False,
            "attach_router_hook": False,
            "async_init_state_hook": False,
        }
    decorators = {
        decorator.id
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Name)
    }
    bases = {
        base.id
        for base in node.bases
        if isinstance(base, ast.Name)
    }
    methods = {
        child.name: child
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return {
        "runtime_checkable_protocol": (
            "runtime_checkable" in decorators and "Protocol" in bases
        ),
        "attach_router_hook": isinstance(
            methods.get("attach_router"),
            ast.FunctionDef,
        ),
        "async_init_state_hook": isinstance(
            methods.get("init_state"),
            ast.AsyncFunctionDef,
        ),
    }


def observe_revision(repo: Path, commit: str) -> dict[str, Any]:
    blobs = {
        name: _source_blob(repo, commit, path)
        for name, path in {
            "loader": LOADER_PATH,
            "server": SERVER_PATH,
            "protocol": PROTOCOL_PATH,
            "documentation": DOC_PATH,
            "tests": TEST_PATH,
        }.items()
    }
    loader = blobs["loader"]
    server = blobs["server"]
    protocol = blobs["protocol"]
    has_loader = (
        loader is not None
        and _definition_node(loader["source"], "load_endpoint_plugins")
        is not None
    )
    has_attach = (
        server is not None
        and _definition_node(server["source"], "_attach_endpoint_plugins")
        is not None
    )
    has_init = (
        server is not None
        and _definition_node(server["source"], "_init_endpoint_plugins_state")
        is not None
    )
    if not (has_loader or has_attach or has_init or protocol is not None):
        outcome = {
            "kind": "unavailable",
            "loader": False,
            "protocol": False,
            "route_phase": False,
            "state_phase": False,
        }
        return {
            "outcome": outcome,
            "evidence": {
                name: (
                    None
                    if blob is None
                    else {
                        "path": blob["path"],
                        "git_blob_sha1": blob["git_blob_sha1"],
                        "source_sha256": blob["source_sha256"],
                    }
                )
                for name, blob in sorted(blobs.items())
            },
        }
    if not (
        has_loader
        and has_attach
        and has_init
        and protocol is not None
        and blobs["documentation"] is not None
        and blobs["tests"] is not None
    ):
        raise EndpointPluginsProbeError(
            "partial endpoint-plugin framework cannot satisfy the candidate"
        )
    loader_checks = _exercise_loader(str(loader["source"]))
    lifecycle_checks = _exercise_lifecycle(str(server["source"]))
    protocol_checks = _protocol_contract(str(protocol["source"]))
    outcome = {
        "kind": "endpoint_plugin_framework",
        **protocol_checks,
        **loader_checks,
        **lifecycle_checks,
        "documentation_present": True,
        "upstream_tests_present": True,
    }
    return {
        "outcome": outcome,
        "evidence": {
            name: {
                "path": blob["path"],
                "git_blob_sha1": blob["git_blob_sha1"],
                "source_sha256": blob["source_sha256"],
            }
            for name, blob in sorted(blobs.items())
            if blob is not None
        },
    }


def evaluate_observations(
    probe: Mapping[str, Any],
    observations: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validate_probe(probe)
    if set(observations) != set(ROLES):
        raise EndpointPluginsProbeError("observations must cover both acceptance roles")
    case = probe["cases"][0]
    expected_before = case["expected_before"]
    expected_after = case["expected_after"]
    observed_before = observations["acceptance_before"]
    observed_after = observations["acceptance_after"]
    checks = {
        "before_matches": observed_before == expected_before,
        "after_matches": observed_after == expected_after,
    }
    return [
        {
            "case_id": case["case_id"],
            "expected_before": expected_before,
            "observed_before": observed_before,
            "expected_after": expected_after,
            "observed_after": observed_after,
            "checks": checks,
            "status": "pass" if all(checks.values()) else "fail",
        }
    ]


def run_probe(
    *,
    probe: Mapping[str, Any],
    repo: Path,
    snapshots_path: Path,
    allow_acceptance: bool,
) -> dict[str, Any]:
    validate_probe(probe)
    if not allow_acceptance:
        raise EndpointPluginsProbeError(
            "acceptance probe requires explicit --allow-acceptance"
        )
    manifest = load_snapshot_manifest(snapshots_path)
    before, after = select_transition(
        manifest,
        "acceptance",
        allow_acceptance=True,
    )
    snapshots = (before, after)
    sources = {}
    observations = {}
    source_evidence = {}
    for snapshot in snapshots:
        verify_snapshot(repo, snapshot)
        sources[snapshot.role] = {
            "revision": snapshot.revision,
            "commit_sha": snapshot.commit_sha,
            "content_hash": snapshot.content_hash,
        }
        observation = observe_revision(
            repo,
            snapshot.commit_sha,
        )
        observations[snapshot.role] = observation["outcome"]
        source_evidence[snapshot.role] = observation["evidence"]
    cases = evaluate_observations(probe, observations)
    status = "pass" if all(case["status"] == "pass" for case in cases) else "fail"
    return {
        "schema": RESULT_SCHEMA,
        "probe_id": probe["probe_id"],
        "candidate_id": probe["candidate_id"],
        "transition": "acceptance",
        "claim_scope": probe["claim_scope"],
        "status": status,
        "sources": sources,
        "source_evidence": source_evidence,
        "cases": cases,
        "runtime": {
            "python": sys.version.split()[0],
            "source_execution": probe["runtime"]["source_execution"],
            "dependency_policy": probe["runtime"]["dependency_policy"],
        },
        "acceptance_accessed": True,
        "eval_environment_frozen": False,
    }


def _serialize_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe endpoint-plugin behavior across acceptance snapshots."
    )
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-acceptance", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_probe(
        probe=load_probe(args.probe),
        repo=args.repo,
        snapshots_path=args.snapshots,
        allow_acceptance=args.allow_acceptance,
    )
    raw = _serialize_json(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(raw)
    print(raw.decode("utf-8"), end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
