from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from experiments.delta_v2.inventory import (
    load_snapshot_manifest,
    select_transition,
    verify_snapshot,
)


PROBE_SCHEMA = "delta.behavior_probe.v1"
RESULT_SCHEMA = "delta.behavior_probe_result.v1"
OUTCOME_KINDS = {"value", "attribute_error", "value_error"}
HEX_SHA1 = re.compile(r"^[0-9a-f]{40}$")
DEVELOPMENT_ROLES = ("development_before", "development_after")


class BehaviorProbeError(ValueError):
    """The probe contract, source evidence, or execution is invalid."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BehaviorProbeError(f"{field} must be a mapping")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise BehaviorProbeError(f"{field} must be a non-empty string")
    return value


def load_probe(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BehaviorProbeError(f"cannot read BehaviorProbe: {path}") from exc
    probe = _mapping(payload, "probe")
    validate_probe(probe)
    return probe


def _validate_outcome(value: Any, field: str) -> None:
    outcome = _mapping(value, field)
    kind = outcome.get("kind")
    if kind not in OUTCOME_KINDS:
        raise BehaviorProbeError(f"{field}.kind is unsupported")
    if kind == "value":
        if set(outcome) != {"kind", "value"}:
            raise BehaviorProbeError(
                f"{field} value outcome requires exactly kind and value"
            )
        if outcome["value"] is not None and not isinstance(outcome["value"], int):
            raise BehaviorProbeError(f"{field}.value must be an integer or null")
    elif set(outcome) != {"kind"}:
        raise BehaviorProbeError(
            f"{field} error outcome requires exactly the kind field"
        )


def validate_probe(probe: Mapping[str, Any]) -> None:
    if probe.get("schema") != PROBE_SCHEMA:
        raise BehaviorProbeError("unsupported BehaviorProbe schema")
    if not _string(probe.get("probe_id"), "probe_id").startswith(
        "behavior-probe:"
    ):
        raise BehaviorProbeError("probe_id must use behavior-probe: prefix")
    if not _string(probe.get("candidate_id"), "candidate_id").startswith(
        "feature-candidate:"
    ):
        raise BehaviorProbeError("candidate_id must use feature-candidate: prefix")
    if probe.get("transition") != "development":
        raise BehaviorProbeError("BehaviorProbe may only use development snapshots")
    if probe.get("claim_scope") != "configuration-interface-only":
        raise BehaviorProbeError("this probe must retain its narrow claim scope")

    setup = _mapping(probe.get("setup"), "setup")
    source_path = _string(setup.get("source_path"), "setup.source_path")
    if source_path.startswith("/") or ".." in Path(source_path).parts:
        raise BehaviorProbeError("setup.source_path must be repository-relative")
    if setup.get("source_loader") != "exact-git-blob-importlib-v1":
        raise BehaviorProbeError("unsupported source_loader")
    _string(setup.get("attribute"), "setup.attribute")

    cases = probe.get("cases")
    if not isinstance(cases, list) or not cases:
        raise BehaviorProbeError("cases must be a non-empty list")
    case_ids: set[str] = set()
    for index, raw_case in enumerate(cases):
        case = _mapping(raw_case, f"cases[{index}]")
        case_id = _string(case.get("case_id"), f"cases[{index}].case_id")
        if case_id in case_ids:
            raise BehaviorProbeError(f"duplicate case_id: {case_id}")
        case_ids.add(case_id)
        environment_value = case.get("environment_value")
        if environment_value is not None and not isinstance(environment_value, str):
            raise BehaviorProbeError(
                f"cases[{index}].environment_value must be a string or null"
            )
        _validate_outcome(case.get("expected_before"), f"cases[{index}].expected_before")
        _validate_outcome(case.get("expected_after"), f"cases[{index}].expected_after")

    environment = _mapping(probe.get("environment"), "environment")
    if environment.get("dependencies") != "standard-library-only":
        raise BehaviorProbeError("probe must remain standard-library-only")
    if environment.get("process_isolation") != "fresh-module-per-case":
        raise BehaviorProbeError("probe must load a fresh module for every case")


def _git_bytes(repo: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise BehaviorProbeError(f"git {' '.join(args)} failed: {stderr}") from exc
    return result.stdout


def read_source_blob(
    repo: Path,
    commit_sha: str,
    source_path: str,
) -> tuple[str, bytes]:
    object_spec = f"{commit_sha}:{source_path}"
    object_id = _git_bytes(repo, "rev-parse", object_spec).decode("ascii").strip()
    if not HEX_SHA1.fullmatch(object_id):
        raise BehaviorProbeError(f"invalid Git object ID for {object_spec}")
    object_type = _git_bytes(repo, "cat-file", "-t", object_id).decode("ascii").strip()
    if object_type != "blob":
        raise BehaviorProbeError(f"{object_spec} is not a blob")
    source = _git_bytes(repo, "cat-file", "blob", object_id)
    return object_id, source


def observe_source(
    source: bytes,
    *,
    attribute: str,
    environment_value: str | None,
    case_token: str,
) -> dict[str, Any]:
    env_key = attribute
    previous = os.environ.get(env_key)
    was_present = env_key in os.environ
    try:
        if environment_value is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = environment_value

        with tempfile.TemporaryDirectory(prefix="delta-v2-behavior-probe-") as temp_dir:
            module_path = Path(temp_dir) / "source.py"
            module_path.write_bytes(source)
            module_name = "delta_v2_probe_" + hashlib.sha256(
                case_token.encode("utf-8")
            ).hexdigest()
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise BehaviorProbeError("could not create source module spec")
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                value = getattr(module, attribute)
            except AttributeError:
                return {"kind": "attribute_error"}
            except ValueError:
                return {"kind": "value_error"}
    finally:
        if was_present:
            assert previous is not None
            os.environ[env_key] = previous
        else:
            os.environ.pop(env_key, None)

    if value is not None and not isinstance(value, int):
        raise BehaviorProbeError(
            f"probe returned unsupported value type: {type(value).__name__}"
        )
    return {"kind": "value", "value": value}


def run_probe(
    repo: Path,
    snapshots_path: Path,
    probe: Mapping[str, Any],
) -> dict[str, Any]:
    validate_probe(probe)
    manifest = load_snapshot_manifest(snapshots_path)
    before, after = select_transition(manifest, "development")
    snapshots = {
        "development_before": before,
        "development_after": after,
    }
    setup = probe["setup"]
    source_path = str(setup["source_path"])
    attribute = str(setup["attribute"])

    sources: dict[str, bytes] = {}
    source_records: dict[str, dict[str, str]] = {}
    for role in DEVELOPMENT_ROLES:
        snapshot = snapshots[role]
        verify_snapshot(repo, snapshot)
        object_id, source = read_source_blob(
            repo,
            snapshot.commit_sha,
            source_path,
        )
        sources[role] = source
        source_records[role] = {
            "commit_sha": snapshot.commit_sha,
            "content_hash": snapshot.content_hash,
            "source_path": source_path,
            "git_blob_sha1": object_id,
            "source_sha256": hashlib.sha256(source).hexdigest(),
        }

    case_results: list[dict[str, Any]] = []
    for raw_case in probe["cases"]:
        case = dict(raw_case)
        observed: dict[str, Any] = {}
        matches: dict[str, bool] = {}
        for role, expectation_key in (
            ("development_before", "expected_before"),
            ("development_after", "expected_after"),
        ):
            outcome = observe_source(
                sources[role],
                attribute=attribute,
                environment_value=case.get("environment_value"),
                case_token=f"{role}:{case['case_id']}",
            )
            observed[role] = outcome
            matches[role] = outcome == case[expectation_key]
        case_results.append(
            {
                "case_id": case["case_id"],
                "environment_value": case.get("environment_value"),
                "expected_before": case["expected_before"],
                "observed_before": observed["development_before"],
                "expected_after": case["expected_after"],
                "observed_after": observed["development_after"],
                "status": "pass" if all(matches.values()) else "fail",
            }
        )

    status = (
        "pass"
        if all(case["status"] == "pass" for case in case_results)
        else "fail"
    )
    return {
        "schema": RESULT_SCHEMA,
        "probe_id": probe["probe_id"],
        "candidate_id": probe["candidate_id"],
        "claim": probe["claim"],
        "claim_scope": probe["claim_scope"],
        "status": status,
        "sources": source_records,
        "cases": case_results,
        "promotion_boundary": probe["promotion_boundary"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute a delta_v2 BehaviorProbe on development snapshots."
    )
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_probe(
        args.repo,
        args.snapshots,
        load_probe(args.probe),
    )
    serialized = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

