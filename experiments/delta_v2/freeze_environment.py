from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


SCHEMA = "delta.eval_environment.v1"
RESULT_SCHEMA = "delta.eval_environment_freeze_result.v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
EVAL_ITEM_SCHEMA = "delta.eval_item.v1"
SPLIT_SCHEMA = "delta.source_split_assignment.v1"
EXPECTED_SNAPSHOT_ROLES = {
    "development_before": (
        "v0.22.0",
        "0b3ba88f165976e77ca5e6a7a3f5bba4562b80af",
        "git-tree-sha1:92701b6c49f4969297d5d11a2cac91c180690482",
    ),
    "development_after": (
        "v0.23.0",
        "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665",
        "git-tree-sha1:e18d76eec620cdfc6559d7152f9911b7e91f84f8",
    ),
    "acceptance_before": (
        "v0.25.1",
        "752a3a504485790a2e8491cacbb35c137339ad34",
        "git-tree-sha1:3ec7a4eb00f9bc8fec399bea6cf7de27a7936372",
    ),
    "acceptance_after": (
        "v0.26.0",
        "568afb3a13806beb53bb2e6bd518269357b237c0",
        "git-tree-sha1:ce348f7622d677acf4ee7bc4e5e8a826c2bc2c1f",
    ),
}


class EnvironmentFreezeError(ValueError):
    """The EvalEnvironment is incomplete, inconsistent, or has drifted."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EnvironmentFreezeError(f"{field} must be a mapping")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise EnvironmentFreezeError(f"{field} must be a non-empty string")
    return value


def _relative_path(value: Any, field: str) -> Path:
    path = Path(_string(value, field))
    if path.is_absolute() or ".." in path.parts:
        raise EnvironmentFreezeError(f"{field} must stay within the repository")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EnvironmentFreezeError(f"cannot read YAML: {path}") from exc
    return _mapping(payload, str(path))


def load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvironmentFreezeError(f"cannot read JSON: {path}") from exc
    return _mapping(payload, str(path))


def load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EnvironmentFreezeError(f"cannot read JSONL: {path}") from exc
    rows = []
    for line_number, line in enumerate(lines, start=1):
        try:
            rows.append(
                _mapping(json.loads(line), f"{path}:{line_number}")
            )
        except json.JSONDecodeError as exc:
            raise EnvironmentFreezeError(
                f"invalid JSON at {path}:{line_number}"
            ) from exc
    return rows


def validate_policy(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != SCHEMA:
        raise EnvironmentFreezeError("unsupported EvalEnvironment schema")
    if (
        manifest.get("experiment_id") != "delta_v2"
        or manifest.get("repository") != "vllm-project/vllm"
        or manifest.get("freeze_state") != "frozen"
        or manifest.get("publication_state")
        != "local-reproducible-not-published"
    ):
        raise EnvironmentFreezeError("environment identity or state changed")
    _string(manifest.get("environment_id"), "environment_id")
    code_commit = _string(manifest.get("code_commit"), "code_commit")
    if not COMMIT_SHA.fullmatch(code_commit):
        raise EnvironmentFreezeError("code_commit must be a full commit SHA")

    modules = _mapping(manifest.get("module_status"), "module_status")
    expected_modules = {
        "sense_source_change": "implemented",
        "sense_target_model_drift": "deferred",
        "build_fact_environment": "implemented",
        "build_development_feature_proof": "implemented",
        "build_acceptance_feature_eval": "partial",
        "decide": "deferred",
        "adapt": "deferred",
        "verify": "deferred",
        "memory": "deferred",
    }
    if dict(modules) != expected_modules:
        raise EnvironmentFreezeError("module status overclaims the loop")

    truth = _mapping(manifest.get("truth_policy"), "truth_policy")
    if (
        truth.get("gold_authority")
        != "deterministic-source-and-executable-evidence"
        or truth.get("llm_role") != "review-only"
        or truth.get("llm_may_change_gold") is not False
        or truth.get("conflict_action") != "fail-build"
    ):
        raise EnvironmentFreezeError("truth policy permits ungrounded gold")

    execution = _mapping(manifest.get("execution"), "execution")
    if (
        execution.get("target_model_runs") != 0
        or execution.get("training_runs") != 0
        or execution.get("fine_tuning_runs") != 0
        or set(execution.get("forbidden_without_separate_authorization", []))
        != {
            "target-model-run",
            "training",
            "fine-tuning",
            "modal-job",
            "lambda-job",
            "b2-write",
        }
    ):
        raise EnvironmentFreezeError("execution boundary changed")

    limitations = manifest.get("limitations")
    required_limitations = {
        "acceptance-facts-only",
        "acceptance-additions-and-stable-controls-only",
        "acceptance-feature-eval-not-generalized",
        "acceptance-selection-amended-after-unseal",
        "acceptance-endpoint-seen-in-prior-broad-e1-work",
        "no-target-model-result",
    }
    if not isinstance(limitations, list) or set(limitations) != required_limitations:
        raise EnvironmentFreezeError("required limitations are incomplete")

    gate = _mapping(manifest.get("test_gate"), "test_gate")
    command = gate.get("command")
    if not isinstance(command, list) or any(
        not isinstance(part, str) or not part for part in command
    ):
        raise EnvironmentFreezeError("test command must be an argument list")
    expected_tests = gate.get("expected_tests")
    if not isinstance(expected_tests, int) or expected_tests < 1:
        raise EnvironmentFreezeError("expected_tests must be positive")


def verify_bindings(
    records: Any,
    *,
    repo_root: Path,
) -> dict[str, Path]:
    if not isinstance(records, list) or not records:
        raise EnvironmentFreezeError("bindings must be a non-empty list")
    result: dict[str, Path] = {}
    seen_paths: set[Path] = set()
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"bindings[{index}]")
        binding_id = _string(record.get("id"), f"bindings[{index}].id")
        relative = _relative_path(
            record.get("path"),
            f"bindings[{index}].path",
        )
        expected = _string(
            record.get("sha256"),
            f"bindings[{index}].sha256",
        )
        if binding_id in result or relative in seen_paths:
            raise EnvironmentFreezeError("duplicate binding ID or path")
        if not SHA256.fullmatch(expected):
            raise EnvironmentFreezeError(f"invalid SHA-256 for {relative}")
        absolute = repo_root / relative
        if not absolute.is_file():
            raise EnvironmentFreezeError(f"bound file is missing: {relative}")
        actual = _sha256(absolute)
        if actual != expected:
            raise EnvironmentFreezeError(
                f"bound file hash mismatch: {relative}; "
                f"expected {expected}, got {actual}"
            )
        result[binding_id] = absolute
        seen_paths.add(relative)
    return result


def validate_snapshots(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema") != "delta.source_snapshots.v1"
        or payload.get("content_hash_algorithm") != "git-tree-sha1"
        or _mapping(payload.get("repository"), "repository").get("id")
        != "vllm-project/vllm"
    ):
        raise EnvironmentFreezeError("snapshot manifest identity changed")
    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list):
        raise EnvironmentFreezeError("snapshots must be a list")
    observed = {}
    for raw_snapshot in snapshots:
        snapshot = _mapping(raw_snapshot, "snapshot")
        role = _string(snapshot.get("role"), "snapshot.role")
        if role in observed:
            raise EnvironmentFreezeError(f"duplicate snapshot role: {role}")
        observed[role] = (
            snapshot.get("revision"),
            snapshot.get("commit_sha"),
            snapshot.get("content_hash"),
        )
    if observed != EXPECTED_SNAPSHOT_ROLES:
        raise EnvironmentFreezeError("snapshot pins changed")


def validate_eval_summary(
    payload: Mapping[str, Any],
    *,
    transition: str,
) -> None:
    if (
        payload.get("schema") != "delta.eval_item_build_audit.v1"
        or payload.get("status") != "pass"
        or payload.get("unique_eval_ids") is not True
        or payload.get("unique_source_ids") is not True
        or payload.get("control_balance_matches") is not True
        or payload.get("deterministic_order") is not True
    ):
        raise EnvironmentFreezeError(f"{transition} EvalItem audit failed")
    if transition == "development":
        if (
            payload.get("items") != 87
            or payload.get("split_counts") != {"dev": 22, "train": 65}
            or payload.get("eval_items") != 0
            or payload.get("acceptance_accessed") is not False
        ):
            raise EnvironmentFreezeError("development EvalItem counts changed")
    elif transition == "acceptance":
        exclusions = _mapping(
            _mapping(payload.get("outputs"), "outputs").get(
                "cross_transition_exclusions"
            ),
            "cross_transition_exclusions",
        )
        if (
            payload.get("contract_id")
            != "delta-v2-acceptance-eval-items-v2"
            or payload.get("items") != 42
            or payload.get("split_counts") != {"eval": 42}
            or payload.get("fact_status_counts")
            != {"added": 21, "stable": 21}
            or payload.get("cross_transition_source_overlap") != []
            or payload.get("acceptance_accessed") is not True
            or exclusions.get("rows") != 995
        ):
            raise EnvironmentFreezeError("acceptance EvalItem counts changed")
    else:
        raise EnvironmentFreezeError(f"unsupported transition: {transition}")


def validate_judge_summary(
    payload: Mapping[str, Any],
    *,
    transition: str,
) -> None:
    rates = _mapping(payload.get("dimension_pass_rates"), "dimension_pass_rates")
    if (
        payload.get("schema") != "delta.llm_judge_audit.v1"
        or payload.get("status") != "pass"
        or payload.get("root_checks") != "pass"
        or payload.get("items") != 32
        or set(rates) != {"truth", "version_status", "answerability"}
        or any(float(rate) < 0.90 for rate in rates.values())
        or payload.get("acceptance_accessed") is (transition == "development")
    ):
        raise EnvironmentFreezeError(f"{transition} grounded LLM audit failed")


def validate_items(
    rows: Iterable[Mapping[str, Any]],
    *,
    allowed_splits: set[str],
    expected_count: int,
) -> set[str]:
    source_ids: set[str] = set()
    eval_ids: set[str] = set()
    scorers: Counter[str] = Counter()
    count = 0
    for row in rows:
        count += 1
        if row.get("schema") != EVAL_ITEM_SCHEMA:
            raise EnvironmentFreezeError("unexpected EvalItem schema")
        source_id = _string(row.get("source_id"), "source_id")
        eval_id = _string(row.get("eval_id"), "eval_id")
        if source_id in source_ids or eval_id in eval_ids:
            raise EnvironmentFreezeError("duplicate EvalItem identity")
        if row.get("split") not in allowed_splits:
            raise EnvironmentFreezeError("EvalItem is in a forbidden split")
        scorer = _string(row.get("scorer"), "scorer")
        if scorer not in {"exact-enum-v1", "exact-structured-json-v1"}:
            raise EnvironmentFreezeError("unfrozen scorer entered environment")
        source_ids.add(source_id)
        eval_ids.add(eval_id)
        scorers[scorer] += 1
    if count != expected_count:
        raise EnvironmentFreezeError(
            f"EvalItem count mismatch: expected {expected_count}, got {count}"
        )
    return source_ids


def development_source_ids(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    source_ids = set()
    for row in rows:
        if (
            row.get("schema") != SPLIT_SCHEMA
            or row.get("split") not in {"train", "dev"}
        ):
            raise EnvironmentFreezeError("invalid development source split")
        source_id = _string(row.get("source_id"), "source_id")
        if source_id in source_ids:
            raise EnvironmentFreezeError("duplicate development source identity")
        source_ids.add(source_id)
    return source_ids


def validate_bound_environment(bindings: Mapping[str, Path]) -> dict[str, Any]:
    required = {
        "snapshots",
        "development_source_splits",
        "development_eval_items",
        "development_eval_summary",
        "development_judge_summary",
        "acceptance_eval_items",
        "acceptance_eval_exclusions",
        "acceptance_eval_summary",
        "acceptance_judge_summary",
    }
    missing = required.difference(bindings)
    if missing:
        raise EnvironmentFreezeError(
            f"required environment bindings missing: {sorted(missing)}"
        )
    validate_snapshots(load_yaml(bindings["snapshots"]))
    validate_eval_summary(
        load_json(bindings["development_eval_summary"]),
        transition="development",
    )
    validate_eval_summary(
        load_json(bindings["acceptance_eval_summary"]),
        transition="acceptance",
    )
    validate_judge_summary(
        load_json(bindings["development_judge_summary"]),
        transition="development",
    )
    validate_judge_summary(
        load_json(bindings["acceptance_judge_summary"]),
        transition="acceptance",
    )
    development_ids = development_source_ids(
        load_jsonl(bindings["development_source_splits"])
    )
    train_dev_ids = validate_items(
        load_jsonl(bindings["development_eval_items"]),
        allowed_splits={"train", "dev"},
        expected_count=87,
    )
    acceptance_ids = validate_items(
        load_jsonl(bindings["acceptance_eval_items"]),
        allowed_splits={"eval"},
        expected_count=42,
    )
    if not train_dev_ids.issubset(development_ids):
        raise EnvironmentFreezeError("development item lacks source assignment")
    overlap = sorted(development_ids.intersection(acceptance_ids))
    if overlap:
        raise EnvironmentFreezeError(
            f"cross-transition source leakage: {overlap}"
        )
    exclusions = load_jsonl(bindings["acceptance_eval_exclusions"])
    if len(exclusions) != 995:
        raise EnvironmentFreezeError("acceptance exclusion count changed")
    excluded_ids = {
        _string(row.get("source_id"), "exclusion.source_id")
        for row in exclusions
    }
    if len(excluded_ids) != len(exclusions):
        raise EnvironmentFreezeError("duplicate acceptance exclusion")
    if not excluded_ids.issubset(development_ids):
        raise EnvironmentFreezeError("exclusion is not a development source")
    return {
        "development_source_ids": len(development_ids),
        "development_eval_items": len(train_dev_ids),
        "acceptance_eval_items": len(acceptance_ids),
        "acceptance_exclusions": len(excluded_ids),
        "cross_transition_source_overlap": 0,
    }


def run_test_gate(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    gate = manifest["test_gate"]
    command = list(gate["command"])
    completed = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + "\n" + completed.stderr
    match = re.search(r"Ran ([0-9]+) tests", combined)
    observed = int(match.group(1)) if match else None
    if completed.returncode != 0:
        raise EnvironmentFreezeError(
            f"test gate failed with exit {completed.returncode}"
        )
    if observed != gate["expected_tests"]:
        raise EnvironmentFreezeError(
            f"test count mismatch: expected {gate['expected_tests']}, "
            f"got {observed}"
        )
    return {
        "command": command,
        "expected_tests": gate["expected_tests"],
        "observed_tests": observed,
        "status": "pass",
    }


def validate_freeze(
    *,
    manifest_path: Path,
    repo_root: Path,
    run_tests: bool = True,
) -> dict[str, Any]:
    manifest = load_yaml(manifest_path)
    validate_policy(manifest)
    commit = str(manifest["code_commit"])
    commit_check = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if commit_check.returncode != 0:
        raise EnvironmentFreezeError("frozen code commit is unavailable")
    bindings = verify_bindings(manifest.get("bindings"), repo_root=repo_root)
    environment = validate_bound_environment(bindings)
    test_gate = (
        run_test_gate(manifest, repo_root=repo_root)
        if run_tests
        else {
            "command": list(manifest["test_gate"]["command"]),
            "expected_tests": manifest["test_gate"]["expected_tests"],
            "observed_tests": None,
            "status": "not-run",
        }
    )
    return {
        "schema": RESULT_SCHEMA,
        "environment_id": manifest["environment_id"],
        "status": "pass",
        "freeze_state": "frozen",
        "publication_state": manifest["publication_state"],
        "manifest_sha256": _sha256(manifest_path),
        "code_commit": commit,
        "bindings": len(bindings),
        "environment": environment,
        "test_gate": test_gate,
        "target_model_runs": 0,
        "training_runs": 0,
        "fine_tuning_runs": 0,
    }


def write_result(result: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and freeze the delta_v2 EvalEnvironment."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_freeze(
        manifest_path=args.manifest,
        repo_root=args.repo_root.resolve(),
    )
    write_result(result, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
