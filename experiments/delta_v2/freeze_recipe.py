from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA = "delta.build_recipe_freeze.v1"
RESULT_SCHEMA = "delta.build_recipe_freeze_result.v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
AUDIT_KINDS = {
    "file_inventory",
    "atomic_facts",
    "behavior_probe",
    "source_split",
    "eval_items",
    "llm_judge",
}


class RecipeFreezeError(ValueError):
    """The development recipe cannot be frozen or unlocked."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecipeFreezeError(f"{field} must be a mapping")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RecipeFreezeError(f"{field} must be a non-empty string")
    return value


def _relative_path(value: Any, field: str) -> Path:
    raw = _string(value, field)
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise RecipeFreezeError(f"{field} must stay within the repository")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RecipeFreezeError(f"cannot read freeze manifest: {path}") from exc
    return _mapping(payload, str(path))


def load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecipeFreezeError(f"cannot read audit summary: {path}") from exc
    return _mapping(payload, str(path))


def validate_policy(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != SCHEMA:
        raise RecipeFreezeError("unsupported recipe-freeze schema")
    _string(manifest.get("recipe_id"), "recipe_id")
    if (
        manifest.get("experiment_id") != "delta_v2"
        or manifest.get("repository") != "vllm-project/vllm"
        or manifest.get("frozen_transition") != "development"
    ):
        raise RecipeFreezeError("freeze manifest targets the wrong experiment")
    commit = _string(manifest.get("code_commit"), "code_commit")
    if not COMMIT_SHA.fullmatch(commit):
        raise RecipeFreezeError("code_commit must be a full commit SHA")

    protocol = _mapping(manifest.get("protocol"), "protocol")
    actor = _mapping(protocol.get("audit_actor"), "audit_actor")
    if (
        actor.get("kind") != "evidence-grounded-llm"
        or actor.get("amendment_timing")
        != "before-development-freeze-and-before-acceptance-unseal"
        or actor.get("truth_authority")
        != "deterministic-sources-and-executable-probes"
    ):
        raise RecipeFreezeError("audit protocol amendment is not explicit")
    feature_policy = _mapping(
        protocol.get("acceptance_feature_policy"),
        "acceptance_feature_policy",
    )
    if (
        feature_policy.get("state") != "excluded-from-v1-eval"
        or feature_policy.get("development_feature_evidence_retained") is not True
    ):
        raise RecipeFreezeError("acceptance feature limitation is not explicit")

    unlock = _mapping(manifest.get("acceptance_unlock"), "acceptance_unlock")
    if (
        unlock.get("initial_state") != "sealed"
        or unlock.get("condition") != "this-manifest-validates"
    ):
        raise RecipeFreezeError("acceptance unlock is not freeze-gated")
    forbidden = set(unlock.get("forbidden_operations", []))
    required_forbidden = {
        "target-model-run",
        "training",
        "fine-tuning",
        "modal-job",
        "lambda-job",
        "b2-write",
    }
    if forbidden != required_forbidden:
        raise RecipeFreezeError("forbidden acceptance operations changed")

    test_gate = _mapping(manifest.get("test_gate"), "test_gate")
    command = test_gate.get("command")
    if not isinstance(command, list) or any(
        not isinstance(part, str) or not part for part in command
    ):
        raise RecipeFreezeError("test command must be an argument list")
    if test_gate.get("expected_tests") != 111:
        raise RecipeFreezeError("test gate must bind all 111 tests")
    if manifest.get("freeze_state") != "frozen":
        raise RecipeFreezeError("freeze manifest must declare frozen")


def verify_file_records(
    records: Any,
    *,
    repo_root: Path,
    field: str,
) -> list[dict[str, str]]:
    if not isinstance(records, list) or not records:
        raise RecipeFreezeError(f"{field} must be a non-empty list")
    verified: list[dict[str, str]] = []
    seen: set[Path] = set()
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"{field}[{index}]")
        relative = _relative_path(
            record.get("path"),
            f"{field}[{index}].path",
        )
        expected = _string(
            record.get("sha256"),
            f"{field}[{index}].sha256",
        )
        if not SHA256.fullmatch(expected):
            raise RecipeFreezeError(f"invalid SHA-256 for {relative}")
        if relative in seen:
            raise RecipeFreezeError(f"duplicate frozen path: {relative}")
        seen.add(relative)
        absolute = repo_root / relative
        if not absolute.is_file():
            raise RecipeFreezeError(f"frozen file is missing: {relative}")
        actual = _sha256(absolute)
        if actual != expected:
            raise RecipeFreezeError(
                f"frozen file hash mismatch: {relative}; "
                f"expected {expected}, got {actual}"
            )
        verified.append({"path": str(relative), "sha256": actual})
    return verified


def _development_sources_only(payload: Mapping[str, Any]) -> None:
    sources = _mapping(payload.get("sources"), "probe.sources")
    if set(sources) != {"development_before", "development_after"}:
        raise RecipeFreezeError("probe did not use exactly development sources")


def validate_audit(kind: str, payload: Mapping[str, Any]) -> None:
    if kind not in AUDIT_KINDS:
        raise RecipeFreezeError(f"unsupported audit kind: {kind}")
    if payload.get("acceptance_accessed") is True:
        raise RecipeFreezeError(f"{kind} accessed acceptance before freeze")
    if kind == "file_inventory":
        if (
            payload.get("schema") != "delta.file_inventory.audit.v1"
            or payload.get("transition") != "development"
            or not all(
                payload.get(field) is True
                for field in (
                    "reconstructs_before",
                    "reconstructs_after",
                    "unique_file_ids",
                    "deterministic_order",
                )
            )
        ):
            raise RecipeFreezeError("file inventory audit failed")
    elif kind == "atomic_facts":
        if (
            payload.get("schema") != "delta.atomic_fact_extraction.audit.v1"
            or payload.get("transition") != "development"
            or not all(
                payload.get(field) is True
                for field in (
                    "reconstructs_before",
                    "reconstructs_after",
                    "unique_fact_ids",
                    "deterministic_order",
                )
            )
        ):
            raise RecipeFreezeError("atomic-fact audit failed")
    elif kind == "behavior_probe":
        if (
            payload.get("schema") != "delta.behavior_probe_result.v1"
            or payload.get("status") != "pass"
        ):
            raise RecipeFreezeError("behavior probe failed")
        _development_sources_only(payload)
    elif kind == "source_split":
        if (
            payload.get("schema") != "delta.source_split_audit.v1"
            or payload.get("status") != "pass"
            or payload.get("transition") != "development"
            or payload.get("eval_sources") != 0
            or payload.get("cross_split_units") != []
        ):
            raise RecipeFreezeError("development source split audit failed")
    elif kind == "eval_items":
        if (
            payload.get("schema") != "delta.eval_item_build_audit.v1"
            or payload.get("status") != "pass"
            or payload.get("eval_items") != 0
            or payload.get("unique_eval_ids") is not True
            or payload.get("unique_source_ids") is not True
            or payload.get("control_balance_matches") is not True
        ):
            raise RecipeFreezeError("development EvalItem audit failed")
    elif kind == "llm_judge":
        rates = _mapping(
            payload.get("dimension_pass_rates"),
            "dimension_pass_rates",
        )
        if (
            payload.get("schema") != "delta.llm_judge_audit.v1"
            or payload.get("status") != "pass"
            or payload.get("root_checks") != "pass"
            or set(rates) != {"truth", "version_status", "answerability"}
            or any(float(rate) < 0.90 for rate in rates.values())
        ):
            raise RecipeFreezeError("evidence-grounded LLM audit failed")


def verify_audits(
    records: Any,
    *,
    repo_root: Path,
) -> list[dict[str, str]]:
    if not isinstance(records, list) or not records:
        raise RecipeFreezeError("audit_summaries must be a non-empty list")
    verified: list[dict[str, str]] = []
    seen_kinds: set[str] = set()
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"audit_summaries[{index}]")
        kind = _string(record.get("kind"), f"audit_summaries[{index}].kind")
        relative = _relative_path(
            record.get("path"),
            f"audit_summaries[{index}].path",
        )
        expected = _string(
            record.get("sha256"),
            f"audit_summaries[{index}].sha256",
        )
        absolute = repo_root / relative
        if not absolute.is_file() or _sha256(absolute) != expected:
            raise RecipeFreezeError(f"audit summary hash mismatch: {relative}")
        validate_audit(kind, load_json(absolute))
        seen_kinds.add(kind)
        verified.append(
            {"kind": kind, "path": str(relative), "sha256": expected}
        )
    if seen_kinds != AUDIT_KINDS:
        raise RecipeFreezeError(
            f"audit kinds incomplete: expected {sorted(AUDIT_KINDS)}"
        )
    return verified


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
        raise RecipeFreezeError(
            f"test gate failed with exit {completed.returncode}"
        )
    if observed != gate["expected_tests"]:
        raise RecipeFreezeError(
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
        raise RecipeFreezeError("frozen code commit is unavailable")
    recipe_files = verify_file_records(
        manifest.get("recipe_files"),
        repo_root=repo_root,
        field="recipe_files",
    )
    evidence_files = verify_file_records(
        manifest.get("development_evidence"),
        repo_root=repo_root,
        field="development_evidence",
    )
    audits = verify_audits(
        manifest.get("audit_summaries"),
        repo_root=repo_root,
    )
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
    manifest_hash = _sha256(manifest_path)
    prior_attempts = manifest.get("prior_acceptance_attempts", [])
    if not isinstance(prior_attempts, list):
        raise RecipeFreezeError("prior_acceptance_attempts must be a list")
    return {
        "schema": RESULT_SCHEMA,
        "recipe_id": manifest["recipe_id"],
        "status": "pass",
        "freeze_state": "frozen",
        "manifest_sha256": manifest_hash,
        "code_commit": commit,
        "recipe_files": len(recipe_files),
        "development_evidence_files": len(evidence_files),
        "audit_summaries": len(audits),
        "test_gate": test_gate,
        "prior_acceptance_attempts_recorded": len(prior_attempts),
        "selected_acceptance_accessed": False,
        "acceptance_unlock": "granted",
        "forbidden_operations_remain_forbidden": True,
    }


def write_result(result: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and freeze the delta_v2 development BUILD recipe."
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
