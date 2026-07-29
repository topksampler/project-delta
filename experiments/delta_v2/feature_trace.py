from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from experiments.delta_v2.inventory import (
    load_snapshot_manifest,
    select_transition,
    verify_snapshot,
)


CANDIDATE_SCHEMA = "delta.feature_candidate.v1"
PR_SCHEMA = "delta.pull_request_evidence.v1"
PENDING = "pending"
HEX_SHA1 = re.compile(r"^[0-9a-f]{40}$")


class FeatureTraceError(ValueError):
    """A feature trace record is malformed or contradicts pinned evidence."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FeatureTraceError(f"{field} must be a mapping")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise FeatureTraceError(f"{field} must be a non-empty string")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise FeatureTraceError(f"{field} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise FeatureTraceError(f"{field} must not contain duplicates")
    return value


def load_yaml_record(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FeatureTraceError(f"cannot read YAML record: {path}") from exc
    return _mapping(payload, str(path))


def validate_candidate(record: Mapping[str, Any]) -> None:
    if record.get("schema") != CANDIDATE_SCHEMA:
        raise FeatureTraceError("unsupported FeatureCandidate schema")
    if not _string(record.get("candidate_id"), "candidate_id").startswith(
        "feature-candidate:"
    ):
        raise FeatureTraceError("candidate_id must use feature-candidate: prefix")
    _string(record.get("repository"), "repository")
    if record.get("transition") != "development":
        raise FeatureTraceError("only the development transition may be traced")
    if record.get("proposed_status") not in {
        "stable",
        "added",
        "removed",
        "changed",
    }:
        raise FeatureTraceError("unsupported proposed_status")
    if record.get("proposal_method") not in {
        "human",
        "deterministic",
        "teacher-assisted",
    }:
        raise FeatureTraceError("unsupported proposal_method")
    if record.get("verification_status") != PENDING:
        raise FeatureTraceError(
            "FeatureCandidate must remain pending until promotion"
        )
    evidence_ids = _string_list(record.get("evidence_ids"), "evidence_ids")
    if not any(item.startswith("fact:") for item in evidence_ids):
        raise FeatureTraceError("FeatureCandidate requires atomic-fact evidence")
    if not any(item.startswith("file:") for item in evidence_ids):
        raise FeatureTraceError("FeatureCandidate requires file evidence")
    if not any(item.startswith("pull-request:") for item in evidence_ids):
        raise FeatureTraceError("FeatureCandidate requires PR provenance")
    if record.get("behavior_probe_ids") != []:
        raise FeatureTraceError(
            "pending FeatureCandidate cannot claim BehaviorProbe evidence"
        )


def validate_pr_evidence(record: Mapping[str, Any]) -> None:
    if record.get("schema") != PR_SCHEMA:
        raise FeatureTraceError("unsupported PullRequestEvidence schema")
    evidence_id = _string(record.get("evidence_id"), "evidence_id")
    repository = _string(record.get("repository"), "repository")
    number = record.get("number")
    if not isinstance(number, int) or number <= 0:
        raise FeatureTraceError("number must be a positive integer")
    expected_id = f"pull-request:{repository}#{number}"
    if evidence_id != expected_id:
        raise FeatureTraceError(f"evidence_id must equal {expected_id}")
    if record.get("state") != "merged":
        raise FeatureTraceError("only merged PR evidence belongs in this record")
    merge_sha = _string(record.get("merge_commit_sha"), "merge_commit_sha")
    if not HEX_SHA1.fullmatch(merge_sha):
        raise FeatureTraceError("merge_commit_sha must be a full SHA-1")
    changed_paths = _string_list(record.get("changed_paths"), "changed_paths")
    counts = _mapping(record.get("counts"), "counts")
    if counts.get("changed_files") != len(changed_paths):
        raise FeatureTraceError("changed_files count does not match changed_paths")
    membership = _mapping(
        record.get("development_transition_membership"),
        "development_transition_membership",
    )
    if membership.get("development_before_contains_merge") is not False:
        raise FeatureTraceError("PR merge must not be contained in development_before")
    if membership.get("development_after_contains_merge") is not True:
        raise FeatureTraceError("PR merge must be contained in development_after")
    if record.get("provenance_role") != "candidate-discovery-and-grouping-only":
        raise FeatureTraceError("PR evidence must not be labeled as truth")


def load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FeatureTraceError(f"cannot read JSONL evidence: {path}") from exc
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FeatureTraceError(
                f"invalid JSON on {path}:{line_number}"
            ) from exc
        records.append(_mapping(record, f"{path}:{line_number}"))
    return records


def audit_evidence_links(
    candidate: Mapping[str, Any],
    pr_evidence: Mapping[str, Any],
    fact_records: Iterable[Mapping[str, Any]],
    file_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_candidate(candidate)
    validate_pr_evidence(pr_evidence)

    evidence_ids = set(candidate["evidence_ids"])
    fact_ids = {record.get("fact_id") for record in fact_records}
    file_by_id = {
        record.get("file_id"): record
        for record in file_records
        if isinstance(record.get("file_id"), str)
    }
    candidate_fact_ids = {
        item for item in evidence_ids if item.startswith("fact:")
    }
    candidate_file_ids = {
        item for item in evidence_ids if item.startswith("file:")
    }
    unresolved = sorted(
        (candidate_fact_ids - fact_ids)
        | (candidate_file_ids - set(file_by_id))
        | ({pr_evidence["evidence_id"]} - evidence_ids)
    )

    linked_paths = {
        file_by_id[file_id].get("path_after")
        or file_by_id[file_id].get("path_before")
        for file_id in candidate_file_ids & set(file_by_id)
    }
    pr_paths = set(pr_evidence["changed_paths"])
    missing_pr_paths = sorted(pr_paths - linked_paths)
    extra_candidate_paths = sorted(linked_paths - pr_paths)
    status = (
        "pass"
        if not unresolved and not missing_pr_paths and not extra_candidate_paths
        else "fail"
    )
    return {
        "schema": "delta.feature_trace_audit.v1",
        "candidate_id": candidate["candidate_id"],
        "pull_request_evidence_id": pr_evidence["evidence_id"],
        "status": status,
        "unresolved_evidence_ids": unresolved,
        "missing_pr_paths": missing_pr_paths,
        "extra_candidate_paths": extra_candidate_paths,
        "linked_fact_count": len(candidate_fact_ids),
        "linked_file_count": len(candidate_file_ids),
    }


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def verify_transition_membership(
    repo: Path,
    snapshots_path: Path,
    pr_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    validate_pr_evidence(pr_evidence)
    manifest = load_snapshot_manifest(snapshots_path)
    before, after = select_transition(manifest, "development")
    verify_snapshot(repo, before)
    verify_snapshot(repo, after)
    merge_sha = str(pr_evidence["merge_commit_sha"])
    _git(repo, "cat-file", "-e", f"{merge_sha}^{{commit}}")

    before_result = _git(
        repo,
        "merge-base",
        "--is-ancestor",
        merge_sha,
        before.commit_sha,
        check=False,
    )
    after_result = _git(
        repo,
        "merge-base",
        "--is-ancestor",
        merge_sha,
        after.commit_sha,
        check=False,
    )
    if before_result.returncode not in {0, 1} or after_result.returncode not in {
        0,
        1,
    }:
        raise FeatureTraceError("git could not determine PR transition membership")
    observed = {
        "development_before_contains_merge": before_result.returncode == 0,
        "development_after_contains_merge": after_result.returncode == 0,
    }
    declared = pr_evidence["development_transition_membership"]
    matches_record = all(
        observed[key] is declared[key]
        for key in (
            "development_before_contains_merge",
            "development_after_contains_merge",
        )
    )
    return {
        "schema": "delta.pull_request_membership_audit.v1",
        "pull_request_evidence_id": pr_evidence["evidence_id"],
        "merge_commit_sha": merge_sha,
        **observed,
        "matches_record": matches_record,
        "status": "pass" if matches_record else "fail",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit one pending delta_v2 FeatureCandidate trace."
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--pull-request", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--files", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidate = load_yaml_record(args.candidate)
    pr_evidence = load_yaml_record(args.pull_request)
    evidence_audit = audit_evidence_links(
        candidate,
        pr_evidence,
        load_jsonl(args.facts),
        load_jsonl(args.files),
    )
    membership_audit = verify_transition_membership(
        args.repo,
        args.snapshots,
        pr_evidence,
    )
    result = {
        "schema": "delta.feature_trace_check.v1",
        "status": (
            "pass"
            if evidence_audit["status"] == membership_audit["status"] == "pass"
            else "fail"
        ),
        "evidence": evidence_audit,
        "transition_membership": membership_audit,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

