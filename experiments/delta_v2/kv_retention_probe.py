from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from experiments.delta_v2.inventory import (
    load_snapshot_manifest,
    select_transition,
)


PROBE_SCHEMA = "delta.behavior_probe.v1"
RESULT_SCHEMA = "delta.behavior_probe_result.v1"
RUNTIME_SENTINEL = "DELTA_V2_RUNTIME_RESULT="
ENV_KEY = "VLLM_PREFIX_CACHE_RETENTION_INTERVAL"
ROLES = ("development_before", "development_after")
ROLE_EXPECTATIONS = {
    "development_before": "expected_before",
    "development_after": "expected_after",
}
OUTCOME_KINDS = {"cache_state", "construction_error"}


class KVRetentionProbeError(ValueError):
    """The mechanics probe contract, source tree, or runtime is invalid."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KVRetentionProbeError(f"{field} must be a mapping")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise KVRetentionProbeError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise KVRetentionProbeError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise KVRetentionProbeError(f"{field} must be a non-negative integer")
    return value


def _validate_outcome(value: Any, field: str) -> None:
    outcome = _mapping(value, field)
    kind = outcome.get("kind")
    if kind not in OUTCOME_KINDS:
        raise KVRetentionProbeError(f"{field}.kind is unsupported")
    if kind == "cache_state":
        if set(outcome) != {
            "kind",
            "cached_indices",
            "replay_computed_tokens",
        }:
            raise KVRetentionProbeError(
                f"{field} cache_state has unexpected fields"
            )
        indices = outcome["cached_indices"]
        if (
            not isinstance(indices, list)
            or any(
                not isinstance(index, int) or isinstance(index, bool) or index < 0
                for index in indices
            )
            or indices != sorted(set(indices))
        ):
            raise KVRetentionProbeError(
                f"{field}.cached_indices must be sorted unique integers"
            )
        _nonnegative_int(
            outcome["replay_computed_tokens"],
            f"{field}.replay_computed_tokens",
        )
    else:
        if set(outcome) != {"kind", "category"}:
            raise KVRetentionProbeError(
                f"{field} construction_error has unexpected fields"
            )
        if outcome["category"] not in {
            "non_negative",
            "scheduler_block_size_multiple",
        }:
            raise KVRetentionProbeError(f"{field}.category is unsupported")


def validate_probe(probe: Mapping[str, Any]) -> None:
    if probe.get("schema") != PROBE_SCHEMA:
        raise KVRetentionProbeError("unsupported BehaviorProbe schema")
    if not _string(probe.get("probe_id"), "probe_id").startswith(
        "behavior-probe:"
    ):
        raise KVRetentionProbeError("probe_id must use behavior-probe: prefix")
    if not _string(probe.get("candidate_id"), "candidate_id").startswith(
        "feature-candidate:"
    ):
        raise KVRetentionProbeError(
            "candidate_id must use feature-candidate: prefix"
        )
    if probe.get("transition") != "development":
        raise KVRetentionProbeError("mechanics probe may only use development")
    if probe.get("claim_scope") != "complete-candidate":
        raise KVRetentionProbeError("mechanics probe must cover the full candidate")

    scenario = _mapping(probe.get("scenario"), "scenario")
    if scenario.get("id") != "pure-swa-16x16-v1":
        raise KVRetentionProbeError("unsupported scenario")
    for field in (
        "block_size",
        "sliding_window",
        "prompt_blocks",
        "num_cache_blocks",
        "max_model_len",
    ):
        _positive_int(scenario.get(field), f"scenario.{field}")

    cases = probe.get("cases")
    if not isinstance(cases, list) or not cases:
        raise KVRetentionProbeError("cases must be a non-empty list")
    case_ids: set[str] = set()
    for index, raw_case in enumerate(cases):
        case = _mapping(raw_case, f"cases[{index}]")
        case_id = _string(case.get("case_id"), f"cases[{index}].case_id")
        if case_id in case_ids:
            raise KVRetentionProbeError(f"duplicate case_id: {case_id}")
        case_ids.add(case_id)
        environment_value = case.get("environment_value")
        if environment_value is not None and not isinstance(environment_value, str):
            raise KVRetentionProbeError(
                f"cases[{index}].environment_value must be a string or null"
            )
        _validate_outcome(
            case.get("expected_before"),
            f"cases[{index}].expected_before",
        )
        _validate_outcome(
            case.get("expected_after"),
            f"cases[{index}].expected_after",
        )

    runtime = _mapping(probe.get("runtime"), "runtime")
    if runtime.get("target") != "local-cpu":
        raise KVRetentionProbeError("mechanics probe must target local CPU")
    if runtime.get("environment_status") != "development-only-unfrozen":
        raise KVRetentionProbeError("development runtime must not claim a freeze")
    dependency_source = _mapping(
        runtime.get("dependency_source"),
        "runtime.dependency_source",
    )
    if dependency_source.get("snapshot_role") != "development_after":
        raise KVRetentionProbeError(
            "dependency source must be the development-after snapshot"
        )
    if dependency_source.get("path") != "requirements/common.txt":
        raise KVRetentionProbeError("unexpected dependency source path")
    if runtime.get("source_execution") != "exact-detached-worktree-v1":
        raise KVRetentionProbeError("unsupported source execution mode")
    if runtime.get("process_isolation") != "one-process-per-snapshot":
        raise KVRetentionProbeError("unsupported process isolation")


def load_probe(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise KVRetentionProbeError(f"cannot read mechanics probe: {path}") from exc
    probe = _mapping(payload, "probe")
    validate_probe(probe)
    return probe


def _git(source_tree: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_tree), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise KVRetentionProbeError(
            f"git {' '.join(args)} failed for {source_tree}: {exc.stderr.strip()}"
        ) from exc
    return result.stdout.strip()


def verify_source_tree(
    source_tree: Path,
    *,
    expected_commit: str,
    expected_tree: str,
) -> dict[str, str]:
    observed_commit = _git(source_tree, "rev-parse", "HEAD")
    observed_tree = _git(source_tree, "rev-parse", "HEAD^{tree}")
    expected_tree_oid = expected_tree.removeprefix("git-tree-sha1:")
    if observed_commit != expected_commit:
        raise KVRetentionProbeError(
            f"source commit mismatch: expected {expected_commit}, "
            f"observed {observed_commit}"
        )
    if observed_tree != expected_tree_oid:
        raise KVRetentionProbeError(
            f"source tree mismatch: expected {expected_tree_oid}, "
            f"observed {observed_tree}"
        )
    return {
        "commit_sha": observed_commit,
        "content_hash": f"git-tree-sha1:{observed_tree}",
    }


def _normalize_construction_error(
    message: str,
    configured_value: str | None,
    scheduler_block_size: int,
) -> str:
    if (
        "must be non-negative" not in message
        or "multiple of scheduler_block_size" not in message
    ):
        raise KVRetentionProbeError(
            f"unexpected KV-cache construction ValueError: {message}"
        )
    try:
        interval = int(configured_value) if configured_value is not None else None
    except ValueError as exc:
        raise KVRetentionProbeError(
            f"invalid preregistered interval: {configured_value}"
        ) from exc
    if interval is not None and interval < 0:
        return "non_negative"
    if interval is not None and interval % scheduler_block_size != 0:
        return "scheduler_block_size_multiple"
    raise KVRetentionProbeError(
        "construction error did not match the configured numeric constraint"
    )


def _runtime_packages() -> dict[str, str]:
    names = ("torch", "transformers", "pydantic", "msgspec", "numpy")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "absent"
    return versions


def _runtime_case(
    case: Mapping[str, Any],
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import (
        get_request_block_hasher,
        init_none_hash,
    )
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        SlidingWindowSpec,
    )
    from vllm.v1.request import Request

    environment_value = case.get("environment_value")
    if environment_value is None:
        os.environ.pop(ENV_KEY, None)
    else:
        os.environ[ENV_KEY] = str(environment_value)

    block_size = int(scenario["block_size"])
    prompt_blocks = int(scenario["prompt_blocks"])
    init_none_hash(sha256)
    kv_cache_config = KVCacheConfig(
        num_blocks=int(scenario["num_cache_blocks"]),
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=int(scenario["sliding_window"]),
                ),
            )
        ],
    )
    manager_kwargs: dict[str, Any] = {
        "kv_cache_config": kv_cache_config,
        "max_model_len": int(scenario["max_model_len"]),
        "enable_caching": True,
        "hash_block_size": block_size,
    }
    if "scheduler_block_size" in inspect.signature(
        KVCacheManager.__init__
    ).parameters:
        manager_kwargs["scheduler_block_size"] = block_size
    try:
        manager = KVCacheManager(**manager_kwargs)
    except ValueError as exc:
        return {
            "kind": "construction_error",
            "category": _normalize_construction_error(
                str(exc),
                str(environment_value) if environment_value is not None else None,
                block_size,
            ),
        }

    token_ids = [
        token
        for token in range(prompt_blocks)
        for _ in range(block_size)
    ]
    sampling_params = SamplingParams(max_tokens=1)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    request = Request(
        request_id=f"fill-{case['case_id']}",
        prompt_token_ids=token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )
    computed_blocks, num_computed_tokens = manager.get_computed_blocks(request)
    blocks = manager.allocate_slots(
        request,
        len(token_ids),
        num_computed_tokens,
        computed_blocks,
    )
    if blocks is None:
        raise KVRetentionProbeError("KV-cache scenario could not allocate blocks")

    pool = manager.block_pool
    cached_indices = [
        index
        for index in range(prompt_blocks)
        if pool.get_cached_block(
            request.block_hashes[index],
            kv_cache_group_ids=[0],
        )
        is not None
    ]
    replay = Request(
        request_id=f"replay-{case['case_id']}",
        prompt_token_ids=token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )
    _, replay_computed_tokens = manager.get_computed_blocks(replay)
    return {
        "kind": "cache_state",
        "cached_indices": cached_indices,
        "replay_computed_tokens": replay_computed_tokens,
    }


def _runtime_main(cases_json: str, scenario_json: str) -> int:
    cases = json.loads(cases_json)
    scenario = json.loads(scenario_json)
    observations = [
        {
            "case_id": case["case_id"],
            "outcome": _runtime_case(case, scenario),
        }
        for case in cases
    ]
    result = {
        "runtime": {
            "python": platform.python_version(),
            "platform": f"{platform.system()}-{platform.machine()}",
            "packages": _runtime_packages(),
        },
        "observations": observations,
    }
    print(
        RUNTIME_SENTINEL
        + json.dumps(result, sort_keys=True, separators=(",", ":"))
    )
    return 0


def run_snapshot_process(
    *,
    python: Path,
    source_tree: Path,
    cases: Sequence[Mapping[str, Any]],
    scenario: Mapping[str, Any],
    timeout_seconds: int,
) -> Mapping[str, Any]:
    environment = os.environ.copy()
    harness_root = Path(__file__).resolve().parents[2]
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(source_tree), str(harness_root))
    )
    environment["VLLM_LOGGING_LEVEL"] = "ERROR"
    environment["VLLM_NO_USAGE_STATS"] = "1"
    environment["VLLM_DO_NOT_TRACK"] = "1"
    command = [
        str(python),
        str(Path(__file__).resolve()),
        "_runtime",
        "--cases-json",
        json.dumps(list(cases), sort_keys=True, separators=(",", ":")),
        "--scenario-json",
        json.dumps(dict(scenario), sort_keys=True, separators=(",", ":")),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise KVRetentionProbeError(
            f"snapshot runtime exceeded {timeout_seconds} seconds"
        ) from exc
    if result.returncode != 0:
        raise KVRetentionProbeError(
            "snapshot runtime failed: " + result.stderr.strip()
        )
    payloads = [
        line.removeprefix(RUNTIME_SENTINEL)
        for line in result.stdout.splitlines()
        if line.startswith(RUNTIME_SENTINEL)
    ]
    if len(payloads) != 1:
        raise KVRetentionProbeError(
            "snapshot runtime did not emit exactly one result"
        )
    try:
        payload = json.loads(payloads[0])
    except json.JSONDecodeError as exc:
        raise KVRetentionProbeError("snapshot runtime emitted invalid JSON") from exc
    return _mapping(payload, "snapshot runtime result")


def evaluate_observations(
    probe: Mapping[str, Any],
    observations_by_role: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_role: dict[str, dict[str, Mapping[str, Any]]] = {}
    for role in ROLES:
        runtime_result = _mapping(observations_by_role.get(role), role)
        raw_observations = runtime_result.get("observations")
        if not isinstance(raw_observations, list):
            raise KVRetentionProbeError(f"{role}.observations must be a list")
        by_role[role] = {}
        for item in raw_observations:
            observation = _mapping(item, f"{role}.observation")
            case_id = _string(observation.get("case_id"), "case_id")
            if case_id in by_role[role]:
                raise KVRetentionProbeError(
                    f"duplicate runtime case for {role}: {case_id}"
                )
            outcome = _mapping(observation.get("outcome"), "outcome")
            _validate_outcome(outcome, "outcome")
            by_role[role][case_id] = outcome

    results: list[dict[str, Any]] = []
    for raw_case in probe["cases"]:
        case = _mapping(raw_case, "case")
        case_id = str(case["case_id"])
        observations: dict[str, Mapping[str, Any]] = {}
        statuses: list[str] = []
        for role in ROLES:
            if case_id not in by_role[role]:
                raise KVRetentionProbeError(
                    f"missing runtime case for {role}: {case_id}"
                )
            outcome = by_role[role][case_id]
            observations[role] = outcome
            expected = case[ROLE_EXPECTATIONS[role]]
            statuses.append("pass" if outcome == expected else "fail")
        results.append(
            {
                "case_id": case_id,
                "environment_value": case.get("environment_value"),
                "expected_before": case["expected_before"],
                "observed_before": observations["development_before"],
                "expected_after": case["expected_after"],
                "observed_after": observations["development_after"],
                "status": "pass" if all(s == "pass" for s in statuses) else "fail",
            }
        )
    return results


def run_probe(
    *,
    probe: Mapping[str, Any],
    snapshots_path: Path,
    python: Path,
    before_tree: Path,
    after_tree: Path,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    validate_probe(probe)
    manifest = load_snapshot_manifest(snapshots_path)
    before, after = select_transition(manifest, "development")
    snapshots = {
        "development_before": before,
        "development_after": after,
    }
    source_trees = {
        "development_before": before_tree,
        "development_after": after_tree,
    }
    source_records: dict[str, dict[str, str]] = {}
    runtime_results: dict[str, Mapping[str, Any]] = {}
    for role in ROLES:
        snapshot = snapshots[role]
        source_records[role] = verify_source_tree(
            source_trees[role],
            expected_commit=snapshot.commit_sha,
            expected_tree=snapshot.content_hash,
        )
        runtime_results[role] = run_snapshot_process(
            python=python,
            source_tree=source_trees[role],
            cases=probe["cases"],
            scenario=probe["scenario"],
            timeout_seconds=timeout_seconds,
        )

    dependency_path = after_tree / probe["runtime"]["dependency_source"]["path"]
    try:
        dependency_bytes = dependency_path.read_bytes()
    except OSError as exc:
        raise KVRetentionProbeError(
            f"cannot read dependency source: {dependency_path}"
        ) from exc
    case_results = evaluate_observations(probe, runtime_results)
    runtime_metadata = runtime_results["development_after"]["runtime"]
    if runtime_results["development_before"]["runtime"] != runtime_metadata:
        raise KVRetentionProbeError(
            "before and after did not execute in the same runtime"
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
        "runtime": {
            **runtime_metadata,
            "environment_status": probe["runtime"]["environment_status"],
            "dependency_source": {
                "snapshot_role": "development_after",
                "path": probe["runtime"]["dependency_source"]["path"],
                "sha256": hashlib.sha256(dependency_bytes).hexdigest(),
            },
        },
        "cases": case_results,
        "promotion_rule": probe["promotion_rule"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the delta_v2 prefix-cache mechanics probe."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--probe", type=Path, required=True)
    run_parser.add_argument("--snapshots", type=Path, required=True)
    run_parser.add_argument("--python", type=Path, required=True)
    run_parser.add_argument("--before-tree", type=Path, required=True)
    run_parser.add_argument("--after-tree", type=Path, required=True)
    run_parser.add_argument("--output", type=Path)
    run_parser.add_argument("--timeout-seconds", type=int, default=180)

    runtime_parser = subparsers.add_parser("_runtime")
    runtime_parser.add_argument("--cases-json", required=True)
    runtime_parser.add_argument("--scenario-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "_runtime":
        return _runtime_main(args.cases_json, args.scenario_json)
    result = run_probe(
        probe=load_probe(args.probe),
        snapshots_path=args.snapshots,
        python=args.python,
        before_tree=args.before_tree,
        after_tree=args.after_tree,
        timeout_seconds=args.timeout_seconds,
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
