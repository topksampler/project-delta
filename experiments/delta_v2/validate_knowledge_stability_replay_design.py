from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA = "delta.knowledge_stability_replay_design.v1"
DESIGN_ID = "delta-v2-knowledge-stability-replay-v1"
SUPPORTED_FAMILIES = (
    "python.config_field.v1",
    "python.environment_variable.v1",
)
PARTITION_ORDER = ("replay_train", "stability_dev", "verify_v2")
EXPECTED_UNCHANGED_TRAINING = {
    "objective": "assistant-only-generative-sft-v1",
    "initialization": "fresh-base",
    "base_revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "learning_rate": 0.0001,
    "optimizer_steps": 60,
    "gradient_accumulation_steps": 4,
    "seed": 20260730,
}
EXPECTED_STEP_TYPES = [
    {
        "count": 45,
        "units": [
            "acquisition_boolean_pair",
            "acquisition_boolean_pair",
            "acquisition_recall",
            "replay_boolean_pair",
        ],
    },
    {
        "count": 15,
        "units": [
            "acquisition_boolean_pair",
            "acquisition_boolean_pair",
            "acquisition_boolean_pair",
            "replay_recall",
        ],
    },
]


class StabilityReplayDesignError(ValueError):
    """The stability-replay design cannot be reproduced safely."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StabilityReplayDesignError(f"{field} must be a mapping")
    return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise StabilityReplayDesignError(
            f"cannot read YAML: {path}"
        ) from exc
    return _mapping(payload, str(path))


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise StabilityReplayDesignError(
            f"cannot read JSONL: {path}"
        ) from exc
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StabilityReplayDesignError(
                f"invalid JSON at {path}:{line_number}"
            ) from exc
        rows.append(_mapping(payload, f"{path}:{line_number}"))
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise StabilityReplayDesignError(f"{field} must be a path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise StabilityReplayDesignError(
            f"{field} must stay in the repository"
        )
    return path


def _rank(source_id: str, salt: str) -> str:
    return hashlib.sha256(
        f"{salt}\0{source_id}".encode("utf-8")
    ).hexdigest()


def _select_partition(
    candidates: Sequence[Mapping[str, Any]],
    *,
    salt: str,
    quotas: Mapping[str, int],
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    for family in SUPPORTED_FAMILIES:
        quota = quotas[family]
        ranked = sorted(
            (
                row
                for row in candidates
                if row.get("family") == family
            ),
            key=lambda row: (
                _rank(str(row["fact_id"]), salt),
                str(row["fact_id"]),
            ),
        )
        if family == "python.config_field.v1":
            paths: set[str] = set()
            family_rows = []
            for row in ranked:
                path = str(
                    _mapping(
                        row.get("evidence_after"), "evidence_after"
                    ).get("path")
                )
                if path in paths:
                    continue
                paths.add(path)
                family_rows.append(row)
                if len(family_rows) == quota:
                    break
        else:
            family_rows = ranked[:quota]
        if len(family_rows) != quota:
            raise StabilityReplayDesignError(
                f"insufficient eligible sources for {family}"
            )
        selected.extend(family_rows)
    return sorted(selected, key=lambda row: str(row["fact_id"]))


def derive_partitions(
    contract: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[dict[str, list[Mapping[str, Any]]], set[str], int]:
    source_pool = _mapping(contract.get("source_pool"), "source_pool")
    facts_binding = _mapping(source_pool.get("facts"), "source_pool.facts")
    eval_binding = _mapping(
        source_pool.get("current_eval"), "source_pool.current_eval"
    )
    facts_path = repo_root / _relative_path(
        facts_binding.get("path"), "source_pool.facts.path"
    )
    eval_path = repo_root / _relative_path(
        eval_binding.get("path"), "source_pool.current_eval.path"
    )
    if (
        _sha256(facts_path) != facts_binding.get("sha256")
        or _sha256(eval_path) != eval_binding.get("sha256")
    ):
        raise StabilityReplayDesignError("source-pool bytes changed")
    eval_rows = _load_jsonl(eval_path)
    blocked = {
        str(row["source_id"])
        for row in eval_rows
        if row.get("stratum")
        in {"acquisition_added", "retention_stable"}
    }
    if len(blocked) != 42:
        raise StabilityReplayDesignError(
            "current acquisition/retention split changed"
        )
    eligible = [
        row
        for row in _load_jsonl(facts_path)
        if row.get("status") == "stable"
        and row.get("family") in SUPPORTED_FAMILIES
        and str(row.get("fact_id")) not in blocked
    ]
    if len(eligible) != 711:
        raise StabilityReplayDesignError(
            "eligible stability source pool changed"
        )
    remaining = {
        str(row["fact_id"]): row for row in eligible
    }
    partition_contract = _mapping(
        contract.get("partitions"), "partitions"
    )
    partitions: dict[str, list[Mapping[str, Any]]] = {}
    for name in PARTITION_ORDER:
        block = _mapping(partition_contract.get(name), name)
        quotas = _mapping(block.get("family_quota"), f"{name}.family_quota")
        normalized_quotas = {
            family: int(quotas[family]) for family in SUPPORTED_FAMILIES
        }
        rows = _select_partition(
            list(remaining.values()),
            salt=str(block["salt"]),
            quotas=normalized_quotas,
        )
        if len(rows) != int(block["sources"]):
            raise StabilityReplayDesignError(
                f"{name} source count changed"
            )
        partitions[name] = rows
        for row in rows:
            remaining.pop(str(row["fact_id"]))
    return partitions, blocked, len(eligible)


def _partition_payload(
    partitions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        {
            "partition": name,
            "source_ids": [
                str(row["fact_id"]) for row in partitions[name]
            ],
        }
        for name in PARTITION_ORDER
    ]


def partition_sha256(
    partitions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    encoded = json.dumps(
        _partition_payload(partitions),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_design(
    contract: Mapping[str, Any],
    *,
    repo_root: Path,
    require_frozen_partition: bool = True,
) -> dict[str, Any]:
    if (
        contract.get("schema") != SCHEMA
        or contract.get("design_id") != DESIGN_ID
        or contract.get("experiment_id") != "delta_v2"
        or contract.get("candidate_id")
        != "d4_stability_replay_sft_control"
        or contract.get("state") != "preregistered-for-data-build"
    ):
        raise StabilityReplayDesignError("design identity changed")
    trigger = _mapping(contract.get("decision_trigger"), "decision_trigger")
    for stage, expected_hash in (
        (
            "stage_1",
            "1bd1f7ee7803f62991b0b751135afdbb7336b9f2a9870fb8f4ed38cbb4b5d46f",
        ),
        (
            "stage_2",
            "ab5a2035ab601635ecafbd8f77b5be6b8661e936e093122c64cd7a60bdbcd5ae",
        ),
    ):
        block = _mapping(trigger.get(stage), stage)
        path = repo_root / _relative_path(
            block.get("report"), f"{stage}.report"
        )
        if _sha256(path) != expected_hash:
            raise StabilityReplayDesignError(
                f"{stage} decision evidence changed"
            )
    if trigger.get("decision") != (
        "preserve-generative-objective-add-source-disjoint-stability-replay"
    ):
        raise StabilityReplayDesignError("design decision changed")

    partitions, blocked, eligible_count = derive_partitions(
        contract, repo_root=repo_root
    )
    derived_hash = partition_sha256(partitions)
    expected_hash = _mapping(
        contract.get("partitions"), "partitions"
    ).get("derived_partition_sha256")
    if require_frozen_partition and derived_hash != expected_hash:
        raise StabilityReplayDesignError(
            "derived stability partition changed"
        )
    selected_ids = {
        name: {str(row["fact_id"]) for row in rows}
        for name, rows in partitions.items()
    }
    if any(selected_ids[name] & blocked for name in PARTITION_ORDER):
        raise StabilityReplayDesignError(
            "current eval source entered a stability partition"
        )
    for index, left in enumerate(PARTITION_ORDER):
        for right in PARTITION_ORDER[index + 1 :]:
            if selected_ids[left] & selected_ids[right]:
                raise StabilityReplayDesignError(
                    "stability partitions overlap"
                )
    family_counts = {
        name: Counter(str(row["family"]) for row in rows)
        for name, rows in partitions.items()
    }
    config_paths = {
        name: [
            str(row["evidence_after"]["path"])
            for row in rows
            if row["family"] == "python.config_field.v1"
        ]
        for name, rows in partitions.items()
    }
    if any(
        len(paths) != len(set(paths)) for paths in config_paths.values()
    ):
        raise StabilityReplayDesignError(
            "config path diversity gate failed"
        )

    replay = _mapping(contract.get("replay_data"), "replay_data")
    if (
        replay.get("replay_sources") != 15
        or replay.get("replay_rows") != 105
        or replay.get("replay_pairs") != 45
        or replay.get("donor_value_verified_true_support_rate") != 1.0
        or replay.get("exact_current_eval_prompt_overlap") != 0
        or replay.get("exact_stability_dev_prompt_overlap") != 0
        or replay.get("exact_verify_v2_prompt_overlap") != 0
    ):
        raise StabilityReplayDesignError("replay-data boundary changed")
    training = _mapping(
        contract.get("candidate_training"), "candidate_training"
    )
    parent = _mapping(training.get("parent_protocol"), "parent_protocol")
    parent_path = repo_root / _relative_path(
        parent.get("path"), "parent_protocol.path"
    )
    if (
        parent.get("sha256")
        != "7c97f2581636510403f7af8792e044fd7543c8be66500f5d923f73ef0feb72cd"
        or _sha256(parent_path) != parent.get("sha256")
    ):
        raise StabilityReplayDesignError("parent protocol changed")
    unchanged = _mapping(training.get("unchanged"), "unchanged")
    schedule = _mapping(training.get("schedule"), "schedule")
    if (
        training.get("only_intervention_factor")
        != "stability-replay-microbatch"
        or dict(unchanged) != EXPECTED_UNCHANGED_TRAINING
        or schedule.get("optimizer_steps") != 60
        or schedule.get("units_per_step") != 4
        or schedule.get("total_units") != 240
        or schedule.get("acquisition_units") != 180
        or schedule.get("replay_units") != 60
        or schedule.get("replay_share") != 0.25
        or schedule.get("total_boolean_pair_units") != 180
        or schedule.get("total_recall_units") != 60
        or schedule.get("step_types") != EXPECTED_STEP_TYPES
        or schedule.get("step_type_order")
        != "deterministic-seeded-shuffle"
        or schedule.get("within_stream_order")
        != "deterministic-seeded-shuffle"
    ):
        raise StabilityReplayDesignError(
            "stability-replay schedule changed"
        )
    gates = _mapping(contract.get("gates"), "gates")
    before_training = _mapping(
        gates.get("before_training"), "before_training"
    )
    before_full_eval = _mapping(
        gates.get("before_full_eval"), "before_full_eval"
    )
    acquisition_gate = _mapping(
        before_full_eval.get("acquisition_pair_margin"),
        "acquisition_pair_margin",
    )
    stability_gate = _mapping(
        before_full_eval.get("held_out_stability_dev"),
        "held_out_stability_dev",
    )
    current_eval_gate = _mapping(
        gates.get("current_full_eval"), "current_full_eval"
    )
    verify_gate = _mapping(gates.get("verify_v2"), "verify_v2")
    if (
        dict(before_training)
        != {
            "source_partition_audit": "required",
            "corruption_shortcut_audit": "required",
            "exact_prompt_overlap": 0,
            "zero_update_stability_dev_base_baseline": "required",
        }
        or dict(acquisition_gate)
        != {
            "verified_true_minimum": 0.80,
            "verified_false_minimum": 0.80,
            "truth_conditioned_pairs_minimum": 0.80,
            "every_corruption_family_false_minimum": 0.80,
        }
        or dict(stability_gate)
        != {
            "sources": 15,
            "probes_per_source": [
                "choice",
                "boolean_true",
                "boolean_false",
                "recall",
            ],
            "items": 60,
            "choice_parseable_minimum": 0.90,
            "boolean_parseable_minimum": 1.0,
            "boolean_true_accuracy_minimum": 0.80,
            "boolean_false_accuracy_minimum": 0.90,
            "recall": "advisory",
            "pooled_score": "forbidden",
        }
        or dict(current_eval_gate)
        != {
            "protocol_id": "delta-v2-knowledge-eval-v1",
            "unchanged_required": True,
            "original_gates_required": True,
        }
        or dict(verify_gate)
        != {
            "required_before_promotion": True,
            "stable_sources": 21,
            "probes_per_source": [
                "choice",
                "boolean_true",
                "boolean_false",
                "recall",
            ],
            "pooled_score": "forbidden",
        }
        or gates.get("promotion_authorized") is not False
    ):
        raise StabilityReplayDesignError("verification gates changed")
    boundary = _mapping(
        contract.get("execution_boundary"), "execution_boundary"
    )
    if boundary.get("partition_build_authorized") is not True or any(
        boundary.get(field) is not False
        for field in (
            "replay_data_build_authorized",
            "base_baseline_authorized",
            "lora_training_authorized",
            "qlora_authorized",
            "full_weight_authorized",
            "reinforcement_authorized",
            "modal_job_authorized",
            "b2_write_authorized",
            "promotion_authorized",
        )
    ):
        raise StabilityReplayDesignError("execution boundary changed")
    return {
        "design_id": DESIGN_ID,
        "status": "pass",
        "eligible_sources": eligible_count,
        "blocked_current_eval_sources": len(blocked),
        "partition_sha256": derived_hash,
        "partitions": {
            name: {
                "sources": len(partitions[name]),
                "families": dict(sorted(family_counts[name].items())),
                "config_paths": len(config_paths[name]),
                "source_ids": sorted(selected_ids[name]),
            }
            for name in PARTITION_ORDER
        },
        "model_invocations_completed": 0,
        "optimizer_steps_completed": 0,
        "training_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--design",
        type=Path,
        default=Path(
            "experiments/delta_v2/"
            "knowledge_stability_replay_design.yaml"
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--derive-only", action="store_true")
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    design_path = args.design
    if not design_path.is_absolute():
        design_path = repo_root / design_path
    design = _load_yaml(design_path)
    result = validate_design(
        design,
        repo_root=repo_root,
        require_frozen_partition=not args.derive_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
