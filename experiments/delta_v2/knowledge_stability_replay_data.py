from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from experiments.delta_v2.knowledge_adaptation import (
    CARD_VERSION,
    _display_name,
    _fact_probe_rows,
    _recall_prompt,
    _verify_prompt,
    canonical_json,
    knowledge_card,
)
from experiments.delta_v2.validate_knowledge_stability_replay_design import (
    derive_partitions,
    validate_design,
)


CONTRACT_SCHEMA = "delta.knowledge_stability_replay_data_contract.v1"
CONTRACT_ID = "delta-v2-knowledge-stability-replay-data-v1"
DATASET_ID = "delta-v2-knowledge-stability-replay-v1"
ROW_SCHEMA = "delta.knowledge_stability_replay_row.v1"
PAIR_SCHEMA = "delta.knowledge_stability_replay_pair.v1"
SUMMARY_SCHEMA = "delta.knowledge_stability_replay_data_audit.v1"
FREEZE_SCHEMA = "delta.knowledge_stability_replay_data_freeze.v1"
CONTRACT_PATH = Path(
    "experiments/delta_v2/knowledge_stability_replay_data_contract.yaml"
)
FREEZE_PATH = Path(
    "experiments/delta_v2/knowledge_stability_replay_data_freeze.yaml"
)


class StabilityReplayDataError(ValueError):
    """Replay data cannot be built without weakening its frozen contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StabilityReplayDataError(f"{field} must be a mapping")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise StabilityReplayDataError(f"{field} must be a non-empty string")
    return value


def _relative_path(value: Any, field: str) -> Path:
    path = Path(_string(value, field))
    if path.is_absolute() or ".." in path.parts:
        raise StabilityReplayDataError(f"{field} must stay in the repository")
    return path


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise StabilityReplayDataError(f"cannot read YAML: {path}") from exc
    return _mapping(payload, str(path))


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise StabilityReplayDataError(f"cannot read JSONL: {path}") from exc
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StabilityReplayDataError(
                f"invalid JSON at {path}:{line_number}"
            ) from exc
        rows.append(_mapping(payload, f"{path}:{line_number}"))
    return rows


def _binding_path(
    binding: Mapping[str, Any],
    *,
    repo_root: Path,
    field: str,
) -> Path:
    path = repo_root / _relative_path(binding.get("path"), f"{field}.path")
    if not path.is_file() or _sha256(path) != binding.get("sha256"):
        raise StabilityReplayDataError(f"{field} bytes changed")
    return path


def _identity(schema: str, *parts: str) -> str:
    value = "\0".join((schema, CONTRACT_ID, *parts)).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _serialize_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (canonical_json(dict(row)) + "\n").encode("utf-8") for row in rows
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _field_value(card: Mapping[str, Any], field: str) -> str:
    if field == "$card":
        return canonical_json(card)
    return canonical_json(card[field])


def _assign_values(
    source_ids: Sequence[str],
    cards: Mapping[str, Mapping[str, Any]],
    *,
    field: str,
    salt: str,
) -> dict[str, tuple[str, str]]:
    values = {
        source_id: _field_value(cards[source_id], field)
        for source_id in source_ids
    }
    true_counts = Counter(values.values())
    allowed = {
        source_id: sorted(
            value for value in true_counts if value != values[source_id]
        )
        for source_id in source_ids
    }
    if any(not candidates for candidates in allowed.values()):
        raise StabilityReplayDataError(
            f"no source-grounded alternate for {field}"
        )
    desired = Counter(true_counts)
    capacities = Counter(
        {
            value: sum(value in allowed[source_id] for source_id in source_ids)
            for value in true_counts
        }
    )
    overflow = 0
    for value in sorted(desired):
        if desired[value] > capacities[value]:
            overflow += desired[value] - capacities[value]
            desired[value] = capacities[value]
    while overflow:
        candidates = [
            value
            for value in sorted(desired)
            if desired[value] < capacities[value]
        ]
        if not candidates:
            raise StabilityReplayDataError(
                f"cannot balance alternate values for {field}"
            )
        chosen = min(
            candidates,
            key=lambda value: (
                desired[value] - true_counts[value],
                hashlib.sha256(
                    f"{CONTRACT_ID}\0{salt}\0{value}".encode()
                ).hexdigest(),
            ),
        )
        desired[chosen] += 1
        overflow -= 1

    ordered = sorted(
        source_ids,
        key=lambda source_id: (
            len(allowed[source_id]),
            hashlib.sha256(
                f"{CONTRACT_ID}\0{salt}\0{source_id}".encode()
            ).hexdigest(),
        ),
    )
    assignment: dict[str, str] = {}

    def solve(position: int) -> bool:
        if position == len(ordered):
            return not any(desired.values())
        source_id = ordered[position]
        candidates = [
            value
            for value in allowed[source_id]
            if desired[value] > 0
        ]
        candidates.sort(
            key=lambda value: (
                -desired[value],
                hashlib.sha256(
                    f"{CONTRACT_ID}\0{salt}\0{source_id}\0{value}".encode()
                ).hexdigest(),
            )
        )
        for value in candidates:
            desired[value] -= 1
            assignment[source_id] = value
            remaining = ordered[position + 1 :]
            feasible = all(
                desired[candidate]
                <= sum(candidate in allowed[item] for item in remaining)
                for candidate in desired
            )
            if feasible and solve(position + 1):
                return True
            desired[value] += 1
            assignment.pop(source_id, None)
        return False

    if not solve(0):
        raise StabilityReplayDataError(
            f"cannot assign alternate values for {field}"
        )
    result: dict[str, tuple[str, str]] = {}
    for source_id in source_ids:
        encoded = assignment[source_id]
        donors = [
            donor_id
            for donor_id in source_ids
            if donor_id != source_id and values[donor_id] == encoded
        ]
        if not donors:
            raise StabilityReplayDataError(
                f"alternate {field} value lacks true donor support"
            )
        donor_id = min(
            donors,
            key=lambda candidate: hashlib.sha256(
                (
                    f"{CONTRACT_ID}\0{salt}\0{source_id}\0{candidate}"
                ).encode()
            ).hexdigest(),
        )
        result[source_id] = encoded, donor_id
    return result


def _row(
    *,
    source_id: str,
    surface: str,
    prompt: str,
    answer: str,
    pair_id: str | None = None,
    pair_role: str | None = None,
    corruption_family: str | None = None,
    changed_field: str | None = None,
    donor_source_id: str | None = None,
) -> dict[str, Any]:
    parts = [source_id, surface]
    if corruption_family:
        parts.append(corruption_family)
    return {
        "schema": ROW_SCHEMA,
        "row_id": "stability-replay:" + _identity(ROW_SCHEMA, *parts),
        "source_id": source_id,
        "surface": surface,
        "truth_authority": CARD_VERSION,
        "pair_id": pair_id,
        "pair_role": pair_role,
        "corruption_family": corruption_family,
        "changed_field": changed_field,
        "donor_source_id": donor_source_id,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def _equalize_pair_prompt_lengths(
    positive: dict[str, Any],
    negative: dict[str, Any],
) -> None:
    positive_prompt = str(positive["messages"][0]["content"])
    negative_prompt = str(negative["messages"][0]["content"])
    target_length = max(len(positive_prompt), len(negative_prompt))

    def pad(prompt: str) -> str:
        contract, separator, answer = prompt.partition(
            "\nAnswer yes or no only."
        )
        if not separator:
            raise StabilityReplayDataError(
                "verification prompt lost its answer boundary"
            )
        return contract + (" " * (target_length - len(prompt))) + separator + answer

    positive["messages"][0]["content"] = pad(positive_prompt)
    negative["messages"][0]["content"] = pad(negative_prompt)


def _value_only_accuracy(
    pairs: Sequence[Mapping[str, Any]],
    cards: Mapping[str, Mapping[str, Any]],
    false_cards: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for pair in pairs:
        family = str(pair["corruption_family"])
        source_id = str(pair["source_id"])
        field = str(pair["changed_field"])
        counts[family][_field_value(cards[source_id], field)]["true"] += 1
        counts[family][_field_value(false_cards[str(pair["pair_id"])], field)][
            "false"
        ] += 1
    result = {}
    for family, by_value in counts.items():
        correct = sum(max(labels.values()) for labels in by_value.values())
        total = sum(sum(labels.values()) for labels in by_value.values())
        result[family] = correct / total
    return dict(sorted(result.items()))


def _prompt_length_only_accuracy(
    rows: Sequence[Mapping[str, Any]],
) -> float:
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        role = row.get("pair_role")
        if role in {"positive", "negative"}:
            prompt = str(row["messages"][0]["content"])
            counts[len(prompt)][str(role)] += 1
    correct = sum(max(labels.values()) for labels in counts.values())
    total = sum(sum(labels.values()) for labels in counts.values())
    return correct / total


def validate_contract(
    contract: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("contract_id") != CONTRACT_ID
        or contract.get("dataset_id") != DATASET_ID
        or contract.get("experiment_id") != "delta_v2"
        or contract.get("candidate_id") != "d4_stability_replay_sft_control"
        or contract.get("target_revision") != "v0.26.0"
        or contract.get("freeze_state") != "preregistered-for-build"
    ):
        raise StabilityReplayDataError("replay-data identity changed")
    bindings = _mapping(contract.get("bindings"), "bindings")
    paths = [
        _binding_path(
            _mapping(bindings.get(name), name),
            repo_root=repo_root,
            field=name,
        )
        for name in (
            "design",
            "partition_validator",
            "knowledge_card_generator",
            "acquisition_train",
            "current_eval",
        )
    ]
    design = _load_yaml(paths[0])
    validate_design(design, repo_root=repo_root)
    replay = _mapping(contract.get("replay_train"), "replay_train")
    if dict(replay) != {
        "partition": "replay_train",
        "source_count": 15,
        "recall_rows_per_source": 1,
        "boolean_pairs_per_source": 3,
        "positive_rows_per_source": 3,
        "negative_rows_per_source": 3,
        "rows": 105,
        "pairs": 45,
        "model_visible_fields": ["messages"],
        "corruption_families": {
            "annotation_swap": 15,
            "family_value_swap": 15,
            "full_card_swap": 15,
        },
    }:
        raise StabilityReplayDataError("replay construction changed")
    stability = _mapping(contract.get("stability_dev"), "stability_dev")
    if dict(stability) != {
        "partition": "stability_dev",
        "source_count": 15,
        "probes_per_source": [
            "choice",
            "boolean_true",
            "boolean_false",
            "recall",
        ],
        "rows": 60,
        "model_visible_fields": ["prompt"],
        "training_overlap": "none",
        "pooled_score": "forbidden",
    }:
        raise StabilityReplayDataError("stability-dev construction changed")
    gates = _mapping(contract.get("shortcut_gates"), "shortcut_gates")
    if dict(gates) != {
        "source_overlap_replay_vs_acquisition": 0,
        "source_overlap_replay_vs_current_eval": 0,
        "source_overlap_replay_vs_stability_dev": 0,
        "source_overlap_replay_vs_verify_v2": 0,
        "exact_prompt_overlap_acquisition_train": 0,
        "exact_prompt_overlap_current_eval": 0,
        "exact_prompt_overlap_stability_dev": 0,
        "verify_v2_prompt_audit": (
            "deferred-until-post-receipt-materialization"
        ),
        "outer_prompt_template_identical_within_pair": "required",
        "prompt_length_identical_within_pair": "required",
        "contract_key_order_identical_within_pair": "required",
        "donor_value_verified_true_support_rate": 1.0,
        "single_corruption_family_share_maximum": 1 / 3,
        "empirical_value_only_accuracy_maximum_per_family": 0.60,
        "empirical_prompt_length_only_accuracy_maximum": 0.50,
    }:
        raise StabilityReplayDataError("shortcut gates changed")
    outputs = _mapping(contract.get("outputs"), "outputs")
    if dict(outputs) != {
        "root": "data/experiments/delta_v2/knowledge_stability_replay_v1",
        "replay_train": "replay_train.jsonl",
        "replay_pairs": "replay_pairs.jsonl",
        "stability_dev": "stability_dev.jsonl",
        "summary": "summary.json",
    }:
        raise StabilityReplayDataError("replay outputs changed")
    boundary = _mapping(
        contract.get("execution_boundary"), "execution_boundary"
    )
    if dict(boundary) != {
        "dataset_build_authorized": True,
        "base_baseline_authorized": False,
        "lora_training_authorized": False,
        "modal_job_authorized": False,
        "b2_write_authorized": False,
        "promotion_authorized": False,
    }:
        raise StabilityReplayDataError("execution boundary changed")
    return tuple(paths)  # type: ignore[return-value]


def validate_freeze(
    freeze: Mapping[str, Any],
    *,
    repo_root: Path,
) -> None:
    if (
        freeze.get("schema") != FREEZE_SCHEMA
        or freeze.get("dataset_id") != DATASET_ID
        or freeze.get("contract_id") != CONTRACT_ID
        or freeze.get("experiment_id") != "delta_v2"
        or freeze.get("candidate_id") != "d4_stability_replay_sft_control"
        or freeze.get("freeze_state") != "frozen-for-baseline"
    ):
        raise StabilityReplayDataError("replay freeze identity changed")
    training = _mapping(freeze.get("replay_train"), "replay_train")
    if dict(training) != {
        "sources": 15,
        "rows": 105,
        "recall_rows": 15,
        "boolean_pairs": 45,
        "positive_rows": 45,
        "negative_rows": 45,
        "model_visible_fields": ["messages"],
    }:
        raise StabilityReplayDataError("frozen replay counts changed")
    stability = _mapping(freeze.get("stability_dev"), "stability_dev")
    if dict(stability) != {
        "sources": 15,
        "rows": 60,
        "probes_per_source": [
            "choice",
            "boolean_true",
            "boolean_false",
            "recall",
        ],
        "pooled_score": "forbidden",
    }:
        raise StabilityReplayDataError("frozen stability-dev counts changed")
    audit = _mapping(freeze.get("shortcut_audit"), "shortcut_audit")
    if dict(audit) != {
        "status": "pass",
        "all_declared_source_overlaps": 0,
        "all_materialized_prompt_overlaps": 0,
        "outer_prompt_template_identical": True,
        "prompt_length_identical_within_pair": True,
        "donor_value_verified_true_support_rate": 1.0,
        "maximum_single_corruption_family_share": 1 / 3,
        "maximum_empirical_value_only_accuracy": 0.50,
        "empirical_prompt_length_only_accuracy": 0.50,
        "verify_v2_prompt_audit": (
            "deferred-until-post-receipt-materialization"
        ),
    }:
        raise StabilityReplayDataError("frozen shortcut audit changed")
    bindings = freeze.get("bindings")
    if not isinstance(bindings, list) or len(bindings) != 10:
        raise StabilityReplayDataError("replay freeze bindings changed")
    expected_ids = {
        "design",
        "contract",
        "builder",
        "builder_tests",
        "replay_train",
        "replay_pairs",
        "stability_dev",
        "summary",
        "acquisition_train",
        "current_eval",
    }
    observed_ids: set[str] = set()
    for index, raw_binding in enumerate(bindings):
        binding = _mapping(raw_binding, f"bindings[{index}]")
        binding_id = _string(binding.get("id"), f"bindings[{index}].id")
        if binding_id in observed_ids:
            raise StabilityReplayDataError("duplicate replay freeze binding")
        observed_ids.add(binding_id)
        _binding_path(
            binding,
            repo_root=repo_root,
            field=f"bindings[{binding_id}]",
        )
    if observed_ids != expected_ids:
        raise StabilityReplayDataError(
            "replay freeze binding ids changed"
        )
    boundary = _mapping(
        freeze.get("execution_boundary"), "execution_boundary"
    )
    if dict(boundary) != {
        "model_invocations_completed": 0,
        "optimizer_steps_completed": 0,
        "dataset_build_completed": True,
        "base_baseline_requires_separate_protocol": True,
        "lora_training_requires_separate_protocol": True,
        "qlora_authorized": False,
        "full_weight_authorized": False,
        "reinforcement_authorized": False,
        "modal_job_authorized": False,
        "b2_write_authorized": False,
        "promotion_authorized": False,
    }:
        raise StabilityReplayDataError("replay freeze boundary changed")


def build_replay_data(
    *,
    contract_path: Path,
    repo_root: Path,
    output_root_override: Path | None = None,
) -> dict[str, Any]:
    contract = _load_yaml(contract_path)
    (
        design_path,
        validator_path,
        generator_path,
        acquisition_path,
        current_eval_path,
    ) = validate_contract(contract, repo_root=repo_root)
    design = _load_yaml(design_path)
    partitions, _blocked, _eligible = derive_partitions(
        design, repo_root=repo_root
    )
    replay_records = {
        str(row["fact_id"]): row for row in partitions["replay_train"]
    }
    dev_records = list(partitions["stability_dev"])
    verify_ids = {
        str(row["fact_id"]) for row in partitions["verify_v2"]
    }
    source_ids = sorted(replay_records)
    cards = {
        source_id: knowledge_card(replay_records[source_id])
        for source_id in source_ids
    }
    config_ids = [
        source_id
        for source_id in source_ids
        if cards[source_id]["kind"] == "python_config_field"
    ]
    env_ids = [
        source_id
        for source_id in source_ids
        if cards[source_id]["kind"] == "python_environment_variable"
    ]
    assignments: dict[str, dict[str, tuple[str, str]]] = {
        "annotation_swap": _assign_values(
            source_ids,
            cards,
            field="annotation",
            salt="annotation-swap-v1",
        ),
        "family_value_swap": {},
        "full_card_swap": {},
    }
    assignments["family_value_swap"].update(
        _assign_values(
            config_ids,
            cards,
            field="default",
            salt="config-default-swap-v1",
        )
    )
    assignments["family_value_swap"].update(
        _assign_values(
            env_ids,
            cards,
            field="getter",
            salt="env-getter-swap-v1",
        )
    )
    assignments["full_card_swap"].update(
        _assign_values(
            config_ids,
            cards,
            field="$card",
            salt="config-card-swap-v1",
        )
    )
    assignments["full_card_swap"].update(
        _assign_values(
            env_ids,
            cards,
            field="$card",
            salt="env-card-swap-v1",
        )
    )

    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    false_cards: dict[str, Mapping[str, Any]] = {}
    changed_counts: Counter[str] = Counter()
    for source_id in source_ids:
        record = replay_records[source_id]
        display = _display_name(record)
        rows.append(
            _row(
                source_id=source_id,
                surface="replay_recall",
                prompt=_recall_prompt(display, variant="train"),
                answer=canonical_json(cards[source_id]),
            )
        )
        family_fields = {
            "annotation_swap": "annotation",
            "family_value_swap": (
                "default"
                if source_id in config_ids
                else "getter"
            ),
            "full_card_swap": "$card",
        }
        for family, field in family_fields.items():
            encoded, donor_id = assignments[family][source_id]
            false_card = (
                json.loads(encoded)
                if field == "$card"
                else {**cards[source_id], field: json.loads(encoded)}
            )
            changed = [
                key
                for key in false_card
                if false_card[key] != cards[source_id][key]
            ]
            if (
                not changed
                or set(false_card) != set(cards[source_id])
                or (field != "$card" and changed != [field])
            ):
                raise StabilityReplayDataError(
                    "false card violates its corruption family"
                )
            changed_counts[str(len(changed))] += 1
            pair_id = "stability-replay-pair:" + _identity(
                PAIR_SCHEMA, source_id, family
            )
            positive = _row(
                source_id=source_id,
                surface="replay_verify_true",
                prompt=_verify_prompt(display, cards[source_id], variant="train"),
                answer="yes",
                pair_id=pair_id,
                pair_role="positive",
                corruption_family=family,
                changed_field=field,
            )
            negative = _row(
                source_id=source_id,
                surface="replay_verify_false",
                prompt=_verify_prompt(display, false_card, variant="train"),
                answer="no",
                pair_id=pair_id,
                pair_role="negative",
                corruption_family=family,
                changed_field=field,
                donor_source_id=donor_id,
            )
            _equalize_pair_prompt_lengths(positive, negative)
            rows.extend([positive, negative])
            false_cards[pair_id] = false_card
            pairs.append(
                {
                    "schema": PAIR_SCHEMA,
                    "pair_id": pair_id,
                    "source_id": source_id,
                    "source_kind": str(cards[source_id]["kind"]),
                    "corruption_family": family,
                    "changed_field": field,
                    "changed_top_level_fields": len(changed),
                    "donor_source_id": donor_id,
                    "positive_row_id": positive["row_id"],
                    "negative_row_id": negative["row_id"],
                    "true_contract_sha256": hashlib.sha256(
                        canonical_json(cards[source_id]).encode()
                    ).hexdigest(),
                    "false_contract_sha256": hashlib.sha256(
                        canonical_json(false_card).encode()
                    ).hexdigest(),
                }
            )

    dev_rows: list[dict[str, Any]] = []
    for record in dev_records:
        dev_rows.extend(
            _fact_probe_rows(
                record,
                stratum="stability_dev",
                training_overlap="none",
            )
        )
    rows.sort(key=lambda row: str(row["row_id"]))
    pairs.sort(key=lambda row: str(row["pair_id"]))
    dev_rows.sort(key=lambda row: str(row["probe_id"]))
    if len(rows) != 105 or len(pairs) != 45 or len(dev_rows) != 60:
        raise StabilityReplayDataError("replay-data row counts changed")

    rows_by_id = {str(row["row_id"]): row for row in rows}
    family_counts = Counter(str(pair["corruption_family"]) for pair in pairs)
    prompt_length_deltas: Counter[str] = Counter()
    for pair in pairs:
        positive = rows_by_id[str(pair["positive_row_id"])]
        negative = rows_by_id[str(pair["negative_row_id"])]
        positive_prompt = str(positive["messages"][0]["content"])
        negative_prompt = str(negative["messages"][0]["content"])
        positive_prefix, _, positive_contract = positive_prompt.partition(
            "\nCONTRACT="
        )
        negative_prefix, _, negative_contract = negative_prompt.partition(
            "\nCONTRACT="
        )
        if (
            positive_prefix != negative_prefix
            or len(positive_prompt) != len(negative_prompt)
            or not positive_contract.endswith("\nAnswer yes or no only.")
            or not negative_contract.endswith("\nAnswer yes or no only.")
        ):
            raise StabilityReplayDataError("pair outer prompt changed")
        prompt_length_deltas[
            str(len(negative_prompt) - len(positive_prompt))
        ] += 1

    replay_sources = set(source_ids)
    acquisition_rows = _load_jsonl(acquisition_path)
    acquisition_sources = {
        str(row["source_id"]) for row in acquisition_rows
    }
    current_eval_rows = _load_jsonl(current_eval_path)
    current_eval_sources = {
        str(row["source_id"])
        for row in current_eval_rows
        if row.get("source_kind") == "atomic_fact_delta"
    }
    dev_sources = {str(row["source_id"]) for row in dev_rows}
    train_prompts = {
        str(row["messages"][0]["content"]) for row in rows
    }
    acquisition_prompts = {
        str(row["messages"][0]["content"]) for row in acquisition_rows
    }
    current_eval_prompts = {
        str(row["prompt"]) for row in current_eval_rows
    }
    dev_prompts = {str(row["prompt"]) for row in dev_rows}
    source_overlaps = {
        "acquisition_train": len(replay_sources & acquisition_sources),
        "current_eval": len(replay_sources & current_eval_sources),
        "stability_dev": len(replay_sources & dev_sources),
        "verify_v2": len(replay_sources & verify_ids),
    }
    prompt_overlaps = {
        "acquisition_train": len(train_prompts & acquisition_prompts),
        "current_eval": len(train_prompts & current_eval_prompts),
        "stability_dev": len(train_prompts & dev_prompts),
    }
    value_only = _value_only_accuracy(pairs, cards, false_cards)
    prompt_length_only = _prompt_length_only_accuracy(rows)
    maximum_family_share = max(family_counts.values()) / len(pairs)
    gates = contract["shortcut_gates"]
    if (
        any(source_overlaps.values())
        or any(prompt_overlaps.values())
        or maximum_family_share
        > gates["single_corruption_family_share_maximum"]
        or max(value_only.values())
        > gates["empirical_value_only_accuracy_maximum_per_family"]
        or prompt_length_only
        > gates["empirical_prompt_length_only_accuracy_maximum"]
    ):
        raise StabilityReplayDataError(
            "replay shortcut gate failed: "
            f"source_overlaps={source_overlaps}, "
            f"prompt_overlaps={prompt_overlaps}, "
            f"family_share={maximum_family_share}, "
            f"value_only={value_only}, "
            f"prompt_length_only={prompt_length_only}"
        )

    output_root = (
        output_root_override
        if output_root_override is not None
        else repo_root / str(contract["outputs"]["root"])
    )
    output_root.mkdir(parents=True, exist_ok=True)
    train_path = output_root / str(contract["outputs"]["replay_train"])
    pairs_path = output_root / str(contract["outputs"]["replay_pairs"])
    dev_path = output_root / str(contract["outputs"]["stability_dev"])
    summary_path = output_root / str(contract["outputs"]["summary"])
    train_path.write_bytes(_serialize_jsonl(rows))
    pairs_path.write_bytes(_serialize_jsonl(pairs))
    dev_path.write_bytes(_serialize_jsonl(dev_rows))
    summary = {
        "schema": SUMMARY_SCHEMA,
        "contract_id": CONTRACT_ID,
        "dataset_id": DATASET_ID,
        "status": "pass",
        "deterministic_order": True,
        "inputs": {
            "contract": {
                "path": str(contract_path.relative_to(repo_root)),
                "sha256": _sha256(contract_path),
            },
            "design": {
                "path": str(design_path.relative_to(repo_root)),
                "sha256": _sha256(design_path),
            },
            "partition_validator": {
                "path": str(validator_path.relative_to(repo_root)),
                "sha256": _sha256(validator_path),
            },
            "knowledge_card_generator": {
                "path": str(generator_path.relative_to(repo_root)),
                "sha256": _sha256(generator_path),
            },
            "acquisition_train": {
                "path": str(acquisition_path.relative_to(repo_root)),
                "sha256": _sha256(acquisition_path),
            },
            "current_eval": {
                "path": str(current_eval_path.relative_to(repo_root)),
                "sha256": _sha256(current_eval_path),
            },
        },
        "outputs": {
            "replay_train": {
                "rows": len(rows),
                "sha256": _sha256(train_path),
            },
            "replay_pairs": {
                "rows": len(pairs),
                "sha256": _sha256(pairs_path),
            },
            "stability_dev": {
                "rows": len(dev_rows),
                "sha256": _sha256(dev_path),
            },
        },
        "replay_train": {
            "sources": len(replay_sources),
            "surface_counts": dict(
                sorted(Counter(str(row["surface"]) for row in rows).items())
            ),
            "pair_roles": {"negative": 45, "positive": 45},
        },
        "stability_dev": {
            "sources": len(dev_sources),
            "probe_counts": dict(
                sorted(
                    Counter(str(row["probe_kind"]) for row in dev_rows).items()
                )
            ),
            "pooled_score": None,
        },
        "shortcut_audit": {
            "status": "pass",
            "source_overlaps": source_overlaps,
            "prompt_overlaps": prompt_overlaps,
            "verify_v2_prompt_audit": (
                "deferred-until-post-receipt-materialization"
            ),
            "outer_prompt_template_identical": True,
            "contract_key_order_identical": True,
            "donor_value_verified_true_support_rate": 1.0,
            "changed_top_level_field_count_distribution": dict(
                sorted(changed_counts.items(), key=lambda item: int(item[0]))
            ),
            "corruption_family_counts": dict(sorted(family_counts.items())),
            "maximum_single_corruption_family_share": maximum_family_share,
            "empirical_value_only_accuracy_by_family": value_only,
            "maximum_empirical_value_only_accuracy": max(value_only.values()),
            "empirical_prompt_length_only_accuracy": prompt_length_only,
            "prompt_length_delta_false_minus_true": dict(
                sorted(
                    prompt_length_deltas.items(),
                    key=lambda item: int(item[0]),
                )
            ),
            "model_visible_metadata_fields": [],
        },
        "execution_boundary": {
            "model_invocations_completed": 0,
            "optimizer_steps_completed": 0,
            "base_baseline_authorized": False,
            "lora_training_authorized": False,
            "modal_job_authorized": False,
            "b2_write_authorized": False,
            "promotion_authorized": False,
        },
    }
    _write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build frozen source-disjoint DELTA stability replay data."
    )
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    contract_path = args.contract
    if not contract_path.is_absolute():
        contract_path = repo_root / contract_path
    output_root = args.output_root
    if output_root is not None and not output_root.is_absolute():
        output_root = repo_root / output_root
    result = build_replay_data(
        contract_path=contract_path,
        repo_root=repo_root,
        output_root_override=output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
