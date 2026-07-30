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
    _recall_prompt,
    _verify_prompt,
    canonical_json,
    knowledge_card,
)


CONTRACT_SCHEMA = "delta.knowledge_paired_data_contract.v1"
CONTRACT_ID = "delta-v2-knowledge-paired-data-v2"
ROW_SCHEMA = "delta.knowledge_paired_sft_row.v1"
PAIR_SCHEMA = "delta.knowledge_boolean_pair.v1"
SUMMARY_SCHEMA = "delta.knowledge_paired_data_audit.v1"
FREEZE_SCHEMA = "delta.knowledge_paired_data_freeze.v1"
CONTRACT_PATH = Path(
    "experiments/delta_v2/knowledge_paired_data_contract.yaml"
)
FREEZE_PATH = Path(
    "experiments/delta_v2/knowledge_paired_data_freeze.yaml"
)


class KnowledgePairedDataError(ValueError):
    """The paired data cannot be built without weakening its frozen gates."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KnowledgePairedDataError(f"{field} must be a mapping")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise KnowledgePairedDataError(f"{field} must be a non-empty string")
    return value


def _relative_path(value: Any, field: str) -> Path:
    path = Path(_string(value, field))
    if path.is_absolute() or ".." in path.parts:
        raise KnowledgePairedDataError(f"{field} must stay in the repository")
    return path


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise KnowledgePairedDataError(f"cannot read YAML: {path}") from exc
    return _mapping(payload, str(path))


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise KnowledgePairedDataError(f"cannot read JSONL: {path}") from exc
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise KnowledgePairedDataError(
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
    expected = _string(binding.get("sha256"), f"{field}.sha256")
    if not path.is_file() or _sha256(path) != expected:
        raise KnowledgePairedDataError(f"{field} bytes changed")
    return path


def _identity(schema: str, *parts: str) -> str:
    encoded = "\0".join((schema, CONTRACT_ID, *parts)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _assign_balanced_values(
    source_ids: Sequence[str],
    cards: Mapping[str, Mapping[str, Any]],
    *,
    field: str,
    salt: str,
    forbidden: Mapping[str, set[str]] | None = None,
) -> dict[str, tuple[str, str]]:
    forbidden = forbidden or {}
    values_by_source = {
        source_id: _field_value(cards[source_id], field)
        for source_id in source_ids
    }
    true_counts = Counter(values_by_source.values())
    allowed: dict[str, list[str]] = {}
    for source_id in source_ids:
        own = values_by_source[source_id]
        blocked = forbidden.get(source_id, set())
        allowed[source_id] = sorted(
            value
            for value in true_counts
            if value != own and value not in blocked
        )
        if not allowed[source_id]:
            raise KnowledgePairedDataError(
                f"no changed {field} value for {source_id}"
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
            raise KnowledgePairedDataError(
                f"cannot balance changed {field} values"
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

    target_order = sorted(
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
        if position == len(target_order):
            return not any(desired.values())
        source_id = target_order[position]
        candidates = [
            value
            for value in allowed[source_id]
            if desired[value] > 0
        ]
        candidates.sort(
            key=lambda value: (
                -desired[value],
                hashlib.sha256(
                    (
                        f"{CONTRACT_ID}\0{salt}\0{source_id}\0{value}"
                    ).encode()
                ).hexdigest(),
            )
        )
        for value in candidates:
            desired[value] -= 1
            assignment[source_id] = value
            remaining_sources = target_order[position + 1 :]
            feasible = all(
                desired[candidate]
                <= sum(
                    candidate in allowed[remaining]
                    for remaining in remaining_sources
                )
                for candidate in desired
            )
            if feasible and solve(position + 1):
                return True
            desired[value] += 1
            assignment.pop(source_id, None)
        return False

    if not solve(0):
        raise KnowledgePairedDataError(
            f"cannot assign balanced changed {field} values"
        )

    result: dict[str, tuple[str, str]] = {}
    for source_id in source_ids:
        value = assignment[source_id]
        donors = [
            donor_id
            for donor_id in source_ids
            if donor_id != source_id
            and values_by_source[donor_id] == value
        ]
        if not donors:
            raise KnowledgePairedDataError(
                f"changed {field} value lacks a donor"
            )
        donor_id = min(
            donors,
            key=lambda candidate: hashlib.sha256(
                (
                    f"{CONTRACT_ID}\0{salt}\0{source_id}\0{candidate}"
                ).encode()
            ).hexdigest(),
        )
        result[source_id] = (value, donor_id)
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
    identity_parts = [source_id, surface]
    if corruption_family is not None:
        identity_parts.append(corruption_family)
    return {
        "schema": ROW_SCHEMA,
        "row_id": "knowledge-paired-train:"
        + _identity(ROW_SCHEMA, *identity_parts),
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
    result: dict[str, float] = {}
    for family, by_value in counts.items():
        correct = sum(max(labels.values()) for labels in by_value.values())
        total = sum(sum(labels.values()) for labels in by_value.values())
        result[family] = correct / total
    return dict(sorted(result.items()))


def validate_contract(
    contract: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("contract_id") != CONTRACT_ID
        or contract.get("experiment_id") != "delta_v2"
        or contract.get("parent_dataset_id")
        != "delta-v2-vllm-knowledge-adaptation-v1"
        or contract.get("target_revision") != "v0.26.0"
        or contract.get("freeze_state") != "preregistered"
    ):
        raise KnowledgePairedDataError("paired-data identity changed")
    trigger = _mapping(contract.get("decision_trigger"), "decision_trigger")
    if dict(trigger) != {
        "diagnostic_protocol_id": "delta-v2-knowledge-boolean-margin-v1",
        "diagnostic_protocol_sha256": (
            "57b79d276d869e85b9323c6bc1730a540bb89503e9628115d85fd1c685c8da2e"
        ),
        "result_report": (
            "artifacts/reports/delta_v2/"
            "boolean_margin_diagnostic_modal_v1.md"
        ),
        "result_report_sha256": (
            "3788aa452e7bb735f8e96417c7b7d7017517809006b5ddce01d1dae6aaadb780"
        ),
        "observed_failure": (
            "global-label-intercept-with-zero-trained-"
            "truth-conditioned-pairs"
        ),
        "blocking_warning": "annotation-only-negative-bank",
    }:
        raise KnowledgePairedDataError("paired-data trigger changed")
    result_path = repo_root / _relative_path(
        trigger.get("result_report"), "decision_trigger.result_report"
    )
    if (
        not result_path.is_file()
        or _sha256(result_path) != trigger.get("result_report_sha256")
    ):
        raise KnowledgePairedDataError("paired-data trigger evidence changed")

    bindings = _mapping(contract.get("bindings"), "bindings")
    generator_path = _binding_path(
        _mapping(
            bindings.get("knowledge_card_generator"),
            "knowledge_card_generator",
        ),
        repo_root=repo_root,
        field="knowledge_card_generator",
    )
    facts_path = _binding_path(
        _mapping(bindings.get("fact_deltas"), "fact_deltas"),
        repo_root=repo_root,
        field="fact_deltas",
    )
    selection_path = _binding_path(
        _mapping(
            bindings.get("frozen_eval_selection"),
            "frozen_eval_selection",
        ),
        repo_root=repo_root,
        field="frozen_eval_selection",
    )
    dev_path = _binding_path(
        _mapping(bindings.get("unchanged_dev"), "unchanged_dev"),
        repo_root=repo_root,
        field="unchanged_dev",
    )
    eval_path = _binding_path(
        _mapping(bindings.get("unchanged_eval"), "unchanged_eval"),
        repo_root=repo_root,
        field="unchanged_eval",
    )

    construction = _mapping(contract.get("construction"), "construction")
    if dict(construction) != {
        "source_count": 21,
        "recall_rows_per_source": 1,
        "boolean_pairs_per_source": 3,
        "positive_rows_per_source": 3,
        "negative_rows_per_source": 3,
        "train_rows": 147,
        "pair_rows": 63,
        "false_contract_rule": (
            "source-grounded-field-value-or-full-card-derangement"
        ),
        "model_visible_fields": ["messages"],
        "metadata_not_model_visible": [
            "pair_id",
            "pair_role",
            "corruption_family",
            "changed_field",
            "donor_source_id",
        ],
        "corruption_families": {
            "annotation_swap": 21,
            "config_default_swap": 9,
            "env_getter_swap": 12,
            "full_card_swap": 21,
        },
    }:
        raise KnowledgePairedDataError("paired-data construction changed")
    gates = _mapping(contract.get("shortcut_gates"), "shortcut_gates")
    if (
        gates.get("outer_prompt_template_identical_within_pair")
        != "required"
        or gates.get("contract_key_order_identical_within_pair")
        != "required"
        or gates.get("changed_top_level_fields_minimum") != 1
        or gates.get("single_field_corruption_pairs") != 42
        or gates.get("full_card_corruption_pairs") != 21
        or gates.get("donor_value_verified_true_support_rate") != 1.0
        or gates.get("annotation_only_negative_share_maximum") != 1 / 3
        or gates.get("single_corruption_family_share_maximum") != 1 / 3
        or gates.get("empirical_value_only_accuracy_maximum_per_family")
        != 0.60
        or gates.get("original_eval_prompt_overlap") != 0
        or gates.get("frozen_eval_prompt_overlap") != 0
        or gates.get("evaluation_wording_added_to_training") is not False
    ):
        raise KnowledgePairedDataError("paired-data shortcut gates changed")
    comparison = _mapping(
        contract.get("causal_comparison"), "causal_comparison"
    )
    if dict(comparison) != {
        "stage_1": (
            "original-assistant-only-generative-sft-on-repaired-data"
        ),
        "stage_2": "paired-boolean-objective-on-identical-repaired-data",
        "stage_2_only_change": "learning-objective",
        "lora_topology_unchanged": True,
        "optimizer_schedule_unchanged": True,
        "seed_unchanged": True,
        "frozen_eval_unchanged": True,
    }:
        raise KnowledgePairedDataError("paired causal comparison changed")
    outputs = _mapping(contract.get("outputs"), "outputs")
    if dict(outputs) != {
        "root": "data/experiments/delta_v2/knowledge_paired_v2",
        "train": "train.jsonl",
        "pairs": "pairs.jsonl",
        "summary": "summary.json",
    }:
        raise KnowledgePairedDataError("paired-data outputs changed")
    boundary = _mapping(
        contract.get("execution_boundary"), "execution_boundary"
    )
    if dict(boundary) != {
        "dataset_build_authorized": True,
        "model_training_authorized": False,
        "promotion_authorized": False,
    }:
        raise KnowledgePairedDataError("paired-data boundary changed")
    return generator_path, facts_path, selection_path, dev_path, eval_path


def validate_freeze(
    freeze: Mapping[str, Any],
    *,
    repo_root: Path,
) -> None:
    if (
        freeze.get("schema") != FREEZE_SCHEMA
        or freeze.get("dataset_id") != "delta-v2-knowledge-paired-v2"
        or freeze.get("contract_id") != CONTRACT_ID
        or freeze.get("experiment_id") != "delta_v2"
        or freeze.get("parent_dataset_id")
        != "delta-v2-vllm-knowledge-adaptation-v1"
        or freeze.get("target_revision") != "v0.26.0"
        or freeze.get("freeze_state") != "frozen"
    ):
        raise KnowledgePairedDataError("paired-data freeze identity changed")
    training = _mapping(freeze.get("training_data"), "training_data")
    if dict(training) != {
        "source_cards": 21,
        "rows": 147,
        "recall_rows": 21,
        "boolean_pairs": 63,
        "positive_rows": 63,
        "negative_rows": 63,
        "model_visible_fields": ["messages"],
    }:
        raise KnowledgePairedDataError("paired-data frozen counts changed")
    audit = _mapping(freeze.get("shortcut_audit"), "shortcut_audit")
    if (
        audit.get("status") != "pass"
        or audit.get("outer_prompt_template_identical") is not True
        or audit.get("contract_key_order_identical") is not True
        or audit.get("donor_value_verified_true_support_rate") != 1.0
        or audit.get("maximum_empirical_value_only_accuracy")
        > audit.get("maximum_allowed_empirical_value_only_accuracy", 0)
        or audit.get("frozen_eval_prompt_overlap") != 0
        or audit.get("original_eval_prompt_overlap") != 0
    ):
        raise KnowledgePairedDataError("paired-data frozen audit changed")
    bindings = freeze.get("bindings")
    if not isinstance(bindings, list) or len(bindings) != 8:
        raise KnowledgePairedDataError("paired-data freeze bindings changed")
    expected_ids = {
        "contract",
        "generator",
        "generator_tests",
        "train",
        "pairs",
        "build_summary",
        "unchanged_dev",
        "unchanged_eval",
    }
    observed_ids: set[str] = set()
    for index, raw_binding in enumerate(bindings):
        binding = _mapping(raw_binding, f"bindings[{index}]")
        binding_id = _string(binding.get("id"), f"bindings[{index}].id")
        if binding_id in observed_ids:
            raise KnowledgePairedDataError("duplicate paired freeze binding")
        observed_ids.add(binding_id)
        _binding_path(
            binding,
            repo_root=repo_root,
            field=f"bindings[{binding_id}]",
        )
    if observed_ids != expected_ids:
        raise KnowledgePairedDataError("paired-data freeze binding ids changed")
    boundary = _mapping(
        freeze.get("execution_boundary"), "execution_boundary"
    )
    if dict(boundary) != {
        "model_invocations_completed": 0,
        "optimizer_steps_completed": 0,
        "dataset_build_authorized": True,
        "stage_1_training_requires_separate_protocol": True,
        "stage_2_training_requires_separate_protocol": True,
    }:
        raise KnowledgePairedDataError("paired-data freeze boundary changed")


def build_paired_data(
    *,
    contract_path: Path,
    repo_root: Path,
    output_root_override: Path | None = None,
) -> dict[str, Any]:
    contract = _load_yaml(contract_path)
    (
        generator_path,
        facts_path,
        selection_path,
        dev_path,
        eval_path,
    ) = validate_contract(contract, repo_root=repo_root)
    output_root = (
        output_root_override
        if output_root_override is not None
        else repo_root / str(contract["outputs"]["root"])
    )

    facts = {
        str(row["fact_id"]): row for row in _load_jsonl(facts_path)
    }
    selection = _load_jsonl(selection_path)
    added_ids = sorted(
        {
            str(row["source_id"])
            for row in selection
            if row.get("source_kind") == "atomic_fact_delta"
            and _mapping(row.get("gold"), "selection.gold").get("status")
            == "added"
        }
    )
    if len(added_ids) != 21:
        raise KnowledgePairedDataError("paired source selection changed")
    records = {source_id: facts[source_id] for source_id in added_ids}
    cards = {
        source_id: knowledge_card(records[source_id])
        for source_id in added_ids
    }
    config_ids = [
        source_id
        for source_id in added_ids
        if cards[source_id]["kind"] == "python_config_field"
    ]
    env_ids = [
        source_id
        for source_id in added_ids
        if cards[source_id]["kind"] == "python_environment_variable"
    ]
    if len(config_ids) != 9 or len(env_ids) != 12:
        raise KnowledgePairedDataError("paired source-family counts changed")

    assignments: dict[str, dict[str, tuple[str, str]]] = {}
    assignments["annotation_swap"] = _assign_balanced_values(
        added_ids,
        cards,
        field="annotation",
        salt="annotation_swap",
    )
    assignments["config_default_swap"] = _assign_balanced_values(
        config_ids,
        cards,
        field="default",
        salt="config_default_swap",
    )
    assignments["env_getter_swap"] = _assign_balanced_values(
        env_ids,
        cards,
        field="getter",
        salt="env_getter_swap",
    )
    full_card_assignments = _assign_balanced_values(
        config_ids,
        cards,
        field="$card",
        salt="full_card_swap_config",
    )
    full_card_assignments.update(
        _assign_balanced_values(
            env_ids,
            cards,
            field="$card",
            salt="full_card_swap_env",
        )
    )
    assignments["full_card_swap"] = full_card_assignments
    family_field = {
        "annotation_swap": "annotation",
        "config_default_swap": "default",
        "env_getter_swap": "getter",
        "full_card_swap": "$card",
    }

    train_rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    false_cards: dict[str, Mapping[str, Any]] = {}
    changed_field_counts: Counter[str] = Counter()
    for source_id in added_ids:
        display = _display_name(records[source_id])
        train_rows.append(
            _row(
                source_id=source_id,
                surface="exact_recall",
                prompt=_recall_prompt(display, variant="train"),
                answer=canonical_json(cards[source_id]),
            )
        )
        source_families = ["annotation_swap"]
        source_families.append(
            "config_default_swap"
            if source_id in config_ids
            else "env_getter_swap"
        )
        source_families.append("full_card_swap")
        for family in source_families:
            field = family_field[family]
            encoded_value, donor_id = assignments[family][source_id]
            if field == "$card":
                false_card = json.loads(encoded_value)
            else:
                false_card = dict(cards[source_id])
                false_card[field] = json.loads(encoded_value)
            changed = [
                key
                for key in false_card
                if false_card[key] != cards[source_id][key]
            ]
            if (
                not changed
                or (field != "$card" and changed != [field])
                or set(false_card) != set(cards[source_id])
            ):
                raise KnowledgePairedDataError(
                    "false contract violates its declared corruption family"
                )
            changed_field_counts[str(len(changed))] += 1
            pair_id = "knowledge-pair:" + _identity(
                PAIR_SCHEMA, source_id, family
            )
            positive = _row(
                source_id=source_id,
                surface="verify_true",
                prompt=_verify_prompt(
                    display, cards[source_id], variant="train"
                ),
                answer="yes",
                pair_id=pair_id,
                pair_role="positive",
                corruption_family=family,
                changed_field=field,
            )
            negative = _row(
                source_id=source_id,
                surface="verify_false",
                prompt=_verify_prompt(display, false_card, variant="train"),
                answer="no",
                pair_id=pair_id,
                pair_role="negative",
                corruption_family=family,
                changed_field=field,
                donor_source_id=donor_id,
            )
            train_rows.extend([positive, negative])
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

    train_rows.sort(key=lambda row: str(row["row_id"]))
    pairs.sort(key=lambda row: str(row["pair_id"]))
    if len(train_rows) != 147 or len(pairs) != 63:
        raise KnowledgePairedDataError("paired row counts changed")

    rows_by_id = {str(row["row_id"]): row for row in train_rows}
    family_counts = Counter(str(pair["corruption_family"]) for pair in pairs)
    prompt_length_deltas: Counter[str] = Counter()
    for pair in pairs:
        positive = rows_by_id[str(pair["positive_row_id"])]
        negative = rows_by_id[str(pair["negative_row_id"])]
        positive_prompt = str(positive["messages"][0]["content"])
        negative_prompt = str(negative["messages"][0]["content"])
        positive_prefix, _separator, positive_contract = positive_prompt.partition(
            "\nCONTRACT="
        )
        negative_prefix, _separator, negative_contract = negative_prompt.partition(
            "\nCONTRACT="
        )
        if (
            positive_prefix != negative_prefix
            or not positive_contract.endswith("\nAnswer yes or no only.")
            or not negative_contract.endswith("\nAnswer yes or no only.")
        ):
            raise KnowledgePairedDataError("pair outer prompt changed")
        prompt_length_deltas[
            str(len(negative_prompt) - len(positive_prompt))
        ] += 1

    train_prompts = {
        str(row["messages"][0]["content"]) for row in train_rows
    }
    eval_prompts = {
        str(row["prompt"]) for row in _load_jsonl(eval_path)
    }
    original_prompts = {
        str(row["prompt"]) for row in selection
    }
    value_only = _value_only_accuracy(pairs, cards, false_cards)
    annotation_share = family_counts["annotation_swap"] / len(pairs)
    maximum_family_share = max(family_counts.values()) / len(pairs)
    gates = contract["shortcut_gates"]
    if (
        annotation_share
        > gates["annotation_only_negative_share_maximum"]
        or maximum_family_share
        > gates["single_corruption_family_share_maximum"]
        or max(value_only.values())
        > gates["empirical_value_only_accuracy_maximum_per_family"]
        or train_prompts & eval_prompts
        or train_prompts & original_prompts
    ):
        raise KnowledgePairedDataError(
            "paired shortcut gate failed: "
            f"annotation_share={annotation_share}, "
            f"maximum_family_share={maximum_family_share}, "
            f"value_only={value_only}, "
            f"frozen_overlap={len(train_prompts & eval_prompts)}, "
            f"original_overlap={len(train_prompts & original_prompts)}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    train_path = output_root / str(contract["outputs"]["train"])
    pairs_path = output_root / str(contract["outputs"]["pairs"])
    summary_path = output_root / str(contract["outputs"]["summary"])
    train_path.write_bytes(_serialize_jsonl(train_rows))
    pairs_path.write_bytes(_serialize_jsonl(pairs))
    summary = {
        "schema": SUMMARY_SCHEMA,
        "contract_id": CONTRACT_ID,
        "status": "pass",
        "deterministic_order": True,
        "inputs": {
            "contract": {
                "path": str(contract_path.relative_to(repo_root)),
                "sha256": _sha256(contract_path),
            },
            "knowledge_card_generator": {
                "path": str(generator_path.relative_to(repo_root)),
                "sha256": _sha256(generator_path),
            },
            "fact_deltas": {
                "path": str(facts_path.relative_to(repo_root)),
                "sha256": _sha256(facts_path),
            },
            "selection": {
                "path": str(selection_path.relative_to(repo_root)),
                "sha256": _sha256(selection_path),
            },
            "unchanged_dev": {
                "path": str(dev_path.relative_to(repo_root)),
                "sha256": _sha256(dev_path),
            },
            "unchanged_eval": {
                "path": str(eval_path.relative_to(repo_root)),
                "sha256": _sha256(eval_path),
            },
        },
        "outputs": {
            "train": {
                "rows": len(train_rows),
                "sha256": _sha256(train_path),
            },
            "pairs": {
                "rows": len(pairs),
                "sha256": _sha256(pairs_path),
            },
        },
        "training": {
            "source_ids": len(added_ids),
            "surface_counts": dict(
                sorted(Counter(str(row["surface"]) for row in train_rows).items())
            ),
            "pair_roles": {
                "positive": 63,
                "negative": 63,
            },
        },
        "shortcut_audit": {
            "outer_prompt_template_identical": True,
            "contract_key_order_identical": True,
            "changed_top_level_field_count_distribution": dict(
                sorted(
                    changed_field_counts.items(),
                    key=lambda item: int(item[0]),
                )
            ),
            "donor_value_verified_true_support_rate": 1.0,
            "corruption_family_counts": dict(sorted(family_counts.items())),
            "annotation_only_negative_share": annotation_share,
            "maximum_single_corruption_family_share": maximum_family_share,
            "empirical_value_only_accuracy_by_family": value_only,
            "maximum_empirical_value_only_accuracy": max(value_only.values()),
            "prompt_length_delta_false_minus_true": dict(
                sorted(prompt_length_deltas.items(), key=lambda item: int(item[0]))
            ),
            "frozen_eval_prompt_overlap": len(train_prompts & eval_prompts),
            "original_eval_prompt_overlap": len(
                train_prompts & original_prompts
            ),
            "model_visible_metadata_fields": [],
        },
        "claim_boundary": {
            "repairs_negative-construction": True,
            "does_not_change_frozen_eval": True,
            "does_not_establish_weight_adaptation": True,
        },
    }
    _write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build shortcut-resistant paired DELTA knowledge data."
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
    result = build_paired_data(
        contract_path=contract_path,
        repo_root=repo_root,
        output_root_override=output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
