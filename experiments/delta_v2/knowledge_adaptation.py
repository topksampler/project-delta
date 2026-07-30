from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from experiments.delta_v2.oracle_source_context import _feature_evidence


CONTRACT_SCHEMA = "delta.knowledge_adaptation_contract.v1"
CONTRACT_ID = "delta-v2-knowledge-adaptation-v1"
TRAIN_SCHEMA = "delta.knowledge_sft_row.v1"
PROBE_SCHEMA = "delta.knowledge_probe.v1"
SUMMARY_SCHEMA = "delta.knowledge_adaptation_build_audit.v1"
CARD_VERSION = "delta-v2-knowledge-card-v1"
SUPPORTED_FAMILIES = {
    "python.config_field.v1",
    "python.environment_variable.v1",
}
LETTERS = ("A", "B", "C", "D")


class KnowledgeAdaptationError(ValueError):
    """The knowledge-adaptation data cannot be built without weakening truth."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KnowledgeAdaptationError(f"{field} must be a mapping")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise KnowledgeAdaptationError(f"{field} must be a non-empty string")
    return value


def _relative_path(value: Any, field: str) -> Path:
    path = Path(_string(value, field))
    if path.is_absolute() or ".." in path.parts:
        raise KnowledgeAdaptationError(f"{field} must stay in the repository")
    return path


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise KnowledgeAdaptationError(f"cannot read YAML: {path}") from exc
    return _mapping(payload, str(path))


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeAdaptationError(f"cannot read JSON: {path}") from exc
    return _mapping(payload, str(path))


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise KnowledgeAdaptationError(f"cannot read JSONL: {path}") from exc
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise KnowledgeAdaptationError(
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
        raise KnowledgeAdaptationError(f"{field} bytes changed")
    return path


def validate_contract(
    contract: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[Path, Path, Path, Path]:
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("contract_id") != CONTRACT_ID
        or contract.get("experiment_id") != "delta_v2"
        or contract.get("parent_environment_id")
        != "delta-v2-vllm-source-build-v2"
        or contract.get("target_revision") != "v0.26.0"
        or contract.get("freeze_state") != "preregistered"
    ):
        raise KnowledgeAdaptationError("knowledge-adaptation identity changed")

    truth = _mapping(contract.get("truth_sources"), "truth_sources")
    facts_path = _binding_path(
        _mapping(truth.get("fact_deltas"), "fact_deltas"),
        repo_root=repo_root,
        field="fact_deltas",
    )
    selection_path = _binding_path(
        _mapping(truth.get("frozen_eval_selection"), "frozen_eval_selection"),
        repo_root=repo_root,
        field="frozen_eval_selection",
    )
    feature_path = _binding_path(
        _mapping(truth.get("feature_probe"), "feature_probe"),
        repo_root=repo_root,
        field="feature_probe",
    )

    card = _mapping(contract.get("knowledge_card"), "knowledge_card")
    if (
        card.get("generator_version") != CARD_VERSION
        or card.get("source_status") != "added"
        or card.get("source_count") != 21
        or set(card.get("supported_families") or []) != SUPPORTED_FAMILIES
        or card.get("authority")
        != "normalized-after-revision-AST-observation"
    ):
        raise KnowledgeAdaptationError("knowledge-card contract changed")

    training = _mapping(contract.get("training"), "training")
    if (
        training.get("format") != "chat-messages"
        or training.get("surfaces_per_source")
        != ["exact_recall", "verify_true", "verify_false"]
        or training.get("rows_per_source") != 3
        or training.get("rows") != 63
        or training.get("truth_balance")
        != {"exact_recall": 21, "yes": 21, "no": 21}
        or training.get("target_model_inputs") != ["messages"]
    ):
        raise KnowledgeAdaptationError("training contract changed")

    evaluation = _mapping(contract.get("evaluation"), "evaluation")
    acquisition = _mapping(evaluation.get("acquisition"), "acquisition")
    retention = _mapping(evaluation.get("retention"), "retention")
    feature = _mapping(
        evaluation.get("feature_retention"),
        "feature_retention",
    )
    probes = ["choice", "boolean_true", "boolean_false", "recall"]
    if (
        acquisition.get("sources") != "same-21-added-source-cards"
        or acquisition.get("claim_overlap_with_training") != "intentional"
        or acquisition.get("exact_prompt_overlap_with_training") != "forbidden"
        or acquisition.get("probes_per_source") != probes
        or retention.get("sources") != "frozen-21-stable-controls"
        or retention.get("source_overlap_with_training") != "forbidden"
        or retention.get("probes_per_source") != probes
        or feature.get("source") != "feature:vllm-endpoint-plugins"
        or feature.get("source_overlap_with_training") != "forbidden"
    ):
        raise KnowledgeAdaptationError("evaluation split contract changed")

    outputs = _mapping(contract.get("outputs"), "outputs")
    output_root = _relative_path(outputs.get("root"), "outputs.root")
    expected_names = {
        "train": "train.jsonl",
        "dev": "dev.jsonl",
        "eval": "eval.jsonl",
        "summary": "summary.json",
    }
    if any(outputs.get(key) != value for key, value in expected_names.items()):
        raise KnowledgeAdaptationError("knowledge output paths changed")
    return facts_path, selection_path, feature_path, output_root


def _decode_canonical_ast(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_canonical_ast(item) for item in value]
    if isinstance(value, Mapping) and set(value) == {"node", "fields"}:
        node_name = _string(value.get("node"), "canonical_ast.node")
        node_type = getattr(ast, node_name, None)
        if not isinstance(node_type, type) or not issubclass(node_type, ast.AST):
            raise KnowledgeAdaptationError(
                f"unsupported canonical AST node: {node_name}"
            )
        fields = _mapping(value.get("fields"), "canonical_ast.fields")
        return node_type(
            **{
                str(name): _decode_canonical_ast(child)
                for name, child in fields.items()
            }
        )
    return value


def _expression(value: Any, field: str) -> str:
    node = _decode_canonical_ast(value)
    if not isinstance(node, ast.AST):
        raise KnowledgeAdaptationError(f"{field} is not a canonical AST")
    try:
        return ast.unparse(ast.fix_missing_locations(node))
    except (AttributeError, TypeError, ValueError) as exc:
        raise KnowledgeAdaptationError(f"cannot render {field}") from exc


def _constraint_expression(value: Any, field: str) -> str:
    payload = _mapping(value, field)
    kind = payload.get("kind")
    if kind == "literal":
        return canonical_json(payload.get("value"))
    if kind == "expression":
        return _expression(payload.get("ast"), f"{field}.ast")
    if kind == "absent":
        return "<absent>"
    raise KnowledgeAdaptationError(f"unsupported constraint value: {field}")


def _display_name(record: Mapping[str, Any]) -> str:
    family = record.get("family")
    semantic_key = _string(record.get("semantic_key"), "semantic_key")
    if family == "python.config_field.v1":
        prefix = "python-config-field:"
        if not semantic_key.startswith(prefix):
            raise KnowledgeAdaptationError("invalid config-field semantic key")
        return f"Python configuration field `{semantic_key[len(prefix):]}`"
    if family == "python.environment_variable.v1":
        prefix = "python-environment-variable:"
        if not semantic_key.startswith(prefix):
            raise KnowledgeAdaptationError("invalid environment semantic key")
        return f"Python environment variable `{semantic_key[len(prefix):]}`"
    raise KnowledgeAdaptationError(f"unsupported family: {family}")


def knowledge_card(record: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("status") not in {"added", "stable"}:
        raise KnowledgeAdaptationError("knowledge cards require after values")
    value = _mapping(record.get("value_after"), "value_after")
    family = record.get("family")
    if family == "python.config_field.v1":
        default_kind = _string(value.get("default_kind"), "default_kind")
        default = (
            "<required>"
            if default_kind == "required"
            else _expression(value.get("default_ast"), "default_ast")
        )
        constraints_payload = _mapping(
            value.get("static_field_constraints"),
            "static_field_constraints",
        )
        constraints = {
            str(name): _constraint_expression(
                constraint,
                f"constraint.{name}",
            )
            for name, constraint in sorted(constraints_payload.items())
        }
        return {
            "kind": "python_config_field",
            "annotation": _expression(
                value.get("annotation_ast"),
                "annotation_ast",
            ),
            "default": default,
            "constraints": constraints,
        }
    if family == "python.environment_variable.v1":
        getter = _expression(
            value.get("getter_expression_ast"),
            "getter_expression_ast",
        )
        if getter.startswith("lambda: "):
            getter = getter[len("lambda: ") :]
        return {
            "kind": "python_environment_variable",
            "annotation": _expression(
                value.get("type_checking_annotation_ast"),
                "type_checking_annotation_ast",
            ),
            "getter": getter,
        }
    raise KnowledgeAdaptationError(f"unsupported family: {family}")


def _alternate_annotation(annotation: str) -> str:
    for candidate in ("str", "int", "bool", "float"):
        if candidate != annotation:
            return candidate
    return "object"


def _alternate_default(default: str) -> str:
    replacements = {
        "False": "True",
        "True": "False",
        "None": "0",
        "<required>": "None",
    }
    return replacements.get(default, "None")


def distractor_cards(
    record: Mapping[str, Any],
    card: Mapping[str, Any],
) -> list[dict[str, Any]]:
    family = record.get("family")
    candidates: list[dict[str, Any]] = []
    if family == "python.config_field.v1":
        first = copy.deepcopy(dict(card))
        first["annotation"] = _alternate_annotation(str(card["annotation"]))
        candidates.append(first)

        second = copy.deepcopy(dict(card))
        second["default"] = _alternate_default(str(card["default"]))
        candidates.append(second)

        third = copy.deepcopy(dict(card))
        constraints = dict(_mapping(third.get("constraints"), "constraints"))
        if constraints:
            key = sorted(constraints)[0]
            constraints[key] = "1" if constraints[key] != "1" else "0"
        else:
            constraints["gt"] = "0"
        third["constraints"] = constraints
        candidates.append(third)
    elif family == "python.environment_variable.v1":
        semantic_key = str(record["semantic_key"])
        key = semantic_key.removeprefix("python-environment-variable:")

        first = copy.deepcopy(dict(card))
        first["annotation"] = _alternate_annotation(str(card["annotation"]))
        candidates.append(first)

        second = copy.deepcopy(dict(card))
        second["getter"] = f"os.environ.get({key!r}, '<unset>')"
        candidates.append(second)

        third = copy.deepcopy(dict(card))
        third["getter"] = f"bool(os.environ.get({key!r}, '0'))"
        candidates.append(third)
    else:
        raise KnowledgeAdaptationError(f"unsupported family: {family}")

    correct = canonical_json(card)
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        encoded = canonical_json(candidate)
        if encoded != correct:
            unique.setdefault(encoded, candidate)
    if len(unique) != 3:
        raise KnowledgeAdaptationError("could not build three distractors")
    return list(unique.values())


def _identity(schema: str, *parts: str) -> str:
    payload = "\0".join((schema, CONTRACT_ID, *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _recall_prompt(display: str, *, variant: str) -> str:
    if variant == "train":
        return (
            f"In vLLM v0.26.0, return the exact verified source contract for "
            f"{display}. Return only one compact JSON object."
        )
    if variant == "dev":
        return (
            f"Give the vLLM v0.26.0 source-level contract for {display}. "
            "Output the compact JSON object only."
        )
    if variant == "eval":
        return (
            f"What exact contract does vLLM v0.26.0 define for {display}? "
            "Answer with only the compact JSON object."
        )
    raise KnowledgeAdaptationError(f"unsupported recall variant: {variant}")


def _verify_prompt(
    display: str,
    card: Mapping[str, Any],
    *,
    variant: str,
) -> str:
    if variant == "train":
        question = (
            f"In vLLM v0.26.0, does this compact contract exactly match "
            f"{display}?"
        )
    elif variant == "eval":
        question = (
            f"Is the following exact source-level claim about {display} "
            "correct for vLLM v0.26.0?"
        )
    else:
        raise KnowledgeAdaptationError(
            f"unsupported verification variant: {variant}"
        )
    return (
        f"{question}\nCONTRACT={canonical_json(card)}\n"
        "Answer yes or no only."
    )


def _choice_probe(
    record: Mapping[str, Any],
    card: Mapping[str, Any],
) -> tuple[str, str]:
    source_id = str(record["fact_id"])
    options = [dict(card), *distractor_cards(record, card)]
    ranked = sorted(
        options,
        key=lambda option: hashlib.sha256(
            (
                "delta-v2-knowledge-choice-v1"
                + "\0"
                + source_id
                + "\0"
                + canonical_json(option)
            ).encode("utf-8")
        ).hexdigest(),
    )
    correct = canonical_json(card)
    gold_index = [canonical_json(option) for option in ranked].index(correct)
    option_lines = [
        f"{letter}. {canonical_json(option)}"
        for letter, option in zip(LETTERS, ranked, strict=True)
    ]
    prompt = (
        f"Which exact vLLM v0.26.0 source contract belongs to "
        f"{_display_name(record)}?\n"
        + "\n".join(option_lines)
        + "\nAnswer A, B, C, or D only."
    )
    return prompt, LETTERS[gold_index]


def _train_row(
    record: Mapping[str, Any],
    *,
    surface: str,
    prompt: str,
    answer: str,
) -> dict[str, Any]:
    source_id = str(record["fact_id"])
    return {
        "schema": TRAIN_SCHEMA,
        "row_id": "knowledge-train:"
        + _identity(TRAIN_SCHEMA, source_id, surface),
        "source_id": source_id,
        "surface": surface,
        "truth_authority": CARD_VERSION,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def _probe_row(
    record: Mapping[str, Any],
    *,
    stratum: str,
    probe_kind: str,
    prompt: str,
    gold: Any,
    scorer: str,
    training_overlap: str,
) -> dict[str, Any]:
    source_id = str(record["fact_id"])
    return {
        "schema": PROBE_SCHEMA,
        "probe_id": "knowledge-probe:"
        + _identity(PROBE_SCHEMA, source_id, stratum, probe_kind),
        "source_id": source_id,
        "source_kind": "atomic_fact_delta",
        "stratum": stratum,
        "probe_kind": probe_kind,
        "prompt": prompt,
        "gold": gold,
        "scorer": scorer,
        "training_overlap": training_overlap,
        "truth_authority": CARD_VERSION,
    }


def _fact_probe_rows(
    record: Mapping[str, Any],
    *,
    stratum: str,
    training_overlap: str,
) -> list[dict[str, Any]]:
    card = knowledge_card(record)
    false_card = distractor_cards(record, card)[0]
    choice_prompt, choice_gold = _choice_probe(record, card)
    return [
        _probe_row(
            record,
            stratum=stratum,
            probe_kind="choice",
            prompt=choice_prompt,
            gold=choice_gold,
            scorer="exact-choice-v1",
            training_overlap=training_overlap,
        ),
        _probe_row(
            record,
            stratum=stratum,
            probe_kind="boolean_true",
            prompt=_verify_prompt(
                _display_name(record),
                card,
                variant="eval",
            ),
            gold="yes",
            scorer="exact-boolean-v1",
            training_overlap=training_overlap,
        ),
        _probe_row(
            record,
            stratum=stratum,
            probe_kind="boolean_false",
            prompt=_verify_prompt(
                _display_name(record),
                false_card,
                variant="eval",
            ),
            gold="no",
            scorer="exact-boolean-v1",
            training_overlap=training_overlap,
        ),
        _probe_row(
            record,
            stratum=stratum,
            probe_kind="recall",
            prompt=_recall_prompt(_display_name(record), variant="eval"),
            gold=card,
            scorer="exact-json-v1",
            training_overlap=training_overlap,
        ),
    ]


def _feature_probe_row(
    feature_item: Mapping[str, Any],
    feature_probe: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = _feature_evidence(feature_probe)
    source_id = str(feature_item["source_id"])
    gold = _mapping(feature_item.get("gold"), "feature.gold")
    prompt = (
        "Use only these verified vLLM v0.26.0 endpoint-plugin observations:\n"
        f"OBSERVED_AFTER={canonical_json(evidence)}\n"
        "Return exactly one JSON object with the six user-facing behavior "
        "states: default_without_allowlist, allowlisted_matching_task, "
        "required_task_mismatch, factory_exception, route_phase, and "
        "post_engine_state_phase. Output JSON only."
    )
    return {
        "schema": PROBE_SCHEMA,
        "probe_id": "knowledge-probe:"
        + _identity(PROBE_SCHEMA, source_id, "feature_retention"),
        "source_id": source_id,
        "source_kind": "feature_delta",
        "stratum": "feature_retention",
        "probe_kind": "verified_behavior_json",
        "prompt": prompt,
        "gold": dict(gold),
        "scorer": "exact-json-v1",
        "training_overlap": "none",
        "truth_authority": "verified-behavior-probe",
    }


def _serialize_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (canonical_json(dict(row)) + "\n").encode("utf-8") for row in rows
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_datasets(
    *,
    contract_path: Path,
    repo_root: Path,
    output_root_override: Path | None = None,
) -> dict[str, Any]:
    contract = _load_yaml(contract_path)
    (
        facts_path,
        selection_path,
        feature_path,
        output_root,
    ) = validate_contract(contract, repo_root=repo_root)
    if output_root_override is not None:
        output_root = output_root_override
    elif not output_root.is_absolute():
        output_root = repo_root / output_root

    facts = {
        str(row["fact_id"]): row for row in _load_jsonl(facts_path)
    }
    selection = _load_jsonl(selection_path)
    added_ids: list[str] = []
    stable_ids: list[str] = []
    feature_items: list[Mapping[str, Any]] = []
    original_prompts: set[str] = set()
    for item in selection:
        original_prompts.add(_string(item.get("prompt"), "item.prompt"))
        if item.get("source_kind") == "feature_delta":
            feature_items.append(item)
            continue
        source_id = _string(item.get("source_id"), "item.source_id")
        status = _mapping(item.get("gold"), "item.gold").get("status")
        if status == "added":
            added_ids.append(source_id)
        elif status == "stable":
            stable_ids.append(source_id)
        else:
            raise KnowledgeAdaptationError("unexpected frozen fact selection")
    if (
        len(added_ids) != 21
        or len(stable_ids) != 21
        or len(feature_items) != 1
        or len(set(added_ids + stable_ids)) != 42
    ):
        raise KnowledgeAdaptationError("frozen selection counts changed")

    added = [facts[source_id] for source_id in sorted(added_ids)]
    stable = [facts[source_id] for source_id in sorted(stable_ids)]
    if any(
        record.get("status") != "added"
        or record.get("family") not in SUPPORTED_FAMILIES
        for record in added
    ):
        raise KnowledgeAdaptationError("added knowledge source changed")
    if any(record.get("status") != "stable" for record in stable):
        raise KnowledgeAdaptationError("stable retention source changed")

    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for record in added:
        card = knowledge_card(record)
        false_card = distractor_cards(record, card)[0]
        display = _display_name(record)
        train_rows.extend(
            [
                _train_row(
                    record,
                    surface="exact_recall",
                    prompt=_recall_prompt(display, variant="train"),
                    answer=canonical_json(card),
                ),
                _train_row(
                    record,
                    surface="verify_true",
                    prompt=_verify_prompt(
                        display,
                        card,
                        variant="train",
                    ),
                    answer="yes",
                ),
                _train_row(
                    record,
                    surface="verify_false",
                    prompt=_verify_prompt(
                        display,
                        false_card,
                        variant="train",
                    ),
                    answer="no",
                ),
            ]
        )
        dev_rows.append(
            _train_row(
                record,
                surface="dev_recall",
                prompt=_recall_prompt(display, variant="dev"),
                answer=canonical_json(card),
            )
        )
        eval_rows.extend(
            _fact_probe_rows(
                record,
                stratum="acquisition_added",
                training_overlap="same-source-new-surface",
            )
        )
    for record in stable:
        eval_rows.extend(
            _fact_probe_rows(
                record,
                stratum="retention_stable",
                training_overlap="none",
            )
        )
    feature_probe = _load_json(feature_path)
    eval_rows.append(_feature_probe_row(feature_items[0], feature_probe))

    train_rows.sort(key=lambda row: str(row["row_id"]))
    dev_rows.sort(key=lambda row: str(row["row_id"]))
    eval_rows.sort(key=lambda row: str(row["probe_id"]))
    if len(train_rows) != 63 or len(dev_rows) != 21 or len(eval_rows) != 169:
        raise KnowledgeAdaptationError("knowledge dataset counts changed")

    training_prompts = {
        str(row["messages"][0]["content"]) for row in train_rows
    }
    eval_prompts = {str(row["prompt"]) for row in eval_rows}
    if training_prompts & eval_prompts:
        raise KnowledgeAdaptationError("exact train/eval prompt leakage")
    if training_prompts & original_prompts:
        raise KnowledgeAdaptationError(
            "original frozen EvalItem wording entered training"
        )
    train_sources = {str(row["source_id"]) for row in train_rows}
    stable_sources = {
        str(row["source_id"])
        for row in eval_rows
        if row["stratum"] == "retention_stable"
    }
    feature_sources = {
        str(row["source_id"])
        for row in eval_rows
        if row["stratum"] == "feature_retention"
    }
    if train_sources & (stable_sources | feature_sources):
        raise KnowledgeAdaptationError("retention source entered training")

    output_root.mkdir(parents=True, exist_ok=True)
    train_path = output_root / "train.jsonl"
    dev_path = output_root / "dev.jsonl"
    eval_path = output_root / "eval.jsonl"
    summary_path = output_root / "summary.json"
    train_path.write_bytes(_serialize_jsonl(train_rows))
    dev_path.write_bytes(_serialize_jsonl(dev_rows))
    eval_path.write_bytes(_serialize_jsonl(eval_rows))

    strata = Counter(str(row["stratum"]) for row in eval_rows)
    probe_kinds = Counter(str(row["probe_kind"]) for row in eval_rows)
    answers = Counter(
        str(row["messages"][1]["content"]) for row in train_rows
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "contract_id": CONTRACT_ID,
        "status": "pass",
        "deterministic_order": True,
        "truth_authority": "frozen-normalized-source-and-behavior-evidence",
        "inputs": {
            "contract": {
                "path": str(contract_path.relative_to(repo_root)),
                "sha256": _sha256(contract_path),
            },
            "fact_deltas": {
                "path": str(facts_path.relative_to(repo_root)),
                "sha256": _sha256(facts_path),
            },
            "selection": {
                "path": str(selection_path.relative_to(repo_root)),
                "sha256": _sha256(selection_path),
            },
            "feature_probe": {
                "path": str(feature_path.relative_to(repo_root)),
                "sha256": _sha256(feature_path),
            },
        },
        "outputs": {
            "train": {
                "rows": len(train_rows),
                "sha256": _sha256(train_path),
            },
            "dev": {
                "rows": len(dev_rows),
                "sha256": _sha256(dev_path),
            },
            "eval": {
                "rows": len(eval_rows),
                "sha256": _sha256(eval_path),
            },
        },
        "training": {
            "source_ids": len(train_sources),
            "answer_counts": dict(sorted(answers.items())),
        },
        "evaluation": {
            "strata": dict(sorted(strata.items())),
            "probe_kinds": dict(sorted(probe_kinds.items())),
            "acquisition_source_overlap": len(
                train_sources
                & {
                    str(row["source_id"])
                    for row in eval_rows
                    if row["stratum"] == "acquisition_added"
                }
            ),
            "retention_source_overlap": len(
                train_sources & stable_sources
            ),
            "feature_source_overlap": len(
                train_sources & feature_sources
            ),
            "exact_prompt_overlap": len(training_prompts & eval_prompts),
            "original_eval_prompt_overlap": len(
                training_prompts & original_prompts
            ),
            "pooled_overall_score": None,
        },
        "claim_boundary": {
            "acquisition": "same-claim-new-surface",
            "held_out_fact_generalization": False,
            "free_recall": "advisory",
        },
    }
    _write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build truth-rooted delta_v2 LoRA train and eval data."
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "experiments/delta_v2/knowledge_adaptation_contract.yaml"
        ),
    )
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
    summary = build_datasets(
        contract_path=contract_path,
        repo_root=repo_root,
        output_root_override=output_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
