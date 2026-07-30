from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2.run_knowledge_eval import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    validate_runtime,
)
from experiments.delta_v2.train_knowledge_lora import (
    TARGET_MODULES_REGEX,
    KnowledgeTrainError,
    _mean_loss,
    _model_inputs,
    _seed_everything,
    _tokenize_rows,
    target_module_allowed,
)


ROW_SCHEMA = "delta.knowledge_paired_sft_row.v1"
PAIR_SCHEMA = "delta.knowledge_boolean_pair.v1"
METRICS_SCHEMA = "delta.knowledge_paired_lora_train_metrics.v1"
RECEIPT_SCHEMA = "delta.knowledge_paired_lora_train_receipt.v1"
OBJECTIVE_GENERATIVE = "assistant-only-generative-sft-v1"
OBJECTIVE_PAIRED = "paired-boolean-logistic-ranking-v1"
EXPECTED_SURFACES = {
    "exact_recall": 21,
    "verify_true": 63,
    "verify_false": 63,
}
EXPECTED_CORRUPTIONS = {
    "annotation_swap": 21,
    "config_default_swap": 9,
    "env_getter_swap": 12,
    "full_card_swap": 21,
}
ROW_FIELDS = {
    "schema",
    "row_id",
    "source_id",
    "surface",
    "truth_authority",
    "pair_id",
    "pair_role",
    "corruption_family",
    "changed_field",
    "donor_source_id",
    "messages",
}
PAIR_FIELDS = {
    "schema",
    "pair_id",
    "source_id",
    "source_kind",
    "corruption_family",
    "changed_field",
    "changed_top_level_fields",
    "donor_source_id",
    "positive_row_id",
    "negative_row_id",
    "true_contract_sha256",
    "false_contract_sha256",
}


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise KnowledgeTrainError(f"{field} must be a non-empty string")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_paired_rows(
    rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
) -> None:
    if len(rows) != 147 or len(pairs) != 63:
        raise KnowledgeTrainError("paired training row counts changed")
    row_ids: list[str] = []
    surfaces: Counter[str] = Counter()
    for row in rows:
        if row.get("schema") != ROW_SCHEMA or set(row) != ROW_FIELDS:
            raise KnowledgeTrainError("paired training row schema changed")
        row_id = _string(row.get("row_id"), "row_id")
        if row_id in row_ids:
            raise KnowledgeTrainError("duplicate paired training row")
        row_ids.append(row_id)
        _string(row.get("source_id"), "source_id")
        surface = _string(row.get("surface"), "surface")
        surfaces[surface] += 1
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or messages[0].get("role") != "user"
            or messages[1].get("role") != "assistant"
            or not isinstance(messages[0].get("content"), str)
            or not isinstance(messages[1].get("content"), str)
        ):
            raise KnowledgeTrainError("paired training messages changed")
        if surface == "exact_recall":
            if any(
                row[field] is not None
                for field in (
                    "pair_id",
                    "pair_role",
                    "corruption_family",
                    "changed_field",
                    "donor_source_id",
                )
            ):
                raise KnowledgeTrainError("recall row entered a boolean pair")
        elif surface == "verify_true":
            if (
                row["messages"][1]["content"] != "yes"
                or row.get("pair_role") != "positive"
                or row.get("donor_source_id") is not None
            ):
                raise KnowledgeTrainError("paired positive row changed")
        elif surface == "verify_false":
            if (
                row["messages"][1]["content"] != "no"
                or row.get("pair_role") != "negative"
                or not isinstance(row.get("donor_source_id"), str)
            ):
                raise KnowledgeTrainError("paired negative row changed")
        else:
            raise KnowledgeTrainError("paired training surface changed")
    if row_ids != sorted(row_ids) or surfaces != Counter(EXPECTED_SURFACES):
        raise KnowledgeTrainError("paired training ordering changed")

    rows_by_id = {str(row["row_id"]): row for row in rows}
    pair_ids: list[str] = []
    corruptions: Counter[str] = Counter()
    paired_rows: set[str] = set()
    for pair in pairs:
        if pair.get("schema") != PAIR_SCHEMA or set(pair) != PAIR_FIELDS:
            raise KnowledgeTrainError("boolean pair schema changed")
        pair_id = _string(pair.get("pair_id"), "pair_id")
        if pair_id in pair_ids:
            raise KnowledgeTrainError("duplicate boolean pair")
        pair_ids.append(pair_id)
        family = _string(
            pair.get("corruption_family"), "corruption_family"
        )
        corruptions[family] += 1
        positive_id = _string(
            pair.get("positive_row_id"), "positive_row_id"
        )
        negative_id = _string(
            pair.get("negative_row_id"), "negative_row_id"
        )
        try:
            positive = rows_by_id[positive_id]
            negative = rows_by_id[negative_id]
        except KeyError as exc:
            raise KnowledgeTrainError("pair references a missing row") from exc
        if (
            positive["pair_id"] != pair_id
            or negative["pair_id"] != pair_id
            or positive["source_id"] != pair["source_id"]
            or negative["source_id"] != pair["source_id"]
            or positive["corruption_family"] != family
            or negative["corruption_family"] != family
            or positive["changed_field"] != pair["changed_field"]
            or negative["changed_field"] != pair["changed_field"]
            or negative["donor_source_id"] != pair["donor_source_id"]
        ):
            raise KnowledgeTrainError("paired row linkage changed")
        paired_rows.update((positive_id, negative_id))
    expected_paired = {
        str(row["row_id"])
        for row in rows
        if row["surface"] != "exact_recall"
    }
    if (
        pair_ids != sorted(pair_ids)
        or corruptions != Counter(EXPECTED_CORRUPTIONS)
        or paired_rows != expected_paired
    ):
        raise KnowledgeTrainError("boolean pair coverage changed")


def build_training_units(
    rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validate_paired_rows(rows, pairs)
    rows_by_id = {str(row["row_id"]): row for row in rows}
    units = [
        {
            "unit_id": "recall:" + str(row["row_id"]),
            "kind": "recall",
            "recall_row_id": str(row["row_id"]),
        }
        for row in rows
        if row["surface"] == "exact_recall"
    ]
    units.extend(
        {
            "unit_id": "pair:" + str(pair["pair_id"]),
            "kind": "boolean_pair",
            "pair_id": str(pair["pair_id"]),
            "positive_row_id": str(pair["positive_row_id"]),
            "negative_row_id": str(pair["negative_row_id"]),
            "corruption_family": str(pair["corruption_family"]),
        }
        for pair in pairs
    )
    units.sort(key=lambda unit: str(unit["unit_id"]))
    if (
        len(units) != 84
        or Counter(str(unit["kind"]) for unit in units)
        != {"recall": 21, "boolean_pair": 63}
        or any(
            row_id not in rows_by_id
            for unit in units
            for row_id in (
                [unit["recall_row_id"]]
                if unit["kind"] == "recall"
                else [unit["positive_row_id"], unit["negative_row_id"]]
            )
        )
    ):
        raise KnowledgeTrainError("paired training units changed")
    return units


def paired_boolean_objective(
    *,
    true_yes_log_probability: Any,
    true_no_log_probability: Any,
    false_yes_log_probability: Any,
    false_no_log_probability: Any,
    ranking_margin: float,
    ranking_weight: float,
    torch: Any,
) -> tuple[Any, Mapping[str, Any]]:
    true_margin = true_yes_log_probability - true_no_log_probability
    false_margin = false_yes_log_probability - false_no_log_probability
    separation = true_margin - false_margin
    classification = (
        torch.nn.functional.softplus(-true_margin)
        + torch.nn.functional.softplus(false_margin)
    ) / 2
    ranking = torch.nn.functional.softplus(ranking_margin - separation)
    loss = classification + ranking_weight * ranking
    return loss, {
        "true_margin": true_margin,
        "false_margin": false_margin,
        "separation": separation,
        "classification_loss": classification,
        "ranking_loss": ranking,
    }


def _candidate_row(row: Mapping[str, Any], answer: str) -> dict[str, Any]:
    return {
        "row_id": f"{row['row_id']}:{answer}",
        "surface": row["surface"],
        "messages": [
            dict(row["messages"][0]),
            {"role": "assistant", "content": answer},
        ],
    }


def _completion_log_probability(
    model: Any,
    example: Mapping[str, Any],
    torch: Any,
) -> tuple[Any, Any]:
    output = model(**_model_inputs(example, torch))
    log_probability = -output.loss * int(example["assistant_tokens"])
    return log_probability, output


def _runtime_inputs(
    example: Mapping[str, Any],
    torch: Any,
) -> dict[str, Any]:
    return _model_inputs(example, torch)


def train_paired_candidate(
    *,
    config_path: Path,
    repo_root: Path,
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
    run_id: str,
    condition_id: str,
    protocol_id: str,
    protocol_path: Path,
    dataset_id: str,
    train_path: Path,
    pairs_path: Path,
    dev_path: Path,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    observed_runtime = validate_runtime(config)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForMultimodalLM,
        AutoProcessor,
        get_linear_schedule_with_warmup,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise KnowledgeTrainError("paired LoRA requires one CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise KnowledgeTrainError("paired LoRA requires bfloat16")
    optimization = protocol["optimization"]
    objective = protocol["objective"]
    objective_id = str(objective["objective_id"])
    if objective_id not in {OBJECTIVE_GENERATIVE, OBJECTIVE_PAIRED}:
        raise KnowledgeTrainError("unsupported paired LoRA objective")
    seed = int(optimization["seed"])
    _seed_everything(seed, torch)

    units = build_training_units(train_rows, pairs)
    rows_by_id = {str(row["row_id"]): row for row in train_rows}
    processor = AutoProcessor.from_pretrained(
        MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )
    max_sequence_length = int(optimization["max_sequence_length"])
    gold_examples = {
        str(row["row_id"]): _tokenize_rows(
            processor,
            [row],
            max_sequence_length=max_sequence_length,
        )[0]
        for row in train_rows
    }
    candidate_examples: dict[tuple[str, str], Mapping[str, Any]] = {}
    if objective_id == OBJECTIVE_PAIRED:
        for row in train_rows:
            if row["surface"] in {"verify_true", "verify_false"}:
                for answer in ("yes", "no"):
                    candidate_examples[(str(row["row_id"]), answer)] = (
                        _tokenize_rows(
                            processor,
                            [_candidate_row(row, answer)],
                            max_sequence_length=max_sequence_length,
                        )[0]
                    )
    dev_examples = _tokenize_rows(
        processor,
        dev_rows,
        max_sequence_length=max_sequence_length,
    )

    base = AutoModelForMultimodalLM.from_pretrained(
        MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
    )
    base.config.use_cache = False
    lora = protocol["lora"]
    model = get_peft_model(
        base,
        LoraConfig(
            r=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            bias=str(lora["bias"]),
            target_modules=str(lora["target_modules_regex"]),
        ),
    ).to("cuda")
    model.train()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise KnowledgeTrainError("paired LoRA created no parameters")
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if unexpected:
        raise KnowledgeTrainError(
            f"non-LoRA parameters became trainable: {unexpected[:3]}"
        )
    matched_base_modules = [
        name
        for name, _module in base.named_modules()
        if target_module_allowed(name)
    ]
    if not matched_base_modules or any(
        not name.startswith("model.language_model.layers.")
        for name in matched_base_modules
    ):
        raise KnowledgeTrainError("paired LoRA entered vision modules")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    max_steps = int(optimization["max_steps"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(optimization["warmup_steps"]),
        num_training_steps=max_steps,
    )
    gradient_accumulation = int(
        optimization["gradient_accumulation_steps"]
    )
    initial_dev_loss = _mean_loss(model, dev_examples, torch)
    order = list(range(len(units)))
    rng = random.Random(seed)
    rng.shuffle(order)
    cursor = 0
    unit_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    model_forwards = 0
    loss_history: list[dict[str, Any]] = []
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        for _micro_step in range(gradient_accumulation):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            unit = units[order[cursor]]
            cursor += 1
            unit_kind = str(unit["kind"])
            unit_counts[unit_kind] += 1
            if unit_kind == "recall":
                example = gold_examples[str(unit["recall_row_id"])]
                output = model(**_runtime_inputs(example, torch))
                unit_loss = output.loss
                model_forwards += 1
            else:
                positive_id = str(unit["positive_row_id"])
                negative_id = str(unit["negative_row_id"])
                family_counts[str(unit["corruption_family"])] += 1
                if objective_id == OBJECTIVE_GENERATIVE:
                    positive_output = model(
                        **_runtime_inputs(gold_examples[positive_id], torch)
                    )
                    negative_output = model(
                        **_runtime_inputs(gold_examples[negative_id], torch)
                    )
                    unit_loss = (
                        positive_output.loss + negative_output.loss
                    ) / 2
                    model_forwards += 2
                else:
                    true_yes, true_yes_output = _completion_log_probability(
                        model,
                        candidate_examples[(positive_id, "yes")],
                        torch,
                    )
                    true_no, true_no_output = _completion_log_probability(
                        model,
                        candidate_examples[(positive_id, "no")],
                        torch,
                    )
                    false_yes, false_yes_output = _completion_log_probability(
                        model,
                        candidate_examples[(negative_id, "yes")],
                        torch,
                    )
                    false_no, false_no_output = _completion_log_probability(
                        model,
                        candidate_examples[(negative_id, "no")],
                        torch,
                    )
                    unit_loss, _components = paired_boolean_objective(
                        true_yes_log_probability=true_yes,
                        true_no_log_probability=true_no,
                        false_yes_log_probability=false_yes,
                        false_no_log_probability=false_no,
                        ranking_margin=float(objective["ranking_margin"]),
                        ranking_weight=float(objective["ranking_weight"]),
                        torch=torch,
                    )
                    model_forwards += 4
                    del (
                        true_yes_output,
                        true_no_output,
                        false_yes_output,
                        false_no_output,
                    )
            (unit_loss / gradient_accumulation).backward()
            micro_losses.append(float(unit_loss.detach().cpu()))
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            float(optimization["max_grad_norm"]),
        )
        optimizer.step()
        scheduler.step()
        loss_history.append(
            {
                "step": step,
                "loss": sum(micro_losses) / len(micro_losses),
                "learning_rate": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm.detach().cpu()),
            }
        )
    final_dev_loss = _mean_loss(model, dev_examples, torch)

    output_dir = repo_root / str(config["output"]["dir"])
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    adapter_path = adapter_dir / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise KnowledgeTrainError("paired LoRA adapter was not saved")
    trainable_parameters = sum(
        parameter.numel() for parameter in trainable
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    metrics = {
        "schema": METRICS_SCHEMA,
        "run_id": run_id,
        "condition_id": condition_id,
        "protocol_id": protocol_id,
        "objective_id": objective_id,
        "status": "pass",
        "steps": max_steps,
        "gradient_accumulation_steps": gradient_accumulation,
        "training_units_seen": sum(unit_counts.values()),
        "unit_counts": dict(sorted(unit_counts.items())),
        "corruption_family_units_seen": dict(sorted(family_counts.items())),
        "model_forwards": model_forwards,
        "initial_dev_loss": initial_dev_loss,
        "final_dev_loss": final_dev_loss,
        "loss_reduction": initial_dev_loss - final_dev_loss,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "matched_language_modules": len(matched_base_modules),
        "vision_modules_trainable": False,
        "max_train_tokens": max(
            int(example["tokens"]) for example in gold_examples.values()
        ),
        "max_dev_tokens": max(
            int(example["tokens"]) for example in dev_examples
        ),
        "loss_history": loss_history,
    }
    metrics_path = output_dir / "train_metrics.json"
    _write_json(metrics_path, metrics)
    config_copy = output_dir / "config.yaml"
    config_copy.write_bytes(config_path.read_bytes())
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "run_id": run_id,
        "condition_id": condition_id,
        "protocol_id": protocol_id,
        "dataset_id": dataset_id,
        "objective_id": objective_id,
        "base_model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
        },
        "config_sha256": _sha256(config_path),
        "protocol_sha256": _sha256(repo_root / protocol_path),
        "train_sha256": _sha256(repo_root / train_path),
        "pairs_sha256": _sha256(repo_root / pairs_path),
        "dev_sha256": _sha256(repo_root / dev_path),
        "adapter_sha256": _sha256(adapter_path),
        "metrics_sha256": _sha256(metrics_path),
        "runtime": {
            **observed_runtime,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "max_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        },
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "status": "pass",
        "candidate_only": True,
        "promotion_authorized": False,
    }
    receipt_path = output_dir / "run_receipt.json"
    _write_json(receipt_path, receipt)
    return {"metrics": metrics, "receipt": receipt}
