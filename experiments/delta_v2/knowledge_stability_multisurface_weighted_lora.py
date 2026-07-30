from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2.knowledge_paired_lora import (
    _candidate_row,
    _completion_log_probability,
    paired_boolean_objective,
)
from experiments.delta_v2.run_knowledge_eval import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    validate_runtime,
)
from experiments.delta_v2.train_knowledge_lora import (
    KnowledgeTrainError,
    _mean_loss,
    _model_inputs,
    _seed_everything,
    _tokenize_rows,
    target_module_allowed,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def train_multisurface_weighted_candidate(
    *,
    config_path: Path,
    repo_root: Path,
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    run_id: str,
    condition_id: str,
    protocol_id: str,
    protocol_path: Path,
    dataset_id: str,
    acquisition_train_path: Path,
    acquisition_pairs_path: Path,
    stability_train_path: Path,
    stability_pairs_path: Path,
    stability_eval_path: Path,
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
        raise KnowledgeTrainError("multisurface LoRA requires one CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise KnowledgeTrainError("multisurface LoRA requires bfloat16")
    objective = protocol["objective"]
    if (
        objective.get("objective_id")
        != "generative-sft-plus-paired-ranking-v2"
        or objective.get("ranking_margin") != 1.0
        or objective.get("ranking_weight") != 0.5
    ):
        raise KnowledgeTrainError("multisurface objective changed")
    optimization = protocol["optimization"]
    _seed_everything(int(optimization["seed"]), torch)
    if len(schedule) != 240:
        raise KnowledgeTrainError("multisurface schedule changed")

    rows_by_id = {str(row["row_id"]): row for row in train_rows}
    processor = AutoProcessor.from_pretrained(
        MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )
    max_length = int(optimization["max_sequence_length"])
    gold_examples = {
        row_id: _tokenize_rows(
            processor, [row], max_sequence_length=max_length
        )[0]
        for row_id, row in rows_by_id.items()
    }
    alternative_examples = {}
    for row_id, row in rows_by_id.items():
        if row["surface"] in {"verify_true", "verify_false"}:
            gold = str(row["messages"][1]["content"])
            answer = "no" if gold == "yes" else "yes"
            alternative_examples[(row_id, answer)] = _tokenize_rows(
                processor,
                [_candidate_row(row, answer)],
                max_sequence_length=max_length,
            )[0]
    dev_examples = _tokenize_rows(
        processor, dev_rows, max_sequence_length=max_length
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
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    matched = [
        name for name, _ in base.named_modules()
        if target_module_allowed(name)
    ]
    if (
        not trainable
        or unexpected
        or not matched
        or any(
            not name.startswith("model.language_model.layers.")
            for name in matched
        )
    ):
        raise KnowledgeTrainError("multisurface trainable scope changed")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    max_steps = int(optimization["max_steps"])
    accumulation = int(optimization["gradient_accumulation_steps"])
    if max_steps * accumulation != len(schedule):
        raise KnowledgeTrainError("multisurface optimizer schedule changed")
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(optimization["warmup_steps"]),
        num_training_steps=max_steps,
    )
    initial_dev_loss = _mean_loss(model, dev_examples, torch)
    cursor = 0
    kind_counts: Counter[str] = Counter()
    stream_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    loss_history = []
    model_forwards = 0
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        step_streams = []
        for _ in range(accumulation):
            unit = schedule[cursor]
            cursor += 1
            kind = str(unit["kind"])
            stream = str(unit["stream"])
            kind_counts[kind] += 1
            stream_counts[stream] += 1
            step_streams.append(stream)
            if kind == "recall":
                output = model(
                    **_model_inputs(
                        gold_examples[str(unit["recall_row_id"])], torch
                    )
                )
                unit_loss = output.loss
                model_forwards += 1
            elif kind == "boolean_pair":
                family_counts[str(unit["corruption_family"])] += 1
                positive_id = str(unit["positive_row_id"])
                negative_id = str(unit["negative_row_id"])
                positive = model(
                    **_model_inputs(gold_examples[positive_id], torch)
                )
                negative = model(
                    **_model_inputs(gold_examples[negative_id], torch)
                )
                true_yes = (
                    -positive.loss
                    * int(gold_examples[positive_id]["assistant_tokens"])
                )
                false_no = (
                    -negative.loss
                    * int(gold_examples[negative_id]["assistant_tokens"])
                )
                true_no, true_no_output = _completion_log_probability(
                    model,
                    alternative_examples[(positive_id, "no")],
                    torch,
                )
                false_yes, false_yes_output = _completion_log_probability(
                    model,
                    alternative_examples[(negative_id, "yes")],
                    torch,
                )
                ranking, _components = paired_boolean_objective(
                    true_yes_log_probability=true_yes,
                    true_no_log_probability=true_no,
                    false_yes_log_probability=false_yes,
                    false_no_log_probability=false_no,
                    ranking_margin=1.0,
                    ranking_weight=1.0,
                    torch=torch,
                )
                generative = (positive.loss + negative.loss) / 2
                unit_loss = generative + 0.5 * ranking
                model_forwards += 4
                del true_no_output, false_yes_output
            else:
                raise KnowledgeTrainError("unknown multisurface unit")
            (unit_loss / accumulation).backward()
            micro_losses.append(float(unit_loss.detach().cpu()))
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(optimization["max_grad_norm"])
        )
        optimizer.step()
        scheduler.step()
        loss_history.append(
            {
                "step": step,
                "loss": sum(micro_losses) / len(micro_losses),
                "learning_rate": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm.detach().cpu()),
                "streams": step_streams,
            }
        )
    final_dev_loss = _mean_loss(model, dev_examples, torch)
    output_dir = repo_root / str(config["output"]["dir"])
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    adapter_path = adapter_dir / "adapter_model.safetensors"
    schedule_hash = hashlib.sha256(
        json.dumps(
            list(schedule), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    metrics = {
        "schema": "delta.knowledge_stability_multisurface_metrics.v1",
        "run_id": run_id,
        "condition_id": condition_id,
        "protocol_id": protocol_id,
        "objective_id": objective["objective_id"],
        "status": "pass",
        "steps": max_steps,
        "training_units_seen": cursor,
        "unit_counts": dict(sorted(kind_counts.items())),
        "source_stream_counts": dict(sorted(stream_counts.items())),
        "corruption_family_units_seen": dict(sorted(family_counts.items())),
        "model_forwards": model_forwards,
        "initial_dev_loss": initial_dev_loss,
        "final_dev_loss": final_dev_loss,
        "loss_reduction": initial_dev_loss - final_dev_loss,
        "schedule_sha256": schedule_hash,
        "ranking_margin": 1.0,
        "ranking_weight": 0.5,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "matched_language_modules": len(matched),
        "vision_modules_trainable": False,
        "loss_history": loss_history,
    }
    metrics_path = output_dir / "train_metrics.json"
    _write_json(metrics_path, metrics)
    (output_dir / "config.yaml").write_bytes(config_path.read_bytes())
    receipt = {
        "schema": "delta.knowledge_stability_multisurface_receipt.v1",
        "run_id": run_id,
        "condition_id": condition_id,
        "protocol_id": protocol_id,
        "dataset_id": dataset_id,
        "objective_id": objective["objective_id"],
        "base_model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
        },
        "config_sha256": _sha256(config_path),
        "protocol_sha256": _sha256(repo_root / protocol_path),
        "acquisition_train_sha256": _sha256(
            repo_root / acquisition_train_path
        ),
        "acquisition_pairs_sha256": _sha256(
            repo_root / acquisition_pairs_path
        ),
        "stability_train_sha256": _sha256(
            repo_root / stability_train_path
        ),
        "stability_pairs_sha256": _sha256(
            repo_root / stability_pairs_path
        ),
        "stability_eval_sha256": _sha256(
            repo_root / stability_eval_path
        ),
        "dev_sha256": _sha256(repo_root / dev_path),
        "schedule_sha256": schedule_hash,
        "schedule_units": len(schedule),
        "adapter_sha256": _sha256(adapter_path),
        "metrics_sha256": _sha256(metrics_path),
        "runtime": {
            **observed_runtime,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "deterministic_algorithms":
                torch.are_deterministic_algorithms_enabled(),
            "max_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        },
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "status": "pass",
        "candidate_only": True,
        "promotion_authorized": False,
    }
    _write_json(output_dir / "run_receipt.json", receipt)
    return {"metrics": metrics, "receipt": receipt}
