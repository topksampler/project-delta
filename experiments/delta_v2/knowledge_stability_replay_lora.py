from __future__ import annotations

import hashlib
import json
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
    KnowledgeTrainError,
    _mean_loss,
    _model_inputs,
    _seed_everything,
    _tokenize_rows,
    target_module_allowed,
)


METRICS_SCHEMA = "delta.knowledge_stability_replay_train_metrics.v1"
RECEIPT_SCHEMA = "delta.knowledge_stability_replay_train_receipt.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def train_stability_candidate(
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
    replay_train_path: Path,
    replay_pairs_path: Path,
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
        raise KnowledgeTrainError("stability LoRA requires one CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise KnowledgeTrainError("stability LoRA requires bfloat16")
    optimization = protocol["optimization"]
    if protocol["objective"]["objective_id"] != (
        "assistant-only-generative-sft-v1"
    ):
        raise KnowledgeTrainError("stability LoRA objective changed")
    seed = int(optimization["seed"])
    _seed_everything(seed, torch)
    if len(schedule) != 240:
        raise KnowledgeTrainError("stability LoRA schedule changed")

    rows_by_id = {str(row["row_id"]): row for row in train_rows}
    if len(rows_by_id) != len(train_rows):
        raise KnowledgeTrainError("duplicate stability training row")
    processor = AutoProcessor.from_pretrained(
        MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )
    max_sequence_length = int(optimization["max_sequence_length"])
    examples = {
        str(row["row_id"]): _tokenize_rows(
            processor,
            [row],
            max_sequence_length=max_sequence_length,
        )[0]
        for row in train_rows
    }
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
        raise KnowledgeTrainError("stability LoRA created no parameters")
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
        raise KnowledgeTrainError("stability LoRA entered vision modules")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    max_steps = int(optimization["max_steps"])
    gradient_accumulation = int(
        optimization["gradient_accumulation_steps"]
    )
    if max_steps * gradient_accumulation != len(schedule):
        raise KnowledgeTrainError("stability optimizer schedule changed")
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(optimization["warmup_steps"]),
        num_training_steps=max_steps,
    )
    initial_dev_loss = _mean_loss(model, dev_examples, torch)
    stream_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    model_forwards = 0
    loss_history = []
    cursor = 0
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        step_streams = []
        for _micro_step in range(gradient_accumulation):
            unit = schedule[cursor]
            cursor += 1
            stream = str(unit["stream"])
            kind = str(unit["kind"])
            stream_counts[stream] += 1
            kind_counts[kind] += 1
            step_streams.append(stream)
            if kind == "recall":
                output = model(
                    **_model_inputs(
                        examples[str(unit["recall_row_id"])],
                        torch,
                    )
                )
                unit_loss = output.loss
                model_forwards += 1
            elif kind == "boolean_pair":
                family_counts[str(unit["corruption_family"])] += 1
                positive = model(
                    **_model_inputs(
                        examples[str(unit["positive_row_id"])],
                        torch,
                    )
                )
                negative = model(
                    **_model_inputs(
                        examples[str(unit["negative_row_id"])],
                        torch,
                    )
                )
                unit_loss = (positive.loss + negative.loss) / 2
                model_forwards += 2
            else:
                raise KnowledgeTrainError("unknown stability training unit")
            (unit_loss / gradient_accumulation).backward()
            micro_losses.append(float(unit_loss.detach().cpu()))
        if tuple(step_streams) not in {
            (
                "acquisition_pair",
                "acquisition_pair",
                "acquisition_recall",
                "replay_pair",
            ),
            (
                "acquisition_pair",
                "acquisition_pair",
                "acquisition_pair",
                "replay_recall",
            ),
        }:
            raise KnowledgeTrainError("stability microbatch shape changed")
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
                "streams": step_streams,
            }
        )
    if cursor != len(schedule):
        raise KnowledgeTrainError("stability schedule was not consumed once")
    final_dev_loss = _mean_loss(model, dev_examples, torch)

    output_dir = repo_root / str(config["output"]["dir"])
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    adapter_path = adapter_dir / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise KnowledgeTrainError("stability adapter was not saved")
    trainable_parameters = sum(
        parameter.numel() for parameter in trainable
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    schedule_sha256 = hashlib.sha256(
        json.dumps(
            list(schedule), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    metrics = {
        "schema": METRICS_SCHEMA,
        "run_id": run_id,
        "condition_id": condition_id,
        "protocol_id": protocol_id,
        "objective_id": protocol["objective"]["objective_id"],
        "status": "pass",
        "steps": max_steps,
        "gradient_accumulation_steps": gradient_accumulation,
        "training_units_seen": cursor,
        "stream_counts": dict(sorted(stream_counts.items())),
        "unit_counts": dict(sorted(kind_counts.items())),
        "corruption_family_units_seen": dict(sorted(family_counts.items())),
        "schedule_sha256": schedule_sha256,
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
            int(example["tokens"]) for example in examples.values()
        ),
        "max_dev_tokens": max(
            int(example["tokens"]) for example in dev_examples
        ),
        "loss_history": loss_history,
    }
    metrics_path = output_dir / "train_metrics.json"
    _write_json(metrics_path, metrics)
    (output_dir / "config.yaml").write_bytes(config_path.read_bytes())
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "run_id": run_id,
        "condition_id": condition_id,
        "protocol_id": protocol_id,
        "dataset_id": dataset_id,
        "objective_id": protocol["objective"]["objective_id"],
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
        "replay_train_sha256": _sha256(repo_root / replay_train_path),
        "replay_pairs_sha256": _sha256(repo_root / replay_pairs_path),
        "dev_sha256": _sha256(repo_root / dev_path),
        "schedule_sha256": schedule_sha256,
        "schedule_units": len(schedule),
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
    _write_json(output_dir / "run_receipt.json", receipt)
    return {"metrics": metrics, "receipt": receipt}
