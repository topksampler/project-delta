from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from experiments.delta_v2.run_knowledge_eval import (
    DATASET_ID,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODAL_APP_PATH,
    RUNTIME_PACKAGES,
    TRAIN_RUN_ID,
    KnowledgeEvalError,
    validate_runtime,
)


PROTOCOL_SCHEMA = "delta.knowledge_lora_protocol.v1"
PROTOCOL_ID = "delta-v2-knowledge-lora-v2"
RUN_CONFIG_SCHEMA = "delta.knowledge_lora_train_config.v1"
TRAIN_SCHEMA = "delta.knowledge_sft_row.v1"
RECEIPT_SCHEMA = "delta.knowledge_lora_train_receipt.v1"
METRICS_SCHEMA = "delta.knowledge_lora_train_metrics.v1"
PROTOCOL_PATH = Path(
    "experiments/delta_v2/knowledge_lora_protocol_v2.yaml"
)
FREEZE_PATH = Path(
    "experiments/delta_v2/knowledge_adaptation_freeze.yaml"
)
TRAIN_PATH = Path(
    "data/experiments/delta_v2/knowledge_adaptation_v1/train.jsonl"
)
DEV_PATH = Path(
    "data/experiments/delta_v2/knowledge_adaptation_v1/dev.jsonl"
)
TARGET_MODULES_REGEX = (
    r"^model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"linear_attn\.(?:in_proj_a|in_proj_b|in_proj_qkv|in_proj_z|out_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)


class KnowledgeTrainError(KnowledgeEvalError):
    """The frozen LoRA candidate cannot be trained safely."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KnowledgeTrainError(f"{field} must be a mapping")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise KnowledgeTrainError(f"{field} must be a non-empty string")
    return value


def _relative_path(value: Any, field: str) -> Path:
    path = Path(_string(value, field))
    if path.is_absolute() or ".." in path.parts:
        raise KnowledgeTrainError(f"{field} must stay within the repository")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise KnowledgeTrainError(f"cannot read YAML: {path}") from exc
    return _mapping(payload, str(path))


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise KnowledgeTrainError(f"cannot read JSONL: {path}") from exc
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise KnowledgeTrainError(
                f"invalid JSON at {path}:{line_number}"
            ) from exc
        rows.append(_mapping(payload, f"{path}:{line_number}"))
    return rows


def validate_protocol(
    protocol: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("experiment_id") != "delta_v2"
        or protocol.get("condition_id") != "c5_knowledge_lora"
        or protocol.get("state") != "preregistered-for-training"
    ):
        raise KnowledgeTrainError("knowledge LoRA protocol identity changed")

    amends = _mapping(protocol.get("amends"), "amends")
    if dict(amends) != {
        "protocol_id": "delta-v2-knowledge-lora-v1",
        "run_id": "delta-v2-c5-knowledge-lora-qwen35-08b-modal-v1",
        "git_commit": "ec7987b1067b7e1ce0eccdc398bb9e73185abc64",
        "status": "failed-before-model-load",
        "failure_stage": "modal-training-module-routing",
        "failure_type": "missing-generic-modal-training-router",
        "model_loaded": False,
        "optimizer_steps": 0,
        "adapter_created": False,
        "estimated_cost_usd": 0.0318,
        "evidence_prefix": (
            "runs/delta-v2-c5-knowledge-lora-qwen35-08b-modal-v1/"
        ),
    }:
        raise KnowledgeTrainError("knowledge LoRA amendment changed")

    trigger = _mapping(protocol.get("decision_trigger"), "decision_trigger")
    if (
        trigger.get("base_run_id")
        != "delta-v2-c4-knowledge-base-qwen35-08b-modal-v1"
        or trigger.get("config_sha256")
        != "87f029a255a25cdfc0272667b5d75d94639aefbea892d9271029c165c8e3bdf0"
        or trigger.get("metrics_sha256")
        != "6b617bf893a2f9c49e16ba5bd17f16abe195b7b69a35c7e8e94681d81de6a0fb"
        or trigger.get("samples_sha256")
        != "50ab84e9800285bf8a3a257338ebd32485cfdec307ddabc4057a12f189dc8315"
        or trigger.get("diagnosis")
        != "false-reject-and-response-format-failure"
    ):
        raise KnowledgeTrainError("knowledge LoRA decision trigger changed")

    dataset = _mapping(protocol.get("dataset"), "dataset")
    freeze_path = repo_root / _relative_path(
        dataset.get("freeze_path"),
        "dataset.freeze_path",
    )
    train_path = repo_root / _relative_path(
        dataset.get("train_path"),
        "dataset.train_path",
    )
    dev_path = repo_root / _relative_path(
        dataset.get("dev_path"),
        "dataset.dev_path",
    )
    if (
        dataset.get("dataset_id") != DATASET_ID
        or freeze_path != repo_root / FREEZE_PATH
        or train_path != repo_root / TRAIN_PATH
        or dev_path != repo_root / DEV_PATH
        or not freeze_path.is_file()
        or not train_path.is_file()
        or not dev_path.is_file()
        or _sha256(freeze_path) != dataset.get("freeze_sha256")
        or _sha256(train_path) != dataset.get("train_sha256")
        or _sha256(dev_path) != dataset.get("dev_sha256")
        or dataset.get("train_rows") != 63
        or dataset.get("dev_rows") != 21
        or dataset.get("post_train_eval_protocol")
        != "delta-v2-knowledge-eval-v1"
    ):
        raise KnowledgeTrainError("knowledge LoRA dataset binding changed")

    model = _mapping(protocol.get("model"), "model")
    if dict(model) != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "base_weights": "frozen",
        "quantization": "none",
        "full_weight_training": "forbidden",
    }:
        raise KnowledgeTrainError("knowledge LoRA model boundary changed")

    lora = _mapping(protocol.get("lora"), "lora")
    if (
        lora.get("implementation") != "peft-0.19.0"
        or lora.get("rank") != 8
        or lora.get("alpha") != 16
        or lora.get("dropout") != 0.05
        or lora.get("bias") != "none"
        or lora.get("target_modules_regex") != TARGET_MODULES_REGEX
        or lora.get("vision_modules_trainable") is not False
    ):
        raise KnowledgeTrainError("knowledge LoRA topology changed")

    optimization = _mapping(
        protocol.get("optimization"),
        "optimization",
    )
    if dict(optimization) != {
        "optimizer": "adamw",
        "max_steps": 60,
        "learning_rate": 0.0001,
        "warmup_steps": 6,
        "weight_decay": 0.0,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_sequence_length": 768,
        "max_grad_norm": 1.0,
        "dtype": "bfloat16",
        "seed": 20260730,
        "assistant_only_loss": True,
        "deterministic_algorithms": "required",
    }:
        raise KnowledgeTrainError("knowledge optimization changed")

    gates = _mapping(
        protocol.get("verification_gates"),
        "verification_gates",
    )
    if (
        gates.get("pooled_overall_score") != "forbidden"
        or gates.get("result") != "candidate-only-no-promotion"
        or gates.get("recall") != "advisory"
        or gates.get("feature") != "report-separately"
    ):
        raise KnowledgeTrainError("knowledge verification gate changed")
    boundary = _mapping(
        protocol.get("execution_boundary"),
        "execution_boundary",
    )
    if dict(boundary) != {
        "modal_training_authorized": True,
        "b2_adapter_write_authorized": True,
        "adapter_eval_required": True,
        "promotion_authorized": False,
    }:
        raise KnowledgeTrainError("knowledge training boundary changed")

    train_rows = _load_jsonl(train_path)
    dev_rows = _load_jsonl(dev_path)
    validate_rows(train_rows, expected_rows=63, expected_surfaces={
        "exact_recall": 21,
        "verify_true": 21,
        "verify_false": 21,
    })
    validate_rows(
        dev_rows,
        expected_rows=21,
        expected_surfaces={"dev_recall": 21},
    )
    return train_rows, dev_rows


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_rows: int,
    expected_surfaces: Mapping[str, int],
) -> None:
    if len(rows) != expected_rows:
        raise KnowledgeTrainError("knowledge SFT row count changed")
    ids: list[str] = []
    surfaces: dict[str, int] = {}
    for row in rows:
        if row.get("schema") != TRAIN_SCHEMA:
            raise KnowledgeTrainError("unexpected knowledge SFT schema")
        row_id = _string(row.get("row_id"), "row_id")
        if row_id in ids:
            raise KnowledgeTrainError("duplicate knowledge SFT row")
        ids.append(row_id)
        _string(row.get("source_id"), "source_id")
        surface = _string(row.get("surface"), "surface")
        surfaces[surface] = surfaces.get(surface, 0) + 1
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or messages[0].get("role") != "user"
            or messages[1].get("role") != "assistant"
            or not isinstance(messages[0].get("content"), str)
            or not isinstance(messages[1].get("content"), str)
        ):
            raise KnowledgeTrainError("knowledge SFT messages changed")
        if set(row) - {
            "schema",
            "row_id",
            "source_id",
            "surface",
            "truth_authority",
            "messages",
        }:
            raise KnowledgeTrainError("unexpected knowledge SFT field")
    if ids != sorted(ids) or surfaces != dict(expected_surfaces):
        raise KnowledgeTrainError("knowledge SFT ordering or surfaces changed")


def validate_run_config(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[
    Mapping[str, Any],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
]:
    if (
        config.get("schema") != RUN_CONFIG_SCHEMA
        or config.get("run_id") != TRAIN_RUN_ID
        or config.get("experiment_id") != "delta_v2"
        or config.get("condition_id") != "c5_knowledge_lora"
        or config.get("eval_environment_id") != DATASET_ID
        or config.get("task") != "train"
    ):
        raise KnowledgeTrainError("knowledge training run identity changed")

    protocol_binding = _mapping(config.get("protocol"), "protocol")
    protocol_path = repo_root / _relative_path(
        protocol_binding.get("path"),
        "protocol.path",
    )
    if (
        protocol_binding.get("protocol_id") != PROTOCOL_ID
        or protocol_path != repo_root / PROTOCOL_PATH
        or not protocol_path.is_file()
        or _sha256(protocol_path) != protocol_binding.get("sha256")
    ):
        raise KnowledgeTrainError("knowledge training protocol changed")
    protocol = _load_yaml(protocol_path)
    train_rows, dev_rows = validate_protocol(
        protocol,
        repo_root=repo_root,
    )

    model = _mapping(config.get("model"), "model")
    if dict(model) != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "adapter": "new-lora",
        "quantization": "none",
    }:
        raise KnowledgeTrainError("knowledge training model changed")

    runtime = _mapping(config.get("runtime"), "runtime")
    if (
        runtime.get("profile") != "delta-v2-knowledge-modal-v1"
        or runtime.get("python") != "3.11.9"
        or runtime.get("packages") != RUNTIME_PACKAGES
        or runtime.get("device") != "cuda"
        or runtime.get("dtype") != "bfloat16"
        or runtime.get("deterministic_algorithms") is not True
    ):
        raise KnowledgeTrainError("knowledge training runtime changed")

    data = _mapping(config.get("data"), "data")
    if dict(data) != {
        "train_path": str(TRAIN_PATH),
        "train_sha256": protocol["dataset"]["train_sha256"],
        "eval_path": str(DEV_PATH),
        "eval_sha256": protocol["dataset"]["dev_sha256"],
    }:
        raise KnowledgeTrainError("knowledge training data config changed")
    if _mapping(config.get("lora"), "lora") != protocol["lora"]:
        raise KnowledgeTrainError("knowledge training LoRA config changed")
    training = _mapping(config.get("training"), "training")
    expected_training = {
        "module": "experiments.delta_v2.train_knowledge_lora",
        **dict(protocol["optimization"]),
    }
    if dict(training) != expected_training:
        raise KnowledgeTrainError("knowledge training optimization changed")
    output = _mapping(config.get("output"), "output")
    if _relative_path(output.get("dir"), "output.dir") != (
        Path("runs") / TRAIN_RUN_ID
    ):
        raise KnowledgeTrainError("knowledge training output changed")
    modal = _mapping(config.get("modal"), "modal")
    if dict(modal) != {
        "app_path": MODAL_APP_PATH,
        "gpu": "H100",
        "timeout_seconds": 7200,
    }:
        raise KnowledgeTrainError("knowledge training Modal binding changed")
    return protocol, train_rows, dev_rows


def assistant_only_labels(
    prompt_ids: Sequence[int],
    full_ids: Sequence[int],
    *,
    max_sequence_length: int,
) -> list[int]:
    if (
        not prompt_ids
        or len(full_ids) <= len(prompt_ids)
        or list(full_ids[: len(prompt_ids)]) != list(prompt_ids)
    ):
        raise KnowledgeTrainError(
            "chat template does not expose an assistant-only prefix"
        )
    if len(full_ids) > max_sequence_length:
        raise KnowledgeTrainError(
            f"knowledge row exceeds max sequence length: {len(full_ids)}"
        )
    labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids) :])
    if all(token == -100 for token in labels):
        raise KnowledgeTrainError("knowledge row has no assistant tokens")
    return labels


def target_module_allowed(module_name: str) -> bool:
    return re.fullmatch(TARGET_MODULES_REGEX, module_name) is not None


def _tokenize_rows(
    processor: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    max_sequence_length: int,
) -> list[dict[str, Any]]:
    tokenized: list[dict[str, Any]] = []
    for row in rows:
        messages = row["messages"]
        prompt = processor.apply_chat_template(
            messages[:1],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        full = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        prompt_ids = prompt["input_ids"][0].tolist()
        full_ids = full["input_ids"][0].tolist()
        labels = assistant_only_labels(
            prompt_ids,
            full_ids,
            max_sequence_length=max_sequence_length,
        )
        tokenized.append(
            {
                "row_id": row["row_id"],
                "input_ids": full["input_ids"],
                "attention_mask": full["attention_mask"],
                "labels": labels,
                "tokens": len(full_ids),
                "assistant_tokens": sum(token != -100 for token in labels),
            }
        )
    return tokenized


def _seed_everything(seed: int, torch: Any) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _model_inputs(example: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    return {
        "input_ids": example["input_ids"].to("cuda"),
        "attention_mask": example["attention_mask"].to("cuda"),
        "labels": torch.tensor(
            [example["labels"]],
            dtype=torch.long,
            device="cuda",
        ),
    }


def _mean_loss(
    model: Any,
    examples: Sequence[Mapping[str, Any]],
    torch: Any,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for example in examples:
            output = model(**_model_inputs(example, torch))
            losses.append(float(output.loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def train(
    *,
    config_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    config = _load_yaml(config_path)
    protocol, train_rows, dev_rows = validate_run_config(
        config,
        repo_root=repo_root,
    )
    observed_runtime = validate_runtime(config)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForMultimodalLM,
        AutoProcessor,
        get_linear_schedule_with_warmup,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise KnowledgeTrainError("knowledge LoRA requires one CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise KnowledgeTrainError("knowledge LoRA requires bfloat16")
    optimization = protocol["optimization"]
    seed = int(optimization["seed"])
    _seed_everything(seed, torch)

    processor = AutoProcessor.from_pretrained(
        MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )
    max_sequence_length = int(optimization["max_sequence_length"])
    train_examples = _tokenize_rows(
        processor,
        train_rows,
        max_sequence_length=max_sequence_length,
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
        raise KnowledgeTrainError("LoRA created no trainable parameters")
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and "lora_" not in name
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
        raise KnowledgeTrainError("LoRA target routing entered vision modules")

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
    losses: list[dict[str, Any]] = []
    order = list(range(len(train_examples)))
    rng = random.Random(seed)
    rng.shuffle(order)
    cursor = 0
    examples_seen = 0
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        for _micro_step in range(gradient_accumulation):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            example = train_examples[order[cursor]]
            cursor += 1
            output = model(**_model_inputs(example, torch))
            loss = output.loss / gradient_accumulation
            loss.backward()
            micro_losses.append(float(output.loss.detach().cpu()))
            examples_seen += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            float(optimization["max_grad_norm"]),
        )
        optimizer.step()
        scheduler.step()
        losses.append(
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
        raise KnowledgeTrainError("LoRA adapter was not saved")

    trainable_parameters = sum(
        parameter.numel() for parameter in trainable
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    metrics = {
        "schema": METRICS_SCHEMA,
        "run_id": TRAIN_RUN_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "pass",
        "steps": max_steps,
        "gradient_accumulation_steps": gradient_accumulation,
        "examples_seen": examples_seen,
        "initial_dev_loss": initial_dev_loss,
        "final_dev_loss": final_dev_loss,
        "loss_reduction": initial_dev_loss - final_dev_loss,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "matched_language_modules": len(matched_base_modules),
        "vision_modules_trainable": False,
        "max_train_tokens": max(row["tokens"] for row in train_examples),
        "max_dev_tokens": max(row["tokens"] for row in dev_examples),
        "loss_history": losses,
    }
    metrics_path = output_dir / "train_metrics.json"
    _write_json(metrics_path, metrics)
    config_copy = output_dir / "config.yaml"
    config_copy.write_bytes(config_path.read_bytes())
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "run_id": TRAIN_RUN_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "base_model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
        },
        "config_sha256": _sha256(config_path),
        "protocol_sha256": _sha256(repo_root / PROTOCOL_PATH),
        "train_sha256": _sha256(repo_root / TRAIN_PATH),
        "dev_sha256": _sha256(repo_root / DEV_PATH),
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


def validate_only(
    *,
    config_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    protocol, train_rows, dev_rows = validate_run_config(
        config,
        repo_root=repo_root,
    )
    return {
        "schema": RUN_CONFIG_SCHEMA,
        "run_id": TRAIN_RUN_ID,
        "protocol_id": PROTOCOL_ID,
        "config_sha256": _sha256(config_path),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "max_steps": protocol["optimization"]["max_steps"],
        "lora_rank": protocol["lora"]["rank"],
        "target_modules_regex": protocol["lora"]["target_modules_regex"],
        "model_invocations_completed": 0,
        "status": "valid-unexecuted",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the frozen delta_v2 knowledge LoRA candidate."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    result = (
        validate_only(config_path=config_path, repo_root=repo_root)
        if args.validate_only
        else train(config_path=config_path, repo_root=repo_root)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
