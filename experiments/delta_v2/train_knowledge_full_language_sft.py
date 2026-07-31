from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2.run_knowledge_eval import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODAL_APP_PATH,
    RUNTIME_PACKAGES,
    validate_runtime,
)
from experiments.delta_v2.train_knowledge_lora import (
    KnowledgeTrainError,
    _load_yaml,
    _mean_loss,
    _model_inputs,
    _relative_path,
    _seed_everything,
    _sha256,
    _tokenize_rows,
)
from experiments.delta_v2.train_knowledge_stability_choice_format_v2 import (
    DATA_BINDINGS,
    DATASET_ID,
    _engine_rows,
)
from experiments.delta_v2.train_knowledge_stability_claim_covered import (
    _load_jsonl,
)
from experiments.delta_v2.train_knowledge_stability_replay import (
    ACQUISITION_PAIRS_PATH,
    ACQUISITION_TRAIN_PATH,
    DEV_PATH,
    _normalize_replay_rows,
)
from experiments.delta_v2.train_knowledge_stability_replay50 import (
    build_replay50_schedule,
)


CONFIG_SCHEMA = "delta.knowledge_full_language_sft_config.v1"
PROTOCOL_SCHEMA = "delta.knowledge_full_language_sft_protocol.v1"
PROTOCOL_ID = "delta-v2-knowledge-full-language-sft-v1"
CONDITION_ID = "f1_full_language_plain_sft"
RUN_ID = "delta-v2-f1-full-language-sft-qwen35-08b-modal-v1"
PROTOCOL_PATH = Path(
    "experiments/delta_v2/knowledge_full_language_sft_protocol.yaml"
)
STABILITY_TRAIN_PATH = DATA_BINDINGS["stability_train"][0]
STABILITY_PAIRS_PATH = DATA_BINDINGS["stability_pairs"][0]
STABILITY_EVAL_PATH = DATA_BINDINGS["stability_eval"][0]
OPTIMIZATION = {
    "optimizer": "adamw",
    "max_steps": 60,
    "learning_rate": 0.00001,
    "warmup_steps": 6,
    "weight_decay": 0.0,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_sequence_length": 768,
    "max_grad_norm": 1.0,
    "dtype": "bfloat16",
    "seed": 20260730,
    "deterministic_algorithms": "required",
}
EVIDENCE = {
    (
        "delta-v2-d20-choice-token-diagnostic-d13-modal-v1",
        "run_receipt.json",
    ): "42425da6bb534b510824d7460d6ed6e94551b4e4be4acd0ce688f8c41669c5dd",
    (
        "delta-v2-d20-choice-token-diagnostic-d13-modal-v1",
        "metrics.json",
    ): "3171dfddba9563574cbcd7131f8c12cda40e38c247ede6e1e30c900bd8f2e9cb",
    (
        "delta-v2-d20-choice-token-diagnostic-d18-modal-v1",
        "run_receipt.json",
    ): "15fb095a8f3ab49278edb27234b92691535faeb76d552fcfc822736d0d6b27ce",
    (
        "delta-v2-d20-choice-token-diagnostic-d18-modal-v1",
        "metrics.json",
    ): "e0ad3c9f6ba3be6310a7d3ddc3acaa0b0068078d8abed6b6861306fe235da587",
    (
        "delta-v2-d20-choice-token-diagnostic-d19-modal-v1",
        "run_receipt.json",
    ): "cff4b3097116103f6bd058a6588982c71183d3ec90004f4c99d628954f84ff97",
    (
        "delta-v2-d20-choice-token-diagnostic-d19-modal-v1",
        "metrics.json",
    ): "d1772afb6435d1d4c98b4bbba260090b3361f9324ed19c6f5e1fc74ebdbba112",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _schedule_sha256(schedule: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            list(schedule), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _full_language_scope(name: str) -> bool:
    return (
        name.startswith("model.language_model.")
        or name.startswith("lm_head.")
    )


def validate_config(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[
    Mapping[str, Any],
    tuple[list[Mapping[str, Any]], ...],
    list[Mapping[str, Any]],
]:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("run_id") != RUN_ID
        or config.get("experiment_id") != "delta_v2"
        or config.get("condition_id") != CONDITION_ID
        or config.get("task") != "train"
    ):
        raise KnowledgeTrainError("full-language config identity changed")
    binding = config["protocol"]
    protocol_path = repo_root / _relative_path(
        binding["path"], "protocol.path"
    )
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or protocol_path != repo_root / PROTOCOL_PATH
        or _sha256(protocol_path) != binding.get("sha256")
    ):
        raise KnowledgeTrainError("full-language protocol binding changed")
    protocol = _load_yaml(protocol_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("condition_id") != CONDITION_ID
        or protocol.get("state") != "preregistered-for-training"
        or protocol.get("decision_trigger", {}).get("decision")
        != "establish-full-weight-supervised-feasibility-before-adapters"
    ):
        raise KnowledgeTrainError("full-language protocol changed")
    for (run_id, name), digest in EVIDENCE.items():
        if _sha256(repo_root / "runs" / run_id / name) != digest:
            raise KnowledgeTrainError("full-language evidence changed")
    loaded = []
    for name, (path, digest, rows) in DATA_BINDINGS.items():
        block = protocol["data"][name]
        if (
            block.get("path") != str(path)
            or block.get("sha256") != digest
            or _sha256(repo_root / path) != digest
        ):
            raise KnowledgeTrainError(f"full-language {name} changed")
        values = _load_jsonl(repo_root / path)
        if len(values) != rows:
            raise KnowledgeTrainError(f"full-language {name} rows changed")
        loaded.append(values)
    if config.get("model") != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "initialization": "fresh-base",
        "update_scope": "all-language-weights",
        "vision_modules_trainable": False,
        "quantization": "none",
    }:
        raise KnowledgeTrainError("full-language model scope changed")
    if config.get("objective") != {
        "objective_id": "plain-assistant-token-sft-v1",
        "recall_loss": "assistant-token-cross-entropy",
        "boolean_pair_loss":
            "mean-positive-and-negative-assistant-token-cross-entropy",
        "ranking_loss": "none",
        "preference_loss": "none",
        "format_negative_loss": "none",
    } or protocol.get("objective") != config.get("objective"):
        raise KnowledgeTrainError("full-language objective changed")
    runtime = config["runtime"]
    if (
        runtime.get("packages") != RUNTIME_PACKAGES
        or runtime.get("device") != "cuda"
        or runtime.get("dtype") != "bfloat16"
        or runtime.get("deterministic_algorithms") is not True
    ):
        raise KnowledgeTrainError("full-language runtime changed")
    if config.get("training") != {
        "module":
            "experiments.delta_v2.train_knowledge_full_language_sft",
        **OPTIMIZATION,
    } or protocol.get("optimization") != OPTIMIZATION:
        raise KnowledgeTrainError("full-language optimization changed")
    if (
        _relative_path(config["output"]["dir"], "output.dir")
        != Path("runs") / RUN_ID
    ):
        raise KnowledgeTrainError("full-language output changed")
    if config.get("modal") != {
        "app_path": MODAL_APP_PATH,
        "gpu": "H100",
        "timeout_seconds": 7200,
    }:
        raise KnowledgeTrainError("full-language Modal changed")
    acq_rows, acq_pairs, _dev, stable_rows, stable_pairs, _eval = loaded
    schedule_rows = _engine_rows(
        [
            row for row in stable_rows
            if row["surface"] != "claim_covered_recall"
        ]
    )
    schedule = build_replay50_schedule(
        acq_rows, acq_pairs, schedule_rows, stable_pairs
    )
    if (
        len(schedule) != 240
        or _schedule_sha256(schedule)
        != "7bfb23a53e2c1f58c9cb9aa5f5a18dbeea1ea66ef1ef84204e43f968a4da6427"
    ):
        raise KnowledgeTrainError("full-language schedule changed")
    return protocol, tuple(loaded), schedule


def train(config_path: Path, repo_root: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    protocol, data, schedule = validate_config(config, repo_root=repo_root)
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    observed_runtime = validate_runtime(config)

    import torch
    from transformers import (
        AutoModelForMultimodalLM,
        AutoProcessor,
        get_linear_schedule_with_warmup,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise KnowledgeTrainError("full-language SFT requires one CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise KnowledgeTrainError("full-language SFT requires bfloat16")
    _seed_everything(OPTIMIZATION["seed"], torch)

    acq_rows, _acq_pairs, dev_rows, stable_rows, _pairs, _eval = data
    transformed = _engine_rows(stable_rows)
    combined = [*acq_rows, *_normalize_replay_rows(transformed)]
    rows_by_id = {str(row["row_id"]): row for row in combined}
    processor = AutoProcessor.from_pretrained(
        MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )
    gold_examples = {
        row_id: _tokenize_rows(
            processor,
            [row],
            max_sequence_length=OPTIMIZATION["max_sequence_length"],
        )[0]
        for row_id, row in rows_by_id.items()
    }
    dev_examples = _tokenize_rows(
        processor,
        dev_rows,
        max_sequence_length=OPTIMIZATION["max_sequence_length"],
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
    )
    model.config.use_cache = False
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(_full_language_scope(name))
    trainable_named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    frozen_named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]
    trainable_parameters = sum(
        parameter.numel() for _, parameter in trainable_named
    )
    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    if (
        not trainable_named
        or not frozen_named
        or trainable_parameters / total_parameters < 0.70
        or any(
            not _full_language_scope(name)
            for name, _ in trainable_named
        )
        or any("visual" in name and parameter.requires_grad
               for name, parameter in model.named_parameters())
    ):
        raise KnowledgeTrainError("full-language trainable scope changed")
    trainable = [parameter for _, parameter in trainable_named]
    model = model.to("cuda")
    model.train()
    optimizer = torch.optim.AdamW(
        trainable,
        lr=OPTIMIZATION["learning_rate"],
        weight_decay=OPTIMIZATION["weight_decay"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=OPTIMIZATION["warmup_steps"],
        num_training_steps=OPTIMIZATION["max_steps"],
    )
    initial_dev_loss = _mean_loss(model, dev_examples, torch)
    cursor = 0
    unit_counts: Counter[str] = Counter()
    stream_counts: Counter[str] = Counter()
    loss_history = []
    model_forwards = 0
    accumulation = OPTIMIZATION["gradient_accumulation_steps"]
    for step in range(1, OPTIMIZATION["max_steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        step_streams = []
        for _ in range(accumulation):
            unit = schedule[cursor]
            cursor += 1
            kind = str(unit["kind"])
            stream = str(unit["source_stream"])
            unit_counts[kind] += 1
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
                positive = model(
                    **_model_inputs(
                        gold_examples[str(unit["positive_row_id"])], torch
                    )
                )
                negative = model(
                    **_model_inputs(
                        gold_examples[str(unit["negative_row_id"])], torch
                    )
                )
                unit_loss = (positive.loss + negative.loss) / 2
                model_forwards += 2
            else:
                raise KnowledgeTrainError("unknown full-language unit")
            if not torch.isfinite(unit_loss):
                raise KnowledgeTrainError("non-finite full-language loss")
            (unit_loss / accumulation).backward()
            micro_losses.append(float(unit_loss.detach().cpu()))
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, OPTIMIZATION["max_grad_norm"]
        )
        if not math.isfinite(float(grad_norm.detach().cpu())):
            raise KnowledgeTrainError("non-finite full-language gradient")
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
    checkpoint_dir = output_dir / "model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(
        checkpoint_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    processor.save_pretrained(checkpoint_dir)
    checkpoint_path = checkpoint_dir / "model.safetensors"
    if not checkpoint_path.is_file():
        raise KnowledgeTrainError("full-language checkpoint layout changed")
    schedule_hash = _schedule_sha256(schedule)
    metrics = {
        "schema": "delta.knowledge_full_language_sft_metrics.v1",
        "run_id": RUN_ID,
        "condition_id": CONDITION_ID,
        "protocol_id": PROTOCOL_ID,
        "objective_id": config["objective"]["objective_id"],
        "status": "pass",
        "steps": OPTIMIZATION["max_steps"],
        "training_units_seen": cursor,
        "unit_counts": dict(sorted(unit_counts.items())),
        "source_stream_counts": dict(sorted(stream_counts.items())),
        "model_forwards": model_forwards,
        "initial_dev_loss": initial_dev_loss,
        "final_dev_loss": final_dev_loss,
        "loss_reduction": initial_dev_loss - final_dev_loss,
        "schedule_sha256": schedule_hash,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "trainable_parameter_prefixes": [
            "model.language_model.",
            "lm_head.",
        ],
        "frozen_parameter_count": sum(
            parameter.numel() for _, parameter in frozen_named
        ),
        "vision_modules_trainable": False,
        "loss_history": loss_history,
    }
    metrics_path = output_dir / "train_metrics.json"
    _write_json(metrics_path, metrics)
    (output_dir / "config.yaml").write_bytes(config_path.read_bytes())
    receipt = {
        "schema": "delta.knowledge_full_language_sft_receipt.v1",
        "run_id": RUN_ID,
        "condition_id": CONDITION_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "objective_id": config["objective"]["objective_id"],
        "base_model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
        },
        "rollback": {
            "kind": "retain-pinned-base",
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
        },
        "config_sha256": _sha256(config_path),
        "protocol_sha256": _sha256(repo_root / PROTOCOL_PATH),
        "acquisition_train_sha256": _sha256(
            repo_root / ACQUISITION_TRAIN_PATH
        ),
        "acquisition_pairs_sha256": _sha256(
            repo_root / ACQUISITION_PAIRS_PATH
        ),
        "stability_train_sha256": _sha256(
            repo_root / STABILITY_TRAIN_PATH
        ),
        "stability_pairs_sha256": _sha256(
            repo_root / STABILITY_PAIRS_PATH
        ),
        "stability_eval_sha256": _sha256(
            repo_root / STABILITY_EVAL_PATH
        ),
        "dev_sha256": _sha256(repo_root / DEV_PATH),
        "schedule_sha256": schedule_hash,
        "schedule_units": len(schedule),
        "checkpoint_sha256": _sha256(checkpoint_path),
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    path = args.config if args.config.is_absolute() else root / args.config
    if args.validate_only:
        config = _load_yaml(path)
        _protocol, _data, schedule = validate_config(
            config, repo_root=root
        )
        result = {
            "run_id": RUN_ID,
            "schedule_units": len(schedule),
            "schedule_sha256": _schedule_sha256(schedule),
            "model_invocations_completed": 0,
            "optimizer_steps_completed": 0,
            "status": "valid-unexecuted",
        }
    else:
        result = train(path, root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
