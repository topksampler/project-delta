from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2.run_knowledge_eval import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODAL_APP_PATH,
    RUNTIME_PACKAGES,
)
from experiments.delta_v2.run_knowledge_train_surface_diagnostic import (
    TRAIN_PATH,
    DiagnosticError,
    _load_jsonl,
    _load_yaml,
    _mapping,
    _relative,
    _sha256,
    execute_validated_diagnostic,
)
from experiments.delta_v2.train_knowledge_lora import validate_rows


RUN_ID = "delta-v2-c7-knowledge-lora-2to1-train-surface-modal-v1"
TRAIN_RUN_ID = "delta-v2-c7-knowledge-lora-2to1-qwen35-08b-modal-v1"
PROTOCOL_ID = "delta-v2-knowledge-c7-2to1-train-surface-v1"
PROTOCOL_PATH = Path(
    "experiments/delta_v2/"
    "knowledge_lora_c7_train_surface_diagnostic_protocol.yaml"
)
ADAPTER_SHA256 = (
    "89f79fe8763e3184ef3d701ac42afa657769f7bf0d5a09df920b9d32d3544549"
)
TRAINING_METRICS_SHA256 = (
    "033bf5024879b421cb1a929aa7320f9f3d6f18603c044dded888f91a88ac646f"
)


def validate_config(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    if (
        config.get("schema")
        != "delta.knowledge_train_surface_diagnostic_config.v1"
        or config.get("run_id") != RUN_ID
        or config.get("experiment_id") != "delta_v2"
        or config.get("condition_id")
        != "c7_knowledge_lora_2to1_diagnostic"
        or config.get("task") != "eval"
    ):
        raise DiagnosticError("c7 diagnostic identity changed")
    binding = _mapping(config.get("protocol"), "protocol")
    protocol_path = repo_root / _relative(
        binding.get("path"), "protocol.path"
    )
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or protocol_path != repo_root / PROTOCOL_PATH
        or _sha256(protocol_path) != binding.get("sha256")
    ):
        raise DiagnosticError("c7 diagnostic protocol binding changed")
    protocol = _load_yaml(protocol_path)
    if (
        protocol.get("schema")
        != "delta.knowledge_train_surface_diagnostic_protocol.v1"
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("experiment_id") != "delta_v2"
        or protocol.get("condition_id")
        != "c7_knowledge_lora_2to1_diagnostic"
        or protocol.get("state") != "preregistered-for-diagnostic"
    ):
        raise DiagnosticError("c7 diagnostic protocol identity changed")
    if dict(_mapping(protocol.get("trigger"), "trigger")) != {
        "training_run_id": TRAIN_RUN_ID,
        "training_metrics_sha256": TRAINING_METRICS_SHA256,
        "adapter_sha256": ADAPTER_SHA256,
    }:
        raise DiagnosticError("c7 diagnostic trigger changed")
    if dict(_mapping(protocol.get("dataset"), "dataset")) != {
        "dataset_id": "delta-v2-vllm-knowledge-adaptation-v1",
        "path": str(TRAIN_PATH),
        "sha256": (
            "16ad0d2c12a823722501800227c739e59131fd7c7e0f774e2f1faab2e4bd4b95"
        ),
        "rows": 63,
        "prompt_source": "messages[0]",
        "gold_source": "messages[1]",
        "mutation": "forbidden",
    }:
        raise DiagnosticError("c7 diagnostic dataset contract changed")
    if _sha256(repo_root / TRAIN_PATH) != protocol["dataset"]["sha256"]:
        raise DiagnosticError("c7 diagnostic dataset bytes changed")
    rows = _load_jsonl(repo_root / TRAIN_PATH)
    validate_rows(
        rows,
        expected_rows=63,
        expected_surfaces={
            "exact_recall": 21,
            "verify_true": 21,
            "verify_false": 21,
        },
    )
    if dict(_mapping(protocol.get("generation"), "generation")) != {
        "method": "greedy",
        "do_sample": False,
        "max_new_tokens": 256,
        "repeats": 2,
        "seed": 20260730,
        "repeat_invariant": "byte-identical-raw-continuation",
    }:
        raise DiagnosticError("c7 diagnostic generation changed")
    if dict(_mapping(protocol.get("scoring"), "scoring")) != {
        "authority": "frozen-training-row-assistant-answer",
        "method": "exact-stripped-string",
        "repair": "forbidden",
        "pooled_overall_score": "forbidden",
    }:
        raise DiagnosticError("c7 diagnostic scoring changed")
    if dict(_mapping(protocol.get("decision"), "decision")) != {
        "boolean_surface_fit": {
            "verify_true_accuracy_minimum": 0.8,
            "verify_false_accuracy_minimum": 0.8,
        },
        "full_frozen_eval_if_passes": True,
        "diagnostic_only": True,
        "promotion_authorized": False,
    }:
        raise DiagnosticError("c7 diagnostic decision changed")

    model = _mapping(config.get("model"), "model")
    adapter = _mapping(model.get("adapter"), "model.adapter")
    adapter_path = _relative(adapter.get("path"), "model.adapter.path")
    if (
        model.get("repository") != MODEL_REPOSITORY
        or model.get("revision") != MODEL_REVISION
        or model.get("quantization") != "none"
        or adapter.get("kind") != "lora"
        or adapter.get("run_id") != TRAIN_RUN_ID
        or adapter.get("sha256") != ADAPTER_SHA256
        or adapter_path != Path("runs") / TRAIN_RUN_ID / "adapter"
    ):
        raise DiagnosticError("c7 diagnostic model binding changed")
    adapter_file = repo_root / adapter_path / "adapter_model.safetensors"
    if (
        not adapter_file.is_file()
        or _sha256(adapter_file) != ADAPTER_SHA256
    ):
        raise DiagnosticError("c7 diagnostic adapter bytes changed")
    runtime = _mapping(config.get("runtime"), "runtime")
    if (
        runtime.get("profile") != "delta-v2-knowledge-modal-v1"
        or runtime.get("python") != "3.11.9"
        or runtime.get("packages") != RUNTIME_PACKAGES
        or runtime.get("device") != "cuda"
        or runtime.get("dtype") != "bfloat16"
        or runtime.get("attention_implementation") != "eager"
        or runtime.get("deterministic_algorithms") is not True
    ):
        raise DiagnosticError("c7 diagnostic runtime changed")
    if dict(_mapping(config.get("data"), "data")) != {
        "train_path": str(TRAIN_PATH)
    }:
        raise DiagnosticError("c7 diagnostic data config changed")
    if dict(_mapping(config.get("eval"), "eval")) != {
        "module": (
            "experiments.delta_v2."
            "run_knowledge_lora_c7_train_surface_diagnostic"
        )
    }:
        raise DiagnosticError("c7 diagnostic module changed")
    if _relative(config["output"]["dir"], "output.dir") != (
        Path("runs") / RUN_ID
    ):
        raise DiagnosticError("c7 diagnostic output changed")
    if dict(_mapping(config.get("modal"), "modal")) != {
        "app_path": MODAL_APP_PATH,
        "gpu": "A10G",
        "timeout_seconds": 3600,
    }:
        raise DiagnosticError("c7 diagnostic Modal binding changed")
    return protocol, rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    config = _load_yaml(config_path)
    protocol, rows = validate_config(config, repo_root=repo_root)
    result = (
        {
            "run_id": RUN_ID,
            "rows": len(rows),
            "requests": len(rows) * 2,
            "model_invocations_completed": 0,
            "status": "valid-unexecuted",
        }
        if args.validate_only
        else execute_validated_diagnostic(
            config_path=config_path,
            repo_root=repo_root,
            config=config,
            protocol=protocol,
            rows=rows,
            run_id=RUN_ID,
            protocol_id=PROTOCOL_ID,
            protocol_path=PROTOCOL_PATH,
            train_path=TRAIN_PATH,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
