from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2 import run_knowledge_stability_replay_eval as shared
from experiments.delta_v2.run_knowledge_eval import (
    MODAL_APP_PATH,
    OUTPUT_SCHEMA,
    RUNTIME_PACKAGES,
    TransformersKnowledgeBackend,
    _load_yaml,
    _mapping,
    _relative_path,
    _sha256,
    serialize_jsonl,
    validate_runtime,
)


CONFIG_SCHEMA = "delta.knowledge_full_language_stability_eval_config.v1"
PROTOCOL_SCHEMA = "delta.knowledge_stability_replay_eval_protocol.v1"
PROTOCOL_ID = "delta-v2-knowledge-full-language-stability-eval-v1"
CONDITION_ID = "f1_full_language_stability_eval"
RUN_ID = "delta-v2-f1-full-language-stability-eval-modal-v1"
TRAINING_RUN_ID = "delta-v2-f1-full-language-sft-qwen35-08b-modal-v1"
PROTOCOL_PATH = Path(
    "experiments/delta_v2/knowledge_full_language_stability_eval_protocol.yaml"
)
EVAL_PATH = Path(
    "data/experiments/delta_v2/knowledge_stability_choice_preserved_v1/eval.jsonl"
)
CHECKPOINT_PATH = Path("runs") / TRAINING_RUN_ID / "model"
CHECKPOINT_SHA256 = "0037e2e25837c0ea7e36766f68c487b80853b6051c0ae382b399a59701e2171a"
TRAINING_RECEIPT_SHA256 = "d2de9cf7eeafac0ba54fad077dcaa8a8cb16422e4d98450d447bdeea17f263c1"
TRAINING_METRICS_SHA256 = "f53ed0cfe52aae73411bbb2b33a9789bc936f413f1e872c4e07a84d9381cfeaa"
EXPECTED_KINDS = {
    "choice": 15,
    "boolean_true": 15,
    "boolean_false": 15,
    "recall": 15,
}


class FullLanguageVerifyError(ValueError):
    """The full-language stability gate cannot run outside its receipt."""


def _bind_shared() -> None:
    shared.CONFIG_SCHEMA = CONFIG_SCHEMA
    shared.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    shared.PROTOCOL_ID = PROTOCOL_ID
    shared.CONDITION_ID = CONDITION_ID
    shared.RUN_ID = RUN_ID
    shared.TRAINING_RUN_ID = TRAINING_RUN_ID
    shared.PROTOCOL_PATH = PROTOCOL_PATH
    shared.EVAL_PATH = EVAL_PATH
    shared.EXPECTED_KINDS = EXPECTED_KINDS


def validate_config(
    config: Mapping[str, Any], *, repo_root: Path
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("run_id") != RUN_ID
        or config.get("condition_id") != CONDITION_ID
        or config.get("task") != "eval"
    ):
        raise FullLanguageVerifyError("full-language VERIFY identity changed")
    binding = _mapping(config.get("protocol"), "protocol")
    protocol_path = repo_root / _relative_path(binding.get("path"), "protocol.path")
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or protocol_path != repo_root / PROTOCOL_PATH
        or _sha256(protocol_path) != binding.get("sha256")
    ):
        raise FullLanguageVerifyError("full-language VERIFY protocol changed")
    protocol = _load_yaml(protocol_path)
    trigger = dict(_mapping(protocol.get("decision_trigger"), "trigger"))
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("condition_id") != CONDITION_ID
        or protocol.get("state") != "preregistered-for-pre-gate"
        or trigger != {
            "training_run_id": TRAINING_RUN_ID,
            "training_launch_commit": "154dd00",
            "training_receipt_sha256": TRAINING_RECEIPT_SHA256,
            "training_metrics_sha256": TRAINING_METRICS_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
        }
    ):
        raise FullLanguageVerifyError("full-language VERIFY trigger changed")
    root = repo_root / "runs" / TRAINING_RUN_ID
    if (
        _sha256(root / "run_receipt.json") != TRAINING_RECEIPT_SHA256
        or _sha256(root / "train_metrics.json") != TRAINING_METRICS_SHA256
    ):
        raise FullLanguageVerifyError("full-language VERIFY evidence changed")
    dataset = dict(_mapping(protocol.get("dataset"), "dataset"))
    if dataset != {
        "path": str(EVAL_PATH),
        "sha256": "3b6def494dc791ff3d4d46bde569485656d4ab0a1084dd7dc059e863f96d0ae7",
        "sources": 15,
        "rows": 60,
        "training_overlap": "same-claim-new-wording",
        "exact_prompt_overlap_with_training": 0,
        "probe_counts": EXPECTED_KINDS,
    } or _sha256(repo_root / EVAL_PATH) != dataset["sha256"]:
        raise FullLanguageVerifyError("full-language VERIFY dataset changed")
    if dict(_mapping(protocol.get("generation"), "generation")) != {
        "method": "greedy", "do_sample": False, "max_new_tokens": 256,
        "repeats": 2, "seed": 20260730,
        "repeat_invariant": "byte-identical-raw-continuation",
    }:
        raise FullLanguageVerifyError("full-language VERIFY generation changed")
    if dict(_mapping(protocol.get("gates"), "gates")) != {
        "choice_parseable_minimum": 0.90,
        "boolean_parseable_minimum": 1.0,
        "boolean_true_accuracy_minimum": 0.80,
        "boolean_false_accuracy_minimum": 0.90,
        "recall": "advisory", "pooled_score": "forbidden",
        "all_gates_required": True,
    }:
        raise FullLanguageVerifyError("full-language VERIFY gates changed")
    if config.get("model") != {
        "kind": "full_language_checkpoint",
        "run_id": TRAINING_RUN_ID,
        "path": str(CHECKPOINT_PATH),
        "sha256": CHECKPOINT_SHA256,
        "quantization": "none",
    }:
        raise FullLanguageVerifyError("full-language VERIFY model changed")
    runtime = config["runtime"]
    if (
        runtime.get("packages") != RUNTIME_PACKAGES
        or runtime.get("device") != "cuda"
        or runtime.get("dtype") != "bfloat16"
        or runtime.get("attention_implementation") != "eager"
        or runtime.get("deterministic_algorithms") is not True
    ):
        raise FullLanguageVerifyError("full-language VERIFY runtime changed")
    if config.get("eval") != {
        "module": "experiments.delta_v2.run_knowledge_full_language_stability_eval",
        "path": str(EVAL_PATH), "sha256": dataset["sha256"],
    }:
        raise FullLanguageVerifyError("full-language VERIFY eval changed")
    if _relative_path(config["output"]["dir"], "output.dir") != Path("runs") / RUN_ID:
        raise FullLanguageVerifyError("full-language VERIFY output changed")
    if config.get("modal") != {
        "app_path": MODAL_APP_PATH, "gpu": "A10G", "timeout_seconds": 3600,
    }:
        raise FullLanguageVerifyError("full-language VERIFY Modal changed")
    probes = shared._load_jsonl(repo_root / EVAL_PATH)
    if len(probes) != 60 or Counter(str(row["probe_kind"]) for row in probes) != Counter(EXPECTED_KINDS):
        raise FullLanguageVerifyError("full-language VERIFY probes changed")
    return protocol, probes


class FullCheckpointBackend(TransformersKnowledgeBackend):
    def __init__(self, *, config: Mapping[str, Any], protocol: Mapping[str, Any], repo_root: Path) -> None:
        self._versions = validate_runtime(config)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise FullLanguageVerifyError("full-language VERIFY requires one CUDA GPU")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        checkpoint = repo_root / CHECKPOINT_PATH
        marker = checkpoint / "model.safetensors"
        if _sha256(marker) != CHECKPOINT_SHA256:
            raise FullLanguageVerifyError("full-language checkpoint changed")
        self._torch = torch
        self._seed = int(protocol["generation"]["seed"])
        self._max_new_tokens = int(protocol["generation"]["max_new_tokens"])
        self._processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=False)
        self._model = AutoModelForMultimodalLM.from_pretrained(
            checkpoint, dtype=torch.bfloat16, attn_implementation="eager",
            trust_remote_code=False,
        ).to("cuda")
        self._model.eval()
        self._adapter_kind = "full_language_checkpoint"


def execute(*, config_path: Path, repo_root: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    protocol, probes = validate_config(config, repo_root=repo_root)
    _bind_shared()
    requests = shared.build_requests(protocol, probes)
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    active = FullCheckpointBackend(config=config, protocol=protocol, repo_root=repo_root)
    outputs = []
    for request in requests:
        raw, finish = active.generate(request)
        outputs.append({
            "schema": OUTPUT_SCHEMA, "request_id": request["request_id"],
            "run_id": RUN_ID, "protocol_id": PROTOCOL_ID,
            "probe_id": request["probe_id"], "repeat_index": request["repeat_index"],
            "raw_response": raw, "finish_reason": finish,
        })
    metrics = shared.score(probes, outputs)
    output_dir = repo_root / config["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    samples_path.write_bytes(serialize_jsonl(outputs))
    metrics_path = output_dir / "metrics.json"
    shared._write_json(metrics_path, metrics)
    receipt = {
        "schema": "delta.knowledge_full_language_stability_eval_receipt.v1",
        "run_id": RUN_ID, "condition_id": CONDITION_ID,
        "protocol_id": PROTOCOL_ID, "training_run_id": TRAINING_RUN_ID,
        "config_sha256": _sha256(config_path),
        "protocol_sha256": _sha256(repo_root / PROTOCOL_PATH),
        "eval_sha256": _sha256(repo_root / EVAL_PATH),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "samples_sha256": _sha256(samples_path),
        "metrics_sha256": _sha256(metrics_path),
        "runtime": dict(active.runtime_receipt()),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "model_updates": 0, "status": "pass",
        "stability_gate_passed": metrics["stability_gate_passed"],
        "promotion_authorized": False,
    }
    shared._write_json(output_dir / "run_receipt.json", receipt)
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
        protocol, probes = validate_config(config, repo_root=root)
        _bind_shared()
        result = {"run_id": RUN_ID, "items": len(probes),
                  "requests": len(shared.build_requests(protocol, probes)),
                  "model_invocations_completed": 0, "status": "valid-unexecuted"}
    else:
        result = execute(config_path=path, repo_root=root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
