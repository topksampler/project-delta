from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from experiments.delta_v2 import run_knowledge_stability_replay_eval as shared
from experiments.delta_v2.run_knowledge_eval import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODAL_APP_PATH,
    OUTPUT_SCHEMA,
    RUNTIME_PACKAGES,
    _load_yaml,
    _mapping,
    _relative_path,
    _sha256,
    serialize_jsonl,
)


CONFIG_SCHEMA = "delta.knowledge_stability_replay50_eval_config.v1"
PROTOCOL_SCHEMA = "delta.knowledge_stability_replay_eval_protocol.v1"
PROTOCOL_ID = "delta-v2-knowledge-stability-replay50-eval-v1"
CONDITION_ID = "d5_stability_replay50_eval"
RUN_ID = "delta-v2-d5-stability-replay50-eval-modal-v1"
TRAINING_RUN_ID = "delta-v2-d5-stability-replay50-lora-qwen35-08b-modal-v1"
PROTOCOL_PATH = Path(
    "experiments/delta_v2/knowledge_stability_replay50_eval_protocol.yaml"
)
EVAL_PATH = Path(
    "data/experiments/delta_v2/knowledge_stability_replay_v1/"
    "stability_dev.jsonl"
)
ADAPTER_PATH = Path("runs") / TRAINING_RUN_ID / "adapter"
EXPECTED_KINDS = {
    "choice": 15,
    "boolean_true": 15,
    "boolean_false": 15,
    "recall": 15,
}
TRIGGER = {
    "training_run_id": TRAINING_RUN_ID,
    "training_launch_commit": "6abd3a7",
    "training_receipt_sha256": "448adf5dcd5b049d4cbf3edb2832c2b951ff11eaf059accb262305aa024f986c",
    "training_metrics_sha256": "2971fb93488aeafcc88979327b00a951378b3b77bf4e0c56c95340d769191c7e",
    "adapter_sha256": "05d5e66653e4c116c1e3d32bd5d7df463fecea3989ab89ffa28bf029482bfda3",
}


class StabilityReplay50EvalError(ValueError):
    """The d5 stability pre-gate cannot run outside its frozen receipt."""


class GenerationBackend(Protocol):
    def generate(self, request: Mapping[str, Any]) -> tuple[str, str]: ...

    def runtime_receipt(self) -> Mapping[str, Any]: ...


def validate_config(
    config: Mapping[str, Any], *, repo_root: Path
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("run_id") != RUN_ID
        or config.get("condition_id") != CONDITION_ID
        or config.get("task") != "eval"
    ):
        raise StabilityReplay50EvalError("d5 stability config identity changed")
    binding = _mapping(config.get("protocol"), "protocol")
    protocol_path = repo_root / _relative_path(
        binding.get("path"), "protocol.path"
    )
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or protocol_path != repo_root / PROTOCOL_PATH
        or _sha256(protocol_path) != binding.get("sha256")
    ):
        raise StabilityReplay50EvalError("d5 protocol binding changed")
    protocol = _load_yaml(protocol_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("condition_id") != CONDITION_ID
        or protocol.get("state") != "preregistered-for-pre-gate"
        or dict(_mapping(protocol.get("decision_trigger"), "trigger"))
        != TRIGGER
    ):
        raise StabilityReplay50EvalError("d5 protocol changed")
    training_root = repo_root / "runs" / TRAINING_RUN_ID
    if (
        _sha256(training_root / "run_receipt.json")
        != TRIGGER["training_receipt_sha256"]
        or _sha256(training_root / "train_metrics.json")
        != TRIGGER["training_metrics_sha256"]
        or _sha256(repo_root / ADAPTER_PATH / "adapter_model.safetensors")
        != TRIGGER["adapter_sha256"]
    ):
        raise StabilityReplay50EvalError("d5 training evidence changed")
    dataset = _mapping(protocol.get("dataset"), "dataset")
    if dict(dataset) != {
        "path": str(EVAL_PATH),
        "sha256": "f9c4c150aaf5db7edd2caba10801acd05b8e34cced443afdfa678aca3ed71b3a",
        "sources": 15,
        "rows": 60,
        "probe_counts": EXPECTED_KINDS,
    } or _sha256(repo_root / EVAL_PATH) != dataset["sha256"]:
        raise StabilityReplay50EvalError("d5 stability dataset changed")
    if dict(_mapping(protocol.get("generation"), "generation")) != {
        "method": "greedy",
        "do_sample": False,
        "max_new_tokens": 256,
        "repeats": 2,
        "seed": 20260730,
        "repeat_invariant": "byte-identical-raw-continuation",
    }:
        raise StabilityReplay50EvalError("d5 generation changed")
    if dict(_mapping(protocol.get("gates"), "gates")) != {
        "choice_parseable_minimum": 0.90,
        "boolean_parseable_minimum": 1.0,
        "boolean_true_accuracy_minimum": 0.80,
        "boolean_false_accuracy_minimum": 0.90,
        "recall": "advisory",
        "pooled_score": "forbidden",
        "all_gates_required": True,
    }:
        raise StabilityReplay50EvalError("d5 gates changed")
    if dict(_mapping(protocol.get("execution_boundary"), "boundary")) != {
        "modal_evaluation_authorized": True,
        "model_updates": 0,
        "adapter_mutation": "forbidden",
        "full_eval_authorized": False,
        "verify_v2_authorized": False,
        "promotion_authorized": False,
    }:
        raise StabilityReplay50EvalError("d5 boundary changed")
    if dict(_mapping(config.get("model"), "model")) != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "adapter": {
            "kind": "lora",
            "run_id": TRAINING_RUN_ID,
            "path": str(ADAPTER_PATH),
            "sha256": TRIGGER["adapter_sha256"],
        },
        "quantization": "none",
    }:
        raise StabilityReplay50EvalError("d5 model changed")
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
        raise StabilityReplay50EvalError("d5 runtime changed")
    if config.get("inputs") != {
        "run_artifacts": [
            {
                "run_id": TRAINING_RUN_ID,
                "path": f"runs/{TRAINING_RUN_ID}/run_receipt.json",
                "sha256": TRIGGER["training_receipt_sha256"],
            },
            {
                "run_id": TRAINING_RUN_ID,
                "path": f"runs/{TRAINING_RUN_ID}/train_metrics.json",
                "sha256": TRIGGER["training_metrics_sha256"],
            },
        ]
    }:
        raise StabilityReplay50EvalError("d5 inputs changed")
    if config.get("eval") != {
        "module": "experiments.delta_v2.run_knowledge_stability_replay50_eval",
        "path": str(EVAL_PATH),
        "sha256": dataset["sha256"],
    }:
        raise StabilityReplay50EvalError("d5 eval binding changed")
    if _relative_path(config["output"]["dir"], "output.dir") != Path("runs") / RUN_ID:
        raise StabilityReplay50EvalError("d5 output changed")
    if dict(_mapping(config.get("modal"), "modal")) != {
        "app_path": MODAL_APP_PATH,
        "gpu": "A10G",
        "timeout_seconds": 3600,
    }:
        raise StabilityReplay50EvalError("d5 Modal changed")
    probes = shared._load_jsonl(repo_root / EVAL_PATH)
    if len(probes) != 60 or Counter(
        str(row["probe_kind"]) for row in probes
    ) != Counter(EXPECTED_KINDS):
        raise StabilityReplay50EvalError("d5 probes changed")
    return protocol, probes


def build_requests(
    protocol: Mapping[str, Any],
    probes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    requests = []
    for probe in probes:
        for repeat in (1, 2):
            digest = hashlib.sha256(
                f"{PROTOCOL_ID}\0{RUN_ID}\0{probe['probe_id']}\0{repeat}".encode()
            ).hexdigest()
            requests.append(
                {
                    "schema": "delta.knowledge_eval_request.v1",
                    "request_id": f"stability-request:{digest}",
                    "run_id": RUN_ID,
                    "protocol_id": PROTOCOL_ID,
                    "probe_id": probe["probe_id"],
                    "repeat_index": repeat,
                    "model_input": {
                        "messages": [
                            {"role": "user", "content": probe["prompt"]}
                        ],
                        "add_generation_prompt": True,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    "generation": {
                        "do_sample": False,
                        "max_new_tokens": protocol["generation"][
                            "max_new_tokens"
                        ],
                        "seed": protocol["generation"]["seed"],
                    },
                }
            )
    return requests


def execute(
    *,
    config_path: Path,
    repo_root: Path,
    backend: GenerationBackend | None = None,
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    protocol, probes = validate_config(config, repo_root=repo_root)
    requests = build_requests(protocol, probes)
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    active = backend or shared.TransformersKnowledgeBackend(
        config=config, protocol=protocol, repo_root=repo_root
    )
    outputs = []
    for request in requests:
        raw, finish = active.generate(request)
        outputs.append(
            {
                "schema": OUTPUT_SCHEMA,
                "request_id": request["request_id"],
                "run_id": RUN_ID,
                "protocol_id": PROTOCOL_ID,
                "probe_id": request["probe_id"],
                "repeat_index": request["repeat_index"],
                "raw_response": raw,
                "finish_reason": finish,
            }
        )
    metrics = shared.score(probes, outputs)
    metrics.update({"run_id": RUN_ID, "condition_id": CONDITION_ID})
    output_dir = repo_root / str(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    samples_path.write_bytes(serialize_jsonl(outputs))
    metrics_path = output_dir / "metrics.json"
    shared._write_json(metrics_path, metrics)
    receipt = {
        "schema": "delta.knowledge_stability_replay_eval_receipt.v1",
        "run_id": RUN_ID,
        "condition_id": CONDITION_ID,
        "protocol_id": PROTOCOL_ID,
        "training_run_id": TRAINING_RUN_ID,
        "config_sha256": _sha256(config_path),
        "protocol_sha256": _sha256(repo_root / PROTOCOL_PATH),
        "eval_sha256": _sha256(repo_root / EVAL_PATH),
        "adapter_sha256": TRIGGER["adapter_sha256"],
        "samples_sha256": _sha256(samples_path),
        "metrics_sha256": _sha256(metrics_path),
        "runtime": dict(active.runtime_receipt()),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "model_updates": 0,
        "status": "pass",
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
    repo_root = args.repo_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    if args.validate_only:
        config = _load_yaml(config_path)
        protocol, probes = validate_config(config, repo_root=repo_root)
        result = {
            "run_id": RUN_ID,
            "items": len(probes),
            "requests": len(build_requests(protocol, probes)),
            "model_invocations_completed": 0,
            "status": "valid-unexecuted",
        }
    else:
        result = execute(config_path=config_path, repo_root=repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
