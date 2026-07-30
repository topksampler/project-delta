from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from experiments.delta_v2.run_knowledge_eval import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODAL_APP_PATH,
    OUTPUT_SCHEMA,
    RUNTIME_PACKAGES,
    TransformersKnowledgeBackend,
    _load_yaml,
    _mapping,
    _relative_path,
    _sha256,
    score_one,
    serialize_jsonl,
)


CONFIG_SCHEMA = "delta.knowledge_stability_replay_eval_config.v1"
PROTOCOL_SCHEMA = "delta.knowledge_stability_replay_eval_protocol.v1"
PROTOCOL_ID = "delta-v2-knowledge-stability-replay-eval-v1"
CONDITION_ID = "d4_stability_replay_eval"
RUN_ID = "delta-v2-d4-stability-replay-eval-modal-v1"
TRAINING_RUN_ID = (
    "delta-v2-d4-stability-replay-lora-qwen35-08b-modal-v1"
)
PROTOCOL_PATH = Path(
    "experiments/delta_v2/"
    "knowledge_stability_replay_eval_protocol.yaml"
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


class StabilityEvalError(ValueError):
    """The d4 stability pre-gate cannot run outside its receipt."""


class GenerationBackend(Protocol):
    def generate(self, request: Mapping[str, Any]) -> tuple[str, str]: ...

    def runtime_receipt(self) -> Mapping[str, Any]: ...


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    return [
        _mapping(json.loads(line), str(path))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_config(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("run_id") != RUN_ID
        or config.get("condition_id") != CONDITION_ID
        or config.get("task") != "eval"
    ):
        raise StabilityEvalError("d4 stability config identity changed")
    binding = _mapping(config.get("protocol"), "protocol")
    protocol_path = repo_root / _relative_path(
        binding.get("path"), "protocol.path"
    )
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or protocol_path != repo_root / PROTOCOL_PATH
        or _sha256(protocol_path) != binding.get("sha256")
    ):
        raise StabilityEvalError("d4 stability protocol binding changed")
    protocol = _load_yaml(protocol_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("condition_id") != CONDITION_ID
        or protocol.get("state") != "preregistered-for-pre-gate"
    ):
        raise StabilityEvalError("d4 stability protocol changed")
    trigger = _mapping(protocol.get("decision_trigger"), "trigger")
    expected = {
        "training_run_id": TRAINING_RUN_ID,
        "training_launch_commit": "922f93b",
        "training_receipt_sha256": "bf4b665b8274012ba6ad1e0e6d1417a3708756c24c0e5369a30f1bb89b1d488b",
        "training_metrics_sha256": "6544aff7b1d426e2ed3bcef51537941e071ea7fa56ce7fcf8ebf1da449c0e1d1",
        "adapter_sha256": "0260638f253090c6ab89e38e10ff025fbdc57ef9b736f5457faa26a339fed4c7",
    }
    if dict(trigger) != expected:
        raise StabilityEvalError("d4 stability trigger changed")
    root = repo_root / "runs" / TRAINING_RUN_ID
    if (
        _sha256(root / "run_receipt.json")
        != trigger["training_receipt_sha256"]
        or _sha256(root / "train_metrics.json")
        != trigger["training_metrics_sha256"]
        or _sha256(repo_root / ADAPTER_PATH / "adapter_model.safetensors")
        != trigger["adapter_sha256"]
    ):
        raise StabilityEvalError("d4 stability evidence changed")
    dataset = _mapping(protocol.get("dataset"), "dataset")
    if dict(dataset) != {
        "path": str(EVAL_PATH),
        "sha256": "f9c4c150aaf5db7edd2caba10801acd05b8e34cced443afdfa678aca3ed71b3a",
        "sources": 15,
        "rows": 60,
        "probe_counts": EXPECTED_KINDS,
    } or _sha256(repo_root / EVAL_PATH) != dataset["sha256"]:
        raise StabilityEvalError("d4 stability dataset changed")
    if dict(_mapping(protocol.get("generation"), "generation")) != {
        "method": "greedy",
        "do_sample": False,
        "max_new_tokens": 256,
        "repeats": 2,
        "seed": 20260730,
        "repeat_invariant": "byte-identical-raw-continuation",
    }:
        raise StabilityEvalError("d4 stability generation changed")
    if dict(_mapping(protocol.get("gates"), "gates")) != {
        "choice_parseable_minimum": 0.90,
        "boolean_parseable_minimum": 1.0,
        "boolean_true_accuracy_minimum": 0.80,
        "boolean_false_accuracy_minimum": 0.90,
        "recall": "advisory",
        "pooled_score": "forbidden",
        "all_gates_required": True,
    }:
        raise StabilityEvalError("d4 stability gates changed")
    if dict(
        _mapping(protocol.get("execution_boundary"), "boundary")
    ) != {
        "modal_evaluation_authorized": True,
        "model_updates": 0,
        "adapter_mutation": "forbidden",
        "full_eval_authorized": False,
        "verify_v2_authorized": False,
        "promotion_authorized": False,
    }:
        raise StabilityEvalError("d4 stability boundary changed")
    if dict(_mapping(config.get("model"), "model")) != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "adapter": {
            "kind": "lora",
            "run_id": TRAINING_RUN_ID,
            "path": str(ADAPTER_PATH),
            "sha256": trigger["adapter_sha256"],
        },
        "quantization": "none",
    }:
        raise StabilityEvalError("d4 stability model changed")
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
        raise StabilityEvalError("d4 stability runtime changed")
    if config.get("inputs") != {
        "run_artifacts": [
            {
                "run_id": TRAINING_RUN_ID,
                "path": f"runs/{TRAINING_RUN_ID}/run_receipt.json",
                "sha256": trigger["training_receipt_sha256"],
            },
            {
                "run_id": TRAINING_RUN_ID,
                "path": f"runs/{TRAINING_RUN_ID}/train_metrics.json",
                "sha256": trigger["training_metrics_sha256"],
            },
        ]
    }:
        raise StabilityEvalError("d4 stability inputs changed")
    if config.get("eval") != {
        "module": "experiments.delta_v2.run_knowledge_stability_replay_eval",
        "path": str(EVAL_PATH),
        "sha256": dataset["sha256"],
    }:
        raise StabilityEvalError("d4 stability eval binding changed")
    if _relative_path(config["output"]["dir"], "output.dir") != Path("runs") / RUN_ID:
        raise StabilityEvalError("d4 stability output changed")
    if dict(_mapping(config.get("modal"), "modal")) != {
        "app_path": MODAL_APP_PATH,
        "gpu": "A10G",
        "timeout_seconds": 3600,
    }:
        raise StabilityEvalError("d4 stability Modal changed")
    probes = _load_jsonl(repo_root / EVAL_PATH)
    if (
        len(probes) != 60
        or Counter(str(row["probe_kind"]) for row in probes)
        != Counter(EXPECTED_KINDS)
    ):
        raise StabilityEvalError("d4 stability probes changed")
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
                        "max_new_tokens": protocol["generation"]["max_new_tokens"],
                        "seed": protocol["generation"]["seed"],
                    },
                }
            )
    return requests


def score(
    probes: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    probe_by_id = {str(row["probe_id"]): row for row in probes}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for output in outputs:
        grouped[str(output["probe_id"])].append(output)
    cells = {}
    for probe_id, probe in probe_by_id.items():
        repeats = sorted(grouped[probe_id], key=lambda row: row["repeat_index"])
        if (
            len(repeats) != 2
            or repeats[0]["raw_response"] != repeats[1]["raw_response"]
        ):
            raise StabilityEvalError("d4 stability repeats changed")
        correct, parseable, _ = score_one(
            probe, str(repeats[0]["raw_response"])
        )
        kind = str(probe["probe_kind"])
        cell = cells.setdefault(
            kind, {"items": 0, "correct": 0, "parseable": 0}
        )
        cell["items"] += 1
        cell["correct"] += int(correct)
        cell["parseable"] += int(parseable)
    for kind, cell in cells.items():
        cell["probe_kind"] = kind
        cell["exact_accuracy"] = cell["correct"] / cell["items"]
        cell["parseable_rate"] = cell["parseable"] / cell["items"]
    gates = {
        "choice_parseable": cells["choice"]["parseable_rate"] >= 0.90,
        "boolean_parseable": (
            cells["boolean_true"]["parseable_rate"] == 1.0
            and cells["boolean_false"]["parseable_rate"] == 1.0
        ),
        "boolean_true": cells["boolean_true"]["exact_accuracy"] >= 0.80,
        "boolean_false": cells["boolean_false"]["exact_accuracy"] >= 0.90,
    }
    return {
        "schema": "delta.knowledge_stability_replay_eval_metrics.v1",
        "run_id": RUN_ID,
        "condition_id": CONDITION_ID,
        "status": "pass" if all(gates.values()) else "fail",
        "items": 60,
        "model_outputs": 120,
        "deterministic_repeats": True,
        "pooled_overall_accuracy": None,
        "cells": dict(sorted(cells.items())),
        "gates": gates,
        "stability_gate_passed": all(gates.values()),
    }


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
    active = backend or TransformersKnowledgeBackend(
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
    metrics = score(probes, outputs)
    output_dir = repo_root / str(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    samples_path.write_bytes(serialize_jsonl(outputs))
    metrics_path = output_dir / "metrics.json"
    _write_json(metrics_path, metrics)
    receipt = {
        "schema": "delta.knowledge_stability_replay_eval_receipt.v1",
        "run_id": RUN_ID,
        "condition_id": CONDITION_ID,
        "protocol_id": PROTOCOL_ID,
        "training_run_id": TRAINING_RUN_ID,
        "config_sha256": _sha256(config_path),
        "protocol_sha256": _sha256(repo_root / PROTOCOL_PATH),
        "eval_sha256": _sha256(repo_root / EVAL_PATH),
        "adapter_sha256": protocol["decision_trigger"]["adapter_sha256"],
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
    _write_json(output_dir / "run_receipt.json", receipt)
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
