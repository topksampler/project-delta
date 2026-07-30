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


PROTOCOL_SCHEMA = "delta.knowledge_stability_eval_protocol.v1"
PROTOCOL_ID = "delta-v2-knowledge-stability-base-v1"
CONFIG_SCHEMA = "delta.knowledge_stability_eval_config.v1"
RUN_ID = "delta-v2-d4-stability-base-qwen35-08b-modal-v1"
CONDITION_ID = "d4_stability_base"
DATASET_ID = "delta-v2-knowledge-stability-replay-v1"
PROTOCOL_PATH = Path(
    "experiments/delta_v2/knowledge_stability_base_protocol.yaml"
)
FREEZE_PATH = Path(
    "experiments/delta_v2/knowledge_stability_replay_data_freeze.yaml"
)
EVAL_PATH = Path(
    "data/experiments/delta_v2/knowledge_stability_replay_v1/"
    "stability_dev.jsonl"
)
MODAL_APP_PATH = "experiments/delta_v2/modal_knowledge_adapt_app.py"
EXPECTED_KINDS = {
    "choice": 15,
    "boolean_true": 15,
    "boolean_false": 15,
    "recall": 15,
}


class StabilityBaseError(ValueError):
    """The zero-update stability baseline cannot run outside its contract."""


class GenerationBackend(Protocol):
    def generate(self, request: Mapping[str, Any]) -> tuple[str, str]:
        """Return raw continuation and finish reason."""

    def runtime_receipt(self) -> Mapping[str, Any]:
        """Return the observed runtime."""


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise StabilityBaseError(f"cannot read JSONL: {path}") from exc
    rows = []
    for line_number, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StabilityBaseError(
                f"invalid JSON at {path}:{line_number}"
            ) from exc
        rows.append(_mapping(payload, f"{path}:{line_number}"))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_protocol(
    protocol: Mapping[str, Any],
    *,
    repo_root: Path,
) -> list[Mapping[str, Any]]:
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("experiment_id") != "delta_v2"
        or protocol.get("condition_id") != CONDITION_ID
        or protocol.get("state") != "preregistered-for-base-baseline"
    ):
        raise StabilityBaseError("stability baseline identity changed")
    dataset = _mapping(protocol.get("dataset"), "dataset")
    freeze_path = repo_root / _relative_path(
        dataset.get("freeze_path"), "dataset.freeze_path"
    )
    eval_path = repo_root / _relative_path(
        dataset.get("eval_path"), "dataset.eval_path"
    )
    if (
        dataset.get("dataset_id") != DATASET_ID
        or freeze_path != repo_root / FREEZE_PATH
        or eval_path != repo_root / EVAL_PATH
        or _sha256(freeze_path) != dataset.get("freeze_sha256")
        or _sha256(eval_path) != dataset.get("eval_sha256")
        or dataset.get("sources") != 15
        or dataset.get("rows") != 60
        or dataset.get("model_visible_fields") != ["prompt"]
    ):
        raise StabilityBaseError("stability dataset binding changed")
    if dict(_mapping(protocol.get("target_model"), "target_model")) != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "adapter": "none",
        "quantization": "none",
    }:
        raise StabilityBaseError("stability base model changed")
    if dict(_mapping(protocol.get("prompting"), "prompting")) != {
        "interface": "official-chat-template",
        "messages": [{"role": "user", "content_from": "prompt"}],
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "system_prompt": "none",
    }:
        raise StabilityBaseError("stability prompting changed")
    if dict(_mapping(protocol.get("generation"), "generation")) != {
        "method": "greedy",
        "do_sample": False,
        "max_new_tokens": 256,
        "repeats": 2,
        "seed": 20260730,
        "repeat_invariant": "byte-identical-raw-continuation",
    }:
        raise StabilityBaseError("stability generation changed")
    reporting = _mapping(protocol.get("reporting"), "reporting")
    if (
        reporting.get("pooled_overall_score") != "forbidden"
        or reporting.get("required_probe_counts") != EXPECTED_KINDS
        or reporting.get("headline_probe_kinds")
        != ["choice", "boolean_true", "boolean_false"]
        or reporting.get("advisory_probe_kinds") != ["recall"]
        or reporting.get("role")
        != "zero-update-reference-not-candidate-gate"
    ):
        raise StabilityBaseError("stability reporting changed")
    if dict(
        _mapping(protocol.get("execution_boundary"), "execution_boundary")
    ) != {
        "base_baseline_authorized": True,
        "model_updates": 0,
        "lora_training_authorized": False,
        "qlora_authorized": False,
        "full_weight_authorized": False,
        "reinforcement_authorized": False,
        "promotion_authorized": False,
    }:
        raise StabilityBaseError("stability execution boundary changed")
    rows = _load_jsonl(eval_path)
    if (
        len(rows) != 60
        or Counter(str(row["probe_kind"]) for row in rows)
        != Counter(EXPECTED_KINDS)
        or any(row.get("stratum") != "stability_dev" for row in rows)
        or [str(row["probe_id"]) for row in rows]
        != sorted(str(row["probe_id"]) for row in rows)
    ):
        raise StabilityBaseError("stability probes changed")
    return rows


def validate_config(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("run_id") != RUN_ID
        or config.get("experiment_id") != "delta_v2"
        or config.get("condition_id") != CONDITION_ID
        or config.get("task") != "eval"
    ):
        raise StabilityBaseError("stability config identity changed")
    binding = _mapping(config.get("protocol"), "protocol")
    protocol_path = repo_root / _relative_path(
        binding.get("path"), "protocol.path"
    )
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or protocol_path != repo_root / PROTOCOL_PATH
        or _sha256(protocol_path) != binding.get("sha256")
    ):
        raise StabilityBaseError("stability protocol binding changed")
    protocol = _load_yaml(protocol_path)
    probes = validate_protocol(protocol, repo_root=repo_root)
    if dict(_mapping(config.get("model"), "model")) != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "adapter": {"kind": "none"},
        "quantization": "none",
    }:
        raise StabilityBaseError("stability config model changed")
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
        raise StabilityBaseError("stability runtime changed")
    eval_block = _mapping(config.get("eval"), "eval")
    eval_path = repo_root / _relative_path(
        eval_block.get("path"), "eval.path"
    )
    if (
        eval_block.get("module")
        != "experiments.delta_v2.run_knowledge_stability_base"
        or eval_path != repo_root / EVAL_PATH
        or _sha256(eval_path) != eval_block.get("sha256")
    ):
        raise StabilityBaseError("stability eval binding changed")
    if _relative_path(
        _mapping(config.get("output"), "output").get("dir"),
        "output.dir",
    ) != Path("runs") / RUN_ID:
        raise StabilityBaseError("stability output changed")
    if dict(_mapping(config.get("modal"), "modal")) != {
        "app_path": MODAL_APP_PATH,
        "gpu": "A10G",
        "timeout_seconds": 3600,
    }:
        raise StabilityBaseError("stability Modal binding changed")
    return protocol, probes


def build_requests(
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    probes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    requests = []
    for probe in probes:
        for repeat_index in (1, 2):
            identity = "\0".join(
                (
                    PROTOCOL_ID,
                    RUN_ID,
                    str(probe["probe_id"]),
                    str(repeat_index),
                )
            ).encode()
            requests.append(
                {
                    "schema": "delta.knowledge_eval_request.v1",
                    "request_id": "stability-request:"
                    + hashlib.sha256(identity).hexdigest(),
                    "run_id": config["run_id"],
                    "protocol_id": PROTOCOL_ID,
                    "probe_id": probe["probe_id"],
                    "repeat_index": repeat_index,
                    "model_input": {
                        "messages": [
                            {
                                "role": "user",
                                "content": probe["prompt"],
                            }
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
    if len(requests) != 120:
        raise StabilityBaseError("stability request count changed")
    return requests


def score(
    probes: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    probe_by_id = {str(row["probe_id"]): row for row in probes}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for output in outputs:
        grouped[str(output["probe_id"])].append(output)
    if set(grouped) != set(probe_by_id):
        raise StabilityBaseError("stability output coverage changed")
    cells: dict[str, dict[str, Any]] = {}
    for probe_id, probe in probe_by_id.items():
        repeats = sorted(
            grouped[probe_id], key=lambda row: int(row["repeat_index"])
        )
        if (
            [row["repeat_index"] for row in repeats] != [1, 2]
            or repeats[0]["raw_response"] != repeats[1]["raw_response"]
        ):
            raise StabilityBaseError("stability repeats changed")
        correct, parseable, _parsed = score_one(
            probe, str(repeats[0]["raw_response"])
        )
        kind = str(probe["probe_kind"])
        cell = cells.setdefault(
            kind,
            {
                "probe_kind": kind,
                "items": 0,
                "correct": 0,
                "parseable": 0,
            },
        )
        cell["items"] += 1
        cell["correct"] += int(correct)
        cell["parseable"] += int(parseable)
    for cell in cells.values():
        cell["exact_accuracy"] = cell["correct"] / cell["items"]
        cell["parseable_rate"] = cell["parseable"] / cell["items"]
    return {
        "schema": "delta.knowledge_stability_eval_metrics.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "pass",
        "items": len(probes),
        "model_outputs": len(outputs),
        "deterministic_repeats": True,
        "pooled_overall_accuracy": None,
        "cells": dict(sorted(cells.items())),
        "role": "zero-update-reference-not-candidate-gate",
    }


def execute(
    *,
    config_path: Path,
    repo_root: Path,
    backend: GenerationBackend | None = None,
    output_dir_override: Path | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    config = _load_yaml(config_path)
    protocol, probes = validate_config(config, repo_root=repo_root)
    requests = build_requests(config, protocol, probes)
    active_backend = backend or TransformersKnowledgeBackend(
        config=config,
        protocol=protocol,
        repo_root=repo_root,
    )
    outputs = []
    for request in requests:
        raw_response, finish_reason = active_backend.generate(request)
        outputs.append(
            {
                "schema": OUTPUT_SCHEMA,
                "request_id": request["request_id"],
                "run_id": RUN_ID,
                "protocol_id": PROTOCOL_ID,
                "probe_id": request["probe_id"],
                "repeat_index": request["repeat_index"],
                "raw_response": raw_response,
                "finish_reason": finish_reason,
            }
        )
    metrics = score(probes, outputs)
    request_bytes = serialize_jsonl(requests)
    output_bytes = serialize_jsonl(outputs)
    output_dir = (
        output_dir_override
        if output_dir_override is not None
        else repo_root / str(config["output"]["dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples.jsonl").write_bytes(output_bytes)
    _write_json(output_dir / "metrics.json", metrics)
    receipt = {
        "schema": "delta.knowledge_stability_eval_receipt.v1",
        "run_id": RUN_ID,
        "condition_id": CONDITION_ID,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "config_sha256": _sha256(config_path),
        "protocol_sha256": _sha256(repo_root / PROTOCOL_PATH),
        "eval_sha256": _sha256(repo_root / EVAL_PATH),
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "samples_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "metrics_sha256": _sha256(output_dir / "metrics.json"),
        "eval_items": len(probes),
        "model_outputs": len(outputs),
        "model_updates": 0,
        "runtime": dict(active_backend.runtime_receipt()),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "status": "pass",
    }
    _write_json(output_dir / "run_receipt.json", receipt)
    return {"metrics": metrics, "receipt": receipt}


def validate_only(config_path: Path, repo_root: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    protocol, probes = validate_config(config, repo_root=repo_root)
    requests = build_requests(config, protocol, probes)
    return {
        "run_id": RUN_ID,
        "protocol_id": PROTOCOL_ID,
        "eval_items": len(probes),
        "requests": len(requests),
        "request_sha256": hashlib.sha256(
            serialize_jsonl(requests)
        ).hexdigest(),
        "model_invocations_completed": 0,
        "status": "valid-unexecuted",
    }


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
    result = (
        validate_only(config_path, repo_root)
        if args.validate_only
        else execute(config_path=config_path, repo_root=repo_root)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
