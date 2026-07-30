from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import yaml

from experiments.delta_v2.run_knowledge_eval import (
    DATASET_ID,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODAL_APP_PATH,
    RUNTIME_PACKAGES,
    TRAIN_RUN_ID,
    TransformersKnowledgeBackend,
)
from experiments.delta_v2.train_knowledge_lora import (
    TRAIN_SCHEMA,
    validate_rows,
)


PROTOCOL_SCHEMA = "delta.knowledge_train_surface_diagnostic_protocol.v1"
PROTOCOL_ID = "delta-v2-knowledge-train-surface-diagnostic-v1"
CONFIG_SCHEMA = "delta.knowledge_train_surface_diagnostic_config.v1"
REQUEST_SCHEMA = "delta.knowledge_train_surface_diagnostic_request.v1"
OUTPUT_SCHEMA = "delta.knowledge_train_surface_diagnostic_output.v1"
METRICS_SCHEMA = "delta.knowledge_train_surface_diagnostic_metrics.v1"
RECEIPT_SCHEMA = "delta.knowledge_train_surface_diagnostic_receipt.v1"
RUN_ID = "delta-v2-c5-knowledge-lora-train-surface-modal-v1"
PROTOCOL_PATH = Path(
    "experiments/delta_v2/"
    "knowledge_train_surface_diagnostic_protocol.yaml"
)
TRAIN_PATH = Path(
    "data/experiments/delta_v2/knowledge_adaptation_v1/train.jsonl"
)
EXPECTED_SURFACES = {
    "exact_recall": 21,
    "verify_true": 21,
    "verify_false": 21,
}


class DiagnosticError(ValueError):
    """The frozen training-surface diagnostic cannot run safely."""


class Backend(Protocol):
    def generate(self, request: Mapping[str, Any]) -> tuple[str, str]: ...

    def runtime_receipt(self) -> Mapping[str, Any]: ...


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DiagnosticError(f"{field} must be a mapping")
    return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DiagnosticError(f"cannot read YAML: {path}") from exc
    return _mapping(payload, str(path))


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        return [
            _mapping(json.loads(line), str(path))
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"cannot read JSONL: {path}") from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(value: Any, field: str) -> Path:
    path = Path(str(value))
    if not str(value) or path.is_absolute() or ".." in path.parts:
        raise DiagnosticError(f"{field} must stay within the repository")
    return path


def validate_config(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("run_id") != RUN_ID
        or config.get("experiment_id") != "delta_v2"
        or config.get("condition_id") != "c5_knowledge_lora_diagnostic"
        or config.get("task") != "eval"
    ):
        raise DiagnosticError("diagnostic run identity changed")
    binding = _mapping(config.get("protocol"), "protocol")
    protocol_path = repo_root / _relative(binding.get("path"), "protocol.path")
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or protocol_path != repo_root / PROTOCOL_PATH
        or _sha256(protocol_path) != binding.get("sha256")
    ):
        raise DiagnosticError("diagnostic protocol binding changed")
    protocol = _load_yaml(protocol_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("state") != "preregistered-for-diagnostic"
    ):
        raise DiagnosticError("diagnostic protocol identity changed")
    trigger = _mapping(protocol.get("trigger"), "trigger")
    if dict(trigger) != {
        "adapter_eval_run_id": (
            "delta-v2-c5-knowledge-lora-eval-qwen35-08b-modal-v1"
        ),
        "metrics_sha256": (
            "592915e674e874e4e5874ed835ed4169127260c2faf88e7e8b92490322670a8b"
        ),
        "samples_sha256": (
            "a6549894df5a8c1d9f9065e494e2fbefded5972460b02f296518996bbc2cafa6"
        ),
        "observed_acquisition_boolean_true": "0/21",
        "observed_acquisition_boolean_false": "21/21",
    }:
        raise DiagnosticError("diagnostic trigger changed")

    dataset = _mapping(protocol.get("dataset"), "dataset")
    train_path = repo_root / _relative(dataset.get("path"), "dataset.path")
    if (
        dataset.get("dataset_id") != DATASET_ID
        or train_path != repo_root / TRAIN_PATH
        or dataset.get("sha256") != _sha256(train_path)
        or dataset.get("rows") != 63
        or dataset.get("prompt_source") != "messages[0]"
        or dataset.get("gold_source") != "messages[1]"
        or dataset.get("mutation") != "forbidden"
    ):
        raise DiagnosticError("diagnostic dataset changed")
    rows = _load_jsonl(train_path)
    if any(row.get("schema") != TRAIN_SCHEMA for row in rows):
        raise DiagnosticError("diagnostic training row schema changed")
    validate_rows(
        rows,
        expected_rows=63,
        expected_surfaces=EXPECTED_SURFACES,
    )

    protocol_model = _mapping(protocol.get("model"), "protocol.model")
    if dict(protocol_model) != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "adapter_run_id": TRAIN_RUN_ID,
        "adapter_path": str(Path("runs") / TRAIN_RUN_ID / "adapter"),
        "adapter_sha256": (
            "83fb04ae52502432df6b17d2aaa12cfeb12f457df6a77c83f28fed433d9966fa"
        ),
    }:
        raise DiagnosticError("diagnostic protocol model changed")
    if dict(_mapping(protocol.get("generation"), "generation")) != {
        "method": "greedy",
        "do_sample": False,
        "max_new_tokens": 256,
        "repeats": 2,
        "seed": 20260730,
        "repeat_invariant": "byte-identical-raw-continuation",
    }:
        raise DiagnosticError("diagnostic generation changed")
    scoring = _mapping(protocol.get("scoring"), "scoring")
    if dict(scoring) != {
        "authority": "frozen-training-row-assistant-answer",
        "method": "exact-stripped-string",
        "repair": "forbidden",
        "report_surfaces": [
            "exact_recall",
            "verify_true",
            "verify_false",
        ],
        "pooled_overall_score": "forbidden",
    }:
        raise DiagnosticError("diagnostic scoring changed")
    if dict(_mapping(protocol.get("decision"), "decision")) != {
        "boolean_surface_fit": {
            "verify_true_accuracy_minimum": 0.8,
            "verify_false_accuracy_minimum": 0.8,
        },
        "if_fit_and_eval_true_failed": "paraphrase-transfer-failure",
        "if_training_true_failed": "optimization-or-surface-fit-failure",
        "result": "diagnostic-only",
        "promotion_authorized": False,
    }:
        raise DiagnosticError("diagnostic decision changed")

    model = _mapping(config.get("model"), "model")
    adapter = _mapping(model.get("adapter"), "model.adapter")
    adapter_path = _relative(adapter.get("path"), "model.adapter.path")
    if (
        model.get("repository") != MODEL_REPOSITORY
        or model.get("revision") != MODEL_REVISION
        or model.get("quantization") != "none"
        or adapter.get("kind") != "lora"
        or adapter.get("run_id") != TRAIN_RUN_ID
        or adapter_path != Path("runs") / TRAIN_RUN_ID / "adapter"
    ):
        raise DiagnosticError("diagnostic model binding changed")
    marker = repo_root / adapter_path / "adapter_model.safetensors"
    if not marker.is_file() or _sha256(marker) != adapter.get("sha256"):
        raise DiagnosticError("diagnostic adapter bytes changed")
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
        raise DiagnosticError("diagnostic runtime changed")
    if dict(_mapping(config.get("data"), "data")) != {
        "train_path": str(TRAIN_PATH)
    }:
        raise DiagnosticError("diagnostic data config changed")
    eval_block = _mapping(config.get("eval"), "eval")
    if dict(eval_block) != {
        "module": (
            "experiments.delta_v2."
            "run_knowledge_train_surface_diagnostic"
        )
    }:
        raise DiagnosticError("diagnostic module changed")
    output = _mapping(config.get("output"), "output")
    if _relative(output.get("dir"), "output.dir") != Path("runs") / RUN_ID:
        raise DiagnosticError("diagnostic output changed")
    if dict(_mapping(config.get("modal"), "modal")) != {
        "app_path": MODAL_APP_PATH,
        "gpu": "A10G",
        "timeout_seconds": 3600,
    }:
        raise DiagnosticError("diagnostic Modal binding changed")
    return protocol, rows


def build_requests(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    run_id: str = RUN_ID,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    generation = protocol["generation"]
    for row in rows:
        for repeat_index in range(1, int(generation["repeats"]) + 1):
            identity = f"{run_id}\0{row['row_id']}\0{repeat_index}"
            requests.append(
                {
                    "schema": REQUEST_SCHEMA,
                    "request_id": hashlib.sha256(
                        identity.encode("utf-8")
                    ).hexdigest(),
                    "row_id": row["row_id"],
                    "surface": row["surface"],
                    "repeat_index": repeat_index,
                    "model_input": {
                        "messages": [dict(row["messages"][0])],
                    },
                    "generation": {
                        "do_sample": False,
                        "max_new_tokens": generation["max_new_tokens"],
                        "seed": generation["seed"],
                    },
                }
            )
    return requests


def score(
    *,
    rows: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    run_id: str = RUN_ID,
) -> dict[str, Any]:
    row_by_id = {str(row["row_id"]): row for row in rows}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for output in outputs:
        grouped[str(output["row_id"])].append(output)
    if set(grouped) != set(row_by_id):
        raise DiagnosticError("diagnostic output coverage changed")
    counts: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for row_id, row in row_by_id.items():
        repeats = sorted(
            grouped[row_id],
            key=lambda item: int(item["repeat_index"]),
        )
        if [item["repeat_index"] for item in repeats] != [1, 2]:
            raise DiagnosticError("diagnostic repeat coverage changed")
        raw = [str(item["raw_response"]) for item in repeats]
        if raw[0] != raw[1]:
            raise DiagnosticError("diagnostic output is nondeterministic")
        surface = str(row["surface"])
        counts[surface] += 1
        correct[surface] += (
            raw[0].strip() == str(row["messages"][1]["content"])
        )
    cells = {
        surface: {
            "items": counts[surface],
            "correct": correct[surface],
            "exact_accuracy": correct[surface] / counts[surface],
        }
        for surface in EXPECTED_SURFACES
    }
    decision = protocol["decision"]["boolean_surface_fit"]
    boolean_fit = (
        cells["verify_true"]["exact_accuracy"]
        >= decision["verify_true_accuracy_minimum"]
        and cells["verify_false"]["exact_accuracy"]
        >= decision["verify_false_accuracy_minimum"]
    )
    return {
        "schema": METRICS_SCHEMA,
        "run_id": run_id,
        "status": "pass",
        "rows": 63,
        "outputs": 126,
        "deterministic_repeats": True,
        "pooled_overall_accuracy": None,
        "cells": cells,
        "boolean_surface_fit": boolean_fit,
        "diagnosis": (
            "paraphrase-transfer-failure"
            if boolean_fit
            else "optimization-or-surface-fit-failure"
        ),
        "diagnostic_only": True,
        "promotion_authorized": False,
    }


def execute_validated_diagnostic(
    *,
    config_path: Path,
    repo_root: Path,
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    run_id: str,
    protocol_id: str,
    protocol_path: Path,
    train_path: Path,
    backend: Backend | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    requests = build_requests(rows, protocol, run_id=run_id)
    active = backend or TransformersKnowledgeBackend(
        config=config,
        protocol=protocol,
        repo_root=repo_root,
    )
    outputs = []
    for request in requests:
        raw, finish_reason = active.generate(request)
        outputs.append(
            {
                "schema": OUTPUT_SCHEMA,
                "request_id": request["request_id"],
                "row_id": request["row_id"],
                "surface": request["surface"],
                "repeat_index": request["repeat_index"],
                "raw_response": raw,
                "finish_reason": finish_reason,
            }
        )
    metrics = score(
        rows=rows,
        outputs=outputs,
        protocol=protocol,
        run_id=run_id,
    )
    output_dir = repo_root / str(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    metrics_path = output_dir / "metrics.json"
    samples_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in outputs
        ),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "run_id": run_id,
        "protocol_id": protocol_id,
        "config_sha256": _sha256(config_path),
        "protocol_sha256": _sha256(repo_root / protocol_path),
        "train_sha256": _sha256(repo_root / train_path),
        "adapter_sha256": config["model"]["adapter"]["sha256"],
        "samples_sha256": _sha256(samples_path),
        "metrics_sha256": _sha256(metrics_path),
        "runtime": dict(active.runtime_receipt()),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "status": "pass",
    }
    receipt_path = output_dir / "run_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"metrics": metrics, "receipt": receipt}


def execute(
    *,
    config_path: Path,
    repo_root: Path,
    backend: Backend | None = None,
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    protocol, rows = validate_config(config, repo_root=repo_root)
    return execute_validated_diagnostic(
        config_path=config_path,
        repo_root=repo_root,
        config=config,
        protocol=protocol,
        rows=rows,
        run_id=RUN_ID,
        protocol_id=PROTOCOL_ID,
        protocol_path=PROTOCOL_PATH,
        train_path=TRAIN_PATH,
        backend=backend,
    )


def validate_only(config_path: Path, repo_root: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    protocol, rows = validate_config(config, repo_root=repo_root)
    requests = build_requests(rows, protocol)
    return {
        "schema": CONFIG_SCHEMA,
        "run_id": RUN_ID,
        "rows": len(rows),
        "requests": len(requests),
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
