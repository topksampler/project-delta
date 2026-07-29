from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2.baseline_protocol import (
    EXPECTED_ENVIRONMENT_ID,
    EXPECTED_MODEL_REVISION,
    OUTPUT_SCHEMA,
    BaselineProtocolError,
    load_yaml,
    score_outputs,
    serialize_jsonl,
)
from experiments.delta_v2.run_baseline import (
    TargetBackend,
    TransformersBackend,
)
from experiments.delta_v2.task_calibration import (
    BASELINE_PROTOCOL_SHA256,
    DEVELOPMENT_PATH,
    DEVELOPMENT_SHA256,
    PROTOCOL_ID,
    PROTOCOL_PATH,
    audit_request_context,
    build_task_calibration_requests,
    scoring_protocol,
    validate_task_calibration_protocol,
)


RUN_CONFIG_SCHEMA = "delta.target_model_run_config.v1"
RUN_RECEIPT_SCHEMA = "delta.prompt_adaptation_run_receipt.v1"
EXPECTED_RUN_ID = "delta-v2-c1-task-calibration-qwen35-08b-modal-v1"
EXPECTED_PROTOCOL_SHA256 = (
    "6981b16d259b6b425d1516532a80ec629581dc87b0cb59f3f0f000d22d22eedd"
)
EXPECTED_EVAL_PATH = Path(
    "data/experiments/delta_v2/acceptance_attempt_2/eval/"
    "acceptance_eval_items_v3.jsonl"
)
EXPECTED_EVAL_SHA256 = (
    "1125a7a5429cf847d16475259ffeb6578b8d4841c81f2f01d20d4065f4c188cb"
)
EXPECTED_OUTPUT_DIR = Path("runs") / EXPECTED_RUN_ID
EXPECTED_MODULE = "experiments.delta_v2.run_task_calibration"
EXPECTED_MODAL_APP = "experiments/delta_v2/modal_baseline_app.py"
EXPECTED_RUNTIME = {
    "profile": "delta-v2-baseline-modal-v2",
    "python": "3.11.9",
    "packages": {
        "torch": "2.10.0",
        "torchvision": "0.25.0",
        "transformers": "5.14.1",
        "pillow": "12.1.0",
        "sentencepiece": "0.2.1",
        "protobuf": "6.33.4",
        "pyyaml": "6.0.3",
        "safetensors": "0.8.0",
    },
    "device": "cuda",
    "dtype": "bfloat16",
    "attention_implementation": "eager",
    "deterministic_algorithms": True,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BaselineProtocolError(f"{field} must be a mapping")
    return value


def _relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BaselineProtocolError(f"{field} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise BaselineProtocolError(f"{field} must stay within the repository")
    return path


def validate_run_config(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    Mapping[str, Any],
]:
    if (
        config.get("schema") != RUN_CONFIG_SCHEMA
        or config.get("run_id") != EXPECTED_RUN_ID
        or config.get("experiment_id") != "delta_v2"
        or config.get("condition_id") != "c1_task_calibration"
        or config.get("eval_environment_id") != EXPECTED_ENVIRONMENT_ID
        or config.get("task") != "eval"
    ):
        raise BaselineProtocolError("task-calibration run identity changed")

    binding = _mapping(config.get("protocol"), "protocol")
    protocol_path = _relative_path(binding.get("path"), "protocol.path")
    if binding != {
        "protocol_id": PROTOCOL_ID,
        "path": str(PROTOCOL_PATH),
        "sha256": EXPECTED_PROTOCOL_SHA256,
        "baseline_protocol_sha256": BASELINE_PROTOCOL_SHA256,
        "request_sha256": (
            "96e209a30ca9b723f8e28e2af9c233f2a48b277c913bbb5da5bc4e7019ffbe77"
        ),
    } or protocol_path != PROTOCOL_PATH:
        raise BaselineProtocolError(
            "task-calibration protocol binding changed"
        )
    absolute_protocol_path = repo_root / protocol_path
    if (
        not absolute_protocol_path.is_file()
        or _sha256(absolute_protocol_path) != EXPECTED_PROTOCOL_SHA256
    ):
        raise BaselineProtocolError("task-calibration protocol bytes changed")
    protocol = load_yaml(absolute_protocol_path)
    (
        baseline_protocol,
        acceptance_items,
        fact_examples,
        feature_example,
    ) = validate_task_calibration_protocol(protocol, repo_root=repo_root)

    if config.get("model") != {
        "repository": "Qwen/Qwen3.5-0.8B",
        "revision": EXPECTED_MODEL_REVISION,
        "adapter": "none",
        "quantization": "none",
    }:
        raise BaselineProtocolError("task-calibration model binding changed")
    if config.get("runtime") != EXPECTED_RUNTIME:
        raise BaselineProtocolError(
            "task-calibration runtime binding changed"
        )

    data = _mapping(config.get("data"), "data")
    data_path = _relative_path(data.get("path"), "data.path")
    if (
        data_path != DEVELOPMENT_PATH
        or data.get("sha256") != DEVELOPMENT_SHA256
        or not (repo_root / data_path).is_file()
        or _sha256(repo_root / data_path) != DEVELOPMENT_SHA256
    ):
        raise BaselineProtocolError(
            "task-calibration development data changed"
        )
    eval_config = _mapping(config.get("eval"), "eval")
    eval_path = _relative_path(eval_config.get("path"), "eval.path")
    if (
        eval_config.get("module") != EXPECTED_MODULE
        or eval_path != EXPECTED_EVAL_PATH
        or eval_config.get("sha256") != EXPECTED_EVAL_SHA256
        or not (repo_root / eval_path).is_file()
        or _sha256(repo_root / eval_path) != EXPECTED_EVAL_SHA256
    ):
        raise BaselineProtocolError(
            "task-calibration acceptance data changed"
        )
    output = _mapping(config.get("output"), "output")
    if _relative_path(output.get("dir"), "output.dir") != EXPECTED_OUTPUT_DIR:
        raise BaselineProtocolError(
            "task-calibration output binding changed"
        )
    if config.get("modal") != {
        "app_path": EXPECTED_MODAL_APP,
        "gpu": "A10G",
        "timeout_seconds": 3600,
    }:
        raise BaselineProtocolError("task-calibration Modal binding changed")
    if set(config) != {
        "schema",
        "run_id",
        "experiment_id",
        "condition_id",
        "eval_environment_id",
        "task",
        "protocol",
        "model",
        "runtime",
        "data",
        "eval",
        "output",
        "modal",
    }:
        raise BaselineProtocolError(
            "unexpected task-calibration run-config field"
        )
    return (
        protocol,
        baseline_protocol,
        acceptance_items,
        fact_examples,
        feature_example,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def execute_task_calibration(
    *,
    config_path: Path,
    repo_root: Path,
    backend: TargetBackend | None = None,
    output_dir_override: Path | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    config = load_yaml(config_path)
    (
        protocol,
        baseline_protocol,
        acceptance_items,
        fact_examples,
        feature_example,
    ) = validate_run_config(config, repo_root=repo_root)
    requests = build_task_calibration_requests(
        protocol,
        baseline_protocol=baseline_protocol,
        acceptance_items=acceptance_items,
        fact_examples=fact_examples,
        feature_example=feature_example,
    )
    request_audit = audit_request_context(
        requests=requests,
        acceptance_items=acceptance_items,
        fact_examples=fact_examples,
        feature_example=feature_example,
    )
    request_bytes = serialize_jsonl(requests)
    if (
        hashlib.sha256(request_bytes).hexdigest()
        != protocol["execution_boundary"]["expected_request_sha256"]
    ):
        raise BaselineProtocolError(
            "task-calibration request bytes changed"
        )
    adapted_protocol = scoring_protocol(baseline_protocol)
    active_backend = backend or TransformersBackend(
        config=config,
        protocol=adapted_protocol,
    )

    outputs: list[dict[str, Any]] = []
    for request in requests:
        raw_response, finish_reason = active_backend.generate(request)
        outputs.append(
            {
                "schema": OUTPUT_SCHEMA,
                "protocol_id": PROTOCOL_ID,
                "request_id": request["request_id"],
                "eval_id": request["eval_id"],
                "repeat_index": request["repeat_index"],
                "model_revision": EXPECTED_MODEL_REVISION,
                "raw_response": raw_response,
                "finish_reason": finish_reason,
            }
        )
    result_audit = score_outputs(
        protocol=adapted_protocol,
        items=acceptance_items,
        outputs=outputs,
    )
    result_audit["adaptation"] = {
        "method": protocol["adaptation"]["method"],
        "request_audit": request_audit,
        "weight_updates": False,
        "retrieval": False,
    }

    output_bytes = serialize_jsonl(outputs)
    output_dir = (
        output_dir_override
        if output_dir_override is not None
        else repo_root / EXPECTED_OUTPUT_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples.jsonl").write_bytes(output_bytes)
    _write_json(output_dir / "metrics.json", result_audit)
    _write_json(output_dir / "request_audit.json", request_audit)
    receipt = {
        "schema": RUN_RECEIPT_SCHEMA,
        "run_id": EXPECTED_RUN_ID,
        "protocol_id": PROTOCOL_ID,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "eval_environment_id": EXPECTED_ENVIRONMENT_ID,
        "eval_items": len(acceptance_items),
        "development_fact_examples": len(fact_examples),
        "development_feature_examples": 1,
        "requests": len(requests),
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "model_outputs": len(outputs),
        "samples_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "runtime": dict(active_backend.runtime_receipt()),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "status": "pass",
        "target_model_results_exist": True,
        "weight_updates": False,
    }
    _write_json(output_dir / "run_receipt.json", receipt)
    return {"audit": result_audit, "receipt": receipt}


def validate_only(
    *,
    config_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    (
        protocol,
        baseline_protocol,
        acceptance_items,
        fact_examples,
        feature_example,
    ) = validate_run_config(config, repo_root=repo_root)
    requests = build_task_calibration_requests(
        protocol,
        baseline_protocol=baseline_protocol,
        acceptance_items=acceptance_items,
        fact_examples=fact_examples,
        feature_example=feature_example,
    )
    request_audit = audit_request_context(
        requests=requests,
        acceptance_items=acceptance_items,
        fact_examples=fact_examples,
        feature_example=feature_example,
    )
    payload = serialize_jsonl(requests)
    if (
        hashlib.sha256(payload).hexdigest()
        != protocol["execution_boundary"]["expected_request_sha256"]
    ):
        raise BaselineProtocolError(
            "task-calibration request bytes changed"
        )
    return {
        **request_audit,
        "run_id": config["run_id"],
        "config_sha256": _sha256(config_path),
        "request_sha256": hashlib.sha256(payload).hexdigest(),
        "model_invocations_completed": 0,
        "target_model_results_exist": False,
        "status": "valid-unexecuted",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run delta_v2 development-only task calibration."
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
    if args.validate_only:
        result = validate_only(
            config_path=config_path,
            repo_root=repo_root,
        )
    else:
        result = execute_task_calibration(
            config_path=config_path,
            repo_root=repo_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
