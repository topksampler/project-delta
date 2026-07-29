from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from experiments.delta_v2.baseline_protocol import (
    EXPECTED_ENVIRONMENT_ID,
    EXPECTED_MODEL_REVISION,
    EXPECTED_PROTOCOL_ID,
    OUTPUT_SCHEMA,
    BaselineProtocolError,
    build_requests,
    load_yaml,
    score_outputs,
    serialize_jsonl,
    validate_protocol,
)


RUN_CONFIG_SCHEMA = "delta.target_model_run_config.v1"
RUN_RECEIPT_SCHEMA = "delta.target_model_run_receipt.v1"
EXPECTED_RUN_ID = "delta-v2-c0-base-qwen35-08b-modal-v1"
EXPECTED_PROTOCOL_PATH = Path(
    "experiments/delta_v2/target_baseline_protocol.yaml"
)
EXPECTED_PROTOCOL_SHA256 = (
    "e26985e78b94365123cb8f4c0d1cf34d377d6b488f70f7ca618946336eb9b98d"
)
EXPECTED_EVAL_PATH = Path(
    "data/experiments/delta_v2/acceptance_attempt_2/eval/"
    "acceptance_eval_items_v3.jsonl"
)
EXPECTED_OUTPUT_DIR = Path("runs") / EXPECTED_RUN_ID
EXPECTED_MODULE = "experiments.delta_v2.run_baseline"
EXPECTED_MODAL_APP = "experiments/delta_v2/modal_baseline_app.py"


class TargetBackend(Protocol):
    """A model backend that can see only one sanitized request envelope."""

    def generate(self, request: Mapping[str, Any]) -> tuple[str, str]:
        """Return raw decoded continuation and finish reason."""

    def runtime_receipt(self) -> Mapping[str, Any]:
        """Describe the runtime that produced the continuations."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BaselineProtocolError(f"{field} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise BaselineProtocolError(f"{field} must stay within the repository")
    return path


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BaselineProtocolError(f"{field} must be a mapping")
    return value


def validate_run_config(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    """Fail closed unless this is the single preregistered c0 baseline run."""

    if dict(config).get("schema") != RUN_CONFIG_SCHEMA:
        raise BaselineProtocolError("unexpected target run-config schema")
    if (
        config.get("run_id") != EXPECTED_RUN_ID
        or config.get("experiment_id") != "delta_v2"
        or config.get("condition_id") != "c0_base"
        or config.get("eval_environment_id") != EXPECTED_ENVIRONMENT_ID
        or config.get("task") != "eval"
    ):
        raise BaselineProtocolError("target run-config identity changed")

    protocol_binding = _mapping(config.get("protocol"), "protocol")
    protocol_path = _relative_path(protocol_binding.get("path"), "protocol.path")
    if (
        protocol_path != EXPECTED_PROTOCOL_PATH
        or protocol_binding.get("protocol_id") != EXPECTED_PROTOCOL_ID
        or protocol_binding.get("sha256") != EXPECTED_PROTOCOL_SHA256
    ):
        raise BaselineProtocolError("target protocol binding changed")
    absolute_protocol_path = repo_root / protocol_path
    if (
        not absolute_protocol_path.is_file()
        or _sha256(absolute_protocol_path) != EXPECTED_PROTOCOL_SHA256
    ):
        raise BaselineProtocolError("target protocol bytes changed")
    protocol = load_yaml(absolute_protocol_path)
    items = list(validate_protocol(protocol, repo_root=repo_root))

    model = _mapping(config.get("model"), "model")
    if dict(model) != {
        "repository": "Qwen/Qwen3.5-0.8B",
        "revision": EXPECTED_MODEL_REVISION,
        "adapter": "none",
        "quantization": "none",
    }:
        raise BaselineProtocolError("target model binding changed")

    runtime = _mapping(config.get("runtime"), "runtime")
    if dict(runtime) != {
        "profile": "delta-v2-baseline-modal-v1",
        "python": "3.11.9",
        "torch": "2.10.0",
        "transformers": "5.14.1",
        "device": "cuda",
        "dtype": "bfloat16",
        "attention_implementation": "eager",
        "deterministic_algorithms": True,
    }:
        raise BaselineProtocolError("target runtime binding changed")

    eval_config = _mapping(config.get("eval"), "eval")
    eval_path = _relative_path(eval_config.get("path"), "eval.path")
    if (
        eval_config.get("module") != EXPECTED_MODULE
        or eval_path != EXPECTED_EVAL_PATH
        or eval_config.get("sha256")
        != protocol["evaluation_input"]["sha256"]
        or not (repo_root / eval_path).is_file()
        or _sha256(repo_root / eval_path)
        != protocol["evaluation_input"]["sha256"]
    ):
        raise BaselineProtocolError("target evaluation binding changed")

    output = _mapping(config.get("output"), "output")
    if _relative_path(output.get("dir"), "output.dir") != EXPECTED_OUTPUT_DIR:
        raise BaselineProtocolError("target output binding changed")

    modal = _mapping(config.get("modal"), "modal")
    if dict(modal) != {
        "app_path": EXPECTED_MODAL_APP,
        "gpu": "A10G",
        "timeout_seconds": 3600,
    }:
        raise BaselineProtocolError("target Modal binding changed")

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
        "eval",
        "output",
        "modal",
    }:
        raise BaselineProtocolError("unexpected target run-config field")
    return protocol, items


def _installed_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError as exc:
        raise BaselineProtocolError(
            f"required runtime package is missing: {package}"
        ) from exc


def validate_runtime(config: Mapping[str, Any]) -> dict[str, Any]:
    """Check the real worker before any model repository access."""

    runtime = _mapping(config.get("runtime"), "runtime")
    observed = {
        "python": platform.python_version(),
        "torch": _installed_version("torch"),
        "transformers": _installed_version("transformers"),
    }
    expected = {
        "python": str(runtime["python"]),
        "torch": str(runtime["torch"]),
        "transformers": str(runtime["transformers"]),
    }
    normalized = {
        key: value.split("+", 1)[0] for key, value in observed.items()
    }
    if normalized != expected:
        raise BaselineProtocolError(
            f"worker runtime mismatch: expected={expected} observed={observed}"
        )
    return observed


class TransformersBackend:
    """Pinned, single-GPU Transformers implementation of the protocol."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        protocol: Mapping[str, Any],
    ) -> None:
        self._observed_versions = validate_runtime(config)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise BaselineProtocolError(
                "baseline requires exactly one CUDA GPU"
            )
        if not torch.cuda.is_bf16_supported():
            raise BaselineProtocolError("worker GPU does not support bfloat16")

        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self._torch = torch
        self._seed = int(protocol["generation"]["seed"])
        self._max_new_tokens = int(
            protocol["generation"]["max_new_tokens"]
        )
        self._repository = str(config["model"]["repository"])
        self._revision = str(config["model"]["revision"])
        self._processor = AutoProcessor.from_pretrained(
            self._repository,
            revision=self._revision,
            trust_remote_code=False,
        )
        self._model = AutoModelForMultimodalLM.from_pretrained(
            self._repository,
            revision=self._revision,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=False,
        ).to("cuda")
        self._model.eval()

    def _reset_seed(self) -> None:
        random.seed(self._seed)
        self._torch.manual_seed(self._seed)
        self._torch.cuda.manual_seed_all(self._seed)

    def generate(self, request: Mapping[str, Any]) -> tuple[str, str]:
        self._reset_seed()
        model_input = _mapping(request.get("model_input"), "model_input")
        messages = model_input.get("messages")
        if not isinstance(messages, list):
            raise BaselineProtocolError("request messages changed")
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self._model.device)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self._max_new_tokens,
                num_return_sequences=1,
            )
        self._torch.cuda.synchronize()
        continuation = generated[0][prompt_tokens:]
        raw_response = self._processor.decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        finish_reason = (
            "length"
            if int(continuation.shape[-1]) >= self._max_new_tokens
            else "stop"
        )
        return raw_response, finish_reason

    def runtime_receipt(self) -> Mapping[str, Any]:
        torch = self._torch
        return {
            **self._observed_versions,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "gpu": torch.cuda.get_device_name(0),
            "dtype": "bfloat16",
            "attention_implementation": "eager",
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
        }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def execute_baseline(
    *,
    config_path: Path,
    repo_root: Path,
    backend: TargetBackend | None = None,
    output_dir_override: Path | None = None,
) -> dict[str, Any]:
    """Execute exactly one complete baseline and write only after it scores."""

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    config = load_yaml(config_path)
    protocol, items = validate_run_config(config, repo_root=repo_root)
    requests = build_requests(protocol, items)
    request_bytes = serialize_jsonl(requests)

    active_backend = backend or TransformersBackend(
        config=config,
        protocol=protocol,
    )
    outputs: list[dict[str, Any]] = []
    for request in requests:
        raw_response, finish_reason = active_backend.generate(request)
        outputs.append(
            {
                "schema": OUTPUT_SCHEMA,
                "protocol_id": protocol["protocol_id"],
                "request_id": request["request_id"],
                "eval_id": request["eval_id"],
                "repeat_index": request["repeat_index"],
                "model_revision": protocol["target_model"]["revision"],
                "raw_response": raw_response,
                "finish_reason": finish_reason,
            }
        )

    audit = score_outputs(
        protocol=protocol,
        items=items,
        outputs=outputs,
    )
    output_bytes = serialize_jsonl(outputs)
    output_dir = (
        output_dir_override
        if output_dir_override is not None
        else repo_root / EXPECTED_OUTPUT_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    metrics_path = output_dir / "metrics.json"
    receipt_path = output_dir / "run_receipt.json"
    samples_path.write_bytes(output_bytes)
    _write_json(metrics_path, audit)
    receipt = {
        "schema": RUN_RECEIPT_SCHEMA,
        "run_id": config["run_id"],
        "protocol_id": protocol["protocol_id"],
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "eval_environment_id": config["eval_environment_id"],
        "eval_items": len(items),
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
    }
    _write_json(receipt_path, receipt)
    return {"audit": audit, "receipt": receipt}


def validate_only(
    *,
    config_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    protocol, items = validate_run_config(config, repo_root=repo_root)
    requests = build_requests(protocol, items)
    payload = serialize_jsonl(requests)
    return {
        "schema": RUN_CONFIG_SCHEMA,
        "run_id": config["run_id"],
        "protocol_id": protocol["protocol_id"],
        "config_sha256": _sha256(config_path),
        "eval_items": len(items),
        "requests": len(requests),
        "request_sha256": hashlib.sha256(payload).hexdigest(),
        "model_invocations_completed": 0,
        "target_model_results_exist": False,
        "status": "valid-unexecuted",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen delta_v2 c0 target-model baseline."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all frozen bindings without importing the model runtime.",
    )
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
        result = execute_baseline(
            config_path=config_path,
            repo_root=repo_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
