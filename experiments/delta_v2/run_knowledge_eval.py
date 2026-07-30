from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import yaml


PROTOCOL_SCHEMA = "delta.knowledge_eval_protocol.v1"
PROTOCOL_ID = "delta-v2-knowledge-eval-v1"
RUN_CONFIG_SCHEMA = "delta.knowledge_eval_run_config.v1"
PROBE_SCHEMA = "delta.knowledge_probe.v1"
REQUEST_SCHEMA = "delta.knowledge_eval_request.v1"
OUTPUT_SCHEMA = "delta.knowledge_eval_output.v1"
METRICS_SCHEMA = "delta.knowledge_eval_metrics.v1"
RECEIPT_SCHEMA = "delta.knowledge_eval_run_receipt.v1"
DATASET_ID = "delta-v2-vllm-knowledge-adaptation-v1"
MODEL_REPOSITORY = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
PROTOCOL_PATH = Path("experiments/delta_v2/knowledge_eval_protocol.yaml")
EVAL_PATH = Path(
    "data/experiments/delta_v2/knowledge_adaptation_v1/eval.jsonl"
)
FREEZE_PATH = Path(
    "experiments/delta_v2/knowledge_adaptation_freeze.yaml"
)
MODAL_APP_PATH = "experiments/delta_v2/modal_knowledge_adapt_app.py"
BASE_RUN_ID = "delta-v2-c4-knowledge-base-qwen35-08b-modal-v1"
TRAIN_RUN_ID = "delta-v2-c5-knowledge-lora-qwen35-08b-modal-v1"
ADAPTER_RUN_ID = "delta-v2-c5-knowledge-lora-eval-qwen35-08b-modal-v1"
EXPECTED_STRATA = {
    "acquisition_added": 84,
    "retention_stable": 84,
    "feature_retention": 1,
}
EXPECTED_PROBE_KINDS = {
    "choice": 42,
    "boolean_true": 42,
    "boolean_false": 42,
    "recall": 42,
    "verified_behavior_json": 1,
}
RUNTIME_PACKAGES = {
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "transformers": "5.14.1",
    "peft": "0.19.0",
    "pillow": "12.1.0",
    "sentencepiece": "0.2.1",
    "protobuf": "6.33.4",
    "pyyaml": "6.0.3",
    "safetensors": "0.8.0",
}


class KnowledgeEvalError(ValueError):
    """The frozen knowledge evaluation cannot execute safely."""


class GenerationBackend(Protocol):
    def generate(self, request: Mapping[str, Any]) -> tuple[str, str]:
        """Return the raw continuation and finish reason."""

    def runtime_receipt(self) -> Mapping[str, Any]:
        """Return the actual inference runtime."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KnowledgeEvalError(f"{field} must be a mapping")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise KnowledgeEvalError(f"{field} must be a non-empty string")
    return value


def _relative_path(value: Any, field: str) -> Path:
    path = Path(_string(value, field))
    if path.is_absolute() or ".." in path.parts:
        raise KnowledgeEvalError(f"{field} must stay within the repository")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise KnowledgeEvalError(f"cannot read YAML: {path}") from exc
    return _mapping(payload, str(path))


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise KnowledgeEvalError(f"cannot read JSONL: {path}") from exc
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise KnowledgeEvalError(
                f"invalid JSON at {path}:{line_number}"
            ) from exc
        rows.append(_mapping(payload, f"{path}:{line_number}"))
    return rows


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def serialize_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (canonical_json(dict(row)) + "\n").encode("utf-8") for row in rows
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
        or protocol.get("state") != "preregistered-for-base-and-lora"
    ):
        raise KnowledgeEvalError("knowledge-eval protocol identity changed")

    dataset = _mapping(protocol.get("dataset"), "dataset")
    freeze_path = repo_root / _relative_path(
        dataset.get("freeze_path"),
        "dataset.freeze_path",
    )
    eval_path = repo_root / _relative_path(
        dataset.get("eval_path"),
        "dataset.eval_path",
    )
    if (
        dataset.get("dataset_id") != DATASET_ID
        or freeze_path != repo_root / FREEZE_PATH
        or eval_path != repo_root / EVAL_PATH
        or not freeze_path.is_file()
        or not eval_path.is_file()
        or _sha256(freeze_path) != dataset.get("freeze_sha256")
        or _sha256(eval_path) != dataset.get("eval_sha256")
        or dataset.get("rows") != 169
        or dataset.get("model_visible_fields") != ["prompt"]
    ):
        raise KnowledgeEvalError("knowledge-eval dataset binding changed")

    model = _mapping(protocol.get("target_model"), "target_model")
    if (
        model.get("repository") != MODEL_REPOSITORY
        or model.get("revision") != MODEL_REVISION
        or model.get("conditions")
        != ["c4_knowledge_base", "c5_knowledge_lora"]
    ):
        raise KnowledgeEvalError("knowledge-eval model binding changed")

    prompting = _mapping(protocol.get("prompting"), "prompting")
    if (
        prompting.get("interface") != "official-chat-template"
        or prompting.get("messages")
        != [{"role": "user", "content_from": "prompt"}]
        or prompting.get("add_generation_prompt") is not True
        or prompting.get("chat_template_kwargs") != {"enable_thinking": False}
        or prompting.get("system_prompt") != "none"
    ):
        raise KnowledgeEvalError("knowledge-eval prompting changed")

    generation = _mapping(protocol.get("generation"), "generation")
    if dict(generation) != {
        "method": "greedy",
        "do_sample": False,
        "max_new_tokens": 256,
        "repeats": 2,
        "seed": 20260730,
        "repeat_invariant": "byte-identical-raw-continuation",
    }:
        raise KnowledgeEvalError("knowledge-eval generation changed")

    scoring = _mapping(protocol.get("scoring"), "scoring")
    if (
        scoring.get("authority") != "frozen-deterministic-probe-gold"
        or _mapping(scoring.get("exact_choice"), "exact_choice").get(
            "valid"
        )
        != ["A", "B", "C", "D"]
        or _mapping(scoring.get("exact_boolean"), "exact_boolean").get(
            "valid"
        )
        != ["yes", "no"]
        or _mapping(scoring.get("exact_json"), "exact_json").get("repair")
        != "forbidden"
        or scoring.get("invalid_or_unparseable") != "incorrect"
        or scoring.get("llm_judge") != "forbidden"
    ):
        raise KnowledgeEvalError("knowledge-eval scoring changed")

    reporting = _mapping(protocol.get("reporting"), "reporting")
    if (
        reporting.get("pooled_overall_score") != "forbidden"
        or reporting.get("required_strata") != EXPECTED_STRATA
        or reporting.get("headline_probe_kinds")
        != ["choice", "boolean_true", "boolean_false"]
        or reporting.get("advisory_probe_kinds") != ["recall"]
        or reporting.get("feature_signal") != "separate"
        or reporting.get("comparison")
        != "identical-probes-base-versus-lora"
        or reporting.get("acquisition_claim") != "same-claim-new-surface"
        or reporting.get("held_out_fact_generalization_claim") != "forbidden"
    ):
        raise KnowledgeEvalError("knowledge-eval reporting changed")

    boundary = _mapping(
        protocol.get("execution_boundary"),
        "execution_boundary",
    )
    if dict(boundary) != {
        "base_run_required_before_lora": True,
        "lora_training_authorized": True,
        "full_weight_training": "forbidden",
        "promotion": "forbidden",
    }:
        raise KnowledgeEvalError("knowledge-eval execution boundary changed")

    rows = _load_jsonl(eval_path)
    if len(rows) != 169:
        raise KnowledgeEvalError("knowledge-eval row count changed")
    ids: list[str] = []
    strata: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    scorers: Counter[str] = Counter()
    for row in rows:
        if row.get("schema") != PROBE_SCHEMA:
            raise KnowledgeEvalError("unexpected knowledge-probe schema")
        probe_id = _string(row.get("probe_id"), "probe_id")
        if probe_id in ids:
            raise KnowledgeEvalError("duplicate knowledge-probe id")
        ids.append(probe_id)
        _string(row.get("source_id"), "source_id")
        _string(row.get("prompt"), "prompt")
        stratum = _string(row.get("stratum"), "stratum")
        kind = _string(row.get("probe_kind"), "probe_kind")
        scorer = _string(row.get("scorer"), "scorer")
        if scorer not in {
            "exact-choice-v1",
            "exact-boolean-v1",
            "exact-json-v1",
        }:
            raise KnowledgeEvalError("unsupported knowledge scorer")
        if "gold" not in row:
            raise KnowledgeEvalError("knowledge probe has no deterministic gold")
        strata[stratum] += 1
        kinds[kind] += 1
        scorers[scorer] += 1
    if ids != sorted(ids):
        raise KnowledgeEvalError("knowledge probes must be ordered")
    if dict(sorted(strata.items())) != EXPECTED_STRATA:
        raise KnowledgeEvalError("knowledge strata changed")
    if dict(sorted(kinds.items())) != EXPECTED_PROBE_KINDS:
        raise KnowledgeEvalError("knowledge probe kinds changed")
    if scorers != {
        "exact-choice-v1": 42,
        "exact-boolean-v1": 84,
        "exact-json-v1": 43,
    }:
        raise KnowledgeEvalError("knowledge scorer counts changed")
    return rows


def validate_run_config(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    if (
        config.get("schema") != RUN_CONFIG_SCHEMA
        or config.get("experiment_id") != "delta_v2"
        or config.get("eval_environment_id") != DATASET_ID
        or config.get("task") != "eval"
    ):
        raise KnowledgeEvalError("knowledge run identity changed")
    run_id = _string(config.get("run_id"), "run_id")
    condition = _string(config.get("condition_id"), "condition_id")
    if (run_id, condition) not in {
        (BASE_RUN_ID, "c4_knowledge_base"),
        (ADAPTER_RUN_ID, "c5_knowledge_lora"),
    }:
        raise KnowledgeEvalError("unsupported knowledge run")

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
        raise KnowledgeEvalError("knowledge protocol binding changed")
    protocol = _load_yaml(protocol_path)
    rows = validate_protocol(protocol, repo_root=repo_root)

    model = _mapping(config.get("model"), "model")
    if (
        model.get("repository") != MODEL_REPOSITORY
        or model.get("revision") != MODEL_REVISION
        or model.get("quantization") != "none"
    ):
        raise KnowledgeEvalError("knowledge model binding changed")
    adapter = _mapping(model.get("adapter"), "model.adapter")
    if condition == "c4_knowledge_base":
        if dict(adapter) != {"kind": "none"}:
            raise KnowledgeEvalError("base run cannot load an adapter")
    else:
        adapter_path = _relative_path(adapter.get("path"), "adapter.path")
        if (
            adapter.get("kind") != "lora"
            or adapter.get("run_id") != TRAIN_RUN_ID
            or adapter_path != Path("runs") / TRAIN_RUN_ID / "adapter"
            or not isinstance(adapter.get("sha256"), str)
            or len(str(adapter["sha256"])) != 64
        ):
            raise KnowledgeEvalError("adapter run binding changed")
        absolute_adapter = repo_root / adapter_path
        marker = absolute_adapter / "adapter_model.safetensors"
        if not marker.is_file() or _sha256(marker) != adapter.get("sha256"):
            raise KnowledgeEvalError("adapter bytes changed")

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
        raise KnowledgeEvalError("knowledge runtime binding changed")

    eval_config = _mapping(config.get("eval"), "eval")
    eval_path = repo_root / _relative_path(eval_config.get("path"), "eval.path")
    if (
        eval_config.get("module")
        != "experiments.delta_v2.run_knowledge_eval"
        or eval_path != repo_root / EVAL_PATH
        or not eval_path.is_file()
        or _sha256(eval_path) != eval_config.get("sha256")
        or eval_config.get("sha256")
        != protocol["dataset"]["eval_sha256"]
    ):
        raise KnowledgeEvalError("knowledge eval binding changed")

    output = _mapping(config.get("output"), "output")
    if _relative_path(output.get("dir"), "output.dir") != Path("runs") / run_id:
        raise KnowledgeEvalError("knowledge output binding changed")
    modal = _mapping(config.get("modal"), "modal")
    if dict(modal) != {
        "app_path": MODAL_APP_PATH,
        "gpu": "A10G",
        "timeout_seconds": 3600,
    }:
        raise KnowledgeEvalError("knowledge Modal binding changed")
    return protocol, rows


def _request_id(run_id: str, probe_id: str, repeat_index: int) -> str:
    identity = "\0".join(
        (REQUEST_SCHEMA, run_id, probe_id, str(repeat_index))
    ).encode("utf-8")
    return "knowledge-request:" + hashlib.sha256(identity).hexdigest()


def build_requests(
    *,
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    probes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    run_id = str(config["run_id"])
    generation = protocol["generation"]
    requests: list[dict[str, Any]] = []
    for probe in probes:
        probe_id = str(probe["probe_id"])
        for repeat_index in range(1, int(generation["repeats"]) + 1):
            requests.append(
                {
                    "schema": REQUEST_SCHEMA,
                    "request_id": _request_id(
                        run_id,
                        probe_id,
                        repeat_index,
                    ),
                    "run_id": run_id,
                    "protocol_id": PROTOCOL_ID,
                    "probe_id": probe_id,
                    "repeat_index": repeat_index,
                    "model_input": {
                        "messages": [
                            {
                                "role": "user",
                                "content": str(probe["prompt"]),
                            }
                        ],
                        "add_generation_prompt": True,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    "generation": {
                        "do_sample": False,
                        "max_new_tokens": generation["max_new_tokens"],
                        "seed": generation["seed"],
                    },
                }
            )
    return requests


def audit_requests(requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(requests) != 338:
        raise KnowledgeEvalError("knowledge request count changed")
    for request in requests:
        model_input = _mapping(request.get("model_input"), "model_input")
        messages = model_input.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 1
            or messages[0].get("role") != "user"
        ):
            raise KnowledgeEvalError("knowledge model-input shape changed")
        if set(model_input) != {
            "messages",
            "add_generation_prompt",
            "chat_template_kwargs",
        }:
            raise KnowledgeEvalError("unexpected model-visible field")
    return {
        "schema": "delta.knowledge_eval_request_audit.v1",
        "requests": len(requests),
        "model_visible_fields": ["prompt"],
        "gold_visible": False,
        "scorer_visible": False,
        "provenance_visible": False,
        "status": "pass",
    }


def _installed_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError as exc:
        raise KnowledgeEvalError(
            f"required runtime package is missing: {package}"
        ) from exc


def validate_runtime(config: Mapping[str, Any]) -> dict[str, str]:
    runtime = _mapping(config.get("runtime"), "runtime")
    observed = {"python": platform.python_version()}
    observed.update(
        {
            package: _installed_version(package)
            for package in RUNTIME_PACKAGES
        }
    )
    expected = {"python": str(runtime["python"]), **RUNTIME_PACKAGES}
    normalized = {
        key: value.split("+", 1)[0] for key, value in observed.items()
    }
    if normalized != expected:
        raise KnowledgeEvalError(
            f"worker runtime mismatch: expected={expected} observed={observed}"
        )
    return observed


class TransformersKnowledgeBackend:
    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        protocol: Mapping[str, Any],
        repo_root: Path,
    ) -> None:
        self._versions = validate_runtime(config)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise KnowledgeEvalError("knowledge eval requires one CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise KnowledgeEvalError("knowledge eval requires bfloat16")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        self._torch = torch
        self._seed = int(protocol["generation"]["seed"])
        self._max_new_tokens = int(protocol["generation"]["max_new_tokens"])
        model_config = config["model"]
        self._processor = AutoProcessor.from_pretrained(
            MODEL_REPOSITORY,
            revision=MODEL_REVISION,
            trust_remote_code=False,
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_REPOSITORY,
            revision=MODEL_REVISION,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=False,
        )
        adapter = model_config["adapter"]
        self._adapter_kind = str(adapter["kind"])
        if self._adapter_kind == "lora":
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                str(repo_root / str(adapter["path"])),
                is_trainable=False,
            )
        self._model = model.to("cuda")
        self._model.eval()

    def _reset_seed(self) -> None:
        random.seed(self._seed)
        self._torch.manual_seed(self._seed)
        self._torch.cuda.manual_seed_all(self._seed)

    def generate(self, request: Mapping[str, Any]) -> tuple[str, str]:
        self._reset_seed()
        messages = request["model_input"]["messages"]
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
        raw = self._processor.decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        finish_reason = (
            "length"
            if int(continuation.shape[-1]) >= self._max_new_tokens
            else "stop"
        )
        return raw, finish_reason

    def runtime_receipt(self) -> Mapping[str, Any]:
        torch = self._torch
        return {
            **self._versions,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "gpu": torch.cuda.get_device_name(0),
            "dtype": "bfloat16",
            "attention_implementation": "eager",
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "adapter_kind": self._adapter_kind,
        }


def score_one(
    probe: Mapping[str, Any],
    raw_response: str,
) -> tuple[bool, bool, Any]:
    scorer = str(probe["scorer"])
    stripped = raw_response.strip()
    if scorer == "exact-choice-v1":
        parsed: Any = stripped if stripped in {"A", "B", "C", "D"} else None
    elif scorer == "exact-boolean-v1":
        parsed = stripped if stripped in {"yes", "no"} else None
    elif scorer == "exact-json-v1":
        if stripped.startswith("```"):
            parsed = None
        else:
            try:
                candidate = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            else:
                parsed = candidate if isinstance(candidate, dict) else None
    else:
        raise KnowledgeEvalError(f"unsupported scorer: {scorer}")
    parseable = parsed is not None
    return parsed == probe["gold"], parseable, parsed


def score_outputs(
    *,
    probes: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    request_audit: Mapping[str, Any],
) -> dict[str, Any]:
    probe_by_id = {str(row["probe_id"]): row for row in probes}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for output in outputs:
        if output.get("schema") != OUTPUT_SCHEMA:
            raise KnowledgeEvalError("unexpected knowledge output schema")
        probe_id = _string(output.get("probe_id"), "output.probe_id")
        if probe_id not in probe_by_id:
            raise KnowledgeEvalError("output references unknown probe")
        grouped[probe_id].append(output)
    if set(grouped) != set(probe_by_id):
        raise KnowledgeEvalError("knowledge outputs have incomplete coverage")

    scored: list[dict[str, Any]] = []
    for probe_id, probe in probe_by_id.items():
        repeats = sorted(
            grouped[probe_id],
            key=lambda row: int(row["repeat_index"]),
        )
        if [row["repeat_index"] for row in repeats] != [1, 2]:
            raise KnowledgeEvalError("knowledge repeat coverage changed")
        raw = [str(row["raw_response"]) for row in repeats]
        if raw[0] != raw[1]:
            raise KnowledgeEvalError(
                f"nondeterministic knowledge output: {probe_id}"
            )
        correct, parseable, parsed = score_one(probe, raw[0])
        scored.append(
            {
                "probe_id": probe_id,
                "stratum": probe["stratum"],
                "probe_kind": probe["probe_kind"],
                "scorer": probe["scorer"],
                "correct": correct,
                "parseable": parseable,
                "parsed": parsed,
            }
        )

    cells: dict[str, dict[str, Any]] = {}
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_cell[(str(row["stratum"]), str(row["probe_kind"]))].append(row)
    for (stratum, probe_kind), rows in sorted(by_cell.items()):
        count = len(rows)
        correct = sum(bool(row["correct"]) for row in rows)
        parseable = sum(bool(row["parseable"]) for row in rows)
        cells[f"{stratum}.{probe_kind}"] = {
            "stratum": stratum,
            "probe_kind": probe_kind,
            "items": count,
            "correct": correct,
            "exact_accuracy": correct / count,
            "parseable": parseable,
            "parseable_rate": parseable / count,
        }

    return {
        "schema": METRICS_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "pass",
        "items": len(probes),
        "model_outputs": len(outputs),
        "deterministic_repeats": True,
        "pooled_overall_accuracy": None,
        "request_audit": dict(request_audit),
        "cells": cells,
        "claim_boundary": {
            "acquisition": "same-claim-new-surface",
            "held_out_fact_generalization": False,
            "recall": "advisory",
            "feature": "separate",
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
    protocol, probes = validate_run_config(config, repo_root=repo_root)
    requests = build_requests(
        config=config,
        protocol=protocol,
        probes=probes,
    )
    request_audit = audit_requests(requests)
    request_bytes = serialize_jsonl(requests)

    active_backend = backend or TransformersKnowledgeBackend(
        config=config,
        protocol=protocol,
        repo_root=repo_root,
    )
    outputs: list[dict[str, Any]] = []
    for request in requests:
        raw_response, finish_reason = active_backend.generate(request)
        outputs.append(
            {
                "schema": OUTPUT_SCHEMA,
                "request_id": request["request_id"],
                "run_id": config["run_id"],
                "protocol_id": PROTOCOL_ID,
                "probe_id": request["probe_id"],
                "repeat_index": request["repeat_index"],
                "raw_response": raw_response,
                "finish_reason": finish_reason,
            }
        )
    metrics = score_outputs(
        probes=probes,
        outputs=outputs,
        request_audit=request_audit,
    )
    output_bytes = serialize_jsonl(outputs)
    output_dir = (
        output_dir_override
        if output_dir_override is not None
        else repo_root / str(config["output"]["dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    metrics_path = output_dir / "metrics.json"
    receipt_path = output_dir / "run_receipt.json"
    samples_path.write_bytes(output_bytes)
    _write_json(metrics_path, metrics)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "run_id": config["run_id"],
        "condition_id": config["condition_id"],
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "config_sha256": _sha256(config_path),
        "protocol_sha256": _sha256(repo_root / PROTOCOL_PATH),
        "eval_sha256": _sha256(repo_root / EVAL_PATH),
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "samples_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "metrics_sha256": _sha256(metrics_path),
        "eval_items": len(probes),
        "model_outputs": len(outputs),
        "runtime": dict(active_backend.runtime_receipt()),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "status": "pass",
    }
    _write_json(receipt_path, receipt)
    return {"metrics": metrics, "receipt": receipt}


def validate_only(
    *,
    config_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    protocol, probes = validate_run_config(config, repo_root=repo_root)
    requests = build_requests(
        config=config,
        protocol=protocol,
        probes=probes,
    )
    audit = audit_requests(requests)
    return {
        "schema": RUN_CONFIG_SCHEMA,
        "run_id": config["run_id"],
        "protocol_id": PROTOCOL_ID,
        "config_sha256": _sha256(config_path),
        "eval_items": len(probes),
        "requests": len(requests),
        "request_sha256": hashlib.sha256(
            serialize_jsonl(requests)
        ).hexdigest(),
        "request_audit": audit,
        "model_invocations_completed": 0,
        "status": "valid-unexecuted",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen delta_v2 knowledge acquisition bank."
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
        else execute(config_path=config_path, repo_root=repo_root)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
