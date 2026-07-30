from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Protocol, Sequence

from experiments.delta_v2.run_knowledge_eval import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODAL_APP_PATH,
    RUNTIME_PACKAGES,
    validate_runtime,
)
from experiments.delta_v2.train_knowledge_lora import (
    TRAIN_PATH,
    KnowledgeTrainError,
    _load_jsonl,
    _load_yaml,
    _mapping,
    _model_inputs,
    _relative_path,
    _sha256,
    _tokenize_rows,
    validate_rows,
)


CONFIG_SCHEMA = "delta.knowledge_boolean_margin_config.v1"
PROTOCOL_SCHEMA = "delta.knowledge_boolean_margin_protocol.v1"
PROTOCOL_ID = "delta-v2-knowledge-boolean-margin-v1"
PROTOCOL_PATH = Path(
    "experiments/delta_v2/knowledge_boolean_margin_protocol.yaml"
)
SAMPLE_SCHEMA = "delta.knowledge_boolean_margin_sample.v1"
METRICS_SCHEMA = "delta.knowledge_boolean_margin_metrics.v1"
AUDIT_SCHEMA = "delta.knowledge_boolean_pair_audit.v1"
RECEIPT_SCHEMA = "delta.knowledge_boolean_margin_receipt.v1"

EXPECTED_CONDITIONS = {
    "d1_margin_base": {
        "run_id": "delta-v2-d1-knowledge-margin-base-qwen35-08b-modal-v1",
        "role": "frozen-base",
        "training_run_id": None,
        "adapter": {"kind": "none"},
        "surface_weights": None,
    },
    "d1_margin_c5_1to1": {
        "run_id": (
            "delta-v2-d1-knowledge-margin-c5-1to1-qwen35-08b-modal-v1"
        ),
        "role": "c5-balanced-negative",
        "training_run_id": (
            "delta-v2-c5-knowledge-lora-qwen35-08b-modal-v2"
        ),
        "adapter": {
            "kind": "lora",
            "sha256": (
                "83fb04ae52502432df6b17d2aaa12cfeb12f457df6a77c83f28fed433d9966fa"
            ),
        },
        "surface_weights": {
            "exact_recall": 1,
            "verify_true": 1,
            "verify_false": 1,
        },
    },
    "d1_margin_c7_2to1": {
        "run_id": (
            "delta-v2-d1-knowledge-margin-c7-2to1-qwen35-08b-modal-v1"
        ),
        "role": "c7-midpoint-negative",
        "training_run_id": (
            "delta-v2-c7-knowledge-lora-2to1-qwen35-08b-modal-v1"
        ),
        "adapter": {
            "kind": "lora",
            "sha256": (
                "89f79fe8763e3184ef3d701ac42afa657769f7bf0d5a09df920b9d32d3544549"
            ),
        },
        "surface_weights": {
            "exact_recall": 1,
            "verify_true": 2,
            "verify_false": 1,
        },
    },
    "d1_margin_c6_4to1": {
        "run_id": (
            "delta-v2-d1-knowledge-margin-c6-4to1-qwen35-08b-modal-v1"
        ),
        "role": "c6-true-weighted-negative",
        "training_run_id": (
            "delta-v2-c6-knowledge-lora-true-weighted-"
            "qwen35-08b-modal-v1"
        ),
        "adapter": {
            "kind": "lora",
            "sha256": (
                "8dae0e5cf7692542efc34f4fc836b1c96279bae680747edc368f646fdf938e36"
            ),
        },
        "surface_weights": {
            "exact_recall": 1,
            "verify_true": 4,
            "verify_false": 1,
        },
    },
}


class MarginError(KnowledgeTrainError):
    """The frozen boolean-margin diagnostic cannot run safely."""


class MarginBackend(Protocol):
    def score_row(self, row: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def runtime_receipt(self) -> Mapping[str, Any]: ...


def _contract_from_prompt(prompt: str) -> tuple[str, dict[str, Any], str]:
    marker = "\nCONTRACT="
    suffix = "\nAnswer yes or no only."
    if marker not in prompt or not prompt.endswith(suffix):
        raise MarginError("boolean prompt envelope changed")
    prefix, encoded = prompt.split(marker, 1)
    encoded = encoded[: -len(suffix)]
    try:
        card = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise MarginError("boolean prompt contract is not JSON") from exc
    if not isinstance(card, dict):
        raise MarginError("boolean prompt contract must be an object")
    return prefix, card, suffix


def audit_pairs(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["source_id"])][str(row["surface"])] = row
    if len(grouped) != 21:
        raise MarginError("boolean source-pair count changed")

    true_annotations: Counter[str] = Counter()
    false_annotations: Counter[str] = Counter()
    length_deltas: Counter[int] = Counter()
    kinds: Counter[str] = Counter()
    for surfaces in grouped.values():
        if set(surfaces) != {
            "exact_recall",
            "verify_true",
            "verify_false",
        }:
            raise MarginError("knowledge source surfaces changed")
        recall = surfaces["exact_recall"]
        true = surfaces["verify_true"]
        false = surfaces["verify_false"]
        if (
            true["messages"][1]["content"] != "yes"
            or false["messages"][1]["content"] != "no"
        ):
            raise MarginError("boolean answer labels changed")
        true_prompt = str(true["messages"][0]["content"])
        false_prompt = str(false["messages"][0]["content"])
        true_prefix, true_card, true_suffix = _contract_from_prompt(
            true_prompt
        )
        false_prefix, false_card, false_suffix = _contract_from_prompt(
            false_prompt
        )
        if (
            true_prefix != false_prefix
            or true_suffix != false_suffix
            or list(true_card) != list(false_card)
        ):
            raise MarginError("true/false prompt formatting diverged")
        changed = [
            key
            for key in true_card
            if true_card[key] != false_card[key]
        ]
        if changed != ["annotation"]:
            raise MarginError("false corruption field changed")
        try:
            recall_card = json.loads(str(recall["messages"][1]["content"]))
        except json.JSONDecodeError as exc:
            raise MarginError("recall card is not JSON") from exc
        if recall_card != true_card:
            raise MarginError("true contract no longer matches recall card")
        true_annotation = str(true_card["annotation"])
        false_annotation = str(false_card["annotation"])
        true_annotations[true_annotation] += 1
        false_annotations[false_annotation] += 1
        length_deltas[len(false_prompt) - len(true_prompt)] += 1
        kinds[str(true_card["kind"])] += 1

    return {
        "schema": AUDIT_SCHEMA,
        "source_pairs": 21,
        "outer_prompt_template_identical": True,
        "contract_key_order_identical": True,
        "changed_fields": ["annotation"],
        "changed_field_count": 1,
        "true_annotations": dict(sorted(true_annotations.items())),
        "false_annotations": dict(sorted(false_annotations.items())),
        "prompt_length_delta_false_minus_true": {
            str(delta): count
            for delta, count in sorted(length_deltas.items())
        },
        "source_kinds": dict(sorted(kinds.items())),
        "gold_label_leak_detected": False,
        "formatting_leak_detected": False,
        "semantic_negative_diversity": "annotation-only",
        "status": "pass-with-narrow-negative-warning",
    }


def _condition_from_protocol(
    protocol: Mapping[str, Any],
    condition_id: str,
) -> Mapping[str, Any]:
    conditions = protocol.get("conditions")
    if not isinstance(conditions, list):
        raise MarginError("margin condition matrix changed")
    observed = {
        str(_mapping(item, "condition")["condition_id"]): dict(item)
        for item in conditions
    }
    expected = {
        condition: {"condition_id": condition, **payload}
        for condition, payload in EXPECTED_CONDITIONS.items()
    }
    if observed != expected or condition_id not in observed:
        raise MarginError("margin condition matrix changed")
    return observed[condition_id]


def validate_config(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    list[Mapping[str, Any]],
    Mapping[str, Any],
]:
    condition_id = str(config.get("condition_id"))
    expected = EXPECTED_CONDITIONS.get(condition_id)
    if (
        expected is None
        or config.get("schema") != CONFIG_SCHEMA
        or config.get("run_id") != expected["run_id"]
        or config.get("experiment_id") != "delta_v2"
        or config.get("task") != "eval"
    ):
        raise MarginError("margin config identity changed")
    binding = _mapping(config.get("protocol"), "protocol")
    protocol_path = repo_root / _relative_path(
        binding.get("path"), "protocol.path"
    )
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or protocol_path != repo_root / PROTOCOL_PATH
        or _sha256(protocol_path) != binding.get("sha256")
    ):
        raise MarginError("margin protocol binding changed")
    protocol = _load_yaml(protocol_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("experiment_id") != "delta_v2"
        or protocol.get("state") != "preregistered-for-diagnostic"
    ):
        raise MarginError("margin protocol identity changed")
    condition = _condition_from_protocol(protocol, condition_id)
    dataset = _mapping(protocol.get("dataset"), "dataset")
    if dict(dataset) != {
        "dataset_id": "delta-v2-vllm-knowledge-adaptation-v1",
        "path": str(TRAIN_PATH),
        "sha256": (
            "16ad0d2c12a823722501800227c739e59131fd7c7e0f774e2f1faab2e4bd4b95"
        ),
        "rows": 63,
        "included_surfaces": ["verify_true", "verify_false"],
        "included_rows": 42,
        "source_pairs": 21,
        "mutation": "forbidden",
    }:
        raise MarginError("margin dataset contract changed")
    if _sha256(repo_root / TRAIN_PATH) != dataset["sha256"]:
        raise MarginError("margin dataset bytes changed")
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
    pair_audit = audit_pairs(rows)

    model = _mapping(config.get("model"), "model")
    adapter = dict(_mapping(model.get("adapter"), "model.adapter"))
    if (
        model.get("repository") != MODEL_REPOSITORY
        or model.get("revision") != MODEL_REVISION
        or model.get("quantization") != "none"
    ):
        raise MarginError("margin model binding changed")
    if expected["adapter"]["kind"] == "none":
        if adapter != {"kind": "none"}:
            raise MarginError("base margin adapter binding changed")
    else:
        training_run_id = str(expected["training_run_id"])
        adapter_path = Path("runs") / training_run_id / "adapter"
        if adapter != {
            "kind": "lora",
            "run_id": training_run_id,
            "path": str(adapter_path),
            "sha256": expected["adapter"]["sha256"],
        }:
            raise MarginError("margin adapter binding changed")
        marker = repo_root / adapter_path / "adapter_model.safetensors"
        if (
            not marker.is_file()
            or _sha256(marker) != expected["adapter"]["sha256"]
        ):
            raise MarginError("margin adapter bytes changed")
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
        raise MarginError("margin runtime changed")
    if dict(_mapping(config.get("data"), "data")) != {
        "train_path": str(TRAIN_PATH)
    }:
        raise MarginError("margin data config changed")
    if dict(_mapping(config.get("eval"), "eval")) != {
        "module": "experiments.delta_v2.run_knowledge_boolean_margin"
    }:
        raise MarginError("margin module changed")
    if _relative_path(config["output"]["dir"], "output.dir") != (
        Path("runs") / str(expected["run_id"])
    ):
        raise MarginError("margin output changed")
    if dict(_mapping(config.get("modal"), "modal")) != {
        "app_path": MODAL_APP_PATH,
        "gpu": "A10G",
        "timeout_seconds": 3600,
    }:
        raise MarginError("margin Modal binding changed")
    boolean_rows = [
        row
        for row in rows
        if row["surface"] in {"verify_true", "verify_false"}
    ]
    if len(boolean_rows) != 42:
        raise MarginError("margin boolean row count changed")
    return protocol, condition, boolean_rows, pair_audit


class TransformersMarginBackend:
    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        repo_root: Path,
    ) -> None:
        self._versions = validate_runtime(config)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise MarginError("margin diagnostic requires one CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise MarginError("margin diagnostic requires bfloat16")
        random.seed(20260730)
        torch.manual_seed(20260730)
        torch.cuda.manual_seed_all(20260730)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        self._torch = torch
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
        adapter = config["model"]["adapter"]
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

    def _candidate(
        self,
        row: Mapping[str, Any],
        answer: str,
    ) -> Mapping[str, Any]:
        synthetic = {
            "row_id": row["row_id"],
            "surface": row["surface"],
            "messages": [
                dict(row["messages"][0]),
                {"role": "assistant", "content": answer},
            ],
        }
        return _tokenize_rows(
            self._processor,
            [synthetic],
            max_sequence_length=768,
        )[0]

    def score_row(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        torch = self._torch
        yes = self._candidate(row, "yes")
        no = self._candidate(row, "no")
        yes_labels = list(yes["labels"])
        no_labels = list(no["labels"])
        yes_start = next(
            index for index, token in enumerate(yes_labels) if token != -100
        )
        no_start = next(
            index for index, token in enumerate(no_labels) if token != -100
        )
        if (
            yes_start != no_start
            or yes["input_ids"][0, :yes_start].tolist()
            != no["input_ids"][0, :no_start].tolist()
        ):
            raise MarginError("yes/no candidate prompt prefixes diverged")
        with torch.inference_mode():
            yes_output = self._model(**_model_inputs(yes, torch))
            no_output = self._model(**_model_inputs(no, torch))
        yes_log_probability = -float(yes_output.loss.detach().cpu()) * int(
            yes["assistant_tokens"]
        )
        no_log_probability = -float(no_output.loss.detach().cpu()) * int(
            no["assistant_tokens"]
        )
        first_logits = yes_output.logits[0, yes_start - 1].float()
        yes_token = int(yes_labels[yes_start])
        no_token = int(no_labels[no_start])
        first_margin = float(
            (first_logits[yes_token] - first_logits[no_token]).detach().cpu()
        )
        sequence_margin = yes_log_probability - no_log_probability
        gold = str(row["messages"][1]["content"])
        if gold not in {"yes", "no"}:
            raise MarginError("margin gold is not boolean")
        gold_log_probability = (
            yes_log_probability if gold == "yes" else no_log_probability
        )
        result = {
            "schema": SAMPLE_SCHEMA,
            "row_id": row["row_id"],
            "source_id": row["source_id"],
            "surface": row["surface"],
            "gold": gold,
            "yes_assistant_tokens": yes["assistant_tokens"],
            "no_assistant_tokens": no["assistant_tokens"],
            "yes_first_token_id": yes_token,
            "no_first_token_id": no_token,
            "log_probability_yes": yes_log_probability,
            "log_probability_no": no_log_probability,
            "teacher_forced_gold_nll": -gold_log_probability,
            "first_token_margin_yes_minus_no": first_margin,
            "sequence_margin_yes_minus_no": sequence_margin,
            "gold_signed_margin": (
                sequence_margin if gold == "yes" else -sequence_margin
            ),
            "predicted": "yes" if sequence_margin > 0 else "no",
        }
        del yes_output, no_output, first_logits
        return result

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


def summarize_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    condition_id: str,
) -> dict[str, Any]:
    if len(samples) % 2 != 0:
        raise MarginError("margin sample count is not paired")
    by_surface: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_source: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for sample in samples:
        surface = str(sample["surface"])
        by_surface[surface].append(sample)
        by_source[str(sample["source_id"])][surface] = sample
    if set(by_surface) != {"verify_true", "verify_false"}:
        raise MarginError("margin sample surfaces changed")
    cells = {}
    for surface in ("verify_true", "verify_false"):
        rows = by_surface[surface]
        cells[surface] = {
            "items": len(rows),
            "mean_teacher_forced_gold_nll": mean(
                float(row["teacher_forced_gold_nll"]) for row in rows
            ),
            "mean_first_token_margin_yes_minus_no": mean(
                float(row["first_token_margin_yes_minus_no"])
                for row in rows
            ),
            "mean_sequence_margin_yes_minus_no": mean(
                float(row["sequence_margin_yes_minus_no"]) for row in rows
            ),
            "mean_gold_signed_margin": mean(
                float(row["gold_signed_margin"]) for row in rows
            ),
            "margin_correct": sum(
                str(row["predicted"]) == str(row["gold"]) for row in rows
            ),
            "margin_accuracy": sum(
                str(row["predicted"]) == str(row["gold"]) for row in rows
            )
            / len(rows),
        }
    separations = []
    truth_conditioned = 0
    for pair in by_source.values():
        if set(pair) != {"verify_true", "verify_false"}:
            raise MarginError("margin source pair coverage changed")
        true_margin = float(
            pair["verify_true"]["sequence_margin_yes_minus_no"]
        )
        false_margin = float(
            pair["verify_false"]["sequence_margin_yes_minus_no"]
        )
        separations.append(true_margin - false_margin)
        truth_conditioned += true_margin > 0 and false_margin < 0
    if len(separations) != len(samples) // 2:
        raise MarginError("margin paired summary changed")
    return {
        "schema": METRICS_SCHEMA,
        "run_id": run_id,
        "condition_id": condition_id,
        "status": "pass",
        "per_surface": cells,
        "paired": {
            "source_pairs": len(separations),
            "mean_pair_separation": mean(separations),
            "truth_conditioned_pairs": truth_conditioned,
            "truth_conditioned_pair_rate": (
                truth_conditioned / len(separations)
            ),
        },
        "pooled_overall_score": None,
        "diagnostic_only": True,
        "optimizer_steps": 0,
        "promotion_authorized": False,
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
    backend: MarginBackend | None = None,
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    protocol, condition, rows, pair_audit = validate_config(
        config, repo_root=repo_root
    )
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    active = backend or TransformersMarginBackend(
        config=config,
        repo_root=repo_root,
    )
    samples = [dict(active.score_row(row)) for row in rows]
    for sample in samples:
        for field in (
            "teacher_forced_gold_nll",
            "first_token_margin_yes_minus_no",
            "sequence_margin_yes_minus_no",
            "gold_signed_margin",
        ):
            if not math.isfinite(float(sample[field])):
                raise MarginError("margin sample is not finite")
    metrics = summarize_samples(
        samples,
        run_id=str(config["run_id"]),
        condition_id=str(config["condition_id"]),
    )
    output_dir = repo_root / str(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    metrics_path = output_dir / "metrics.json"
    audit_path = output_dir / "pair_audit.json"
    samples_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in samples
        ),
        encoding="utf-8",
    )
    _write_json(metrics_path, metrics)
    _write_json(audit_path, pair_audit)
    adapter = config["model"]["adapter"]
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "run_id": config["run_id"],
        "condition_id": config["condition_id"],
        "protocol_id": PROTOCOL_ID,
        "condition_role": condition["role"],
        "config_sha256": _sha256(config_path),
        "protocol_sha256": _sha256(repo_root / PROTOCOL_PATH),
        "train_sha256": _sha256(repo_root / TRAIN_PATH),
        "adapter_sha256": adapter.get("sha256"),
        "samples_sha256": _sha256(samples_path),
        "metrics_sha256": _sha256(metrics_path),
        "pair_audit_sha256": _sha256(audit_path),
        "runtime": dict(active.runtime_receipt()),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "model_updates": 0,
        "status": "pass",
    }
    receipt_path = output_dir / "run_receipt.json"
    _write_json(receipt_path, receipt)
    return {
        "metrics": metrics,
        "pair_audit": pair_audit,
        "receipt": receipt,
    }


def validate_only(config_path: Path, repo_root: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    _protocol, condition, rows, pair_audit = validate_config(
        config, repo_root=repo_root
    )
    return {
        "schema": CONFIG_SCHEMA,
        "run_id": config["run_id"],
        "condition_id": config["condition_id"],
        "condition_role": condition["role"],
        "boolean_rows": len(rows),
        "source_pairs": pair_audit["source_pairs"],
        "model_invocations_completed": 0,
        "optimizer_steps": 0,
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
