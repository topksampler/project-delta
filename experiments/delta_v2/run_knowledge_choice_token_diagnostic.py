from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
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
    KnowledgeTrainError,
    _load_jsonl,
    _load_yaml,
    _mapping,
    _model_inputs,
    _relative_path,
    _sha256,
    _tokenize_rows,
)


CONFIG_SCHEMA = "delta.knowledge_choice_token_diagnostic_config.v1"
PROTOCOL_SCHEMA = "delta.knowledge_choice_token_diagnostic_protocol.v1"
PROTOCOL_ID = "delta-v2-knowledge-choice-token-diagnostic-v1"
PROTOCOL_PATH = Path(
    "experiments/delta_v2/knowledge_choice_token_diagnostic_protocol.yaml"
)
DATA_PATH = Path(
    "data/experiments/delta_v2/knowledge_stability_choice_preserved_v1/"
    "eval.jsonl"
)
DATA_SHA256 = (
    "3b6def494dc791ff3d4d46bde569485656d4ab0a1084dd7dc059e863f96d0ae7"
)
SAMPLE_SCHEMA = "delta.knowledge_choice_token_diagnostic_sample.v1"
METRICS_SCHEMA = "delta.knowledge_choice_token_diagnostic_metrics.v1"
RECEIPT_SCHEMA = "delta.knowledge_choice_token_diagnostic_receipt.v1"
LETTERS = ("A", "B", "C", "D")
EXPECTED_CONDITIONS = {
    "d20_choice_token_d13": {
        "run_id": "delta-v2-d20-choice-token-diagnostic-d13-modal-v1",
        "role": "best-exact-choice-control",
        "training_run_id": (
            "delta-v2-d13-stability-choice-preserved-rank025-"
            "lora-qwen35-08b-modal-v1"
        ),
        "training_receipt_sha256": (
            "a9892d79fbbf1325d956ba73f567ed203e84075b184436b3d70fdfc5cba0729c"
        ),
        "training_metrics_sha256": (
            "ce5fe837554286cc84bd0f15037ebe8dbd8df28fedcd8a314f5b1db65aadcfc4"
        ),
        "adapter_sha256": (
            "31fbf1199f948c0fc7d6e9d16f867f3004e5f9cd180f3aa480a075a70d1c3107"
        ),
    },
    "d20_choice_token_d18": {
        "run_id": "delta-v2-d20-choice-token-diagnostic-d18-modal-v1",
        "role": "strong-sequence-format-pressure",
        "training_run_id": (
            "delta-v2-d18-stability-choice-format-weight4-"
            "lora-qwen35-08b-modal-v1"
        ),
        "training_receipt_sha256": (
            "4d50a701196ec15fa45d8f113a5b9feb08b0148ebdda0004c1eaa0518f9900b9"
        ),
        "training_metrics_sha256": (
            "43bbff026c475a5bb80ab4f6b56afbf8ecae4bb4e286ce97a7a7e086f3dffe0c"
        ),
        "adapter_sha256": (
            "7e9e614a83e1a20dfd8087617265094b73d326f97f85257bf9aaebb87659155c"
        ),
    },
    "d20_choice_token_d19": {
        "run_id": "delta-v2-d20-choice-token-diagnostic-d19-modal-v1",
        "role": "broad-residual-format-coverage",
        "training_run_id": (
            "delta-v2-d19-stability-choice-format-multineg-"
            "lora-qwen35-08b-modal-v1"
        ),
        "training_receipt_sha256": (
            "4b26f0f55dc626914a7cab06b5316ee4b2a02dc1290c563ce8b3c5e1ba6da554"
        ),
        "training_metrics_sha256": (
            "705837ea05a4ae3f841ed5d84e0bf2baa7347fd1d59c54e32b1a307bdcf6aadf"
        ),
        "adapter_sha256": (
            "68f60d78dcf10c555de46b89ed25347327b989902e6256c4f396ed7c475ba951"
        ),
    },
}


class ChoiceTokenError(KnowledgeTrainError):
    """The frozen token-local choice diagnostic cannot run safely."""


class ChoiceTokenBackend(Protocol):
    def score_row(self, row: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def runtime_receipt(self) -> Mapping[str, Any]: ...


def _condition_matrix(
    protocol: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    conditions = protocol.get("conditions")
    if not isinstance(conditions, list):
        raise ChoiceTokenError("choice-token condition matrix changed")
    observed = {
        str(_mapping(item, "condition")["condition_id"]): dict(item)
        for item in conditions
    }
    expected = {
        condition_id: {
            "condition_id": condition_id,
            **payload,
        }
        for condition_id, payload in EXPECTED_CONDITIONS.items()
    }
    if observed != expected:
        raise ChoiceTokenError("choice-token condition matrix changed")
    return observed


def validate_config(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], list[Mapping[str, Any]]]:
    condition_id = str(config.get("condition_id"))
    expected = EXPECTED_CONDITIONS.get(condition_id)
    if (
        expected is None
        or config.get("schema") != CONFIG_SCHEMA
        or config.get("run_id") != expected["run_id"]
        or config.get("experiment_id") != "delta_v2"
        or config.get("task") != "eval"
    ):
        raise ChoiceTokenError("choice-token config identity changed")
    binding = _mapping(config.get("protocol"), "protocol")
    protocol_path = repo_root / _relative_path(
        binding.get("path"), "protocol.path"
    )
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or protocol_path != repo_root / PROTOCOL_PATH
        or _sha256(protocol_path) != binding.get("sha256")
    ):
        raise ChoiceTokenError("choice-token protocol binding changed")
    protocol = _load_yaml(protocol_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("experiment_id") != "delta_v2"
        or protocol.get("state") != "preregistered-for-diagnostic"
        or protocol.get("decision_trigger") != {
            "stop_decision_commit": "42a15f1",
            "d18_eval_receipt_sha256": (
                "e6d5b1d1340a28067dde362670073cfa2ffefe6d008b941c11382d4503486a9d"
            ),
            "d18_eval_metrics_sha256": (
                "9ae2f6b7094e332ea405243b94a6ac9217b4609d7076a5ae83ea1411bee93db8"
            ),
            "d19_eval_receipt_sha256": (
                "3ec84cff44e5e11f5114fd7a2b53421a927cb4f06b6d3f62a8ac707450250a63"
            ),
            "d19_eval_metrics_sha256": (
                "4493a39a2d251eea369ec8836615841d3bf9a21053f68ca5e90b88b1dcce78e2"
            ),
        }
    ):
        raise ChoiceTokenError("choice-token protocol identity changed")
    condition = _condition_matrix(protocol)[condition_id]
    dataset = _mapping(protocol.get("dataset"), "dataset")
    if dict(dataset) != {
        "path": str(DATA_PATH),
        "sha256": DATA_SHA256,
        "rows": 60,
        "included_probe_kind": "choice",
        "included_rows": 15,
        "mutation": "forbidden",
    } or _sha256(repo_root / DATA_PATH) != DATA_SHA256:
        raise ChoiceTokenError("choice-token dataset changed")
    if protocol.get("interpretation") != {
        "content_signal_minimum_correct": 9,
        "termination_signal_minimum_argmax": 13,
        "advisory_only": True,
        "promotion_authorized": False,
    }:
        raise ChoiceTokenError("choice-token interpretation changed")
    model = _mapping(config.get("model"), "model")
    training_run_id = str(expected["training_run_id"])
    adapter_path = Path("runs") / training_run_id / "adapter"
    if dict(model) != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "adapter": {
            "kind": "lora",
            "run_id": training_run_id,
            "path": str(adapter_path),
            "sha256": expected["adapter_sha256"],
        },
        "quantization": "none",
    }:
        raise ChoiceTokenError("choice-token model changed")
    evidence_root = repo_root / "runs" / training_run_id
    if (
        _sha256(evidence_root / "run_receipt.json")
        != expected["training_receipt_sha256"]
        or _sha256(evidence_root / "train_metrics.json")
        != expected["training_metrics_sha256"]
        or _sha256(repo_root / adapter_path / "adapter_model.safetensors")
        != expected["adapter_sha256"]
    ):
        raise ChoiceTokenError("choice-token training evidence changed")
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
        raise ChoiceTokenError("choice-token runtime changed")
    if config.get("eval") != {
        "module":
            "experiments.delta_v2.run_knowledge_choice_token_diagnostic",
        "path": str(DATA_PATH),
        "sha256": DATA_SHA256,
    }:
        raise ChoiceTokenError("choice-token eval changed")
    if _relative_path(config["output"]["dir"], "output.dir") != (
        Path("runs") / str(expected["run_id"])
    ):
        raise ChoiceTokenError("choice-token output changed")
    if dict(_mapping(config.get("modal"), "modal")) != {
        "app_path": MODAL_APP_PATH,
        "gpu": "A10G",
        "timeout_seconds": 3600,
    }:
        raise ChoiceTokenError("choice-token Modal changed")
    rows = _load_jsonl(repo_root / DATA_PATH)
    choice_rows = [row for row in rows if row.get("probe_kind") == "choice"]
    if (
        len(rows) != 60
        or len(choice_rows) != 15
        or {str(row.get("gold")) for row in choice_rows} != set(LETTERS)
        or any(
            not str(row.get("prompt", "")).endswith(
                "Answer A, B, C, or D only."
            )
            for row in choice_rows
        )
    ):
        raise ChoiceTokenError("choice-token probes changed")
    return protocol, condition, choice_rows


class TransformersChoiceTokenBackend:
    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        repo_root: Path,
    ) -> None:
        self._versions = validate_runtime(config)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        import torch
        from peft import PeftModel
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ChoiceTokenError("choice-token diagnostic needs one CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise ChoiceTokenError("choice-token diagnostic needs bfloat16")
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
            "row_id": row["probe_id"],
            "surface": "choice",
            "messages": [
                {"role": "user", "content": row["prompt"]},
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
        candidates = {
            letter: self._candidate(row, letter) for letter in LETTERS
        }
        starts = {}
        first_tokens = {}
        termination_tokens = {}
        prefix = None
        for letter, candidate in candidates.items():
            labels = list(candidate["labels"])
            active = [
                index for index, token in enumerate(labels) if token != -100
            ]
            if len(active) < 2:
                raise ChoiceTokenError("choice candidate lacks termination")
            start = active[0]
            starts[letter] = start
            first_tokens[letter] = int(labels[start])
            termination_tokens[letter] = int(labels[start + 1])
            observed_prefix = candidate["input_ids"][0, :start].tolist()
            if prefix is None:
                prefix = observed_prefix
            elif observed_prefix != prefix:
                raise ChoiceTokenError("choice candidate prefixes diverged")
        if (
            len(set(starts.values())) != 1
            or len(set(first_tokens.values())) != 4
            or len(set(termination_tokens.values())) != 1
        ):
            raise ChoiceTokenError("choice candidate token contract changed")

        outputs = {}
        sequence_log_probabilities = {}
        with torch.inference_mode():
            for letter, candidate in candidates.items():
                output = self._model(**_model_inputs(candidate, torch))
                outputs[letter] = output
                sequence_log_probabilities[letter] = (
                    -float(output.loss.detach().cpu())
                    * int(candidate["assistant_tokens"])
                )
        start = next(iter(starts.values()))
        reference = outputs["A"]
        first_logits = reference.logits[0, start - 1].float()
        first_scores = {
            letter: float(first_logits[token].detach().cpu())
            for letter, token in first_tokens.items()
        }
        forced_prediction = max(LETTERS, key=first_scores.__getitem__)
        sequence_prediction = max(
            LETTERS, key=sequence_log_probabilities.__getitem__
        )
        greedy_first_token = int(first_logits.argmax().detach().cpu())
        token_to_letter = {
            token: letter for letter, token in first_tokens.items()
        }
        greedy_first_letter = token_to_letter.get(greedy_first_token)
        gold = str(row["gold"])
        gold_alternatives = [
            score for letter, score in first_scores.items() if letter != gold
        ]
        gold_first_margin = first_scores[gold] - max(gold_alternatives)

        termination_token = termination_tokens[gold]

        def termination_stats(letter: str) -> tuple[float, bool, int]:
            logits = outputs[letter].logits[0, start].float()
            target = float(logits[termination_token].detach().cpu())
            competing = logits.clone()
            competing[termination_token] = -torch.inf
            best_other = float(competing.max().detach().cpu())
            greedy = int(logits.argmax().detach().cpu())
            return target - best_other, greedy == termination_token, greedy

        gold_term_margin, gold_term_argmax, gold_next_token = (
            termination_stats(gold)
        )
        selected_term_margin, selected_term_argmax, selected_next_token = (
            termination_stats(forced_prediction)
        )
        tokenizer = self._processor.tokenizer
        result = {
            "schema": SAMPLE_SCHEMA,
            "probe_id": row["probe_id"],
            "source_id": row["source_id"],
            "gold": gold,
            "first_token_ids": first_tokens,
            "first_token_scores": first_scores,
            "forced_prediction": forced_prediction,
            "forced_correct": forced_prediction == gold,
            "greedy_first_token_id": greedy_first_token,
            "greedy_first_token": tokenizer.convert_ids_to_tokens(
                greedy_first_token
            ),
            "greedy_first_letter": greedy_first_letter,
            "greedy_first_is_choice": greedy_first_letter is not None,
            "sequence_log_probabilities": sequence_log_probabilities,
            "sequence_prediction": sequence_prediction,
            "sequence_correct": sequence_prediction == gold,
            "gold_first_token_margin": gold_first_margin,
            "termination_token_id": termination_token,
            "termination_token": tokenizer.convert_ids_to_tokens(
                termination_token
            ),
            "gold_termination_margin": gold_term_margin,
            "gold_termination_argmax": gold_term_argmax,
            "gold_next_token_id": gold_next_token,
            "gold_next_token": tokenizer.convert_ids_to_tokens(
                gold_next_token
            ),
            "selected_termination_margin": selected_term_margin,
            "selected_termination_argmax": selected_term_argmax,
            "selected_next_token_id": selected_next_token,
            "selected_next_token": tokenizer.convert_ids_to_tokens(
                selected_next_token
            ),
        }
        del outputs, reference, first_logits
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
            "adapter_kind": "lora",
        }


def summarize_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    condition_id: str,
) -> dict[str, Any]:
    if len(samples) != 15:
        raise ChoiceTokenError("choice-token sample count changed")
    forced_correct = sum(bool(row["forced_correct"]) for row in samples)
    sequence_correct = sum(bool(row["sequence_correct"]) for row in samples)
    greedy_choice = sum(
        bool(row["greedy_first_is_choice"]) for row in samples
    )
    gold_term = sum(
        bool(row["gold_termination_argmax"]) for row in samples
    )
    selected_term = sum(
        bool(row["selected_termination_argmax"]) for row in samples
    )
    content_signal = forced_correct >= 9
    termination_signal = gold_term >= 13
    if content_signal and termination_signal:
        diagnosis = "content-and-termination-signals-present"
    elif content_signal:
        diagnosis = "content-signal-with-termination-defect"
    elif termination_signal:
        diagnosis = "termination-signal-with-content-defect"
    else:
        diagnosis = "content-and-termination-defects"
    return {
        "schema": METRICS_SCHEMA,
        "run_id": run_id,
        "condition_id": condition_id,
        "status": "pass",
        "items": 15,
        "forced_choice_correct": forced_correct,
        "forced_choice_accuracy": forced_correct / 15,
        "sequence_candidate_correct": sequence_correct,
        "sequence_candidate_accuracy": sequence_correct / 15,
        "greedy_first_is_choice": greedy_choice,
        "greedy_first_is_choice_rate": greedy_choice / 15,
        "gold_termination_argmax": gold_term,
        "gold_termination_argmax_rate": gold_term / 15,
        "selected_termination_argmax": selected_term,
        "selected_termination_argmax_rate": selected_term / 15,
        "mean_gold_first_token_margin": mean(
            float(row["gold_first_token_margin"]) for row in samples
        ),
        "mean_gold_termination_margin": mean(
            float(row["gold_termination_margin"]) for row in samples
        ),
        "mean_selected_termination_margin": mean(
            float(row["selected_termination_margin"]) for row in samples
        ),
        "content_signal": content_signal,
        "termination_signal": termination_signal,
        "diagnosis": diagnosis,
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
    backend: ChoiceTokenBackend | None = None,
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    _protocol, condition, rows = validate_config(
        config, repo_root=repo_root
    )
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    active = backend or TransformersChoiceTokenBackend(
        config=config, repo_root=repo_root
    )
    samples = [dict(active.score_row(row)) for row in rows]
    for sample in samples:
        for field in (
            "gold_first_token_margin",
            "gold_termination_margin",
            "selected_termination_margin",
        ):
            if not math.isfinite(float(sample[field])):
                raise ChoiceTokenError("choice-token margin is not finite")
    metrics = summarize_samples(
        samples,
        run_id=str(config["run_id"]),
        condition_id=str(config["condition_id"]),
    )
    output_dir = repo_root / str(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    metrics_path = output_dir / "metrics.json"
    samples_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in samples
        ),
        encoding="utf-8",
    )
    _write_json(metrics_path, metrics)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "run_id": config["run_id"],
        "condition_id": config["condition_id"],
        "condition_role": condition["role"],
        "protocol_id": PROTOCOL_ID,
        "config_sha256": _sha256(config_path),
        "protocol_sha256": _sha256(repo_root / PROTOCOL_PATH),
        "eval_sha256": _sha256(repo_root / DATA_PATH),
        "adapter_sha256": config["model"]["adapter"]["sha256"],
        "samples_sha256": _sha256(samples_path),
        "metrics_sha256": _sha256(metrics_path),
        "runtime": dict(active.runtime_receipt()),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "model_updates": 0,
        "optimizer_steps": 0,
        "promotion_authorized": False,
        "status": "pass",
    }
    receipt_path = output_dir / "run_receipt.json"
    _write_json(receipt_path, receipt)
    return {"metrics": metrics, "receipt": receipt}


def validate_only(config_path: Path, repo_root: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    _protocol, condition, rows = validate_config(
        config, repo_root=repo_root
    )
    return {
        "schema": CONFIG_SCHEMA,
        "run_id": config["run_id"],
        "condition_id": config["condition_id"],
        "condition_role": condition["role"],
        "choice_rows": len(rows),
        "candidate_completions_per_row": 4,
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
