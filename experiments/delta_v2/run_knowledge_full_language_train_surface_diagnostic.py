from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2.run_knowledge_eval import (
    MODAL_APP_PATH,
    RUNTIME_PACKAGES,
    _load_yaml,
    _mapping,
    _relative_path,
    _sha256,
    serialize_jsonl,
)
from experiments.delta_v2.run_knowledge_full_language_stability_eval import (
    CHECKPOINT_PATH,
    CHECKPOINT_SHA256,
    FullCheckpointBackend,
)
from experiments.delta_v2.train_knowledge_full_language_sft import (
    RUN_ID as TRAINING_RUN_ID,
    _schedule_sha256,
    validate_config as validate_training_config,
)
from experiments.delta_v2.train_knowledge_stability_choice_format_v2 import (
    _engine_rows,
)
from experiments.delta_v2.train_knowledge_stability_replay import (
    _normalize_replay_rows,
)


CONFIG_SCHEMA = "delta.knowledge_full_language_train_surface_config.v1"
PROTOCOL_SCHEMA = "delta.knowledge_full_language_train_surface_protocol.v1"
PROTOCOL_ID = "delta-v2-knowledge-full-language-train-surface-v1"
CONDITION_ID = "f2_full_language_train_surface_diagnostic"
RUN_ID = "delta-v2-f2-full-language-train-surface-diagnostic-modal-v1"
PROTOCOL_PATH = Path(
    "experiments/delta_v2/knowledge_full_language_train_surface_protocol.yaml"
)
TRAINING_CONFIG_PATH = Path(
    "configs/experiments/delta_v2/f1_full_language_sft_qwen35_08b_modal_v1.yaml"
)
STABILITY_RUN_ID = "delta-v2-f1-full-language-stability-eval-modal-v1"
TRAINING_RECEIPT_SHA256 = "d2de9cf7eeafac0ba54fad077dcaa8a8cb16422e4d98450d447bdeea17f263c1"
TRAINING_METRICS_SHA256 = "f53ed0cfe52aae73411bbb2b33a9789bc936f413f1e872c4e07a84d9381cfeaa"
STABILITY_METRICS_SHA256 = "cfb54dbb3f7010c5ca6ab4be55142a27deef31e8df2324a6e7a64a0512f25d69"
STABILITY_SAMPLES_SHA256 = "212e5fefeeabba78398306447ec6b92639ee5c5c836212da275ce752a3370f13"
SCHEDULE_SHA256 = "7bfb23a53e2c1f58c9cb9aa5f5a18dbeea1ea66ef1ef84204e43f968a4da6427"
EXPECTED_SURFACES = {
    "exact_recall": 36,
    "verify_true": 123,
    "verify_false": 123,
}


class FullLanguageSurfaceError(ValueError):
    """The frozen full-language training-surface diagnostic changed."""


def _training_rows(
    data: tuple[list[Mapping[str, Any]], ...],
    schedule: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    acquisition_rows, _acq_pairs, _dev, stability_rows, _pairs, _eval = data
    combined = [
        *acquisition_rows,
        *_normalize_replay_rows(_engine_rows(stability_rows)),
    ]
    rows_by_id = {str(row["row_id"]): row for row in combined}
    ordered_ids: list[str] = []
    for unit in schedule:
        if unit["kind"] == "recall":
            ordered_ids.append(str(unit["recall_row_id"]))
        elif unit["kind"] == "boolean_pair":
            ordered_ids.extend(
                [str(unit["positive_row_id"]), str(unit["negative_row_id"])]
            )
        else:
            raise FullLanguageSurfaceError("training schedule unit changed")
    distinct_ids = list(dict.fromkeys(ordered_ids))
    if any(row_id not in rows_by_id for row_id in distinct_ids):
        raise FullLanguageSurfaceError("training schedule references unknown row")
    return [rows_by_id[row_id] for row_id in distinct_ids]


def validate_config(
    config: Mapping[str, Any], *, repo_root: Path
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("run_id") != RUN_ID
        or config.get("experiment_id") != "delta_v2"
        or config.get("condition_id") != CONDITION_ID
        or config.get("task") != "eval"
    ):
        raise FullLanguageSurfaceError("diagnostic identity changed")
    binding = _mapping(config.get("protocol"), "protocol")
    protocol_path = repo_root / _relative_path(binding.get("path"), "protocol.path")
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or protocol_path != repo_root / PROTOCOL_PATH
        or _sha256(protocol_path) != binding.get("sha256")
    ):
        raise FullLanguageSurfaceError("diagnostic protocol binding changed")
    protocol = _load_yaml(protocol_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("condition_id") != CONDITION_ID
        or protocol.get("state") != "preregistered-for-diagnostic"
    ):
        raise FullLanguageSurfaceError("diagnostic protocol changed")
    trigger = dict(_mapping(protocol.get("decision_trigger"), "decision_trigger"))
    if trigger != {
        "training_run_id": TRAINING_RUN_ID,
        "training_receipt_sha256": TRAINING_RECEIPT_SHA256,
        "training_metrics_sha256": TRAINING_METRICS_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "stability_eval_run_id": STABILITY_RUN_ID,
        "stability_metrics_sha256": STABILITY_METRICS_SHA256,
        "stability_samples_sha256": STABILITY_SAMPLES_SHA256,
        "observed_stability": {
            "choice": "11/15", "boolean_true": "13/15",
            "boolean_false": "10/15", "recall": "0/15", "gate": "fail",
        },
    }:
        raise FullLanguageSurfaceError("diagnostic trigger changed")
    if (
        _sha256(repo_root / "runs" / TRAINING_RUN_ID / "run_receipt.json")
        != TRAINING_RECEIPT_SHA256
        or _sha256(repo_root / "runs" / TRAINING_RUN_ID / "train_metrics.json")
        != TRAINING_METRICS_SHA256
        or _sha256(repo_root / "runs" / STABILITY_RUN_ID / "metrics.json")
        != STABILITY_METRICS_SHA256
        or _sha256(repo_root / "runs" / STABILITY_RUN_ID / "samples.jsonl")
        != STABILITY_SAMPLES_SHA256
    ):
        raise FullLanguageSurfaceError("diagnostic evidence changed")

    training_block = dict(_mapping(config.get("training_contract"), "training_contract"))
    if training_block != {"config_path": str(TRAINING_CONFIG_PATH)}:
        raise FullLanguageSurfaceError("training contract binding changed")
    training_config = _load_yaml(repo_root / TRAINING_CONFIG_PATH)
    _training_protocol, data, schedule = validate_training_config(
        training_config, repo_root=repo_root
    )
    if _schedule_sha256(schedule) != SCHEDULE_SHA256:
        raise FullLanguageSurfaceError("training schedule changed")
    rows = _training_rows(data, schedule)
    counts = Counter(str(row["surface"]) for row in rows)
    if len(rows) != 282 or counts != Counter(EXPECTED_SURFACES):
        raise FullLanguageSurfaceError("diagnostic surface changed")
    if any(
        not isinstance(row.get("messages"), list)
        or len(row["messages"]) != 2
        or row["messages"][0].get("role") != "user"
        or row["messages"][1].get("role") != "assistant"
        for row in rows
    ):
        raise FullLanguageSurfaceError("diagnostic row contract changed")
    if dict(_mapping(protocol.get("surface"), "surface")) != {
        "authority": "exact-rows-referenced-by-frozen-training-schedule",
        "schedule_sha256": SCHEDULE_SHA256,
        "source_claims": 36,
        "distinct_rows": 282,
        "row_counts": EXPECTED_SURFACES,
        "prompt_source": "messages[0]",
        "gold_source": "messages[1]",
        "mutation": "forbidden",
    }:
        raise FullLanguageSurfaceError("protocol surface changed")
    if dict(_mapping(protocol.get("generation"), "generation")) != {
        "method": "greedy", "do_sample": False, "max_new_tokens": 256,
        "repeats": 2, "seed": 20260730,
        "repeat_invariant": "byte-identical-raw-continuation",
    }:
        raise FullLanguageSurfaceError("diagnostic generation changed")
    if dict(_mapping(protocol.get("scoring"), "scoring")) != {
        "authority": "frozen-training-row-assistant-answer",
        "method": "exact-stripped-string", "repair": "forbidden",
        "pooled_overall_score": "forbidden",
    }:
        raise FullLanguageSurfaceError("diagnostic scoring changed")
    if dict(_mapping(protocol.get("diagnostic_gates"), "diagnostic_gates")) != {
        "exact_recall_accuracy_minimum": 0.80,
        "verify_true_accuracy_minimum": 0.95,
        "verify_false_accuracy_minimum": 0.95,
        "all_required_for_training_surface_fit": True,
    }:
        raise FullLanguageSurfaceError("diagnostic gates changed")
    model = dict(_mapping(config.get("model"), "model"))
    if model != {
        "kind": "full_language_checkpoint", "run_id": TRAINING_RUN_ID,
        "path": str(CHECKPOINT_PATH), "sha256": CHECKPOINT_SHA256,
        "quantization": "none",
    }:
        raise FullLanguageSurfaceError("diagnostic model changed")
    runtime = _mapping(config.get("runtime"), "runtime")
    if (
        runtime.get("packages") != RUNTIME_PACKAGES
        or runtime.get("device") != "cuda"
        or runtime.get("dtype") != "bfloat16"
        or runtime.get("attention_implementation") != "eager"
        or runtime.get("deterministic_algorithms") is not True
    ):
        raise FullLanguageSurfaceError("diagnostic runtime changed")
    if config.get("eval") != {
        "module": "experiments.delta_v2.run_knowledge_full_language_train_surface_diagnostic"
    }:
        raise FullLanguageSurfaceError("diagnostic module changed")
    if _relative_path(config["output"]["dir"], "output.dir") != Path("runs") / RUN_ID:
        raise FullLanguageSurfaceError("diagnostic output changed")
    if config.get("modal") != {
        "app_path": MODAL_APP_PATH, "gpu": "A10G", "timeout_seconds": 3600,
    }:
        raise FullLanguageSurfaceError("diagnostic Modal binding changed")
    return protocol, rows


def build_requests(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    requests = []
    for row in rows:
        for repeat_index in (1, 2):
            identity = f"{RUN_ID}\0{row['row_id']}\0{repeat_index}"
            requests.append({
                "schema": "delta.knowledge_full_language_train_surface_request.v1",
                "request_id": hashlib.sha256(identity.encode()).hexdigest(),
                "row_id": row["row_id"], "surface": row["surface"],
                "repeat_index": repeat_index,
                "model_input": {"messages": [dict(row["messages"][0])]},
                "generation": {
                    "do_sample": False,
                    "max_new_tokens": protocol["generation"]["max_new_tokens"],
                    "seed": protocol["generation"]["seed"],
                },
            })
    return requests


def score(
    rows: Sequence[Mapping[str, Any]], outputs: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    row_by_id = {str(row["row_id"]): row for row in rows}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for output in outputs:
        grouped[str(output["row_id"])].append(output)
    if set(grouped) != set(row_by_id):
        raise FullLanguageSurfaceError("diagnostic output coverage changed")
    counts: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for row_id, row in row_by_id.items():
        repeats = sorted(grouped[row_id], key=lambda item: item["repeat_index"])
        if len(repeats) != 2 or repeats[0]["raw_response"] != repeats[1]["raw_response"]:
            raise FullLanguageSurfaceError("diagnostic repeats changed")
        surface = str(row["surface"])
        counts[surface] += 1
        correct[surface] += int(
            str(repeats[0]["raw_response"]).strip()
            == str(row["messages"][1]["content"])
        )
    cells = {
        surface: {
            "items": counts[surface], "correct": correct[surface],
            "exact_accuracy": correct[surface] / counts[surface],
        }
        for surface in EXPECTED_SURFACES
    }
    gates = protocol["diagnostic_gates"]
    passed = (
        cells["exact_recall"]["exact_accuracy"]
        >= gates["exact_recall_accuracy_minimum"]
        and cells["verify_true"]["exact_accuracy"]
        >= gates["verify_true_accuracy_minimum"]
        and cells["verify_false"]["exact_accuracy"]
        >= gates["verify_false_accuracy_minimum"]
    )
    return {
        "schema": "delta.knowledge_full_language_train_surface_metrics.v1",
        "run_id": RUN_ID, "condition_id": CONDITION_ID, "status": "pass",
        "rows": 282, "model_outputs": 564,
        "deterministic_repeats": True, "pooled_overall_accuracy": None,
        "cells": cells, "training_surface_fit": passed,
        "diagnosis": (
            "heldout-transfer-or-calibration-failure" if passed
            else "optimization-or-surface-fit-failure"
        ),
        "diagnostic_only": True, "promotion_authorized": False,
    }


def execute(*, config_path: Path, repo_root: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    protocol, rows = validate_config(config, repo_root=repo_root)
    requests = build_requests(rows, protocol)
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    active = FullCheckpointBackend(
        config=config, protocol=protocol, repo_root=repo_root
    )
    outputs = []
    for request in requests:
        raw, finish = active.generate(request)
        outputs.append({
            "schema": "delta.knowledge_full_language_train_surface_output.v1",
            "request_id": request["request_id"], "row_id": request["row_id"],
            "surface": request["surface"],
            "repeat_index": request["repeat_index"],
            "raw_response": raw, "finish_reason": finish,
        })
    metrics = score(rows, outputs, protocol)
    output_dir = repo_root / config["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    samples_path.write_bytes(serialize_jsonl(outputs))
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema": "delta.knowledge_full_language_train_surface_receipt.v1",
        "run_id": RUN_ID, "condition_id": CONDITION_ID,
        "protocol_id": PROTOCOL_ID, "training_run_id": TRAINING_RUN_ID,
        "stability_eval_run_id": STABILITY_RUN_ID,
        "config_sha256": _sha256(config_path),
        "protocol_sha256": _sha256(repo_root / PROTOCOL_PATH),
        "schedule_sha256": SCHEDULE_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "samples_sha256": _sha256(samples_path),
        "metrics_sha256": _sha256(metrics_path),
        "runtime": dict(active.runtime_receipt()),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "model_updates": 0, "checkpoint_mutated": False, "status": "pass",
        "training_surface_fit": metrics["training_surface_fit"],
        "promotion_authorized": False,
    }
    receipt_path = output_dir / "run_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
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
        protocol, rows = validate_config(config, repo_root=root)
        result = {
            "run_id": RUN_ID, "rows": len(rows),
            "requests": len(build_requests(rows, protocol)),
            "model_invocations_completed": 0, "status": "valid-unexecuted",
        }
    else:
        result = execute(config_path=path, repo_root=root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
