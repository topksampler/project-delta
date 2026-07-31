from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2 import (
    run_knowledge_stability_replay50_eval as runner,
)
from experiments.delta_v2.run_knowledge_eval import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODAL_APP_PATH,
    RUNTIME_PACKAGES,
    _load_yaml,
    _mapping,
    _relative_path,
    _sha256,
)


CONFIG_SCHEMA = (
    "delta.knowledge_stability_choice_preserved_rank025_eval_config.v1"
)
PROTOCOL_SCHEMA = "delta.knowledge_stability_replay_eval_protocol.v1"
PROTOCOL_ID = (
    "delta-v2-knowledge-stability-choice-preserved-rank025-eval-v1"
)
CONDITION_ID = "d13_stability_choice_preserved_rank025_eval"
RUN_ID = (
    "delta-v2-d13-stability-choice-preserved-rank025-eval-modal-v1"
)
TRAINING_RUN_ID = (
    "delta-v2-d13-stability-choice-preserved-rank025-lora-"
    "qwen35-08b-modal-v1"
)
PROTOCOL_PATH = Path(
    "experiments/delta_v2/"
    "knowledge_stability_choice_preserved_rank025_eval_protocol.yaml"
)
EVAL_PATH = Path(
    "data/experiments/delta_v2/knowledge_stability_choice_preserved_v1/"
    "eval.jsonl"
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
    "training_launch_commit": "68f9478",
    "training_receipt_sha256": "a9892d79fbbf1325d956ba73f567ed203e84075b184436b3d70fdfc5cba0729c",
    "training_metrics_sha256": "ce5fe837554286cc84bd0f15037ebe8dbd8df28fedcd8a314f5b1db65aadcfc4",
    "adapter_sha256": "31fbf1199f948c0fc7d6e9d16f867f3004e5f9cd180f3aa480a075a70d1c3107",
}


class ChoicePreservedRank025EvalError(ValueError):
    """The d13 stability pre-gate cannot run outside its frozen receipt."""


def validate_config(
    config: Mapping[str, Any], *, repo_root: Path
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("run_id") != RUN_ID
        or config.get("condition_id") != CONDITION_ID
        or config.get("task") != "eval"
    ):
        raise ChoicePreservedRank025EvalError(
            "d13 evaluation identity changed"
        )
    binding = _mapping(config.get("protocol"), "protocol")
    protocol_path = repo_root / _relative_path(
        binding.get("path"), "protocol.path"
    )
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or protocol_path != repo_root / PROTOCOL_PATH
        or _sha256(protocol_path) != binding.get("sha256")
    ):
        raise ChoicePreservedRank025EvalError(
            "d13 protocol binding changed"
        )
    protocol = _load_yaml(protocol_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("condition_id") != CONDITION_ID
        or protocol.get("state") != "preregistered-for-pre-gate"
        or dict(_mapping(protocol.get("decision_trigger"), "trigger"))
        != TRIGGER
    ):
        raise ChoicePreservedRank025EvalError("d13 protocol changed")
    training_root = repo_root / "runs" / TRAINING_RUN_ID
    if (
        _sha256(training_root / "run_receipt.json")
        != TRIGGER["training_receipt_sha256"]
        or _sha256(training_root / "train_metrics.json")
        != TRIGGER["training_metrics_sha256"]
        or _sha256(repo_root / ADAPTER_PATH / "adapter_model.safetensors")
        != TRIGGER["adapter_sha256"]
    ):
        raise ChoicePreservedRank025EvalError("d13 evidence changed")
    dataset = _mapping(protocol.get("dataset"), "dataset")
    if dict(dataset) != {
        "path": str(EVAL_PATH),
        "sha256": "3b6def494dc791ff3d4d46bde569485656d4ab0a1084dd7dc059e863f96d0ae7",
        "sources": 15,
        "rows": 60,
        "training_overlap": "same-claim-new-wording",
        "exact_prompt_overlap_with_training": 0,
        "probe_counts": EXPECTED_KINDS,
    } or _sha256(repo_root / EVAL_PATH) != dataset["sha256"]:
        raise ChoicePreservedRank025EvalError("d13 dataset changed")
    if dict(_mapping(protocol.get("generation"), "generation")) != {
        "method": "greedy",
        "do_sample": False,
        "max_new_tokens": 256,
        "repeats": 2,
        "seed": 20260730,
        "repeat_invariant": "byte-identical-raw-continuation",
    }:
        raise ChoicePreservedRank025EvalError("d13 generation changed")
    if dict(_mapping(protocol.get("gates"), "gates")) != {
        "choice_parseable_minimum": 0.90,
        "boolean_parseable_minimum": 1.0,
        "boolean_true_accuracy_minimum": 0.80,
        "boolean_false_accuracy_minimum": 0.90,
        "recall": "advisory",
        "pooled_score": "forbidden",
        "all_gates_required": True,
    }:
        raise ChoicePreservedRank025EvalError("d13 gates changed")
    boundary = _mapping(protocol.get("execution_boundary"), "boundary")
    if (
        boundary.get("modal_evaluation_authorized") is not True
        or boundary.get("model_updates") != 0
        or boundary.get("adapter_mutation") != "forbidden"
        or boundary.get("acquisition_margin_authorized_only_after_pass")
        is not True
        or boundary.get("full_eval_authorized") is not False
        or boundary.get("verify_v2_authorized") is not False
        or boundary.get("promotion_authorized") is not False
    ):
        raise ChoicePreservedRank025EvalError("d13 boundary changed")
    if config.get("model") != {
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
        raise ChoicePreservedRank025EvalError("d13 model changed")
    runtime = config["runtime"]
    if (
        runtime.get("packages") != RUNTIME_PACKAGES
        or runtime.get("device") != "cuda"
        or runtime.get("dtype") != "bfloat16"
        or runtime.get("attention_implementation") != "eager"
        or runtime.get("deterministic_algorithms") is not True
    ):
        raise ChoicePreservedRank025EvalError("d13 runtime changed")
    if config.get("eval") != {
        "module":
            "experiments.delta_v2."
            "run_knowledge_stability_choice_preserved_rank025_eval",
        "path": str(EVAL_PATH),
        "sha256": dataset["sha256"],
    }:
        raise ChoicePreservedRank025EvalError("d13 eval changed")
    if _relative_path(config["output"]["dir"], "output.dir") != Path("runs") / RUN_ID:
        raise ChoicePreservedRank025EvalError("d13 output changed")
    if config.get("modal") != {
        "app_path": MODAL_APP_PATH,
        "gpu": "A10G",
        "timeout_seconds": 3600,
    }:
        raise ChoicePreservedRank025EvalError("d13 Modal changed")
    probes = runner.shared._load_jsonl(repo_root / EVAL_PATH)
    if (
        len(probes) != 60
        or Counter(str(row["probe_kind"]) for row in probes)
        != Counter(EXPECTED_KINDS)
        or {str(row["training_overlap"]) for row in probes}
        != {"same-claim-new-wording"}
    ):
        raise ChoicePreservedRank025EvalError("d13 probes changed")
    return protocol, probes


def _bind_runner() -> None:
    runner.CONFIG_SCHEMA = CONFIG_SCHEMA
    runner.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    runner.PROTOCOL_ID = PROTOCOL_ID
    runner.CONDITION_ID = CONDITION_ID
    runner.RUN_ID = RUN_ID
    runner.TRAINING_RUN_ID = TRAINING_RUN_ID
    runner.PROTOCOL_PATH = PROTOCOL_PATH
    runner.EVAL_PATH = EVAL_PATH
    runner.ADAPTER_PATH = ADAPTER_PATH
    runner.EXPECTED_KINDS = EXPECTED_KINDS
    runner.TRIGGER = TRIGGER
    runner.validate_config = validate_config


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
    _bind_runner()
    if args.validate_only:
        config = _load_yaml(config_path)
        protocol, probes = validate_config(config, repo_root=repo_root)
        result = {
            "run_id": RUN_ID,
            "items": len(probes),
            "requests": len(runner.build_requests(protocol, probes)),
            "model_invocations_completed": 0,
            "status": "valid-unexecuted",
        }
    else:
        result = runner.execute(
            config_path=config_path, repo_root=repo_root
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
