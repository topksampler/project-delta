from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2.knowledge_stability_multisurface_weighted_lora import (
    train_multisurface_weighted_candidate,
)
from experiments.delta_v2.run_knowledge_eval import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODAL_APP_PATH,
    RUNTIME_PACKAGES,
)
from experiments.delta_v2.train_knowledge_lora import (
    KnowledgeTrainError,
    _load_yaml,
    _relative_path,
    _sha256,
)
from experiments.delta_v2.train_knowledge_stability_claim_covered import (
    _load_jsonl,
)
from experiments.delta_v2.train_knowledge_stability_replay import (
    ACQUISITION_PAIRS_PATH,
    ACQUISITION_TRAIN_PATH,
    DEV_PATH,
    OPTIMIZATION,
    _normalize_replay_rows,
)
from experiments.delta_v2.train_knowledge_stability_replay50 import (
    build_replay50_schedule,
)


PROTOCOL_SCHEMA = (
    "delta.knowledge_stability_choice_preserved_lora_protocol.v1"
)
CONFIG_SCHEMA = (
    "delta.knowledge_stability_choice_preserved_lora_config.v1"
)
PROTOCOL_ID = "delta-v2-knowledge-stability-choice-preserved-lora-v1"
CONDITION_ID = "d12_stability_choice_preserved_sft_control"
RUN_ID = (
    "delta-v2-d12-stability-choice-preserved-lora-qwen35-08b-modal-v1"
)
DATASET_ID = "delta-v2-knowledge-stability-choice-preserved-v1"
PROTOCOL_PATH = Path(
    "experiments/delta_v2/"
    "knowledge_stability_choice_preserved_lora_protocol.yaml"
)
D11_TRAIN = (
    "delta-v2-d11-stability-hard-negative-lora-qwen35-08b-modal-v1"
)
D11_EVAL = "delta-v2-d11-stability-hard-negative-eval-modal-v1"
EVIDENCE = {
    (D11_TRAIN, "run_receipt.json"):
        "656ba2de76177713cbeb02a355684a0310f7428f5c3edee5e6619fe24aff5dbc",
    (D11_TRAIN, "train_metrics.json"):
        "3fc85d6eaee54ff486f26444b99cab5770050993e6dd18050bb4cb42d547da45",
    (D11_EVAL, "run_receipt.json"):
        "2834f9816bdcd9806399da134a2b4622257010ce58283ced2a95dc2bf6b16a23",
    (D11_EVAL, "metrics.json"):
        "281f3e7bebf2b2bf5f07f3da6b1ac21e6f637988bc6302cb8388f07fe7ac7c4a",
}
STABILITY_TRAIN_PATH = Path(
    "data/experiments/delta_v2/knowledge_stability_choice_preserved_v1/"
    "train.jsonl"
)
STABILITY_PAIRS_PATH = Path(
    "data/experiments/delta_v2/knowledge_stability_choice_preserved_v1/"
    "pairs.jsonl"
)
STABILITY_EVAL_PATH = Path(
    "data/experiments/delta_v2/knowledge_stability_choice_preserved_v1/"
    "eval.jsonl"
)
DATA_BINDINGS = {
    "acquisition_train": (
        ACQUISITION_TRAIN_PATH,
        "5d1fced8285c18684464a84c12d659a6f68fd50cacabe6c2ef598230a8be0db9",
        147,
    ),
    "acquisition_pairs": (
        ACQUISITION_PAIRS_PATH,
        "0fc176cbb17e533591b7ac980285d3665a1cb062f26a091070bfbfebc9646819",
        63,
    ),
    "acquisition_dev": (
        DEV_PATH,
        "dc8dc048de89c6306d1950acbda2a5e67ace7a712a174eb68d2d22ed3e68cff9",
        21,
    ),
    "stability_train": (
        STABILITY_TRAIN_PATH,
        "2957d28fd6e2476e4acaba49b52fda5983dc65aaad4e125130fe1b58fd68a283",
        150,
    ),
    "stability_pairs": (
        STABILITY_PAIRS_PATH,
        "db6f70d354d74df30c14847061f2487de4446d60ce03e6263f87ab1474b02aad",
        60,
    ),
    "stability_eval": (
        STABILITY_EVAL_PATH,
        "3b6def494dc791ff3d4d46bde569485656d4ab0a1084dd7dc059e863f96d0ae7",
        60,
    ),
}


def _engine_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    surfaces = {
        "choice_preserved_select": "replay_recall",
        "claim_covered_recall": "replay_recall",
        "claim_covered_verify_true": "replay_verify_true",
        "claim_covered_verify_false": "replay_verify_false",
    }
    result = []
    for row in rows:
        item = dict(row)
        try:
            item["surface"] = surfaces[str(row["surface"])]
        except KeyError as exc:
            raise KnowledgeTrainError(
                "choice-preserved surface changed"
            ) from exc
        result.append(item)
    return result


def validate_config(
    config: Mapping[str, Any], *, repo_root: Path
) -> tuple[Mapping[str, Any], tuple[list[Mapping[str, Any]], ...]]:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("run_id") != RUN_ID
        or config.get("condition_id") != CONDITION_ID
        or config.get("task") != "train"
    ):
        raise KnowledgeTrainError("multisurface config identity changed")
    binding = config["protocol"]
    protocol_path = repo_root / _relative_path(
        binding["path"], "protocol.path"
    )
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or protocol_path != repo_root / PROTOCOL_PATH
        or _sha256(protocol_path) != binding.get("sha256")
    ):
        raise KnowledgeTrainError("multisurface protocol binding changed")
    protocol = _load_yaml(protocol_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("condition_id") != CONDITION_ID
        or protocol.get("state") != "preregistered-for-training"
    ):
        raise KnowledgeTrainError("multisurface protocol changed")
    for (run_id, name), expected in EVIDENCE.items():
        if _sha256(repo_root / "runs" / run_id / name) != expected:
            raise KnowledgeTrainError("multisurface evidence changed")
    loaded = []
    for name, (path, digest, rows) in DATA_BINDINGS.items():
        block = protocol["data"][name]
        if (
            block.get("path") != str(path)
            or block.get("sha256") != digest
            or _sha256(repo_root / path) != digest
        ):
            raise KnowledgeTrainError(f"multisurface {name} changed")
        values = _load_jsonl(repo_root / path)
        if len(values) != rows:
            raise KnowledgeTrainError(f"multisurface {name} rows changed")
        loaded.append(values)
    objective = protocol["objective"]
    if objective != {
        "objective_id": "generative-sft-plus-paired-ranking-v2",
        "recall_loss": "teacher-forced-assistant-token-cross-entropy",
        "boolean_generative_loss":
            "mean-positive-and-negative-gold-cross-entropy",
        "boolean_ranking_loss":
            "truth-conditioned-logistic-plus-pair-separation",
        "ranking_margin": 1.0,
        "ranking_weight": 0.5,
    } or config.get("objective") != objective:
        raise KnowledgeTrainError("multisurface objective changed")
    if config.get("model") != {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "adapter": "new-lora",
        "quantization": "none",
    }:
        raise KnowledgeTrainError("multisurface model changed")
    if config["runtime"].get("packages") != RUNTIME_PACKAGES:
        raise KnowledgeTrainError("multisurface runtime changed")
    if config.get("training") != {
        "module":
            "experiments.delta_v2."
            "train_knowledge_stability_choice_preserved",
        **OPTIMIZATION,
    }:
        raise KnowledgeTrainError("multisurface optimization changed")
    if _relative_path(config["output"]["dir"], "output.dir") != Path("runs") / RUN_ID:
        raise KnowledgeTrainError("multisurface output changed")
    if config.get("modal") != {
        "app_path": MODAL_APP_PATH,
        "gpu": "H100",
        "timeout_seconds": 7200,
    }:
        raise KnowledgeTrainError("multisurface Modal changed")
    return protocol, tuple(loaded)


def train(config_path: Path, repo_root: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    protocol, data = validate_config(config, repo_root=repo_root)
    acq_rows, acq_pairs, dev_rows, stable_rows, stable_pairs, _eval = data
    transformed = _engine_rows(stable_rows)
    schedule = build_replay50_schedule(
        acq_rows, acq_pairs, transformed, stable_pairs
    )
    combined = [*acq_rows, *_normalize_replay_rows(transformed)]
    training_protocol = {
        "objective": protocol["objective"],
        "optimization": OPTIMIZATION,
        "lora": config["lora"],
    }
    return train_multisurface_weighted_candidate(
        config_path=config_path,
        repo_root=repo_root,
        config=config,
        protocol=training_protocol,
        train_rows=combined,
        dev_rows=dev_rows,
        schedule=schedule,
        run_id=RUN_ID,
        condition_id=CONDITION_ID,
        protocol_id=PROTOCOL_ID,
        protocol_path=PROTOCOL_PATH,
        dataset_id=DATASET_ID,
        acquisition_train_path=ACQUISITION_TRAIN_PATH,
        acquisition_pairs_path=ACQUISITION_PAIRS_PATH,
        stability_train_path=STABILITY_TRAIN_PATH,
        stability_pairs_path=STABILITY_PAIRS_PATH,
        stability_eval_path=STABILITY_EVAL_PATH,
        dev_path=DEV_PATH,
    )


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
        _protocol, data = validate_config(config, repo_root=root)
        transformed = _engine_rows(data[3])
        schedule = build_replay50_schedule(
            data[0], data[1], transformed, data[4]
        )
        result = {
            "run_id": RUN_ID,
            "schedule_units": len(schedule),
            "schedule_sha256": hashlib.sha256(
                json.dumps(
                    schedule, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "model_invocations_completed": 0,
            "status": "valid-unexecuted",
        }
    else:
        result = train(path, root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
