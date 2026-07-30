from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2.run_knowledge_eval import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
)
from experiments.delta_v2.run_knowledge_train_surface_diagnostic import (
    TRAIN_PATH,
    _load_jsonl,
    _load_yaml,
    _mapping,
    _relative,
    _sha256,
    execute_validated_diagnostic,
)
from experiments.delta_v2.train_knowledge_lora import validate_rows


RUN_ID = "delta-v2-c6-knowledge-lora-true-weighted-train-surface-modal-v1"
TRAIN_RUN_ID = (
    "delta-v2-c6-knowledge-lora-true-weighted-qwen35-08b-modal-v1"
)
PROTOCOL_ID = "delta-v2-knowledge-true-weighted-train-surface-v1"
PROTOCOL_PATH = Path(
    "experiments/delta_v2/"
    "knowledge_true_weighted_train_surface_diagnostic_protocol.yaml"
)
ADAPTER_SHA256 = (
    "8dae0e5cf7692542efc34f4fc836b1c96279bae680747edc368f646fdf938e36"
)


def validate_config(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    if (
        config.get("schema")
        != "delta.knowledge_train_surface_diagnostic_config.v1"
        or config.get("run_id") != RUN_ID
        or config.get("condition_id")
        != "c6_knowledge_lora_true_weighted_diagnostic"
        or config.get("task") != "eval"
    ):
        raise ValueError("c6 diagnostic identity changed")
    binding = _mapping(config.get("protocol"), "protocol")
    protocol_path = repo_root / _relative(
        binding.get("path"), "protocol.path"
    )
    if (
        binding.get("protocol_id") != PROTOCOL_ID
        or protocol_path != repo_root / PROTOCOL_PATH
        or _sha256(protocol_path) != binding.get("sha256")
    ):
        raise ValueError("c6 diagnostic protocol changed")
    protocol = _load_yaml(protocol_path)
    if (
        protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("state") != "preregistered-for-diagnostic"
        or protocol["trigger"]["training_run_id"] != TRAIN_RUN_ID
        or protocol["trigger"]["adapter_sha256"] != ADAPTER_SHA256
        or protocol["dataset"]["sha256"] != _sha256(repo_root / TRAIN_PATH)
        or protocol["dataset"]["mutation"] != "forbidden"
        or protocol["decision"]["full_frozen_eval_if_passes"] is not True
        or protocol["decision"]["promotion_authorized"] is not False
    ):
        raise ValueError("c6 diagnostic contract changed")
    rows = _load_jsonl(repo_root / TRAIN_PATH)
    validate_rows(rows, expected_rows=63, expected_surfaces={
        "exact_recall": 21,
        "verify_true": 21,
        "verify_false": 21,
    })
    model = config["model"]
    adapter = model["adapter"]
    adapter_path = _relative(adapter["path"], "adapter.path")
    if (
        model["repository"] != MODEL_REPOSITORY
        or model["revision"] != MODEL_REVISION
        or adapter["run_id"] != TRAIN_RUN_ID
        or adapter["sha256"] != ADAPTER_SHA256
        or adapter_path != Path("runs") / TRAIN_RUN_ID / "adapter"
        or _sha256(
            repo_root / adapter_path / "adapter_model.safetensors"
        )
        != ADAPTER_SHA256
    ):
        raise ValueError("c6 diagnostic adapter changed")
    if config["eval"]["module"] != (
        "experiments.delta_v2.run_true_weighted_train_surface_diagnostic"
    ):
        raise ValueError("c6 diagnostic module changed")
    if _relative(config["output"]["dir"], "output.dir") != (
        Path("runs") / RUN_ID
    ):
        raise ValueError("c6 diagnostic output changed")
    return protocol, rows


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
    config = _load_yaml(config_path)
    protocol, rows = validate_config(config, repo_root=repo_root)
    result = (
        {
            "run_id": RUN_ID,
            "rows": len(rows),
            "requests": len(rows) * 2,
            "model_invocations_completed": 0,
            "status": "valid-unexecuted",
        }
        if args.validate_only
        else execute_validated_diagnostic(
            config_path=config_path,
            repo_root=repo_root,
            config=config,
            protocol=protocol,
            rows=rows,
            run_id=RUN_ID,
            protocol_id=PROTOCOL_ID,
            protocol_path=PROTOCOL_PATH,
            train_path=TRAIN_PATH,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
