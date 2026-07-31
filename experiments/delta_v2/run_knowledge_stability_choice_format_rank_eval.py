from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2 import (
    run_knowledge_stability_choice_weight4_eval as base,
)
from experiments.delta_v2.run_knowledge_eval import _load_yaml


CONFIG_SCHEMA = "delta.knowledge_stability_choice_format_rank_eval_config.v1"
PROTOCOL_SCHEMA = "delta.knowledge_stability_replay_eval_protocol.v1"
PROTOCOL_ID = "delta-v2-knowledge-stability-choice-format-rank-eval-v1"
CONDITION_ID = "d17_stability_choice_format_rank_eval"
RUN_ID = "delta-v2-d17-stability-choice-format-rank-eval-modal-v1"
TRAINING_RUN_ID = (
    "delta-v2-d17-stability-choice-format-rank-lora-qwen35-08b-modal-v1"
)
PROTOCOL_PATH = Path(
    "experiments/delta_v2/"
    "knowledge_stability_choice_format_rank_eval_protocol.yaml"
)
EVAL_PATH = base.EVAL_PATH
ADAPTER_PATH = Path("runs") / TRAINING_RUN_ID / "adapter"
EXPECTED_KINDS = base.EXPECTED_KINDS
TRIGGER = {
    "training_run_id": TRAINING_RUN_ID,
    "training_launch_commit": "217f074",
    "training_receipt_sha256":
        "d51de680e374704bcccce254e7a6fb1bea3746aa0e748696ac8c1ba5a152adf7",
    "training_metrics_sha256":
        "10fee116efa6db26f10ad213378048b51b8118fa6b368780eca871005e4a774c",
    "adapter_sha256":
        "1c2dae08aac80b0cb8641cdf1abd2e7925743c224b1877353a7e6b374f4a7d53",
}
EVAL_MODULE = (
    "experiments.delta_v2."
    "run_knowledge_stability_choice_format_rank_eval"
)
BASE_EVAL_MODULE = (
    "experiments.delta_v2."
    "run_knowledge_stability_choice_weight4_eval"
)


def _bind_base() -> None:
    base.CONFIG_SCHEMA = CONFIG_SCHEMA
    base.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    base.PROTOCOL_ID = PROTOCOL_ID
    base.CONDITION_ID = CONDITION_ID
    base.RUN_ID = RUN_ID
    base.TRAINING_RUN_ID = TRAINING_RUN_ID
    base.PROTOCOL_PATH = PROTOCOL_PATH
    base.EVAL_PATH = EVAL_PATH
    base.ADAPTER_PATH = ADAPTER_PATH
    base.EXPECTED_KINDS = EXPECTED_KINDS
    base.TRIGGER = TRIGGER


def validate_config(
    config: Mapping[str, Any], *, repo_root: Path
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    _bind_base()
    adjusted = copy.deepcopy(dict(config))
    if adjusted.get("eval", {}).get("module") != EVAL_MODULE:
        raise ValueError("d17 evaluation module changed")
    adjusted["eval"]["module"] = BASE_EVAL_MODULE
    return base.validate_config(adjusted, repo_root=repo_root)


def _bind_runner() -> None:
    _bind_base()
    runner = base.runner
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
            "requests": len(base.runner.build_requests(protocol, probes)),
            "model_invocations_completed": 0,
            "status": "valid-unexecuted",
        }
    else:
        result = base.runner.execute(
            config_path=config_path, repo_root=repo_root
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
