from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from experiments.delta_v2.knowledge_adaptation import (
    CONTRACT_ID,
    KnowledgeAdaptationError,
)


FREEZE_SCHEMA = "delta.knowledge_adaptation_freeze.v1"
DATASET_ID = "delta-v2-vllm-knowledge-adaptation-v1"


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KnowledgeAdaptationError(f"{field} must be a mapping")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise KnowledgeAdaptationError(f"{field} must be a path string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise KnowledgeAdaptationError(f"{field} must stay in the repository")
    return path


def validate_freeze(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    if (
        manifest.get("schema") != FREEZE_SCHEMA
        or manifest.get("dataset_id") != DATASET_ID
        or manifest.get("experiment_id") != "delta_v2"
        or manifest.get("parent_environment_id")
        != "delta-v2-vllm-source-build-v2"
        or manifest.get("target_revision") != "v0.26.0"
        or manifest.get("freeze_state") != "frozen"
    ):
        raise KnowledgeAdaptationError("knowledge freeze identity changed")

    model = _mapping(manifest.get("model_boundary"), "model_boundary")
    if (
        model.get("repository") != "Qwen/Qwen3.5-0.8B"
        or model.get("revision")
        != "2fc06364715b967f1860aea9cf38778875588b17"
        or model.get("base_baseline_required_before_training") is not True
        or model.get("permitted_weight_update") != "lora-only"
        or model.get("full_weight_fine_tuning") != "deferred"
    ):
        raise KnowledgeAdaptationError("model boundary changed")

    scoreboards = _mapping(manifest.get("scoreboards"), "scoreboards")
    acquisition = _mapping(scoreboards.get("acquisition"), "acquisition")
    retention = _mapping(scoreboards.get("retention"), "retention")
    feature = _mapping(
        scoreboards.get("feature_retention"),
        "feature_retention",
    )
    if (
        acquisition.get("sources") != 21
        or acquisition.get("probes") != 84
        or acquisition.get("training_relation")
        != "same-claim-new-surface"
        or retention.get("sources") != 21
        or retention.get("probes") != 84
        or retention.get("training_relation") != "source-disjoint"
        or feature.get("sources") != 1
        or feature.get("probes") != 1
        or feature.get("training_relation") != "source-disjoint"
        or scoreboards.get("pooled_overall_score") != "forbidden"
    ):
        raise KnowledgeAdaptationError("scoreboard boundary changed")

    leakage = _mapping(manifest.get("leakage_gate"), "leakage_gate")
    if dict(leakage) != {
        "exact_train_eval_prompt_overlap": 0,
        "original_eval_prompt_overlap": 0,
        "retention_source_overlap": 0,
        "feature_source_overlap": 0,
        "acquisition_source_overlap": 21,
        "acquisition_overlap_is_preregistered": True,
    }:
        raise KnowledgeAdaptationError("leakage gate changed")

    bindings = manifest.get("bindings")
    if not isinstance(bindings, list) or len(bindings) != 8:
        raise KnowledgeAdaptationError("freeze bindings changed")
    seen: set[str] = set()
    checked: dict[str, str] = {}
    for index, raw in enumerate(bindings):
        binding = _mapping(raw, f"bindings[{index}]")
        binding_id = binding.get("id")
        if not isinstance(binding_id, str) or not binding_id:
            raise KnowledgeAdaptationError("binding id changed")
        if binding_id in seen:
            raise KnowledgeAdaptationError(f"duplicate binding: {binding_id}")
        seen.add(binding_id)
        path = repo_root / _relative_path(
            binding.get("path"),
            f"bindings[{index}].path",
        )
        expected = binding.get("sha256")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or not path.is_file()
            or _sha256(path) != expected
        ):
            raise KnowledgeAdaptationError(
                f"freeze binding bytes changed: {binding_id}"
            )
        checked[binding_id] = expected

    summary_path = repo_root / next(
        Path(str(binding["path"]))
        for binding in bindings
        if binding["id"] == "build_summary"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("contract_id") != CONTRACT_ID
        or summary.get("status") != "pass"
        or summary.get("outputs", {}).get("train", {}).get("rows") != 63
        or summary.get("outputs", {}).get("dev", {}).get("rows") != 21
        or summary.get("outputs", {}).get("eval", {}).get("rows") != 169
        or summary.get("evaluation", {}).get("exact_prompt_overlap") != 0
        or summary.get("evaluation", {}).get("retention_source_overlap") != 0
        or summary.get("evaluation", {}).get("pooled_overall_score") is not None
    ):
        raise KnowledgeAdaptationError("frozen build summary changed")

    execution = _mapping(
        manifest.get("execution_boundary"),
        "execution_boundary",
    )
    if dict(execution) != {
        "base_knowledge_runs": 0,
        "lora_training_runs": 0,
        "adapter_eval_runs": 0,
        "dispatch_authorized": True,
        "promotion_authorized": False,
    }:
        raise KnowledgeAdaptationError("execution boundary changed")

    return {
        "schema": "delta.knowledge_adaptation_freeze_validation.v1",
        "dataset_id": DATASET_ID,
        "bindings_checked": len(checked),
        "train_rows": 63,
        "dev_rows": 21,
        "eval_rows": 169,
        "status": "pass",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the frozen delta_v2 knowledge-adaptation data."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "experiments/delta_v2/knowledge_adaptation_freeze.yaml"
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    result = validate_freeze(
        _mapping(manifest, "manifest"),
        repo_root=repo_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
