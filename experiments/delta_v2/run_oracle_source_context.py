from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2.baseline_protocol import (
    EXPECTED_ENVIRONMENT_ID,
    EXPECTED_MODEL_REVISION,
    OUTPUT_SCHEMA,
    BaselineProtocolError,
    load_yaml,
    score_outputs,
    serialize_jsonl,
)
from experiments.delta_v2.oracle_source_context import (
    BASELINE_PROTOCOL_SHA256,
    FACT_DELTAS_PATH,
    FACT_DELTAS_SHA256,
    FEATURE_PROBE_PATH,
    FEATURE_PROBE_SHA256,
    PROTOCOL_ID,
    PROTOCOL_PATH,
    audit_oracle_requests,
    build_oracle_requests,
    scoring_protocol,
    validate_oracle_protocol,
)
from experiments.delta_v2.run_baseline import (
    TargetBackend,
    TransformersBackend,
)
from experiments.delta_v2.run_task_calibration import EXPECTED_RUNTIME


RUN_CONFIG_SCHEMA = "delta.target_model_run_config.v1"
RUN_RECEIPT_SCHEMA = "delta.oracle_source_context_run_receipt.v1"
EXPECTED_RUN_ID = "delta-v2-c2-oracle-source-context-qwen35-08b-modal-v1"
EXPECTED_PROTOCOL_SHA256 = (
    "c16136937d87bb6b51269ae39f6f4ab5f4e37fa6ead48aa49a0d9239294c322a"
)
EXPECTED_REQUEST_SHA256 = (
    "b4b39f0e325b502d3cf8fb7d5e7fb9329b92ce1301d26e4f2758619c91a98023"
)
EXPECTED_EVAL_PATH = Path(
    "data/experiments/delta_v2/acceptance_attempt_2/eval/"
    "acceptance_eval_items_v3.jsonl"
)
EXPECTED_EVAL_SHA256 = (
    "1125a7a5429cf847d16475259ffeb6578b8d4841c81f2f01d20d4065f4c188cb"
)
EXPECTED_OUTPUT_DIR = Path("runs") / EXPECTED_RUN_ID
EXPECTED_MODULE = "experiments.delta_v2.run_oracle_source_context"
EXPECTED_MODAL_APP = "experiments/delta_v2/modal_baseline_app.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BaselineProtocolError(f"{field} must be a mapping")
    return value


def _relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BaselineProtocolError(f"{field} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise BaselineProtocolError(f"{field} must stay within the repository")
    return path


def validate_run_config(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    list[Mapping[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, bool],
]:
    if (
        config.get("schema") != RUN_CONFIG_SCHEMA
        or config.get("run_id") != EXPECTED_RUN_ID
        or config.get("experiment_id") != "delta_v2"
        or config.get("condition_id") != "c2_oracle_source_context"
        or config.get("eval_environment_id") != EXPECTED_ENVIRONMENT_ID
        or config.get("task") != "eval"
    ):
        raise BaselineProtocolError("oracle run identity changed")

    binding = _mapping(config.get("protocol"), "protocol")
    protocol_path = _relative_path(binding.get("path"), "protocol.path")
    if binding != {
        "protocol_id": PROTOCOL_ID,
        "path": str(PROTOCOL_PATH),
        "sha256": EXPECTED_PROTOCOL_SHA256,
        "baseline_protocol_sha256": BASELINE_PROTOCOL_SHA256,
        "request_sha256": EXPECTED_REQUEST_SHA256,
    } or protocol_path != PROTOCOL_PATH:
        raise BaselineProtocolError("oracle protocol binding changed")
    absolute_protocol_path = repo_root / protocol_path
    if (
        not absolute_protocol_path.is_file()
        or _sha256(absolute_protocol_path) != EXPECTED_PROTOCOL_SHA256
    ):
        raise BaselineProtocolError("oracle protocol bytes changed")
    protocol = load_yaml(absolute_protocol_path)
    (
        baseline_protocol,
        items,
        evidence_by_eval,
        feature_evidence,
    ) = validate_oracle_protocol(protocol, repo_root=repo_root)

    if config.get("model") != {
        "repository": "Qwen/Qwen3.5-0.8B",
        "revision": EXPECTED_MODEL_REVISION,
        "adapter": "none",
        "quantization": "none",
    }:
        raise BaselineProtocolError("oracle model binding changed")
    if config.get("runtime") != EXPECTED_RUNTIME:
        raise BaselineProtocolError("oracle runtime binding changed")

    truth = _mapping(config.get("truth"), "truth")
    fact_path = _relative_path(
        truth.get("fact_deltas_path"),
        "truth.fact_deltas_path",
    )
    probe_path = _relative_path(
        truth.get("feature_probe_path"),
        "truth.feature_probe_path",
    )
    if (
        fact_path != FACT_DELTAS_PATH
        or truth.get("fact_deltas_sha256") != FACT_DELTAS_SHA256
        or probe_path != FEATURE_PROBE_PATH
        or truth.get("feature_probe_sha256") != FEATURE_PROBE_SHA256
        or not (repo_root / fact_path).is_file()
        or _sha256(repo_root / fact_path) != FACT_DELTAS_SHA256
        or not (repo_root / probe_path).is_file()
        or _sha256(repo_root / probe_path) != FEATURE_PROBE_SHA256
    ):
        raise BaselineProtocolError("oracle truth data changed")

    eval_config = _mapping(config.get("eval"), "eval")
    eval_path = _relative_path(eval_config.get("path"), "eval.path")
    if (
        eval_config.get("module") != EXPECTED_MODULE
        or eval_path != EXPECTED_EVAL_PATH
        or eval_config.get("sha256") != EXPECTED_EVAL_SHA256
        or not (repo_root / eval_path).is_file()
        or _sha256(repo_root / eval_path) != EXPECTED_EVAL_SHA256
    ):
        raise BaselineProtocolError("oracle acceptance data changed")
    if _relative_path(
        _mapping(config.get("output"), "output").get("dir"),
        "output.dir",
    ) != EXPECTED_OUTPUT_DIR:
        raise BaselineProtocolError("oracle output binding changed")
    if config.get("modal") != {
        "app_path": EXPECTED_MODAL_APP,
        "gpu": "A10G",
        "timeout_seconds": 3600,
    }:
        raise BaselineProtocolError("oracle Modal binding changed")
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
        "truth",
        "eval",
        "output",
        "modal",
    }:
        raise BaselineProtocolError("unexpected oracle run-config field")
    return (
        protocol,
        baseline_protocol,
        items,
        evidence_by_eval,
        feature_evidence,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prepare(
    *,
    config_path: Path,
    repo_root: Path,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    list[Mapping[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    config = load_yaml(config_path)
    (
        protocol,
        baseline_protocol,
        items,
        evidence_by_eval,
        feature_evidence,
    ) = validate_run_config(config, repo_root=repo_root)
    requests = build_oracle_requests(
        protocol,
        baseline_protocol=baseline_protocol,
        items=items,
        evidence_by_eval=evidence_by_eval,
        feature_evidence=feature_evidence,
    )
    request_audit = audit_oracle_requests(requests)
    request_hash = hashlib.sha256(serialize_jsonl(requests)).hexdigest()
    if (
        request_hash != EXPECTED_REQUEST_SHA256
        or request_hash
        != protocol["execution_boundary"]["expected_request_sha256"]
    ):
        raise BaselineProtocolError("oracle request bytes changed")
    return config, baseline_protocol, items, requests, request_audit


def execute_oracle_source_context(
    *,
    config_path: Path,
    repo_root: Path,
    backend: TargetBackend | None = None,
    output_dir_override: Path | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    config, baseline_protocol, items, requests, request_audit = _prepare(
        config_path=config_path,
        repo_root=repo_root,
    )
    adapted_protocol = scoring_protocol(baseline_protocol)
    active_backend = backend or TransformersBackend(
        config=config,
        protocol=adapted_protocol,
    )
    outputs: list[dict[str, Any]] = []
    for request in requests:
        raw_response, finish_reason = active_backend.generate(request)
        outputs.append(
            {
                "schema": OUTPUT_SCHEMA,
                "protocol_id": PROTOCOL_ID,
                "request_id": request["request_id"],
                "eval_id": request["eval_id"],
                "repeat_index": request["repeat_index"],
                "model_revision": EXPECTED_MODEL_REVISION,
                "raw_response": raw_response,
                "finish_reason": finish_reason,
            }
        )
    result_audit = score_outputs(
        protocol=adapted_protocol,
        items=items,
        outputs=outputs,
    )
    result_audit["adaptation"] = {
        "method": "oracle-routed-frozen-source-context-v1",
        "role": "upper-bound-control",
        "request_audit": request_audit,
        "production_retrieval_claim": False,
        "weight_updates": False,
    }
    output_bytes = serialize_jsonl(outputs)
    output_dir = (
        output_dir_override
        if output_dir_override is not None
        else repo_root / EXPECTED_OUTPUT_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples.jsonl").write_bytes(output_bytes)
    _write_json(output_dir / "metrics.json", result_audit)
    _write_json(output_dir / "request_audit.json", request_audit)
    receipt = {
        "schema": RUN_RECEIPT_SCHEMA,
        "run_id": EXPECTED_RUN_ID,
        "protocol_id": PROTOCOL_ID,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "eval_environment_id": EXPECTED_ENVIRONMENT_ID,
        "eval_items": len(items),
        "requests": len(requests),
        "request_sha256": EXPECTED_REQUEST_SHA256,
        "model_outputs": len(outputs),
        "samples_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "runtime": dict(active_backend.runtime_receipt()),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "status": "pass",
        "target_model_results_exist": True,
        "production_retrieval_claim": False,
        "weight_updates": False,
    }
    _write_json(output_dir / "run_receipt.json", receipt)
    return {"audit": result_audit, "receipt": receipt}


def validate_only(
    *,
    config_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    config, _, items, requests, request_audit = _prepare(
        config_path=config_path,
        repo_root=repo_root,
    )
    return {
        **request_audit,
        "run_id": config["run_id"],
        "config_sha256": _sha256(config_path),
        "request_sha256": EXPECTED_REQUEST_SHA256,
        "eval_items": len(items),
        "requests": len(requests),
        "model_invocations_completed": 0,
        "target_model_results_exist": False,
        "status": "valid-unexecuted",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run delta_v2 oracle source-context upper bound."
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
    if args.validate_only:
        result = validate_only(
            config_path=config_path,
            repo_root=repo_root,
        )
    else:
        result = execute_oracle_source_context(
            config_path=config_path,
            repo_root=repo_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
