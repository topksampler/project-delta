from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from experiments.delta_v2.knowledge_adaptation import (
    _display_name,
    _fact_probe_rows,
    _recall_prompt,
    _verify_prompt,
    canonical_json,
    knowledge_card,
)
from experiments.delta_v2.knowledge_stability_replay_data import (
    PAIR_SCHEMA,
    _assign_values,
    _equalize_pair_prompt_lengths,
    _row,
    _serialize_jsonl,
)
from experiments.delta_v2.validate_knowledge_stability_replay_design import (
    derive_partitions,
    validate_design,
)


PROTOCOL_PATH = Path(
    "experiments/delta_v2/"
    "knowledge_stability_claim_covered_data_protocol.yaml"
)
DESIGN_PATH = Path(
    "experiments/delta_v2/knowledge_stability_replay_design.yaml"
)
OUTPUT_ROOT = Path(
    "data/experiments/delta_v2/knowledge_stability_claim_covered_v1"
)
DATASET_ID = "delta-v2-knowledge-stability-claim-covered-v1"


class ClaimCoveredDataError(ValueError):
    """Claim-covered stability data failed its frozen audit."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ClaimCoveredDataError("protocol must be a mapping")
    return value


def _false_card(
    card: Mapping[str, Any], *, field: str, encoded: str
) -> Mapping[str, Any]:
    if field == "$card":
        value = json.loads(encoded)
    else:
        value = {**card, field: json.loads(encoded)}
    if not isinstance(value, Mapping) or set(value) != set(card):
        raise ClaimCoveredDataError("false card shape changed")
    return value


def validate_protocol(
    protocol: Mapping[str, Any], *, repo_root: Path
) -> None:
    if (
        protocol.get("schema")
        != "delta.knowledge_stability_claim_covered_data_protocol.v1"
        or protocol.get("protocol_id")
        != "delta-v2-knowledge-stability-claim-covered-data-v1"
        or protocol.get("dataset_id") != DATASET_ID
        or protocol.get("state") != "preregistered-for-build"
    ):
        raise ClaimCoveredDataError("data protocol identity changed")
    builder = protocol.get("builder")
    if not isinstance(builder, Mapping) or (
        builder.get("path")
        != "experiments/delta_v2/build_knowledge_stability_claim_covered.py"
        or _sha256(repo_root / str(builder["path"]))
        != builder.get("sha256")
    ):
        raise ClaimCoveredDataError("builder binding changed")
    parent = protocol.get("parent_partition")
    if not isinstance(parent, Mapping) or (
        parent.get("path") != str(DESIGN_PATH)
        or parent.get("sha256")
        != "301fc0de6e1961c67c40e1fec25a40444c301678b0b1b1a4e3dafbe6b5f99c01"
        or _sha256(repo_root / DESIGN_PATH) != parent.get("sha256")
    ):
        raise ClaimCoveredDataError("parent partition changed")
    if protocol.get("construction") != {
        "sources": 15,
        "source_partition": "stability_dev",
        "train_rows": 105,
        "recall_rows": 15,
        "boolean_pairs": 45,
        "positive_rows": 45,
        "negative_rows": 45,
        "eval_rows": 60,
        "eval_probes_per_source": [
            "choice",
            "boolean_true",
            "boolean_false",
            "recall",
        ],
        "source_overlap_train_eval": "required-same-claims",
        "exact_prompt_overlap_train_eval": 0,
        "eval_wording": "held-out",
    }:
        raise ClaimCoveredDataError("construction changed")
    if protocol.get("execution_boundary") != {
        "dataset_build_authorized": True,
        "model_invocations_completed": 0,
        "optimizer_steps_completed": 0,
        "modal_job_authorized": False,
        "promotion_authorized": False,
    }:
        raise ClaimCoveredDataError("data boundary changed")


def build(
    *, protocol_path: Path, repo_root: Path
) -> dict[str, Any]:
    protocol = _load_yaml(protocol_path)
    validate_protocol(protocol, repo_root=repo_root)
    design = _load_yaml(repo_root / DESIGN_PATH)
    validate_design(design, repo_root=repo_root)
    partitions, _blocked, _eligible = derive_partitions(
        design, repo_root=repo_root
    )
    records = {
        str(row["fact_id"]): row
        for row in partitions["stability_dev"]
    }
    source_ids = sorted(records)
    cards = {
        source_id: knowledge_card(records[source_id])
        for source_id in source_ids
    }
    config_ids = [
        source_id
        for source_id in source_ids
        if cards[source_id]["kind"] == "python_config_field"
    ]
    env_ids = [
        source_id
        for source_id in source_ids
        if cards[source_id]["kind"] == "python_environment_variable"
    ]
    assignments: dict[str, dict[str, tuple[str, str]]] = {
        "annotation_swap": _assign_values(
            source_ids,
            cards,
            field="annotation",
            salt="claim-covered-annotation-v1",
        ),
        "family_value_swap": {},
        "full_card_swap": {},
    }
    assignments["family_value_swap"].update(
        _assign_values(
            config_ids,
            cards,
            field="default",
            salt="claim-covered-default-v1",
        )
    )
    assignments["family_value_swap"].update(
        _assign_values(
            env_ids,
            cards,
            field="getter",
            salt="claim-covered-getter-v1",
        )
    )
    assignments["full_card_swap"].update(
        _assign_values(
            config_ids,
            cards,
            field="$card",
            salt="claim-covered-config-card-v1",
        )
    )
    assignments["full_card_swap"].update(
        _assign_values(
            env_ids,
            cards,
            field="$card",
            salt="claim-covered-env-card-v1",
        )
    )
    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for source_id in source_ids:
        record = records[source_id]
        card = cards[source_id]
        display = _display_name(record)
        rows.append(
            _row(
                source_id=source_id,
                surface="claim_covered_recall",
                prompt=_recall_prompt(display, variant="train"),
                answer=canonical_json(card),
            )
        )
        family_fields = {
            "annotation_swap": "annotation",
            "family_value_swap": (
                "default"
                if source_id in config_ids
                else "getter"
            ),
            "full_card_swap": "$card",
        }
        for family, field in family_fields.items():
            encoded, donor_id = assignments[family][source_id]
            false_card = _false_card(card, field=field, encoded=encoded)
            pair_id = (
                "stability-claim-pair:"
                + hashlib.sha256(
                    f"{DATASET_ID}\0{source_id}\0{family}".encode()
                ).hexdigest()
            )
            positive = _row(
                source_id=source_id,
                surface="claim_covered_verify_true",
                prompt=_verify_prompt(display, card, variant="train"),
                answer="yes",
                pair_id=pair_id,
                pair_role="positive",
                corruption_family=family,
                changed_field=field,
            )
            negative = _row(
                source_id=source_id,
                surface="claim_covered_verify_false",
                prompt=_verify_prompt(
                    display, false_card, variant="train"
                ),
                answer="no",
                pair_id=pair_id,
                pair_role="negative",
                corruption_family=family,
                changed_field=field,
                donor_source_id=donor_id,
            )
            _equalize_pair_prompt_lengths(positive, negative)
            rows.extend([positive, negative])
            pairs.append(
                {
                    "schema": PAIR_SCHEMA,
                    "pair_id": pair_id,
                    "source_id": source_id,
                    "source_kind": str(card["kind"]),
                    "corruption_family": family,
                    "changed_field": field,
                    "donor_source_id": donor_id,
                    "positive_row_id": positive["row_id"],
                    "negative_row_id": negative["row_id"],
                    "true_contract_sha256": hashlib.sha256(
                        canonical_json(card).encode()
                    ).hexdigest(),
                    "false_contract_sha256": hashlib.sha256(
                        canonical_json(false_card).encode()
                    ).hexdigest(),
                }
            )
    eval_rows: list[dict[str, Any]] = []
    for source_id in source_ids:
        eval_rows.extend(
            _fact_probe_rows(
                records[source_id],
                stratum="stability_claim_dev",
                training_overlap="same-claim-new-wording",
            )
        )
    rows.sort(key=lambda row: str(row["row_id"]))
    pairs.sort(key=lambda row: str(row["pair_id"]))
    eval_rows.sort(key=lambda row: str(row["probe_id"]))
    train_prompts = {
        str(row["messages"][0]["content"]) for row in rows
    }
    eval_prompts = {str(row["prompt"]) for row in eval_rows}
    train_sources = {str(row["source_id"]) for row in rows}
    eval_sources = {str(row["source_id"]) for row in eval_rows}
    if (
        len(rows) != 105
        or len(pairs) != 45
        or len(eval_rows) != 60
        or train_sources != eval_sources
        or train_prompts & eval_prompts
        or Counter(str(row["pair_role"]) for row in rows)
        != {str(None): 15, "positive": 45, "negative": 45}
    ):
        raise ClaimCoveredDataError("claim-covered audit failed")
    output = repo_root / OUTPUT_ROOT
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "train.jsonl"
    pairs_path = output / "pairs.jsonl"
    eval_path = output / "eval.jsonl"
    train_path.write_bytes(_serialize_jsonl(rows))
    pairs_path.write_bytes(_serialize_jsonl(pairs))
    eval_path.write_bytes(_serialize_jsonl(eval_rows))
    summary = {
        "schema": "delta.knowledge_stability_claim_covered_data_audit.v1",
        "dataset_id": DATASET_ID,
        "sources": 15,
        "train_rows": len(rows),
        "pairs": len(pairs),
        "eval_rows": len(eval_rows),
        "source_overlap_train_eval": len(train_sources & eval_sources),
        "exact_prompt_overlap_train_eval": len(
            train_prompts & eval_prompts
        ),
        "train_sha256": _sha256(train_path),
        "pairs_sha256": _sha256(pairs_path),
        "eval_sha256": _sha256(eval_path),
        "model_invocations_completed": 0,
        "optimizer_steps_completed": 0,
        "status": "pass",
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**summary, "summary_sha256": _sha256(summary_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    protocol_path = args.protocol
    if not protocol_path.is_absolute():
        protocol_path = repo_root / protocol_path
    result = build(protocol_path=protocol_path, repo_root=repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
