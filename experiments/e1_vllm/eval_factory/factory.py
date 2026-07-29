"""Build deterministic version-delta claims and pre-freeze probes."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import yaml

from .cli_flags import extract_cli_flags_with_rejections, semantic_contract
from .entities import (
    extract_config_fields,
    extract_env_contracts,
    extract_public_exports,
)

SCHEMA = "delta.eval_environment.v1"
SPLIT_SEED = "e1_eval_factory_v1:split:20260728"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split(claim_id: str) -> str:
    bucket = int(
        hashlib.sha256(f"{SPLIT_SEED}:{claim_id}".encode()).hexdigest()[:8],
        16,
    ) % 100
    if bucket < 60:
        return "train"
    if bucket < 80:
        return "dev"
    return "eval"


def _split_rank(claim_id: str) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}:{claim_id}".encode()).hexdigest()


def _assign_stratified_splits(claims: list[dict]) -> None:
    """Keep stable 60/20/20; allocate scarce deltas 40/20/40 per family/status."""
    delta_strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for claim in claims:
        if claim["status"] == "stable":
            claim["split"] = _split(claim["claim_id"])
        else:
            delta_strata[(claim["entity_type"], claim["status"])].append(claim)
    for rows in delta_strata.values():
        ordered = sorted(rows, key=lambda row: _split_rank(row["claim_id"]))
        n_train = round(len(ordered) * 0.4)
        n_dev = round(len(ordered) * 0.2)
        for index, claim in enumerate(ordered):
            if index < n_train:
                claim["split"] = "train"
            elif index < n_train + n_dev:
                claim["split"] = "dev"
            else:
                claim["split"] = "eval"


def _snapshot_spec(snapshots_path: Path, alias: str) -> dict:
    payload = yaml.safe_load(snapshots_path.read_text(encoding="utf-8"))
    try:
        spec = payload["vllm_docs"][alias]
    except KeyError as exc:
        raise KeyError(f"unknown snapshot alias {alias!r}") from exc
    return {
        "alias": alias,
        "version": str(spec["version"]),
        "git_tag": str(spec["git_tag"]),
    }


def _snapshot_dir(repo_root: Path, version: str) -> Path:
    return (
        repo_root
        / "data"
        / "experiments"
        / "e1_vllm"
        / "snapshots"
        / f"v{version}"
    )


def _meta(snapshot: Path) -> dict:
    path = snapshot / "snapshot_meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _group(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["entity_type"], row["entity"])].append(row)
    return dict(grouped)


def _value(rows: list[dict]) -> list[dict]:
    return [
        {
            "scope": row["scope"],
            "contract": row["contract"],
        }
        for row in rows
    ]


def _evidence(rows: list[dict]) -> list[dict]:
    return [row["evidence"] for row in rows]


def _status(before: list[dict], after: list[dict]) -> str:
    if not before:
        return "added"
    if not after:
        return "removed"
    before_contracts = sorted(semantic_contract(row) for row in before)
    after_contracts = sorted(semantic_contract(row) for row in after)
    return "stable" if before_contracts == after_contracts else "changed"


def build_claims(
    *,
    before_rows: list[dict],
    after_rows: list[dict],
    before_spec: dict,
    after_spec: dict,
) -> list[dict]:
    before_by_entity = _group(before_rows)
    after_by_entity = _group(after_rows)
    claims: list[dict] = []
    for entity_type, entity in sorted(set(before_by_entity) | set(after_by_entity)):
        key = (entity_type, entity)
        before = before_by_entity.get(key, [])
        after = after_by_entity.get(key, [])
        claim_id = f"vllm:{entity_type}:{entity}"
        exemplar = (before or after)[0]
        claims.append(
            {
                "schema": SCHEMA,
                "claim_id": claim_id,
                "entity_type": entity_type,
                "entity": entity,
                "display_entity": exemplar.get("display_entity", entity),
                "source_before": before_spec["git_tag"],
                "source_after": after_spec["git_tag"],
                "status": _status(before, after),
                "value_before": {"exists": bool(before), "occurrences": _value(before)},
                "value_after": {"exists": bool(after), "occurrences": _value(after)},
                "evidence_before": _evidence(before),
                "evidence_after": _evidence(after),
                "verifier": {"id": "python_ast_argparse_v1", "result": "pass"},
                "split": "",
            }
        )
    _assign_stratified_splits(claims)
    return claims


def _existence_probe(claim: dict, *, side: str, version: str) -> dict:
    value = claim[f"value_{side}"]
    expected = "yes" if value["exists"] else "no"
    object_names = {
        "cli_flag": "CLI flag",
        "env_var": "recognized environment variable",
        "public_export": "top-level public export",
        "config_field": "typed configuration field",
    }
    object_name = object_names[claim["entity_type"]]
    return {
        "schema": SCHEMA,
        "id": f"{claim['claim_id']}:exists:{version}",
        "claim_id": claim["claim_id"],
        "entity_type": claim["entity_type"],
        "display_entity": claim["display_entity"],
        "split": claim["split"],
        "probe_form": "versioned_existence",
        "question": (
            f"In vLLM {version}, is `{claim['display_entity']}` a {object_name}? "
            "Answer yes or no."
        ),
        "gold": {"boolean": expected},
        "eval_class": "A" if claim["status"] == "stable" else "D",
        "drift_type": claim["status"],
        "requires_doc": version,
        "evidence": claim[f"evidence_{side}"],
        "generator": {
            "id": "deterministic_claim_probe_v1",
            "seed": None,
        },
    }


def _delta_probe(claim: dict, before_version: str, after_version: str) -> dict:
    status_words = {
        "stable": [["unchanged", "stable", "both"]],
        "added": [["added", "new", "only"], [after_version]],
        "removed": [["removed", "no longer", "only"], [before_version]],
        "changed": [["changed", "different"]],
    }
    question = (
        f"Between vLLM {before_version} and {after_version}, what happened to "
        f"`{claim['display_entity']}` ({claim['entity_type']})?"
    )
    gold: dict = {
        "must_contain": [claim["display_entity"]],
        "must_contain_any": status_words[claim["status"]],
    }
    expected_change: dict | None = None
    if claim["status"] == "changed":
        question, gold, expected_change = _changed_contract_probe(
            claim, before_version, after_version
        )
    row = {
        "schema": SCHEMA,
        "id": f"{claim['claim_id']}:delta",
        "claim_id": claim["claim_id"],
        "entity_type": claim["entity_type"],
        "display_entity": claim["display_entity"],
        "split": claim["split"],
        "probe_form": "version_delta",
        "source_before_version": before_version,
        "source_after_version": after_version,
        "question": question,
        "gold": gold,
        "eval_class": "A" if claim["status"] == "stable" else "B",
        "drift_type": claim["status"],
        "requires_doc": after_version,
        "evidence": claim["evidence_before"] + claim["evidence_after"],
        "generator": {
            "id": "deterministic_claim_probe_v1",
            "seed": None,
        },
    }
    if expected_change is not None:
        row["expected_change"] = expected_change
    return row


def _value_forms(value: object) -> list[str]:
    if value is None:
        return ["none", "null", "unspecified"]
    if isinstance(value, bool):
        return [str(value).lower()]
    return [str(value)]


def _constraint_forms(name: str, value: object) -> list[str]:
    symbol = {"ge": ">=", "gt": ">", "le": "<=", "lt": "<"}.get(name)
    words = {
        "ge": "greater than or equal to",
        "gt": "greater than",
        "le": "less than or equal to",
        "lt": "less than",
    }.get(name)
    forms = [f"{name}={value}", f"{name} {value}"]
    if symbol:
        forms.append(f"{symbol} {value}")
    if words:
        forms.append(f"{words} {value}")
    return forms


def _changed_contract_probe(
    claim: dict, before_version: str, after_version: str
) -> tuple[str, dict, dict]:
    before_occ = claim["value_before"]["occurrences"]
    after_occ = claim["value_after"]["occurrences"]
    before_contracts = [row["contract"] for row in before_occ]
    after_contracts = [row["contract"] for row in after_occ]
    display = claim["display_entity"]

    if claim["entity_type"] == "cli_flag":
        before_choices = {
            choice
            for contract in before_contracts
            for choice in (contract.get("choices") or [])
        }
        after_choices = {
            choice
            for contract in after_contracts
            for choice in (contract.get("choices") or [])
        }
        added = sorted(after_choices - before_choices)
        removed = sorted(before_choices - after_choices)
        if added or removed:
            question = (
                f"Which accepted value changed for `{display}` between vLLM "
                f"{before_version} and {after_version}?"
            )
            groups = [[value] for value in added + removed]
            return question, {"must_contain_any": groups}, {
                "dimension": "choices",
                "added": added,
                "removed": removed,
            }

    if len(before_contracts) == 1 and len(after_contracts) == 1:
        before, after = before_contracts[0], after_contracts[0]
        before_constraints = before.get("constraints") or {}
        after_constraints = after.get("constraints") or {}
        changed_constraints = {
            key: {"before": before_constraints.get(key), "after": value}
            for key, value in after_constraints.items()
            if before_constraints.get(key) != value
        }
        if changed_constraints:
            groups = [
                _constraint_forms(key, values["after"])
                for key, values in sorted(changed_constraints.items())
            ]
            question = (
                f"What validation constraint was introduced or changed for "
                f"`{display}` in vLLM {after_version}?"
            )
            return question, {"must_contain_any": groups}, {
                "dimension": "constraints",
                "changes": changed_constraints,
            }
        if before.get("default") != after.get("default"):
            question = (
                f"How did the default for `{display}` change from vLLM "
                f"{before_version} to {after_version}?"
            )
            return question, {
                "must_contain_any": [
                    _value_forms(before.get("default")),
                    _value_forms(after.get("default")),
                ]
            }, {
                "dimension": "default",
                "before": before.get("default"),
                "after": after.get("default"),
            }
        if before.get("annotation") != after.get("annotation"):
            question = (
                f"How did the accepted type for `{display}` change from vLLM "
                f"{before_version} to {after_version}?"
            )
            return question, {
                "must_contain_any": [
                    [str(before.get("annotation"))],
                    [str(after.get("annotation"))],
                ]
            }, {
                "dimension": "annotation",
                "before": before.get("annotation"),
                "after": after.get("annotation"),
            }

    return (
        f"What executable contract difference affected `{display}` between "
        f"vLLM {before_version} and {after_version}?",
        {"must_contain_any": [["changed", "different"]]},
        {
            "dimension": "compound",
            "before": before_contracts,
            "after": after_contracts,
        },
    )


def build_probes(
    claims: list[dict], *, before_version: str, after_version: str
) -> list[dict]:
    probes: list[dict] = []
    for claim in claims:
        probes.extend(
            [
                _existence_probe(claim, side="before", version=before_version),
                _existence_probe(claim, side="after", version=after_version),
                _delta_probe(claim, before_version, after_version),
            ]
        )
    return probes


def _extract_all(snapshot: Path) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    rejected: list[dict] = []
    extractors = (
        ("cli_flag", extract_cli_flags_with_rejections),
        ("env_var", extract_env_contracts),
        ("public_export", extract_public_exports),
        ("config_field", extract_config_fields),
    )
    for family, extractor in extractors:
        family_rows, family_rejected = extractor(snapshot)
        rows.extend(family_rows)
        rejected.extend({"entity_type": family, **row} for row in family_rejected)
    return rows, rejected


def build(
    *,
    repo_root: Path,
    snapshots_path: Path,
    before_alias: str,
    after_alias: str,
    protocol_id: str,
) -> dict:
    before_spec = _snapshot_spec(snapshots_path, before_alias)
    after_spec = _snapshot_spec(snapshots_path, after_alias)
    before_snapshot = _snapshot_dir(repo_root, before_spec["version"])
    after_snapshot = _snapshot_dir(repo_root, after_spec["version"])
    before_meta = _meta(before_snapshot)
    after_meta = _meta(after_snapshot)

    before_rows, before_rejected = _extract_all(before_snapshot)
    after_rows, after_rejected = _extract_all(after_snapshot)
    before_rerun, before_rejected_rerun = _extract_all(before_snapshot)
    after_rerun, after_rejected_rerun = _extract_all(after_snapshot)
    verifier_rerun_equal = (
        before_rows == before_rerun
        and after_rows == after_rerun
        and before_rejected == before_rejected_rerun
        and after_rejected == after_rejected_rerun
    )
    if not verifier_rerun_equal:
        raise RuntimeError("executable verifier output changed across immediate rerun")
    claims = build_claims(
        before_rows=before_rows,
        after_rows=after_rows,
        before_spec=before_spec,
        after_spec=after_spec,
    )
    probes = build_probes(
        claims,
        before_version=before_spec["version"],
        after_version=after_spec["version"],
    )

    transition = f"v{before_spec['version']}_to_v{after_spec['version']}"
    out = (
        repo_root
        / "data"
        / "experiments"
        / "e1_vllm"
        / "eval_factory"
        / transition
        / protocol_id
    )
    out.mkdir(parents=True, exist_ok=True)
    for derived_name in (
        "probes_eval.jsonl",
        "audit_50.jsonl",
        "validation.json",
        "surface_generation.json",
        "surface_finalize.json",
        "surface_unresolved.jsonl",
    ):
        derived = out / derived_name
        if derived.exists():
            derived.unlink()
    rejected = [
        {"revision": before_spec["git_tag"], **row} for row in before_rejected
    ] + [{"revision": after_spec["git_tag"], **row} for row in after_rejected]
    _write_jsonl(out / "claims.jsonl", claims)
    _write_jsonl(out / "claims_verified.jsonl", claims)
    _write_jsonl(out / "claims_rejected.jsonl", rejected)
    for split in ("train", "dev", "eval"):
        name = "probes_eval_seed.jsonl" if split == "eval" else f"probes_{split}.jsonl"
        _write_jsonl(
            out / name,
            [probe for probe in probes if probe["split"] == split],
        )

    by_status: dict[str, int] = defaultdict(int)
    by_split: dict[str, int] = defaultdict(int)
    by_family: dict[str, int] = defaultdict(int)
    by_family_status: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for claim in claims:
        by_status[claim["status"]] += 1
        by_split[claim["split"]] += 1
        by_family[claim["entity_type"]] += 1
        by_family_status[claim["entity_type"]][claim["status"]] += 1

    manifest = {
        "schema": SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol_id,
        "transition": transition,
        "split_seed": SPLIT_SEED,
        "source_before": {
            **before_spec,
            "commit_sha": before_meta["commit_sha"],
            "content_sha": before_meta["content_sha"],
        },
        "source_after": {
            **after_spec,
            "commit_sha": after_meta["commit_sha"],
            "content_sha": after_meta["content_sha"],
        },
        "verifiers": [
            "python_ast_argparse_v1",
            "python_ast_env_registry_v1",
            "python_ast_module_attrs_v1",
            "python_ast_vllm_config_v1",
        ],
        "generators": ["deterministic_claim_probe_v1"],
        "verifier_rerun_equal": verifier_rerun_equal,
        "claim_bank": {
            "all": "claims.jsonl",
            "verified": "claims_verified.jsonl",
            "rejected": "claims_rejected.jsonl",
        },
        "probe_banks": {
            "train": "probes_train.jsonl",
            "development": "probes_dev.jsonl",
            "evaluation_seed": "probes_eval_seed.jsonl",
            "evaluation_final": "probes_eval.jsonl",
        },
        "contamination_policy": {
            "split_unit": "claim_id",
            "stable_split": {"train": 0.6, "development": 0.2, "evaluation": 0.2},
            "delta_split": {"train": 0.4, "development": 0.2, "evaluation": 0.4},
            "delta_strata": ["entity_type", "status"],
            "exact_question_overlap_allowed": False,
            "max_train_eval_token_jaccard": 0.85,
        },
        "freeze": {
            "ready": False,
            "validation": "validation.json",
            "human_audit": "audit_50.jsonl",
        },
        "counts": {
            "claims": len(claims),
            "claims_rejected": len(rejected),
            "probes": len(probes),
            "by_status": dict(sorted(by_status.items())),
            "by_split": dict(sorted(by_split.items())),
            "by_family": dict(sorted(by_family.items())),
            "by_family_status": {
                family: dict(sorted(statuses.items()))
                for family, statuses in sorted(by_family_status.items())
            },
        },
    }
    _write_json(out / "manifest.json", manifest)
    manifest["artifact_sha256"] = {
        path.name: _sha256(path)
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    _write_json(out / "manifest.json", manifest)
    return {"artifact_dir": str(out), **manifest["counts"]}
