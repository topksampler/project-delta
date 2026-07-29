"""Factory-backed SENSE: combine structural and behavioral drift evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import DriftEventV2


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_factory_drift_event(
    *,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    probes: Iterable[Mapping[str, Any]],
    samples: Iterable[Mapping[str, Any]] = (),
    paths: Mapping[str, Any] | None = None,
) -> DriftEventV2:
    """Create DriftEvent v2 without mixing in eval_v3/profile scoreboards."""

    counts = dict(manifest.get("counts") or {})
    by_status = dict(counts.get("by_status") or {})
    scores = {
        key: float(value)
        for key, value in (summary.get("by_drift_type") or {}).items()
    }
    failure_mass = {key: round(1.0 - value, 6) for key, value in scores.items()}

    failed_probe_ids = {
        str(row.get("id"))
        for row in samples
        if row.get("id")
        and float(row.get("content_score", row.get("score", 0.0))) < 1.0
    }
    affected_claim_ids: set[str] = set()
    drift_types: set[str] = set()
    for probe in probes:
        drift = str(probe.get("drift_type") or "")
        if drift and drift != "stable":
            drift_types.add(drift)
        if drift == "stable":
            continue
        if failed_probe_ids and str(probe.get("id")) not in failed_probe_ids:
            continue
        claim_id = probe.get("claim_id")
        if claim_id:
            affected_claim_ids.add(str(claim_id))

    affected_scores = [
        score for drift, score in scores.items() if drift in {"added", "changed", "removed"}
    ]
    drift_score = max((1.0 - score for score in affected_scores), default=0.0)
    return DriftEventV2(
        source_before=dict(manifest.get("source_before") or {}),
        source_after=dict(manifest.get("source_after") or {}),
        structural_census={
            "claims": counts.get("claims"),
            "by_status": by_status,
            "by_family_status": counts.get("by_family_status") or {},
        },
        behavioral_failures={
            "scores": scores,
            "failure_mass": failure_mass,
            "n_samples": summary.get("n_samples"),
            "scoreboard": "eval_factory",
        },
        affected_claim_ids=tuple(sorted(affected_claim_ids)),
        drift_types=tuple(sorted(drift_types)),
        drift_score=round(drift_score, 6),
        paths=dict(paths or {}),
        metadata={
            "eval_environment_schema": manifest.get("schema"),
            "summary_schema": summary.get("schema"),
        },
    )


def build_factory_drift_event_from_paths(
    *,
    manifest_path: Path,
    summary_path: Path,
    probes_path: Path,
    samples_path: Path | None = None,
    path_root: Path | None = None,
) -> DriftEventV2:
    def recorded(path: Path) -> str:
        if path_root is None:
            return str(path)
        return str(path.resolve().relative_to(path_root.resolve()))

    samples = _jsonl(samples_path) if samples_path else ()
    return build_factory_drift_event(
        manifest=_json(manifest_path),
        summary=_json(summary_path),
        probes=_jsonl(probes_path),
        samples=samples,
        paths={
            "manifest": recorded(manifest_path),
            "summary": recorded(summary_path),
            "probes": recorded(probes_path),
            **({"samples": recorded(samples_path)} if samples_path else {}),
        },
    )
