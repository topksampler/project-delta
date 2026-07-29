from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.delta_v2.fact_family_manifest import (
    FactFamilyManifest,
    load_fact_family_manifest,
)
from experiments.delta_v2.facts.model import (
    EXTRACTOR_VERSION,
    AtomicFactDelta,
    FactExtractionError,
    FactObservation,
    ObservationStore,
    RejectedFactCandidate,
    audit_fact_deltas,
    join_observations,
    write_json,
    write_jsonl,
)
from experiments.delta_v2.facts.python_ast import (
    FAMILY_IDS,
    PYTHON_FEATURE_VERSION,
    PythonSource,
    extract_python_source,
    should_parse_path,
)
from experiments.delta_v2.inventory import (
    Snapshot,
    list_tree,
    load_snapshot_manifest,
    select_transition,
    verify_snapshot,
)


SUMMARY_SCHEMA = "delta.atomic_fact_extraction.audit.v1"


@dataclass(frozen=True)
class SnapshotExtraction:
    snapshot: Snapshot
    python_files: int
    observations: tuple[FactObservation, ...]
    rejections: tuple[RejectedFactCandidate, ...]


@dataclass(frozen=True)
class DevelopmentFactBuild:
    repository_id: str
    fact_manifest_sha256: str
    before: SnapshotExtraction
    after: SnapshotExtraction
    deltas: tuple[AtomicFactDelta, ...]
    summary: Mapping[str, Any]


class GitBlobBatch:
    """Read exact Git blobs efficiently without checking out a worktree."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.process: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> "GitBlobBatch":
        self.process = subprocess.Popen(
            ["git", "-C", str(self.repo), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return self

    def read(self, object_id: str) -> bytes:
        process = self.process
        if (
            process is None
            or process.stdin is None
            or process.stdout is None
        ):
            raise FactExtractionError("Git blob reader is not open")
        process.stdin.write(object_id.encode("ascii") + b"\n")
        process.stdin.flush()
        header = process.stdout.readline()
        if not header:
            raise FactExtractionError(
                f"git cat-file ended before reading {object_id}"
            )
        parts = header.rstrip(b"\n").split(b" ")
        if len(parts) == 2 and parts[1] == b"missing":
            raise FactExtractionError(f"Git object is missing: {object_id}")
        if len(parts) != 3:
            raise FactExtractionError(
                f"malformed git cat-file header: {header!r}"
            )
        returned_id, object_type, raw_size = parts
        if returned_id.decode("ascii") != object_id:
            raise FactExtractionError(
                f"Git returned {returned_id!r} for requested {object_id}"
            )
        if object_type != b"blob":
            raise FactExtractionError(
                f"expected blob {object_id}, got {object_type!r}"
            )
        try:
            size = int(raw_size)
        except ValueError as exc:
            raise FactExtractionError(
                f"invalid Git blob size: {raw_size!r}"
            ) from exc
        content = process.stdout.read(size)
        separator = process.stdout.read(1)
        if len(content) != size or separator != b"\n":
            raise FactExtractionError(
                f"truncated Git blob response for {object_id}"
            )
        actual_id = hashlib.sha1(
            f"blob {len(content)}\0".encode("ascii") + content
        ).hexdigest()
        if actual_id != object_id:
            raise FactExtractionError(
                f"Git blob hash mismatch: expected {object_id}, got {actual_id}"
            )
        return content

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        process = self.process
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        return_code = process.wait()
        stderr = (
            process.stderr.read().decode("utf-8", errors="replace").strip()
            if process.stderr is not None
            else ""
        )
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        self.process = None
        if exc_type is None and return_code != 0:
            raise FactExtractionError(
                f"git cat-file failed with exit {return_code}: {stderr}"
            )


def _count_by_family(rows: Sequence[Any]) -> dict[str, int]:
    counts = Counter(row.family_id for row in rows)
    return {
        family_id: counts.get(family_id, 0)
        for family_id in FAMILY_IDS
    }


def _validate_extractor_contract(
    manifest: FactFamilyManifest,
) -> None:
    registered_ids = tuple(
        family.family_id for family in manifest.families
    )
    if registered_ids != FAMILY_IDS:
        raise FactExtractionError(
            "extractor family order does not match preregistration: "
            f"{registered_ids}"
        )
    expected_feature_version = ".".join(
        str(part) for part in PYTHON_FEATURE_VERSION
    )
    if manifest.parser_implementation != "cpython.ast":
        raise FactExtractionError("extractor requires cpython.ast")
    if manifest.parser_feature_version != expected_feature_version:
        raise FactExtractionError(
            "extractor parser version does not match preregistration"
        )


def extract_snapshot(
    *,
    repo: Path,
    repository_id: str,
    snapshot: Snapshot,
) -> SnapshotExtraction:
    verify_snapshot(repo, snapshot)
    tree = list_tree(repo, snapshot.commit_sha)
    python_entries = [
        entry
        for path, entry in sorted(tree.items())
        if should_parse_path(path) and entry.object_type == "blob"
    ]
    store = ObservationStore()
    with GitBlobBatch(repo) as blobs:
        for entry in python_entries:
            source = PythonSource(
                snapshot_role=snapshot.role,
                path=entry.path,
                git_blob_sha1=entry.object_id,
                content=blobs.read(entry.object_id),
            )
            extract_python_source(
                repository_id=repository_id,
                source=source,
                store=store,
            )
    return SnapshotExtraction(
        snapshot=snapshot,
        python_files=len(python_entries),
        observations=tuple(store.observations),
        rejections=tuple(store.rejections),
    )


def build_development_facts(
    *,
    repo: Path,
    fact_manifest_path: Path,
) -> DevelopmentFactBuild:
    fact_manifest_bytes = fact_manifest_path.read_bytes()
    fact_manifest = load_fact_family_manifest(fact_manifest_path)
    _validate_extractor_contract(fact_manifest)

    snapshot_manifest_path = (
        fact_manifest_path.parent / fact_manifest.source_manifest
    )
    snapshot_manifest = load_snapshot_manifest(snapshot_manifest_path)
    before_snapshot, after_snapshot = select_transition(
        snapshot_manifest,
        "development",
        allow_acceptance=False,
    )

    before = extract_snapshot(
        repo=repo,
        repository_id=snapshot_manifest.repository_id,
        snapshot=before_snapshot,
    )
    after = extract_snapshot(
        repo=repo,
        repository_id=snapshot_manifest.repository_id,
        snapshot=after_snapshot,
    )
    fact_manifest_sha256 = hashlib.sha256(
        fact_manifest_bytes
    ).hexdigest()
    verifier = {
        "extractor_version": EXTRACTOR_VERSION,
        "fact_family_manifest_sha256": fact_manifest_sha256,
        "parser": {
            "implementation": fact_manifest.parser_implementation,
            "feature_version": fact_manifest.parser_feature_version,
            "canonical_form": "ast-structural-json-v1",
        },
    }
    deltas = tuple(
        join_observations(
            before.observations,
            after.observations,
            verifier=verifier,
        )
    )
    audit = audit_fact_deltas(
        deltas,
        before.observations,
        after.observations,
    )
    if not all(
        (
            audit["reconstructs_before"],
            audit["reconstructs_after"],
            audit["unique_fact_ids"],
            audit["deterministic_order"],
        )
    ):
        raise FactExtractionError(f"AtomicFactDelta audit failed: {audit}")

    summary = {
        "schema": SUMMARY_SCHEMA,
        "repository": {
            "id": snapshot_manifest.repository_id,
            "url": snapshot_manifest.repository_url,
        },
        "transition": "development",
        "fact_family_manifest_sha256": fact_manifest_sha256,
        "extractor_version": EXTRACTOR_VERSION,
        "source_before": before.snapshot.as_dict(),
        "source_after": after.snapshot.as_dict(),
        "python_files_before": before.python_files,
        "python_files_after": after.python_files,
        "observation_counts_before": _count_by_family(
            before.observations
        ),
        "observation_counts_after": _count_by_family(
            after.observations
        ),
        "rejection_counts_before": _count_by_family(before.rejections),
        "rejection_counts_after": _count_by_family(after.rejections),
        **audit,
    }
    return DevelopmentFactBuild(
        repository_id=snapshot_manifest.repository_id,
        fact_manifest_sha256=fact_manifest_sha256,
        before=before,
        after=after,
        deltas=deltas,
        summary=summary,
    )


def write_development_outputs(
    build: DevelopmentFactBuild,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "observations_before": (
            "atomic_fact_observations_development_before.jsonl",
            build.before.observations,
        ),
        "observations_after": (
            "atomic_fact_observations_development_after.jsonl",
            build.after.observations,
        ),
        "rejections_before": (
            "atomic_fact_rejections_development_before.jsonl",
            build.before.rejections,
        ),
        "rejections_after": (
            "atomic_fact_rejections_development_after.jsonl",
            build.after.rejections,
        ),
        "deltas": (
            "atomic_fact_deltas_development.jsonl",
            build.deltas,
        ),
    }
    output_records: dict[str, Any] = {}
    for output_id, (filename, rows) in outputs.items():
        digest = write_jsonl(rows, output_dir / filename)
        output_records[output_id] = {
            "filename": filename,
            "rows": len(rows),
            "sha256": digest,
        }

    summary = {
        **dict(build.summary),
        "outputs": output_records,
    }
    write_json(
        summary,
        output_dir / "atomic_fact_extraction_development.summary.json",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.delta_v2.facts.build",
        description=(
            "Extract and compare preregistered delta_v2 atomic facts from "
            "the development snapshots."
        )
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--fact-families",
        type=Path,
        default=Path(__file__).parents[1] / "fact_families.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build = build_development_facts(
        repo=args.repo,
        fact_manifest_path=args.fact_families,
    )
    summary = write_development_outputs(build, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
