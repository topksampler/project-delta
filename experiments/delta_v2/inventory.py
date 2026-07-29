from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


SCHEMA = "delta.source_snapshots.v1"
CONTENT_HASH_ALGORITHM = "git-tree-sha1"
OBJECT_HASH_PREFIX = "git-object-sha1:"
TREE_HASH_PREFIX = "git-tree-sha1:"
RENAME_POLICY = "exact-object-unique-v1"
DEVELOPMENT_ROLES = ("development_before", "development_after")
ACCEPTANCE_ROLES = ("acceptance_before", "acceptance_after")
ALL_ROLES = DEVELOPMENT_ROLES + ACCEPTANCE_ROLES
STATUSES = ("stable", "added", "removed", "modified", "renamed")
HEX_SHA1 = re.compile(r"^[0-9a-f]{40}$")


class SnapshotManifestError(ValueError):
    """The snapshot contract or its checked-out Git objects are invalid."""


@dataclass(frozen=True)
class Snapshot:
    role: str
    revision: str
    commit_sha: str
    content_hash: str

    @property
    def tree_oid(self) -> str:
        return self.content_hash.removeprefix(TREE_HASH_PREFIX)

    def with_pins(self, *, commit_sha: str, tree_oid: str) -> "Snapshot":
        return replace(
            self,
            commit_sha=commit_sha,
            content_hash=f"{TREE_HASH_PREFIX}{tree_oid}",
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "revision": self.revision,
            "commit_sha": self.commit_sha,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class SnapshotManifest:
    repository_id: str
    repository_url: str
    content_hash_algorithm: str
    snapshots: Mapping[str, Snapshot]


@dataclass(frozen=True, order=True)
class TreeEntry:
    path: str
    mode: str
    object_type: str
    object_id: str


@dataclass(frozen=True)
class FileDelta:
    file_id: str
    status: str
    path_before: str | None
    path_after: str | None
    hash_before: str | None
    hash_after: str | None
    mode_before: str | None
    mode_after: str | None
    type_before: str | None
    type_after: str | None
    rename_policy: str = RENAME_POLICY

    @classmethod
    def create(
        cls,
        *,
        repository_id: str,
        status: str,
        before: TreeEntry | None,
        after: TreeEntry | None,
    ) -> "FileDelta":
        if status not in STATUSES:
            raise ValueError(f"unsupported FileDelta status: {status}")
        _validate_delta_shape(status, before, after)
        path_before = before.path if before else None
        path_after = after.path if after else None
        identity = "\0".join(
            (
                "delta.file.v1",
                repository_id,
                path_before or "",
                path_after or "",
            )
        ).encode("utf-8")
        file_id = f"file:{hashlib.sha256(identity).hexdigest()}"
        return cls(
            file_id=file_id,
            status=status,
            path_before=path_before,
            path_after=path_after,
            hash_before=(
                f"{OBJECT_HASH_PREFIX}{before.object_id}" if before else None
            ),
            hash_after=f"{OBJECT_HASH_PREFIX}{after.object_id}" if after else None,
            mode_before=before.mode if before else None,
            mode_after=after.mode if after else None,
            type_before=before.object_type if before else None,
            type_after=after.object_type if after else None,
        )

    def as_dict(self) -> dict[str, str | None]:
        return {
            "file_id": self.file_id,
            "status": self.status,
            "path_before": self.path_before,
            "path_after": self.path_after,
            "hash_before": self.hash_before,
            "hash_after": self.hash_after,
            "mode_before": self.mode_before,
            "mode_after": self.mode_after,
            "type_before": self.type_before,
            "type_after": self.type_after,
            "rename_policy": self.rename_policy,
        }


def _validate_delta_shape(
    status: str,
    before: TreeEntry | None,
    after: TreeEntry | None,
) -> None:
    if status == "added" and (before is not None or after is None):
        raise ValueError("added FileDelta requires only an after entry")
    if status == "removed" and (before is None or after is not None):
        raise ValueError("removed FileDelta requires only a before entry")
    if status in {"stable", "modified"}:
        if before is None or after is None or before.path != after.path:
            raise ValueError(f"{status} FileDelta requires matching paths")
    if status == "renamed":
        if before is None or after is None or before.path == after.path:
            raise ValueError("renamed FileDelta requires two different paths")


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SnapshotManifestError(f"{field} must be a mapping")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SnapshotManifestError(f"{field} must be a non-empty string")
    return value


def load_snapshot_manifest(path: Path) -> SnapshotManifest:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SnapshotManifestError(f"cannot read snapshot manifest: {path}") from exc
    root = _require_mapping(payload, "manifest")
    schema = _require_string(root.get("schema"), "schema")
    if schema != SCHEMA:
        raise SnapshotManifestError(f"unsupported schema: {schema}")

    repository = _require_mapping(root.get("repository"), "repository")
    repository_id = _require_string(repository.get("id"), "repository.id")
    repository_url = _require_string(repository.get("url"), "repository.url")
    algorithm = _require_string(
        root.get("content_hash_algorithm"),
        "content_hash_algorithm",
    )
    if algorithm != CONTENT_HASH_ALGORITHM:
        raise SnapshotManifestError(
            f"unsupported content hash algorithm: {algorithm}"
        )

    raw_snapshots = root.get("snapshots")
    if not isinstance(raw_snapshots, list):
        raise SnapshotManifestError("snapshots must be a list")
    snapshots: dict[str, Snapshot] = {}
    for index, raw_snapshot in enumerate(raw_snapshots):
        record = _require_mapping(raw_snapshot, f"snapshots[{index}]")
        role = _require_string(record.get("role"), f"snapshots[{index}].role")
        if role not in ALL_ROLES:
            raise SnapshotManifestError(f"unsupported snapshot role: {role}")
        if role in snapshots:
            raise SnapshotManifestError(f"duplicate snapshot role: {role}")
        revision = _require_string(
            record.get("revision"),
            f"snapshots[{index}].revision",
        )
        commit_sha = _require_string(
            record.get("commit_sha"),
            f"snapshots[{index}].commit_sha",
        )
        content_hash = _require_string(
            record.get("content_hash"),
            f"snapshots[{index}].content_hash",
        )
        if not HEX_SHA1.fullmatch(commit_sha):
            raise SnapshotManifestError(f"invalid commit SHA for {role}")
        if not content_hash.startswith(TREE_HASH_PREFIX):
            raise SnapshotManifestError(f"invalid content hash prefix for {role}")
        tree_oid = content_hash.removeprefix(TREE_HASH_PREFIX)
        if not HEX_SHA1.fullmatch(tree_oid):
            raise SnapshotManifestError(f"invalid tree hash for {role}")
        snapshots[role] = Snapshot(
            role=role,
            revision=revision,
            commit_sha=commit_sha,
            content_hash=content_hash,
        )

    missing = sorted(set(ALL_ROLES) - set(snapshots))
    extra = sorted(set(snapshots) - set(ALL_ROLES))
    if missing or extra:
        raise SnapshotManifestError(
            f"snapshot roles mismatch; missing={missing}, extra={extra}"
        )
    return SnapshotManifest(
        repository_id=repository_id,
        repository_url=repository_url,
        content_hash_algorithm=algorithm,
        snapshots=snapshots,
    )


def select_transition(
    manifest: SnapshotManifest,
    transition: str,
    *,
    allow_acceptance: bool = False,
) -> tuple[Snapshot, Snapshot]:
    if transition == "development":
        roles = DEVELOPMENT_ROLES
    elif transition == "acceptance":
        if not allow_acceptance:
            raise SnapshotManifestError(
                "acceptance transition is sealed; pass --allow-acceptance only "
                "after the development recipe freezes"
            )
        roles = ACCEPTANCE_ROLES
    else:
        raise SnapshotManifestError(f"unknown transition: {transition}")
    return manifest.snapshots[roles[0]], manifest.snapshots[roles[1]]


def _run_git(repo: Path, arguments: Sequence[str]) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise SnapshotManifestError(
            f"git {' '.join(arguments)} failed in {repo}: {detail}"
        )
    return result.stdout


def verify_snapshot(repo: Path, snapshot: Snapshot) -> None:
    resolved_commit = _run_git(
        repo,
        ["rev-parse", "--verify", f"{snapshot.commit_sha}^{{commit}}"],
    ).decode("ascii").strip()
    if resolved_commit != snapshot.commit_sha:
        raise SnapshotManifestError(
            f"commit mismatch for {snapshot.role}: "
            f"expected {snapshot.commit_sha}, got {resolved_commit}"
        )
    resolved_tree = _run_git(
        repo,
        ["rev-parse", "--verify", f"{snapshot.commit_sha}^{{tree}}"],
    ).decode("ascii").strip()
    if resolved_tree != snapshot.tree_oid:
        raise SnapshotManifestError(
            f"tree hash mismatch for {snapshot.role}: "
            f"expected {snapshot.tree_oid}, got {resolved_tree}"
        )


def list_tree(repo: Path, commit_sha: str) -> dict[str, TreeEntry]:
    raw = _run_git(
        repo,
        ["ls-tree", "-r", "-z", "--full-tree", commit_sha],
    )
    entries: dict[str, TreeEntry] = {}
    for raw_record in raw.split(b"\0"):
        if not raw_record:
            continue
        try:
            metadata, raw_path = raw_record.split(b"\t", 1)
            raw_mode, raw_type, raw_object_id = metadata.split(b" ", 2)
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise SnapshotManifestError(
                "Git tree contains a malformed or non-UTF-8 path"
            ) from exc
        entry = TreeEntry(
            path=path,
            mode=raw_mode.decode("ascii"),
            object_type=raw_type.decode("ascii"),
            object_id=raw_object_id.decode("ascii"),
        )
        if path in entries:
            raise SnapshotManifestError(f"duplicate path in Git tree: {path!r}")
        entries[path] = entry
    return entries


def _entry_identity(entry: TreeEntry) -> tuple[str, str, str]:
    return entry.mode, entry.object_type, entry.object_id


def _row_sort_key(row: FileDelta) -> tuple[str, str, str]:
    return row.path_before or "", row.path_after or "", row.status


def build_file_deltas(
    repository_id: str,
    before: Mapping[str, TreeEntry],
    after: Mapping[str, TreeEntry],
) -> list[FileDelta]:
    rows: list[FileDelta] = []
    common_paths = set(before) & set(after)
    for path in common_paths:
        before_entry = before[path]
        after_entry = after[path]
        status = "stable" if before_entry == after_entry else "modified"
        rows.append(
            FileDelta.create(
                repository_id=repository_id,
                status=status,
                before=before_entry,
                after=after_entry,
            )
        )

    removed_paths = set(before) - set(after)
    added_paths = set(after) - set(before)
    removed_by_identity: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    added_by_identity: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for path in removed_paths:
        removed_by_identity[_entry_identity(before[path])].append(path)
    for path in added_paths:
        added_by_identity[_entry_identity(after[path])].append(path)

    matched_removed: set[str] = set()
    matched_added: set[str] = set()
    for identity in set(removed_by_identity) & set(added_by_identity):
        old_candidates = removed_by_identity[identity]
        new_candidates = added_by_identity[identity]
        if len(old_candidates) != 1 or len(new_candidates) != 1:
            continue
        old_path = old_candidates[0]
        new_path = new_candidates[0]
        matched_removed.add(old_path)
        matched_added.add(new_path)
        rows.append(
            FileDelta.create(
                repository_id=repository_id,
                status="renamed",
                before=before[old_path],
                after=after[new_path],
            )
        )

    for path in removed_paths - matched_removed:
        rows.append(
            FileDelta.create(
                repository_id=repository_id,
                status="removed",
                before=before[path],
                after=None,
            )
        )
    for path in added_paths - matched_added:
        rows.append(
            FileDelta.create(
                repository_id=repository_id,
                status="added",
                before=None,
                after=after[path],
            )
        )
    return sorted(rows, key=_row_sort_key)


def _entry_from_delta(
    *,
    path: str | None,
    mode: str | None,
    object_type: str | None,
    object_hash: str | None,
) -> TreeEntry | None:
    if path is None:
        if any(value is not None for value in (mode, object_type, object_hash)):
            raise ValueError("absent path has entry metadata")
        return None
    if mode is None or object_type is None or object_hash is None:
        raise ValueError(f"incomplete metadata for path: {path!r}")
    if not object_hash.startswith(OBJECT_HASH_PREFIX):
        raise ValueError(f"unsupported object hash: {object_hash}")
    return TreeEntry(
        path=path,
        mode=mode,
        object_type=object_type,
        object_id=object_hash.removeprefix(OBJECT_HASH_PREFIX),
    )


def project_inventory(
    rows: Iterable[FileDelta],
) -> tuple[dict[str, TreeEntry], dict[str, TreeEntry]]:
    before: dict[str, TreeEntry] = {}
    after: dict[str, TreeEntry] = {}
    for row in rows:
        before_entry = _entry_from_delta(
            path=row.path_before,
            mode=row.mode_before,
            object_type=row.type_before,
            object_hash=row.hash_before,
        )
        after_entry = _entry_from_delta(
            path=row.path_after,
            mode=row.mode_after,
            object_type=row.type_after,
            object_hash=row.hash_after,
        )
        if before_entry:
            if before_entry.path in before:
                raise ValueError(f"duplicate before path: {before_entry.path!r}")
            before[before_entry.path] = before_entry
        if after_entry:
            if after_entry.path in after:
                raise ValueError(f"duplicate after path: {after_entry.path!r}")
            after[after_entry.path] = after_entry
    return before, after


def audit_inventory(
    rows: Sequence[FileDelta],
    before: Mapping[str, TreeEntry],
    after: Mapping[str, TreeEntry],
) -> dict[str, Any]:
    projected_before, projected_after = project_inventory(rows)
    counts = {status: 0 for status in STATUSES}
    for row in rows:
        counts[row.status] += 1
    return {
        "rows": len(rows),
        "entries_before": len(before),
        "entries_after": len(after),
        "counts": counts,
        "reconstructs_before": projected_before == dict(before),
        "reconstructs_after": projected_after == dict(after),
        "unique_file_ids": len({row.file_id for row in rows}) == len(rows),
        "deterministic_order": list(rows) == sorted(rows, key=_row_sort_key),
        "rename_policy": RENAME_POLICY,
    }


def write_inventory(rows: Sequence[FileDelta], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for row in rows:
            line = (
                json.dumps(
                    row.as_dict(),
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            handle.write(line)
            digest.update(line)
    return digest.hexdigest()


def write_summary(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_inventory(
    *,
    manifest_path: Path,
    repo: Path,
    transition: str,
    allow_acceptance: bool = False,
) -> tuple[list[FileDelta], dict[str, Any]]:
    manifest = load_snapshot_manifest(manifest_path)
    before_snapshot, after_snapshot = select_transition(
        manifest,
        transition,
        allow_acceptance=allow_acceptance,
    )
    verify_snapshot(repo, before_snapshot)
    verify_snapshot(repo, after_snapshot)
    before_tree = list_tree(repo, before_snapshot.commit_sha)
    after_tree = list_tree(repo, after_snapshot.commit_sha)
    rows = build_file_deltas(manifest.repository_id, before_tree, after_tree)
    audit = audit_inventory(rows, before_tree, after_tree)
    if not all(
        (
            audit["reconstructs_before"],
            audit["reconstructs_after"],
            audit["unique_file_ids"],
            audit["deterministic_order"],
        )
    ):
        raise RuntimeError(f"FileDelta inventory failed audit: {audit}")
    summary = {
        "schema": "delta.file_inventory.audit.v1",
        "repository": {
            "id": manifest.repository_id,
            "url": manifest.repository_url,
        },
        "transition": transition,
        "source_before": before_snapshot.as_dict(),
        "source_after": after_snapshot.as_dict(),
        **audit,
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a complete FileDelta inventory from pinned Git trees."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("snapshots.yaml"),
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--transition",
        choices=("development", "acceptance"),
        default="development",
    )
    parser.add_argument(
        "--allow-acceptance",
        action="store_true",
        help="Unlock the sealed acceptance pair after the recipe freezes.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = build_inventory(
        manifest_path=args.manifest,
        repo=args.repo,
        transition=args.transition,
        allow_acceptance=args.allow_acceptance,
    )
    summary["inventory_sha256"] = write_inventory(rows, args.out)
    summary["inventory_path"] = str(args.out)
    write_summary(summary, args.summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
