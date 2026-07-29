from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from experiments.delta_v2.inventory import (
    ACCEPTANCE_ROLES,
    DEVELOPMENT_ROLES,
    FileDelta,
    SnapshotManifestError,
    TreeEntry,
    audit_inventory,
    build_file_deltas,
    list_tree,
    load_snapshot_manifest,
    select_transition,
    verify_snapshot,
    write_inventory,
)


SNAPSHOTS_PATH = Path(__file__).with_name("snapshots.yaml")


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


class SnapshotManifestTest(unittest.TestCase):
    def test_real_manifest_has_exact_roles_and_hashes(self) -> None:
        manifest = load_snapshot_manifest(SNAPSHOTS_PATH)

        self.assertEqual(manifest.repository_id, "vllm-project/vllm")
        self.assertEqual(
            set(manifest.snapshots),
            set(DEVELOPMENT_ROLES + ACCEPTANCE_ROLES),
        )
        for snapshot in manifest.snapshots.values():
            self.assertEqual(len(snapshot.commit_sha), 40)
            self.assertEqual(len(snapshot.tree_oid), 40)

    def test_manifest_rejects_duplicate_roles_and_malformed_hashes(self) -> None:
        original = SNAPSHOTS_PATH.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "snapshots.yaml"
            path.write_text(
                original.replace(
                    "role: acceptance_after",
                    "role: acceptance_before",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                SnapshotManifestError,
                "duplicate snapshot role",
            ):
                load_snapshot_manifest(path)

            path.write_text(
                original.replace(
                    "git-tree-sha1:92701b6c49f4969297d5d11a2cac91c180690482",
                    "git-tree-sha1:not-a-tree",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                SnapshotManifestError,
                "invalid tree hash",
            ):
                load_snapshot_manifest(path)

    def test_acceptance_requires_explicit_unlock(self) -> None:
        manifest = load_snapshot_manifest(SNAPSHOTS_PATH)

        with self.assertRaisesRegex(
            SnapshotManifestError,
            "acceptance transition is sealed",
        ):
            select_transition(manifest, "acceptance", allow_acceptance=False)

        before, after = select_transition(
            manifest,
            "acceptance",
            allow_acceptance=True,
        )
        self.assertEqual(before.role, "acceptance_before")
        self.assertEqual(after.role, "acceptance_after")


class FileDeltaClassificationTest(unittest.TestCase):
    def entry(
        self,
        path: str,
        object_id: str,
        *,
        mode: str = "100644",
        object_type: str = "blob",
    ) -> TreeEntry:
        return TreeEntry(
            path=path,
            mode=mode,
            object_type=object_type,
            object_id=object_id,
        )

    def test_classifies_all_statuses_and_reconstructs_both_trees(self) -> None:
        before = {
            "stable.txt": self.entry("stable.txt", "1" * 40),
            "modified.txt": self.entry("modified.txt", "2" * 40),
            "mode.txt": self.entry("mode.txt", "3" * 40),
            "removed.txt": self.entry("removed.txt", "4" * 40),
            "old-name.txt": self.entry("old-name.txt", "5" * 40),
        }
        after = {
            "stable.txt": self.entry("stable.txt", "1" * 40),
            "modified.txt": self.entry("modified.txt", "6" * 40),
            "mode.txt": self.entry("mode.txt", "3" * 40, mode="100755"),
            "added.txt": self.entry("added.txt", "7" * 40),
            "new-name.txt": self.entry("new-name.txt", "5" * 40),
        }

        rows = build_file_deltas("owner/repo", before, after)
        audit = audit_inventory(rows, before, after)

        self.assertEqual(
            audit["counts"],
            {
                "stable": 1,
                "added": 1,
                "removed": 1,
                "modified": 2,
                "renamed": 1,
            },
        )
        self.assertTrue(audit["reconstructs_before"])
        self.assertTrue(audit["reconstructs_after"])
        self.assertTrue(audit["unique_file_ids"])

    def test_ambiguous_exact_content_is_not_guessed_as_rename(self) -> None:
        shared = "a" * 40
        before = {
            "old-a.txt": self.entry("old-a.txt", shared),
            "old-b.txt": self.entry("old-b.txt", shared),
        }
        after = {
            "new-a.txt": self.entry("new-a.txt", shared),
            "new-b.txt": self.entry("new-b.txt", shared),
        }

        rows = build_file_deltas("owner/repo", before, after)

        self.assertEqual(
            [row.status for row in rows],
            ["added", "added", "removed", "removed"],
        )

    def test_rename_plus_edit_stays_removed_plus_added(self) -> None:
        before = {"old.txt": self.entry("old.txt", "b" * 40)}
        after = {"new.txt": self.entry("new.txt", "c" * 40)}

        rows = build_file_deltas("owner/repo", before, after)

        self.assertEqual({row.status for row in rows}, {"added", "removed"})

    def test_file_id_is_deterministic(self) -> None:
        row = FileDelta.create(
            repository_id="owner/repo",
            status="modified",
            before=self.entry("path.txt", "d" * 40),
            after=self.entry("path.txt", "e" * 40),
        )

        repeated = FileDelta.create(
            repository_id="owner/repo",
            status="modified",
            before=self.entry("path.txt", "d" * 40),
            after=self.entry("path.txt", "e" * 40),
        )

        self.assertEqual(row.file_id, repeated.file_id)

    def test_gitlink_entries_are_preserved(self) -> None:
        before = {
            "vendor/dependency": self.entry(
                "vendor/dependency",
                "f" * 40,
                mode="160000",
                object_type="commit",
            )
        }
        after = {
            "vendor/dependency": self.entry(
                "vendor/dependency",
                "0" * 40,
                mode="160000",
                object_type="commit",
            )
        }

        rows = build_file_deltas("owner/repo", before, after)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].status, "modified")
        self.assertEqual(rows[0].type_before, "commit")
        self.assertEqual(rows[0].mode_before, "160000")


class GitTreeIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp_dir.name)
        run_git(self.repo, "init")
        run_git(self.repo, "config", "user.name", "delta-v2-test")
        run_git(self.repo, "config", "user.email", "delta-v2-test@example.invalid")
        run_git(self.repo, "config", "core.filemode", "true")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def commit(self, message: str) -> str:
        run_git(self.repo, "add", "--all")
        run_git(self.repo, "commit", "-m", message)
        return run_git(self.repo, "rev-parse", "HEAD")

    def write(self, relative_path: str, content: str) -> Path:
        path = self.repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_real_git_trees_reconstruct_and_serialize_deterministically(self) -> None:
        self.write("stable.txt", "stable\n")
        self.write("modified.txt", "before\n")
        self.write("removed.txt", "removed\n")
        self.write("renamed.txt", "rename-me\n")
        mode_path = self.write("mode.txt", "mode\n")
        mode_path.chmod(0o644)
        self.write("ambiguous-a.txt", "same\n")
        self.write("ambiguous-b.txt", "same\n")
        self.write("unicodé\nname.txt", "odd path\n")
        os.symlink("stable.txt", self.repo / "link")
        before_sha = self.commit("before")

        self.write("modified.txt", "after\n")
        (self.repo / "removed.txt").unlink()
        (self.repo / "renamed.txt").rename(self.repo / "moved.txt")
        mode_path.chmod(0o755)
        self.write("added.txt", "added\n")
        (self.repo / "ambiguous-a.txt").unlink()
        (self.repo / "ambiguous-b.txt").unlink()
        self.write("ambiguous-c.txt", "same\n")
        self.write("ambiguous-d.txt", "same\n")
        (self.repo / "link").unlink()
        os.symlink("modified.txt", self.repo / "link")
        after_sha = self.commit("after")

        before = list_tree(self.repo, before_sha)
        after = list_tree(self.repo, after_sha)
        rows = build_file_deltas("fixture/repo", before, after)
        audit = audit_inventory(rows, before, after)

        self.assertEqual(
            audit["counts"],
            {
                "stable": 2,
                "added": 3,
                "removed": 3,
                "modified": 3,
                "renamed": 1,
            },
        )
        self.assertTrue(audit["reconstructs_before"])
        self.assertTrue(audit["reconstructs_after"])

        first = self.repo / "inventory-first.jsonl"
        second = self.repo / "inventory-second.jsonl"
        first_hash = write_inventory(rows, first)
        second_hash = write_inventory(rows, second)
        self.assertEqual(first_hash, second_hash)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        decoded = [
            json.loads(line)
            for line in first.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(decoded), len(rows))
        self.assertIn(
            "unicodé\nname.txt",
            {
                record["path_before"] or record["path_after"]
                for record in decoded
            },
        )

    def test_snapshot_verification_checks_commit_and_tree(self) -> None:
        self.write("file.txt", "content\n")
        commit_sha = self.commit("snapshot")
        tree_oid = run_git(self.repo, "rev-parse", f"{commit_sha}^{{tree}}")

        manifest = load_snapshot_manifest(SNAPSHOTS_PATH)
        template = manifest.snapshots["development_before"]
        snapshot = template.with_pins(commit_sha=commit_sha, tree_oid=tree_oid)

        verify_snapshot(self.repo, snapshot)

        bad = snapshot.with_pins(commit_sha=commit_sha, tree_oid="0" * 40)
        with self.assertRaisesRegex(SnapshotManifestError, "tree hash mismatch"):
            verify_snapshot(self.repo, bad)


if __name__ == "__main__":
    unittest.main()
