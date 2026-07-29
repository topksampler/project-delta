from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lab.dispatch.b2 import upload_active_pointer, upload_state
from lab.dispatch.manifest import build_manifest


class DeltaManifestTest(unittest.TestCase):
    def test_delta_lineage_is_copied_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.yaml"
            config.write_text(
                "run_id: run-1\n"
                "reconcile_id: rec-1\n"
                "drift_event_id: drift-1\n"
                "plan_id: plan-1\n"
                "eval_environment_id: eval-1\n",
                encoding="utf-8",
            )
            with patch("lab.dispatch.manifest.git_commit", return_value="abc"), patch(
                "lab.dispatch.manifest.git_branch", return_value="main"
            ):
                manifest = build_manifest(
                    run_id="run-1",
                    task="eval",
                    target="modal",
                    config_path=config,
                    repo_root=root,
                )
        self.assertEqual(
            manifest["delta_lineage"],
            {
                "reconcile_id": "rec-1",
                "drift_event_id": "drift-1",
                "plan_id": "plan-1",
                "eval_environment_id": "eval-1",
            },
        )

    def test_state_storage_uses_canonical_b2_prefixes(self) -> None:
        with patch("lab.dispatch.b2.upload_dir") as upload_dir, patch(
            "lab.dispatch.b2.upload_file"
        ) as upload_file:
            upload_state(Path("/tmp/state"), "state-123")
            upload_active_pointer(Path("/tmp/active.json"))
        upload_dir.assert_called_once_with(Path("/tmp/state"), "states/state-123")
        upload_file.assert_called_once_with(
            Path("/tmp/active.json"), "states/active.json"
        )


if __name__ == "__main__":
    unittest.main()
