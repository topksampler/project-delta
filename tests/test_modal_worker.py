from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from lab.dispatch import modal_worker


class ModalWorkerTrainingRoutingTest(unittest.TestCase):
    def test_training_module_wins(self) -> None:
        config = {
            "training": {"module": "experiments.example.train"},
            "train": {"module": "legacy.train"},
        }
        self.assertEqual(
            modal_worker.train_module(config),
            "experiments.example.train",
        )

    def test_train_module_is_legacy_fallback(self) -> None:
        self.assertEqual(
            modal_worker.train_module(
                {"train": {"module": "experiments.example.legacy"}}
            ),
            "experiments.example.legacy",
        )

    def test_default_is_shared_sft_lora(self) -> None:
        self.assertEqual(
            modal_worker.train_module({}),
            "lab.train_sft_lora",
        )


class ModalWorkerAdapterSyncTest(unittest.TestCase):
    def _write_config(self, root: Path, model: dict) -> Path:
        config_path = root / "config.yaml"
        config_path.write_text(
            yaml.safe_dump({"model": model}),
            encoding="utf-8",
        )
        return config_path

    def test_structured_lora_adapter_is_pulled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_config(
                root,
                {
                    "adapter": {
                        "kind": "lora",
                        "run_id": "train-run",
                        "path": "runs/train-run/adapter",
                    }
                },
            )
            with (
                mock.patch.dict(
                    "os.environ",
                    {
                        "S3_ENDPOINT_URL": "https://example.invalid",
                        "S3_BUCKET": "bucket",
                    },
                ),
                mock.patch.object(modal_worker, "_s5cmd") as s5cmd,
            ):
                result = modal_worker.sync_inputs(
                    repo_root=root,
                    run_id="eval-run",
                    config_rel="config.yaml",
                )

            self.assertEqual(result, root / "config.yaml")
            s5cmd.assert_called_once_with(
                [
                    "cp",
                    "s3://bucket/runs/train-run/adapter/*",
                    f"{root}/runs/train-run/adapter/",
                ]
            )

    def test_existing_adapter_is_not_pulled_again(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_config(
                root,
                {
                    "adapter": {
                        "kind": "lora",
                        "run_id": "train-run",
                        "path": "runs/train-run/adapter",
                    }
                },
            )
            marker = (
                root
                / "runs/train-run/adapter/adapter_model.safetensors"
            )
            marker.parent.mkdir(parents=True)
            marker.write_bytes(b"adapter")
            with mock.patch.object(modal_worker, "_s5cmd") as s5cmd:
                modal_worker.sync_inputs(
                    repo_root=root,
                    run_id="eval-run",
                    config_rel="config.yaml",
                )

            s5cmd.assert_not_called()

    def test_structured_adapter_path_must_match_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_config(
                root,
                {
                    "adapter": {
                        "kind": "lora",
                        "run_id": "train-run",
                        "path": "../outside",
                    }
                },
            )
            with self.assertRaisesRegex(
                ValueError,
                "runs/<run_id>/adapter",
            ):
                modal_worker.sync_inputs(
                    repo_root=root,
                    run_id="eval-run",
                    config_rel="config.yaml",
                )


if __name__ == "__main__":
    unittest.main()
