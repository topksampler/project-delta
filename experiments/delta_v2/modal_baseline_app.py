"""Pinned Modal execution image for the frozen delta_v2 c0 baseline."""

from __future__ import annotations

from pathlib import Path

import modal
import yaml

from infra.modal import app as platform


REPO_ROOT = platform.REPO_ROOT
EVAL_DATA = (
    "data/experiments/delta_v2/acceptance_attempt_2/eval/"
    "acceptance_eval_items_v3.jsonl"
)

app = modal.App("lalith-ai-lab-delta-v2-baseline")

image = (
    modal.Image.debian_slim(python_version="3.11.9")
    .apt_install("wget")
    .run_commands(
        "wget -q https://github.com/peak/s5cmd/releases/download/"
        "v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz",
        "tar -xzf s5cmd_2.2.2_Linux-64bit.tar.gz",
        "mv s5cmd /usr/local/bin/s5cmd",
        "rm s5cmd_2.2.2_Linux-64bit.tar.gz",
    )
    .pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers==5.14.1",
        "pyyaml==6.0.3",
        "pillow==12.1.0",
        "sentencepiece==0.2.1",
        "protobuf==6.33.4",
        "safetensors==0.7.0",
    )
    .env(
        {
            "PYTHONPATH": "/root/lalith-ai-lab/src",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir("src", remote_path="/root/lalith-ai-lab/src")
    .add_local_dir(
        "configs",
        remote_path="/root/lalith-ai-lab/configs",
    )
    .add_local_dir(
        "experiments",
        remote_path="/root/lalith-ai-lab/experiments",
    )
    .add_local_file(
        EVAL_DATA,
        remote_path=f"/root/lalith-ai-lab/{EVAL_DATA}",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    secrets=platform.secrets,
    timeout=60 * 60,
)
def eval_run(run_id: str, config: str) -> None:
    platform._timed(
        "env_receipts",
        lambda: platform._write_receipts(run_id),
    )
    config_path = platform._timed(
        "b2_pull_inputs",
        lambda: platform._sync_inputs(run_id, config),
    )

    def _compute() -> dict:
        with open(config_path, "r", encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
        return platform._run_module(platform._eval_module(cfg), config_path)

    cfg = platform._timed("compute", _compute)
    platform._timed(
        "b2_push_outputs",
        lambda: platform._sync_outputs(run_id, cfg),
    )
    print(f"delta_v2 baseline complete: {run_id}")


@app.local_entrypoint()
def main(
    run_id: str,
    config: str,
    task: str = "eval",
    gpu: str = "A10G",
) -> None:
    if task != "eval":
        raise ValueError("delta_v2 baseline Modal app supports eval only")
    print(f"modal: task=eval gpu={gpu} run_id={run_id}")
    eval_run.with_options(gpu=gpu).remote(
        run_id=run_id,
        config=config,
    )
