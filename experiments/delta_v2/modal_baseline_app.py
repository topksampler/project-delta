"""Pinned Modal execution image for the frozen delta_v2 c0 baseline."""

from __future__ import annotations

from pathlib import Path

import modal
import yaml

from lab.dispatch import modal_worker


REPO_ROOT = Path("/root/lalith-ai-lab")
EVAL_DATA = (
    "data/experiments/delta_v2/acceptance_attempt_2/eval/"
    "acceptance_eval_items_v3.jsonl"
)

app = modal.App("lalith-ai-lab-delta-v2-baseline")
secrets = [modal.Secret.from_name("lalith-lab")]

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
        "safetensors==0.8.0",
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
    secrets=secrets,
    timeout=60 * 60,
)
def eval_run(run_id: str, config: str) -> None:
    modal_worker.timed(
        "env_receipts",
        lambda: modal_worker.write_receipts(
            repo_root=REPO_ROOT,
            run_id=run_id,
        ),
    )
    config_path = modal_worker.timed(
        "b2_pull_inputs",
        lambda: modal_worker.sync_inputs(
            repo_root=REPO_ROOT,
            run_id=run_id,
            config_rel=config,
        ),
    )

    def _compute() -> dict:
        with open(config_path, "r", encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
        return modal_worker.run_module(
            repo_root=REPO_ROOT,
            module=modal_worker.eval_module(cfg),
            config_path=config_path,
        )

    cfg = modal_worker.timed("compute", _compute)
    modal_worker.timed(
        "b2_push_outputs",
        lambda: modal_worker.sync_outputs(
            repo_root=REPO_ROOT,
            run_id=run_id,
            config=cfg,
        ),
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
