"""Pinned Modal plane for delta_v2 knowledge evaluation and LoRA training."""

from __future__ import annotations

from pathlib import Path

import modal
import yaml

from lab.dispatch import modal_worker


REPO_ROOT = Path("/root/lalith-ai-lab")
TRAIN_DATA = "data/experiments/delta_v2/knowledge_adaptation_v1/train.jsonl"
DEV_DATA = "data/experiments/delta_v2/knowledge_adaptation_v1/dev.jsonl"
EVAL_DATA = "data/experiments/delta_v2/knowledge_adaptation_v1/eval.jsonl"
PAIRED_TRAIN_DATA = (
    "data/experiments/delta_v2/knowledge_paired_v2/train.jsonl"
)
PAIRED_DATA = "data/experiments/delta_v2/knowledge_paired_v2/pairs.jsonl"

app = modal.App("lalith-ai-lab-delta-v2-knowledge-adapt")
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
        "peft==0.19.0",
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
    .add_local_dir("configs", remote_path="/root/lalith-ai-lab/configs")
    .add_local_dir(
        "experiments",
        remote_path="/root/lalith-ai-lab/experiments",
    )
    .add_local_file(
        TRAIN_DATA,
        remote_path=f"/root/lalith-ai-lab/{TRAIN_DATA}",
    )
    .add_local_file(
        DEV_DATA,
        remote_path=f"/root/lalith-ai-lab/{DEV_DATA}",
    )
    .add_local_file(
        EVAL_DATA,
        remote_path=f"/root/lalith-ai-lab/{EVAL_DATA}",
    )
    .add_local_file(
        PAIRED_TRAIN_DATA,
        remote_path=f"/root/lalith-ai-lab/{PAIRED_TRAIN_DATA}",
    )
    .add_local_file(
        PAIRED_DATA,
        remote_path=f"/root/lalith-ai-lab/{PAIRED_DATA}",
    )
)


def _config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


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
    cfg = modal_worker.timed(
        "compute",
        lambda: modal_worker.run_module(
            repo_root=REPO_ROOT,
            module=modal_worker.eval_module(_config(config_path)),
            config_path=config_path,
        ),
    )
    modal_worker.timed(
        "b2_push_outputs",
        lambda: modal_worker.sync_outputs(
            repo_root=REPO_ROOT,
            run_id=run_id,
            config=cfg,
        ),
    )


@app.function(
    image=image,
    gpu="H100",
    secrets=secrets,
    timeout=60 * 60 * 2,
)
def train(run_id: str, config: str) -> None:
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
    cfg = modal_worker.timed(
        "compute",
        lambda: modal_worker.run_module(
            repo_root=REPO_ROOT,
            module=modal_worker.train_module(_config(config_path)),
            config_path=config_path,
        ),
    )
    modal_worker.timed(
        "b2_push_outputs",
        lambda: modal_worker.sync_outputs(
            repo_root=REPO_ROOT,
            run_id=run_id,
            config=cfg,
        ),
    )


@app.local_entrypoint()
def main(
    run_id: str,
    config: str,
    task: str = "eval",
    gpu: str = "A10G",
) -> None:
    if task == "eval":
        eval_run.with_options(gpu=gpu).remote(
            run_id=run_id,
            config=config,
        )
    elif task == "train":
        train.with_options(gpu=gpu).remote(
            run_id=run_id,
            config=config,
        )
    else:
        raise ValueError(f"unsupported delta_v2 knowledge task: {task}")
