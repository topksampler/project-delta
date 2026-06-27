"""Modal execution plane for lalith-ai-lab.

Setup once on Mac:
  pip install modal
  modal setup
  modal secret create lalith-lab \\
    S3_BUCKET=... S3_ENDPOINT_URL=... \\
    AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \\
    AWS_REGION=... AWS_DEFAULT_REGION=...

Run from Mac cockpit (no SSH):
  ./scripts/lab run --target modal --config configs/evals/e0b_run_spec_base.yaml
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal
import yaml

APP_NAME = "lalith-ai-lab"
REPO_ROOT = Path("/root/lalith-ai-lab")

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "git")
    .run_commands(
        "wget -q https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz",
        "tar -xzf s5cmd_2.2.2_Linux-64bit.tar.gz",
        "mv s5cmd /usr/local/bin/s5cmd",
        "rm s5cmd_2.2.2_Linux-64bit.tar.gz",
    )
    .pip_install_from_requirements("requirements.txt")
    .env({"PYTHONPATH": "/root/lalith-ai-lab/src"})
)

src_mount = modal.Mount.from_local_dir("src", remote_path="/root/lalith-ai-lab/src")
cfg_mount = modal.Mount.from_local_dir("configs", remote_path="/root/lalith-ai-lab/configs")
secrets = [modal.Secret.from_name("lalith-lab")]


def _s5cmd(args: list[str]) -> None:
    endpoint = os.environ["S3_ENDPOINT_URL"]
    cmd = ["s5cmd", "--endpoint-url", endpoint, *args]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _s3(key: str) -> str:
    return f"s3://{os.environ['S3_BUCKET']}/{key.lstrip('/')}"


def _sync_inputs(run_id: str, config_rel: str) -> Path:
    config_path = REPO_ROOT / config_rel
    if not config_path.exists():
        local_cfg = Path(f"/tmp/{run_id}-config.yaml")
        _s5cmd(["cp", _s3(f"runs/{run_id}/config.yaml"), str(local_cfg)])
        config_path = local_cfg

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths: list[str] = []
    for block_key in ("data", "eval"):
        block = cfg.get(block_key, {})
        for field in ("train_path", "eval_path", "path"):
            value = block.get(field)
            if value:
                paths.append(value)

    for rel in paths:
        local = REPO_ROOT / rel
        if local.exists():
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        remote_dataset = _s3(f"datasets/{rel}")
        remote_flat = _s3(rel)
        try:
            _s5cmd(["cp", remote_dataset, str(local)])
        except subprocess.CalledProcessError:
            _s5cmd(["cp", remote_flat, str(local)])

    return config_path


def _sync_outputs(run_id: str, cfg: dict) -> None:
    if "output" in cfg:
        run_dir = Path(cfg["output"]["dir"])
    else:
        run_dir = Path(cfg["eval"]["output_dir"])
    _s5cmd(["cp", f"{run_dir}/*", _s3(f"runs/{run_id}/")])


def _run_module(module: str, config_path: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    cmd = ["python", "-m", module, "--config", str(config_path)]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@app.function(
    image=image,
    gpu="A10G",
    secrets=secrets,
    mounts=[src_mount, cfg_mount],
    timeout=60 * 60,
)
def eval_run(run_id: str, config: str) -> None:
    config_path = _sync_inputs(run_id, config)
    cfg = _run_module("lab.eval_run_spec", config_path)
    _sync_outputs(run_id, cfg)
    print(f"eval_run complete: {run_id}")


@app.function(
    image=image,
    gpu="A10G",
    secrets=secrets,
    mounts=[src_mount, cfg_mount],
    timeout=60 * 60 * 4,
)
def train(run_id: str, config: str) -> None:
    config_path = _sync_inputs(run_id, config)
    cfg = _run_module("lab.train_sft_lora", config_path)
    _sync_outputs(run_id, cfg)
    print(f"train complete: {run_id}")
