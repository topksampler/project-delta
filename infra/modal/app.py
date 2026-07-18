"""Modal execution plane for lalith-ai-lab.

Setup and ownership: infra/modal/README.md.
Dispatch through ./scripts/lab; do not call experiment modules from the cockpit.
"""

from __future__ import annotations

import os
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal
import torch
import yaml

APP_NAME = "lalith-ai-lab"
REPO_ROOT = Path("/root/lalith-ai-lab")


def _timed(label: str, fn):
    t0 = time.perf_counter()
    result = fn()
    print(f"[timing] {label}: {time.perf_counter() - t0:.1f}s")
    return result

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
    .pip_install_from_requirements("requirements/torch-cu128.txt")
    .pip_install_from_requirements("requirements/runtime.txt")
    .env({"PYTHONPATH": "/root/lalith-ai-lab/src"})
    .add_local_dir("src", remote_path="/root/lalith-ai-lab/src")
    .add_local_dir("configs", remote_path="/root/lalith-ai-lab/configs")
)

secrets = [modal.Secret.from_name("lalith-lab")]


def _s5cmd(args: list[str]) -> None:
    endpoint = os.environ["S3_ENDPOINT_URL"].strip()
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
        candidates = [f"datasets/{rel}", rel]
        if rel.startswith("data/"):
            candidates.insert(0, f"datasets/{rel[len('data/'):]}")
        last_err: subprocess.CalledProcessError | None = None
        for key in candidates:
            try:
                _s5cmd(["cp", _s3(key), str(local)])
                break
            except subprocess.CalledProcessError as exc:
                last_err = exc
        else:
            assert last_err is not None
            raise last_err

    adapter_run = cfg.get("model", {}).get("adapter_run_id")
    if adapter_run:
        local_adapter = REPO_ROOT / "runs" / adapter_run / "adapter"
        local_adapter.mkdir(parents=True, exist_ok=True)
        _s5cmd(["cp", f"{_s3(f'runs/{adapter_run}/adapter')}/*", f"{local_adapter}/"])

    return config_path


def _eval_module(cfg: dict) -> str:
    if cfg.get("experiment_id") == "e1_vllm":
        return "lab.eval_vllm_qa"
    module = cfg.get("eval", {}).get("module")
    if module:
        return module
    return "lab.eval_run_spec"


def _sync_outputs(run_id: str, cfg: dict) -> None:
    if "output" in cfg:
        run_dir = Path(cfg["output"]["dir"])
    else:
        run_dir = Path(cfg["eval"]["output_dir"])
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    _s5cmd(["cp", f"{run_dir}/*", _s3(f"runs/{run_id}/")])


def _write_receipts(run_id: str) -> None:
    run_dir = REPO_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    hardware = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": "modal",
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cuda_available": cuda_available,
        "device_count": device_count,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(device_count)] if cuda_available else [],
    }
    env = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": "modal",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "profile": "modal-cu128",
    }
    (run_dir / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n", encoding="utf-8")
    (run_dir / "env.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    _s5cmd(["cp", str(run_dir / "hardware.json"), _s3(f"runs/{run_id}/hardware.json")])
    _s5cmd(["cp", str(run_dir / "env.json"), _s3(f"runs/{run_id}/env.json")])


def _run_module(module: str, config_path: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    cmd = ["python", "-m", module, "--config", str(config_path)]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=str(REPO_ROOT))
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@app.function(
    image=image,
    gpu="T4",
    secrets=secrets,
    timeout=60 * 60,
)
def eval_run(run_id: str, config: str) -> None:
    _timed("env_receipts", lambda: _write_receipts(run_id))
    config_path = _timed("b2_pull_inputs", lambda: _sync_inputs(run_id, config))

    def _compute() -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return _run_module(_eval_module(cfg), config_path)

    cfg = _timed("compute", _compute)
    _timed("b2_push_outputs", lambda: _sync_outputs(run_id, cfg))
    print(f"eval_run complete: {run_id}")


@app.function(
    image=image,
    gpu="T4",
    secrets=secrets,
    timeout=60 * 60 * 6,
)
def train(run_id: str, config: str) -> None:
    _timed("env_receipts", lambda: _write_receipts(run_id))
    config_path = _timed("b2_pull_inputs", lambda: _sync_inputs(run_id, config))
    cfg = _timed("compute", lambda: _run_module("lab.train_sft_lora", config_path))
    _timed("b2_push_outputs", lambda: _sync_outputs(run_id, cfg))
    print(f"train complete: {run_id}")
