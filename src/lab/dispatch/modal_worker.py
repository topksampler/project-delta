from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar

import yaml


T = TypeVar("T")


def timed(label: str, fn: Callable[[], T]) -> T:
    started = time.perf_counter()
    result = fn()
    print(f"[timing] {label}: {time.perf_counter() - started:.1f}s")
    return result


def _s5cmd(args: list[str]) -> None:
    endpoint = os.environ["S3_ENDPOINT_URL"].strip()
    command = ["s5cmd", "--endpoint-url", endpoint, *args]
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def _s3(key: str) -> str:
    return f"s3://{os.environ['S3_BUCKET']}/{key.lstrip('/')}"


def sync_inputs(
    *,
    repo_root: Path,
    run_id: str,
    config_rel: str,
) -> Path:
    config_path = repo_root / config_rel
    if not config_path.exists():
        local_config = Path(f"/tmp/{run_id}-config.yaml")
        _s5cmd(
            [
                "cp",
                _s3(f"runs/{run_id}/config.yaml"),
                str(local_config),
            ]
        )
        config_path = local_config

    with open(config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    paths: list[str] = []
    for block_key in ("data", "eval"):
        block = config.get(block_key, {})
        for field in ("train_path", "eval_path", "path"):
            value = block.get(field)
            if value:
                paths.append(value)

    retrieval = config.get("retrieval") or {}
    if retrieval.get("enabled"):
        for field in (
            "corpus_path",
            "index_path",
            "evidence_map_path",
            "symbol_index_path",
            "diff_paths_path",
            "diff_corpus_path",
            "diff_index_path",
        ):
            value = retrieval.get(field)
            if value:
                paths.append(value)

    for relative in paths:
        local = repo_root / relative
        if local.exists():
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        candidates = [f"datasets/{relative}", relative]
        if relative.startswith("data/"):
            candidates.insert(0, f"datasets/{relative[len('data/'):]}")
        last_error: subprocess.CalledProcessError | None = None
        for key in candidates:
            try:
                _s5cmd(["cp", _s3(key), str(local)])
                break
            except subprocess.CalledProcessError as exc:
                last_error = exc
        else:
            if last_error is None:
                raise RuntimeError(f"no B2 candidate for input: {relative}")
            raise last_error

    model = config.get("model") or {}
    adapter_run = model.get("adapter_run_id")
    adapter = model.get("adapter")
    if isinstance(adapter, dict) and adapter.get("kind") == "lora":
        declared_run = adapter.get("run_id")
        declared_path = adapter.get("path")
        if not isinstance(declared_run, str) or not declared_run:
            raise ValueError("LoRA adapter run_id must be a non-empty string")
        expected_path = Path("runs") / declared_run / "adapter"
        if Path(str(declared_path)) != expected_path:
            raise ValueError(
                "LoRA adapter path must be runs/<run_id>/adapter"
            )
        if adapter_run and adapter_run != declared_run:
            raise ValueError("conflicting LoRA adapter run IDs")
        adapter_run = declared_run

    if adapter_run:
        if not isinstance(adapter_run, str):
            raise ValueError("adapter_run_id must be a string")
        local_adapter = repo_root / "runs" / adapter_run / "adapter"
        marker = local_adapter / "adapter_model.safetensors"
        if marker.exists():
            print(f"adapter already local at {local_adapter}, skipping B2 pull")
        else:
            local_adapter.mkdir(parents=True, exist_ok=True)
            _s5cmd(
                [
                    "cp",
                    f"{_s3(f'runs/{adapter_run}/adapter')}/*",
                    f"{local_adapter}/",
                ]
            )
    return config_path


def eval_module(config: dict) -> str:
    module = config.get("eval", {}).get("module")
    if module:
        return str(module)
    if config.get("experiment_id") == "e1_vllm":
        return "lab.eval_vllm_qa"
    return "lab.eval_run_spec"


def train_module(config: dict) -> str:
    """Select a configured training entrypoint."""
    module = (
        (config.get("training") or {}).get("module")
        or (config.get("train") or {}).get("module")
    )
    if module:
        return str(module)
    return "lab.train_sft_lora"


def run_module(
    *,
    repo_root: Path,
    module: str,
    config_path: Path,
) -> dict:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root / "src")
    command = ["python", "-m", module, "--config", str(config_path)]
    print("+", " ".join(command))
    subprocess.run(
        command,
        check=True,
        env=environment,
        cwd=str(repo_root),
    )
    with open(config_path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def sync_outputs(
    *,
    repo_root: Path,
    run_id: str,
    config: dict,
) -> None:
    if "output" in config:
        run_dir = Path(config["output"]["dir"])
    else:
        run_dir = Path(config["eval"]["output_dir"])
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir
    _s5cmd(["cp", f"{run_dir}/*", _s3(f"runs/{run_id}/")])


def write_receipts(
    *,
    repo_root: Path,
    run_id: str,
    target: str = "modal",
) -> None:
    import torch

    run_dir = repo_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    hardware = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cuda_available": cuda_available,
        "device_count": device_count,
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(device_count)
        ],
    }
    environment = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "profile": "modal-cu128",
    }
    hardware_path = run_dir / "hardware.json"
    environment_path = run_dir / "env.json"
    hardware_path.write_text(
        json.dumps(hardware, indent=2) + "\n",
        encoding="utf-8",
    )
    environment_path.write_text(
        json.dumps(environment, indent=2) + "\n",
        encoding="utf-8",
    )
    _s5cmd(
        [
            "cp",
            str(hardware_path),
            _s3(f"runs/{run_id}/hardware.json"),
        ]
    )
    _s5cmd(
        [
            "cp",
            str(environment_path),
            _s3(f"runs/{run_id}/env.json"),
        ]
    )
