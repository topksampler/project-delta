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
    .add_local_dir("experiments", remote_path="/root/lalith-ai-lab/experiments")
    # Local data mount: skip B2 Class-B pulls when cockpit already has corpora/indexes.
    .add_local_dir(
        "data/experiments/e1_vllm",
        remote_path="/root/lalith-ai-lab/data/experiments/e1_vllm",
    )
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

    retrieval = cfg.get("retrieval") or {}
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
        marker = local_adapter / "adapter_model.safetensors"
        if marker.exists():
            print(f"adapter already local at {local_adapter}, skipping B2 pull")
        else:
            local_adapter.mkdir(parents=True, exist_ok=True)
            _s5cmd(["cp", f"{_s3(f'runs/{adapter_run}/adapter')}/*", f"{local_adapter}/"])

    return config_path


def _eval_module(cfg: dict) -> str:
    module = cfg.get("eval", {}).get("module")
    if module:
        return module
    if cfg.get("experiment_id") == "e1_vllm":
        return "lab.eval_vllm_qa"
    return "lab.eval_run_spec"


def _train_module(cfg: dict) -> str:
    """Select train entrypoint; default remains SFT LoRA."""
    module = (
        (cfg.get("training") or {}).get("module")
        or (cfg.get("train") or {}).get("module")
    )
    if module:
        return module
    return "lab.train_sft_lora"


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
    gpu="A10G",
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
    gpu="H100",
    secrets=secrets,
    timeout=60 * 60 * 6,
)
def train(run_id: str, config: str) -> None:
    _timed("env_receipts", lambda: _write_receipts(run_id))
    config_path = _timed("b2_pull_inputs", lambda: _sync_inputs(run_id, config))
    def _compute_train() -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return _run_module(_train_module(cfg), config_path)

    cfg = _timed("compute", _compute_train)
    _timed("b2_push_outputs", lambda: _sync_outputs(run_id, cfg))
    print(f"train complete: {run_id}")


def _run_train_stage(config_rel: str) -> dict:
    """Train one config; prefer local adapter continue (Class-B resilient)."""
    with open(REPO_ROOT / config_rel, "r", encoding="utf-8") as f:
        stage_cfg = yaml.safe_load(f)
    stage_run_id = str(stage_cfg["run_id"])
    _timed(f"env_receipts:{stage_run_id}", lambda rid=stage_run_id: _write_receipts(rid))
    config_path = _timed(
        f"b2_pull_inputs:{stage_run_id}",
        lambda rid=stage_run_id, rel=config_rel: _sync_inputs(rid, rel),
    )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = _timed(
        f"compute_train:{stage_run_id}",
        lambda p=config_path, c=cfg: _run_module(_train_module(c), p),
    )
    try:
        _timed(
            f"b2_push_train:{stage_run_id}",
            lambda rid=stage_run_id, c=cfg: _sync_outputs(rid, c),
        )
    except Exception as exc:  # noqa: BLE001 — uploads may hit caps; keep local chain
        print(f"[warn] train upload skipped for {stage_run_id}: {exc}")
    return cfg


def _run_eval_stage(eval_rel: str) -> None:
    """Eval one config; reuse local adapter after first pull (Class-B resilient)."""
    eval_path = REPO_ROOT / eval_rel
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)
    eval_run_id = str(eval_cfg["run_id"])
    _timed(f"env_receipts:{eval_run_id}", lambda rid=eval_run_id: _write_receipts(rid))
    eval_path = _timed(
        f"b2_pull_inputs:{eval_run_id}",
        lambda rid=eval_run_id, rel=eval_rel: _sync_inputs(rid, rel),
    )
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)
    _timed(
        f"compute_eval:{eval_run_id}",
        lambda p=eval_path, c=eval_cfg: _run_module(_eval_module(c), p),
    )
    probes = eval_cfg.get("eval", {}).get("path")
    samples = Path(eval_cfg["eval"]["output_dir"]) / "samples.jsonl"
    if not samples.is_absolute():
        samples = REPO_ROOT / samples
    if probes and samples.exists():
        summary_cmd = [
            "python",
            str(REPO_ROOT / "experiments/e1_vllm/eval_factory/summarize_run.py"),
            "--samples",
            str(samples),
            "--probes",
            str(REPO_ROOT / probes),
            "--out",
            str(samples.with_name("summary.json")),
        ]
        print("+", " ".join(summary_cmd))
        subprocess.run(summary_cmd, check=False, cwd=str(REPO_ROOT))
    try:
        _timed(
            f"b2_push_eval:{eval_run_id}",
            lambda rid=eval_run_id, c=eval_cfg: _sync_outputs(rid, c),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] eval upload skipped for {eval_run_id}: {exc}")


@app.function(
    image=image,
    gpu="H100",
    secrets=secrets,
    timeout=60 * 60 * 6,
)
def train_then_eval(
    run_id: str,
    config: str,
    eval_configs: list[str],
    train_chain: list[str] | None = None,
) -> None:
    """Train (optionally a chain), then eval against the local final adapter."""
    chain = [c.strip() for c in (train_chain or []) if c and c.strip()]
    # Prior stages first (e.g. teacher), then the primary config (e.g. consolidate).
    stages = chain + [config]
    for stage_rel in stages:
        _run_train_stage(stage_rel)
    print(f"train stages complete: {[s for s in stages]} → eval run_id={run_id}")
    for eval_rel in eval_configs:
        _run_eval_stage(eval_rel)
    print(f"train_then_eval complete: {run_id}")


@app.function(
    image=image,
    gpu="H100",
    secrets=secrets,
    timeout=60 * 60 * 6,
)
def eval_batch(run_id: str, eval_configs: list[str]) -> None:
    """Eval-only chain (e.g. attach vs merge) without retraining."""
    print(f"eval_batch start: run_id={run_id} n={len(eval_configs)}")
    for eval_rel in eval_configs:
        _run_eval_stage(eval_rel)
    print(f"eval_batch complete: {run_id}")


@app.local_entrypoint()
def main(
    run_id: str,
    config: str,
    task: str = "train",
    gpu: str = "",
    eval_configs: str = "",
    train_chain: str = "",
) -> None:
    """Dispatch train/eval with an optional GPU override (e.g. H100, A100, A10G)."""
    if task == "train_then_eval":
        chosen = gpu or "H100"
        evals = [x.strip() for x in eval_configs.split(",") if x.strip()]
        chain = [x.strip() for x in train_chain.split(",") if x.strip()]
        print(
            f"modal: task={task} gpu={chosen} run_id={run_id} "
            f"chain={len(chain)} evals={len(evals)}"
        )
        train_then_eval.with_options(gpu=chosen).remote(
            run_id=run_id,
            config=config,
            eval_configs=evals,
            train_chain=chain,
        )
        return
    if task == "eval_batch":
        chosen = gpu or "H100"
        evals = [x.strip() for x in eval_configs.split(",") if x.strip()]
        print(f"modal: task={task} gpu={chosen} run_id={run_id} evals={len(evals)}")
        eval_batch.with_options(gpu=chosen).remote(
            run_id=run_id,
            eval_configs=evals,
        )
        return
    fn = train if task == "train" else eval_run
    default_gpu = "H100" if task == "train" else "A10G"
    chosen = gpu or default_gpu
    print(f"modal: task={task} gpu={chosen} run_id={run_id}")
    fn.with_options(gpu=chosen).remote(run_id=run_id, config=config)
