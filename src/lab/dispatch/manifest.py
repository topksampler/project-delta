import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def infer_task(config_path: Path, task: str | None) -> str:
    if task:
        return task
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "training" in cfg or "lora" in cfg:
        return "train"
    return "eval"


def build_manifest(
    *,
    run_id: str,
    task: str,
    target: str,
    config_path: Path,
    repo_root: Path,
) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return {
        "run_id": run_id,
        "task": task,
        "target": target,
        "config_path": str(config_path.relative_to(repo_root)),
        "git_commit": git_commit(),
        "git_branch": git_branch(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
    }


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
