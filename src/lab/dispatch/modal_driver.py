import subprocess
import time
from pathlib import Path

import yaml


def _modal_bin(repo_root: Path) -> str:
    venv_modal = repo_root / ".venv" / "bin" / "modal"
    if venv_modal.exists():
        return str(venv_modal)
    return "modal"


def dispatch(
    *,
    repo_root: Path,
    task: str,
    config_path: Path,
    run_id: str,
    gpu: str | None,
) -> None:
    with open(config_path, "r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    configured_app = (cfg.get("modal") or {}).get("app_path")
    app_rel = Path(configured_app or "infra/modal/app.py")
    if app_rel.is_absolute() or ".." in app_rel.parts:
        raise ValueError("modal.app_path must stay within the repository")
    app_path = repo_root / app_rel
    if not app_path.is_file():
        raise FileNotFoundError(f"Modal app not found: {app_path}")
    # Use local entrypoint so --gpu actually overrides the decorator default.
    default_gpu = "H100" if task == "train" else "A10G"
    chosen = gpu or default_gpu

    cmd = [
        _modal_bin(repo_root),
        "run",
        str(app_path),
        "--run-id",
        run_id,
        "--config",
        str(config_path.relative_to(repo_root)),
        "--task",
        "train" if task == "train" else "eval",
        "--gpu",
        chosen,
    ]

    print(f"+ {' '.join(cmd)}")
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=str(repo_root), check=True)
    print(f"[timing] modal_run: {time.perf_counter() - t0:.1f}s")
