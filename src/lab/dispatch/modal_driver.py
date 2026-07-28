import subprocess
import time
from pathlib import Path


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
    app_path = repo_root / "infra/modal/app.py"
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
