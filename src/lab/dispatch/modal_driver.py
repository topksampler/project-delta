import subprocess
from pathlib import Path


def dispatch(
    *,
    repo_root: Path,
    task: str,
    config_path: Path,
    run_id: str,
    gpu: str | None,
) -> None:
    app_path = repo_root / "infra/modal/app.py"
    fn_name = "train" if task == "train" else "eval_run"
    fn = f"{app_path}::{fn_name}"

    cmd = ["modal", "run"]
    if gpu:
        cmd.extend(["--gpu", gpu])
    cmd.extend(
        [
            fn,
            "--run-id",
            run_id,
            "--config",
            str(config_path.relative_to(repo_root)),
        ]
    )

    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(repo_root), check=True)
