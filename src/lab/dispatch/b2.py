import os
import subprocess
from pathlib import Path


def require_env(*keys: str) -> dict[str, str]:
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")
    return {k: os.environ[k] for k in keys}


def s5cmd_base() -> list[str]:
    env = require_env("S3_ENDPOINT_URL")
    return ["s5cmd", "--endpoint-url", env["S3_ENDPOINT_URL"]]


def s3_uri(key: str) -> str:
    bucket = os.environ["S3_BUCKET"]
    key = key.lstrip("/")
    return f"s3://{bucket}/{key}"


def run_s5cmd(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    cmd = s5cmd_base() + args
    print(f"+ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def upload_file(local_path: Path, remote_key: str) -> None:
    run_s5cmd(["cp", str(local_path), s3_uri(remote_key)])


def upload_dir(local_dir: Path, remote_prefix: str) -> None:
    remote_prefix = remote_prefix.rstrip("/")
    run_s5cmd(["cp", f"{local_dir}/*", f"{s3_uri(remote_prefix)}/"])


def download_file(remote_key: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    run_s5cmd(["cp", s3_uri(remote_key), str(local_path)])


def download_prefix(remote_prefix: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_prefix = remote_prefix.rstrip("/")
    run_s5cmd(["cp", f"{s3_uri(remote_prefix)}/*", f"{local_dir}/"], check=False)


def sync_run_outputs(local_run_dir: Path, run_id: str) -> None:
    upload_dir(local_run_dir, f"runs/{run_id}")


def sync_dataset_if_needed(local_path: Path, dataset_key: str) -> None:
    if local_path.exists():
        return
    download_file(dataset_key, local_path)


def pull_run_manifest(run_id: str, local_path: Path) -> None:
    download_file(f"runs/{run_id}/manifest.yaml", local_path)
