import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import requests
import yaml

from lab.dispatch.timing import RunTimer

LAMBDA_API = os.environ.get("LAMBDA_API_BASE", "https://cloud.lambdalabs.com/api/v1")
LAMBDA_RUN_JOB_PATH = Path("infra/lambda/run_job.sh")
LAMBDA_BOOTSTRAP_PATH = Path("infra/lambda/bootstrap_env.sh")
SSH_OPTS = [
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "ConnectTimeout=30",
]


@contextmanager
def _stage(timer: RunTimer | None, name: str):
    if timer:
        with timer.stage(name):
            yield
    else:
        yield


def _auth() -> tuple[str, str]:
    api_key = os.environ.get("LAMBDA_API_KEY")
    if not api_key:
        raise RuntimeError("LAMBDA_API_KEY is not set")
    return (api_key, "")


def _request(method: str, path: str, **kwargs) -> dict:
    url = f"{LAMBDA_API}{path}"
    resp = requests.request(method, url, auth=_auth(), timeout=60, **kwargs)
    if not resp.ok:
        raise RuntimeError(f"Lambda API {method} {path} failed: {resp.status_code} {resp.text}")
    return resp.json()


def list_instances() -> list[dict]:
    data = _request("GET", "/instances")
    return data.get("data", [])


def get_instance(instance_id: str) -> dict | None:
    for inst in list_instances():
        if inst.get("id") == instance_id:
            return inst
    return None


def launch_instance(
    *,
    region_name: str,
    instance_type_name: str,
    ssh_key_names: list[str],
    name: str,
) -> str:
    payload = {
        "region_name": region_name,
        "instance_type_name": instance_type_name,
        "ssh_key_names": ssh_key_names,
        "name": name,
    }
    data = _request("POST", "/instance-operations/launch", json=payload)
    ids = data.get("data", {}).get("instance_ids", [])
    if not ids:
        raise RuntimeError(f"Lambda launch returned no instance ids: {data}")
    return ids[0]


def terminate_instance(instance_id: str) -> None:
    payload = {"instance_ids": [instance_id]}
    _request("POST", "/instance-operations/terminate", json=payload)


def wait_for_ssh(instance_id: str, *, timeout_sec: int = 900, poll_sec: int = 15) -> dict:
    deadline = time.time() + timeout_sec
    t0 = time.perf_counter()
    while time.time() < deadline:
        inst = get_instance(instance_id)
        if inst and inst.get("status") == "active" and inst.get("ip"):
            return inst
        status = inst.get("status") if inst else "missing"
        elapsed = time.perf_counter() - t0
        print(f"waiting for instance {instance_id}: status={status} ({elapsed:.0f}s elapsed)")
        time.sleep(poll_sec)
    raise TimeoutError(f"Instance {instance_id} did not become SSH-ready in {timeout_sec}s")


def load_runtime_defaults(repo_root: Path) -> dict:
    path = repo_root / "configs/runtime/lambda.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ssh_env_exports() -> str:
    keys = [
        "S3_BUCKET",
        "S3_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "LAB_ENV_PROFILE",
        "LAB_PYTHON_VERSION",
        "LAB_REQUIRE_GPU",
    ]
    parts = []
    for key in keys:
        val = os.environ.get(key)
        if val:
            if key == "S3_ENDPOINT_URL":
                val = val.strip()
            parts.append(f'export {key}={_shell_quote(val)}')
    return "\n".join(parts)


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def runtime_ssh_user(repo_root: Path) -> str:
    return load_runtime_defaults(repo_root).get("ssh_user", "ubuntu")


def resolve_host(*, instance_id: str | None, instance_ip: str | None) -> str:
    if instance_ip:
        return instance_ip
    if instance_id:
        inst = get_instance(instance_id)
        if not inst:
            raise RuntimeError(f"Instance not found: {instance_id}")
        host = inst.get("ip")
        if not host:
            raise RuntimeError(f"Instance has no IP yet: {instance_id}")
        return host
    raise RuntimeError("Provide --instance-id or --instance-ip")


def remote_exec(*, host: str, user: str, body: str) -> None:
    cmd = ["ssh", *SSH_OPTS, f"{user}@{host}", "bash -s"]
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, input=body, text=True, check=True)


def rsync_repo(*, host: str, user: str, repo_root: Path, remote_dir: str) -> None:
    subprocess.run(
        ["ssh", *SSH_OPTS, f"{user}@{host}", f"mkdir -p {remote_dir}"],
        check=True,
    )
    cmd = [
        "rsync",
        "-az",
        "-e",
        "ssh " + " ".join(SSH_OPTS),
        "--exclude",
        ".git/",
        "--exclude",
        ".venv/",
        "--exclude",
        "runs/",
        "--exclude",
        "tmp/",
        "--exclude",
        "data/",
        f"{repo_root}/",
        f"{user}@{host}:{remote_dir}/",
    ]
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def probe_hardware(*, repo_root: Path, instance_id: str | None, instance_ip: str | None) -> None:
    user = runtime_ssh_user(repo_root)
    host = resolve_host(instance_id=instance_id, instance_ip=instance_ip)
    remote_exec(
        host=host,
        user=user,
        body=r"""
set -euo pipefail
echo "=== hardware probe ==="
hostname
uname -a
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi: missing"
fi
""",
    )


def build_env(
    *,
    repo_root: Path,
    instance_id: str | None,
    instance_ip: str | None,
    run_id: str,
) -> None:
    runtime = load_runtime_defaults(repo_root)
    user = runtime.get("ssh_user", "ubuntu")
    host = resolve_host(instance_id=instance_id, instance_ip=instance_ip)
    remote_dir = os.environ.get("LAB_REPO_DIR", "/home/ubuntu/lalith-ai-lab")
    rsync_repo(host=host, user=user, repo_root=repo_root, remote_dir=remote_dir)
    preamble = "\n".join(
        [
            "set -euo pipefail",
            ssh_env_exports(),
            f'export LAB_RUN_ID={_shell_quote(run_id)}',
            f'export LAB_REPO_DIR={_shell_quote(remote_dir)}',
            "",
        ]
    )
    remote_exec(
        host=host,
        user=user,
        body=preamble
        + f"cd {_shell_quote(remote_dir)}\n{LAMBDA_BOOTSTRAP_PATH.as_posix()}\n",
    )


def run_remote_job(
    *,
    host: str,
    user: str,
    run_id: str,
    repo_url: str,
    repo_branch: str,
    remote_script: Path,
) -> None:
    remote_body = remote_script.read_text(encoding="utf-8")
    preamble = "\n".join(
        [
            "set -euo pipefail",
            ssh_env_exports(),
            f'export LAB_RUN_ID={_shell_quote(run_id)}',
            f'export LAB_REPO_URL={_shell_quote(repo_url)}',
            f'export LAB_REPO_BRANCH={_shell_quote(repo_branch)}',
            "export LAB_SKIP_GIT=1",
            "",
        ]
    )
    cmd = [
        "ssh",
        *SSH_OPTS,
        f"{user}@{host}",
        "bash -s",
    ]
    print(f"+ {' '.join(cmd)}  # run_id={run_id}")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write(preamble + remote_body)
    proc.stdin.close()
    for line in proc.stdout:
        print(line, end="")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Remote job failed with exit code {rc}")


def dispatch(
    *,
    repo_root: Path,
    run_id: str,
    launch: bool,
    terminate: bool,
    instance_id: str | None,
    instance_ip: str | None,
    timer: RunTimer | None = None,
) -> None:
    runtime = load_runtime_defaults(repo_root)
    remote_script = repo_root / LAMBDA_RUN_JOB_PATH
    repo_url = os.environ.get("LAB_REPO_URL", runtime.get("repo_url", ""))
    repo_branch = os.environ.get("LAB_REPO_BRANCH", runtime.get("repo_branch", "main"))
    ssh_user = runtime.get("ssh_user", "ubuntu")

    launched_id = None
    host = instance_ip

    try:
        if launch:
            with _stage(timer, "lambda_launch"):
                launched_id = launch_instance(
                    region_name=runtime["region_name"],
                    instance_type_name=runtime["instance_type_name"],
                    ssh_key_names=runtime["ssh_key_names"],
                    name=f"lab-{run_id}"[:60],
                )
                print(f"launched instance: {launched_id}")
            with _stage(timer, "lambda_boot"):
                inst = wait_for_ssh(launched_id)
            host = inst["ip"]
        elif instance_id:
            inst = get_instance(instance_id)
            if not inst:
                raise RuntimeError(f"Instance not found: {instance_id}")
            host = inst.get("ip")
            launched_id = instance_id
        elif not host:
            raise RuntimeError("Provide --launch, --instance-id, or --instance-ip for lambda runs")

        if not host:
            raise RuntimeError("No SSH host available for lambda run")

        print(f"dispatching to lambda host={host} run_id={run_id}")
        remote_dir = os.environ.get("LAB_REPO_DIR", "/home/ubuntu/lalith-ai-lab")
        with _stage(timer, "rsync_code"):
            rsync_repo(host=host, user=ssh_user, repo_root=repo_root, remote_dir=remote_dir)
        with _stage(timer, "remote_job"):
            run_remote_job(
                host=host,
                user=ssh_user,
                run_id=run_id,
                repo_url=repo_url,
                repo_branch=repo_branch,
                remote_script=remote_script,
            )
    finally:
        if terminate and launched_id and launch:
            with _stage(timer, "lambda_terminate"):
                print(f"terminating instance: {launched_id}")
                terminate_instance(launched_id)
