import argparse
import os
import subprocess
from pathlib import Path

import yaml
from rich import print

from lab.dispatch import b2, lambda_driver, manifest, modal_driver


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


def run_id_from_config(config_path: Path) -> str:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rid = cfg.get("run_id")
    if not rid:
        raise RuntimeError(f"Config missing run_id: {config_path}")
    return rid


def cmd_run(args: argparse.Namespace) -> None:
    root = repo_root()
    load_dotenv(root / ".env")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    task = manifest.infer_task(config_path, args.task)
    run_id = run_id_from_config(config_path)

    mf = manifest.build_manifest(
        run_id=run_id,
        task=task,
        target=args.target,
        config_path=config_path,
        repo_root=root,
    )
    manifest_path = root / "runs" / run_id / "manifest.yaml"
    manifest.write_manifest(mf, manifest_path)

    print(f"[bold]lab run[/bold] run_id={run_id} task={task} target={args.target}")
    b2.upload_file(manifest_path, f"runs/{run_id}/manifest.yaml")
    b2.upload_file(config_path, f"runs/{run_id}/config.yaml")

    if args.target == "modal":
        modal_driver.dispatch(
            repo_root=root,
            task=task,
            config_path=config_path,
            run_id=run_id,
            gpu=args.gpu,
        )
    elif args.target == "lambda":
        lambda_driver.dispatch(
            repo_root=root,
            run_id=run_id,
            launch=args.launch,
            terminate=args.terminate,
            instance_id=args.instance_id,
            instance_ip=args.instance_ip,
        )
    else:
        raise SystemExit(f"Unknown target: {args.target}")


def cmd_status(args: argparse.Namespace) -> None:
    root = repo_root()
    load_dotenv(root / ".env")
    run_id = args.run_id
    local_dir = root / "runs" / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        b2.download_file(f"runs/{run_id}/metrics.json", local_dir / "metrics.json")
        print((local_dir / "metrics.json").read_text(encoding="utf-8"))
    except Exception:
        print(f"No metrics.json in B2 for run_id={run_id}")
    try:
        b2.download_file(f"runs/{run_id}/ledger.yaml", local_dir / "ledger.yaml")
        print((local_dir / "ledger.yaml").read_text(encoding="utf-8"))
    except Exception:
        print(f"No ledger.yaml in B2 for run_id={run_id}")


def cmd_lambda_list(_: argparse.Namespace) -> None:
    root = repo_root()
    load_dotenv(root / ".env")
    for inst in lambda_driver.list_instances():
        print(
            f"{inst.get('id')}  {inst.get('status'):10}  "
            f"{inst.get('ip') or '-':16}  {inst.get('name') or '-'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="lab", description="Dispatch lab runs from the Mac cockpit.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Dispatch a train or eval job.")
    run.add_argument("--target", choices=["modal", "lambda"], required=True)
    run.add_argument("--config", required=True, help="Path to YAML config.")
    run.add_argument("--task", choices=["train", "eval"], help="Override task inference.")
    run.add_argument("--gpu", help="Modal GPU type, e.g. A10G, A100.")
    run.add_argument("--launch", action="store_true", help="Lambda: launch a fresh instance via API.")
    run.add_argument(
        "--terminate",
        action="store_true",
        help="Lambda: terminate instance after job (only with --launch).",
    )
    run.add_argument("--instance-id", help="Lambda: use an existing instance id.")
    run.add_argument("--instance-ip", help="Lambda: SSH directly to this IP.")
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="Pull run artifacts from B2.")
    status.add_argument("--run-id", required=True)
    status.set_defaults(func=cmd_status)

    lam = sub.add_parser("lambda", help="Lambda helpers.")
    lam_sub = lam.add_subparsers(dest="lambda_cmd", required=True)
    lam_list = lam_sub.add_parser("list", help="List running Lambda instances.")
    lam_list.set_defaults(func=cmd_lambda_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
