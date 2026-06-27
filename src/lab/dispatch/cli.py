import argparse
import json
import os
from pathlib import Path

import yaml
from rich import print

from lab.dispatch import b2, costs, lambda_driver, manifest, modal_driver
from lab.dispatch.timing import RunTimer


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
    if args.target == "lambda":
        os.environ["LAB_ENV_PROFILE"] = args.env_profile
        os.environ["LAB_PYTHON_VERSION"] = args.python_version
        os.environ["LAB_REQUIRE_GPU"] = "0" if args.allow_cpu else "1"

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
    timer = RunTimer(run_id=run_id, target=args.target)

    with timer.stage("b2_upload_manifest"):
        b2.upload_file(manifest_path, f"runs/{run_id}/manifest.yaml")
        b2.upload_file(config_path, f"runs/{run_id}/config.yaml")

    try:
        if args.target == "modal":
            with timer.stage("modal_dispatch"):
                modal_driver.dispatch(
                    repo_root=root,
                    task=task,
                    config_path=config_path,
                    run_id=run_id,
                    gpu=args.gpu,
                )
        elif args.target == "lambda":
            with timer.stage("lambda_dispatch"):
                lambda_driver.dispatch(
                    repo_root=root,
                    run_id=run_id,
                    launch=args.launch,
                    terminate=args.terminate,
                    instance_id=args.instance_id,
                    instance_ip=args.instance_ip,
                    timer=timer,
                )
        else:
            raise SystemExit(f"Unknown target: {args.target}")
    finally:
        timings_path = root / "runs" / run_id / "timings.json"
        timer.write(timings_path)
        try:
            b2.upload_file(timings_path, f"runs/{run_id}/timings.json")
        except Exception as exc:
            print(f"[yellow]timings upload skipped: {exc}[/yellow]")

        pricing = costs.load_pricing(root)
        modal_gpu = args.gpu or pricing.get("modal", {}).get("default_gpu", "T4")
        lambda_type = pricing.get("lambda", {}).get("default_instance_type", "gpu_1x_a10")
        if args.target == "lambda":
            lam_cfg = root / "configs/runtime/lambda.yaml"
            if lam_cfg.exists():
                with open(lam_cfg, "r", encoding="utf-8") as f:
                    lambda_type = yaml.safe_load(f).get("instance_type_name", lambda_type)

        cost = costs.estimate_run_cost(
            repo_root=root,
            run_id=run_id,
            target=args.target,
            stages=timer.stages,
            total_sec=timer.total_sec(),
            launched=bool(args.launch),
            modal_gpu=modal_gpu if args.target == "modal" else None,
            lambda_instance_type=lambda_type if args.target == "lambda" else None,
        )
        cost_path = root / "runs" / run_id / "cost.json"
        costs.write_cost(cost_path, cost)
        try:
            b2.upload_file(cost_path, f"runs/{run_id}/cost.json")
        except Exception as exc:
            print(f"[yellow]cost upload skipped: {exc}[/yellow]")
        print(
            f"[green]estimated cost[/green] ${cost['estimated_cost_usd']:.4f} "
            f"({cost['billable_seconds']:.0f}s @ ${cost['rate_usd_per_hour']:.2f}/hr)"
        )
        timer.summary()


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


def _collect_costs(root: Path, *, from_b2: bool) -> list[dict]:
    runs_dir = root / "runs"
    if from_b2:
        try:
            lines = b2.list_prefix("runs/")
        except Exception as exc:
            print(f"[yellow]B2 list skipped: {exc}[/yellow]")
            lines = []
        run_ids = set()
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            key = parts[-1]
            if key.startswith("runs/") and key.count("/") >= 2:
                run_ids.add(key.split("/")[1])
        for rid in sorted(run_ids):
            local = runs_dir / rid
            local.mkdir(parents=True, exist_ok=True)
            for name in ("cost.json", "timings.json"):
                try:
                    b2.download_file(f"runs/{rid}/{name}", local / name)
                except Exception:
                    pass

    pricing = costs.load_pricing(root)
    modal_gpu = pricing.get("modal", {}).get("default_gpu", "T4")
    lambda_type = pricing.get("lambda", {}).get("default_instance_type", "gpu_1x_a10")
    lam_cfg = root / "configs/runtime/lambda.yaml"
    if lam_cfg.exists():
        with open(lam_cfg, "r", encoding="utf-8") as f:
            lambda_type = yaml.safe_load(f).get("instance_type_name", lambda_type)

    out: list[dict] = []
    if not runs_dir.exists():
        return out
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        cost_path = run_dir / "cost.json"
        if cost_path.exists():
            try:
                out.append(json.loads(cost_path.read_text(encoding="utf-8")))
                continue
            except Exception:
                pass
        timings_path = run_dir / "timings.json"
        if not timings_path.exists():
            continue
        timings = json.loads(timings_path.read_text(encoding="utf-8"))
        stages = timings.get("stages_sec", {})
        launched = "lambda_launch" in stages
        target = timings.get("target", "unknown")
        estimate = costs.estimate_from_timings(
            repo_root=root,
            timings_path=timings_path,
            launched=launched,
            modal_gpu=modal_gpu if target == "modal" else None,
            lambda_instance_type=lambda_type if target == "lambda" else None,
        )
        if estimate:
            costs.write_cost(cost_path, estimate)
            out.append(estimate)
    return out


def cmd_credits_runs(args: argparse.Namespace) -> None:
    root = repo_root()
    load_dotenv(root / ".env")
    rows = _collect_costs(root, from_b2=args.from_b2)
    if args.target:
        rows = [r for r in rows if r.get("target") == args.target]
    if not rows:
        print("No cost.json files found. Run jobs first or pass --from-b2.")
        return
    for row in rows:
        print(
            f"{row.get('run_id'):40}  {row.get('target'):6}  "
            f"${row.get('estimated_cost_usd', 0):8.4f}  "
            f"{row.get('billable_seconds', 0):7.0f}s  {row.get('resource', '-')}"
        )


def cmd_credits_summary(args: argparse.Namespace) -> None:
    root = repo_root()
    load_dotenv(root / ".env")
    rows = _collect_costs(root, from_b2=args.from_b2)
    if args.target:
        rows = [r for r in rows if r.get("target") == args.target]
    by_target: dict[str, float] = {}
    for row in rows:
        t = row.get("target", "unknown")
        by_target[t] = by_target.get(t, 0.0) + float(row.get("estimated_cost_usd", 0))
    total = sum(by_target.values())
    print(f"[bold]tracked spend[/bold] ${total:.4f} across {len(rows)} runs")
    for target, amount in sorted(by_target.items()):
        print(f"  {target:8} ${amount:.4f}")


def cmd_credits_balance(args: argparse.Namespace) -> None:
    root = repo_root()
    load_dotenv(root / ".env")
    rows = _collect_costs(root, from_b2=args.from_b2)
    by_target: dict[str, float] = {}
    for row in rows:
        t = row.get("target", "unknown")
        by_target[t] = by_target.get(t, 0.0) + float(row.get("estimated_cost_usd", 0))
    baselines = costs.credit_baselines()
    print("[bold]credit baselines[/bold] (set LAMBDA_CREDITS_USD / MODAL_CREDITS_USD in .env)")
    for target, baseline in (("lambda", baselines["lambda_usd"]), ("modal", baselines["modal_usd"])):
        spent = by_target.get(target, 0.0)
        if baseline is None:
            print(f"  {target:8} baseline not set  tracked spend ${spent:.4f}")
        else:
            remaining = baseline - spent
            print(f"  {target:8} ${remaining:.2f} remaining of ${baseline:.2f}  (spent ${spent:.4f})")


def cmd_hardware_probe(args: argparse.Namespace) -> None:
    root = repo_root()
    load_dotenv(root / ".env")
    if args.target != "lambda":
        raise SystemExit("hardware probe currently supports --target lambda")
    lambda_driver.probe_hardware(
        repo_root=root,
        instance_id=args.instance_id,
        instance_ip=args.instance_ip,
    )


def cmd_env_build(args: argparse.Namespace) -> None:
    root = repo_root()
    load_dotenv(root / ".env")
    if args.target != "lambda":
        raise SystemExit("env build currently supports --target lambda")
    os.environ["LAB_ENV_PROFILE"] = args.profile
    os.environ["LAB_PYTHON_VERSION"] = args.python_version
    os.environ["LAB_REQUIRE_GPU"] = "0" if args.allow_cpu else "1"
    lambda_driver.build_env(
        repo_root=root,
        instance_id=args.instance_id,
        instance_ip=args.instance_ip,
        run_id=args.run_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="lab", description="Dispatch lab runs from the Mac cockpit.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Dispatch a train or eval job.")
    run.add_argument("--target", choices=["modal", "lambda"], required=True)
    run.add_argument("--config", required=True, help="Path to YAML config.")
    run.add_argument("--task", choices=["train", "eval"], help="Override task inference.")
    run.add_argument("--gpu", help="Modal GPU type, e.g. A10G, A100.")
    run.add_argument("--env-profile", default="auto", help="Lambda env profile: auto, cu128, cu126, cu124, cu121, cpu.")
    run.add_argument("--python-version", default="3.11", help="Lambda uv-managed Python version.")
    run.add_argument("--allow-cpu", action="store_true", help="Lambda: allow CPU fallback instead of failing the run.")
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

    hardware = sub.add_parser("hardware", help="Hardware lifecycle commands.")
    hardware_sub = hardware.add_subparsers(dest="hardware_cmd", required=True)
    probe = hardware_sub.add_parser("probe", help="Probe worker hardware without running a job.")
    probe.add_argument("--target", choices=["lambda", "modal"], required=True)
    probe.add_argument("--instance-id")
    probe.add_argument("--instance-ip")
    probe.set_defaults(func=cmd_hardware_probe)

    env = sub.add_parser("env", help="Environment lifecycle commands.")
    env_sub = env.add_subparsers(dest="env_cmd", required=True)
    build = env_sub.add_parser("build", help="Build and verify a worker environment.")
    build.add_argument("--target", choices=["lambda", "modal"], required=True)
    build.add_argument("--instance-id")
    build.add_argument("--instance-ip")
    build.add_argument("--run-id", default="env-build")
    build.add_argument("--profile", default="auto", help="auto, cu128, cu126, cu124, cu121, cpu")
    build.add_argument("--python-version", default="3.11")
    build.add_argument("--allow-cpu", action="store_true")
    build.set_defaults(func=cmd_env_build)

    credits = sub.add_parser("credits", help="Per-run cost estimates and credit tracking.")
    credits_sub = credits.add_subparsers(dest="credits_cmd", required=True)
    cr_runs = credits_sub.add_parser("runs", help="List per-run cost.json entries.")
    cr_runs.add_argument("--target", choices=["modal", "lambda"])
    cr_runs.add_argument("--from-b2", action="store_true", help="Pull cost.json from B2 first.")
    cr_runs.set_defaults(func=cmd_credits_runs)
    cr_sum = credits_sub.add_parser("summary", help="Aggregate tracked spend by target.")
    cr_sum.add_argument("--target", choices=["modal", "lambda"])
    cr_sum.add_argument("--from-b2", action="store_true")
    cr_sum.set_defaults(func=cmd_credits_summary)
    cr_bal = credits_sub.add_parser("balance", help="Remaining credits vs tracked spend.")
    cr_bal.add_argument("--from-b2", action="store_true")
    cr_bal.set_defaults(func=cmd_credits_balance)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
