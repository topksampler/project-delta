"""Project DELTA control-plane command line."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .reconcile import ReconcileRequest, Reconciler
from .registry import plugin_for_repo

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_ROOT = Path(
    os.environ.get("DELTA_ARTIFACT_ROOT", PROJECT_ROOT / "artifacts" / "delta")
)


def _load_receipt(root: Path, run_id: str) -> dict[str, Any]:
    path = root / "reconciles" / run_id / "receipt.json"
    if not path.is_file():
        raise FileNotFoundError(f"unknown reconcile run: {run_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def _reconciler_from_receipt(root: Path, run_id: str) -> Reconciler:
    receipt = _load_receipt(root, run_id)
    plugin = plugin_for_repo(receipt["request"]["repo"], project_root=PROJECT_ROOT)
    return Reconciler(artifact_root=root, plugin=plugin)


def _cmd_reconcile(args: argparse.Namespace) -> dict[str, Any]:
    request = ReconcileRequest(
        repo=args.repo,
        source_from=args.source_from,
        source_to=args.source_to,
        target=args.target,
        dry_run=args.dry_run,
    )
    plugin = plugin_for_repo(request.repo, project_root=PROJECT_ROOT)
    return Reconciler(artifact_root=args.artifact_root, plugin=plugin).reconcile(
        request,
        resume=args.resume,
        through=args.through,
    )


def _cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    return _load_receipt(args.artifact_root, args.run_id)


def _cmd_approve(args: argparse.Namespace) -> dict[str, Any]:
    return _reconciler_from_receipt(args.artifact_root, args.run_id).approve(args.run_id)


def _cmd_rollback(args: argparse.Namespace) -> Any:
    # MEMORY owns pointer mutation; the CLI is only the explicit control surface.
    from .memory import LocalStateRegistry

    transition_id = LocalStateRegistry(args.artifact_root).rollback(args.state_id)
    return {
        "transition_id": transition_id,
        "state_id": args.state_id,
        "reason": args.reason,
    }


def _artifact_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Local durable DELTA artifact root",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="delta")
    sub = parser.add_subparsers(dest="command", required=True)

    reconcile = sub.add_parser("reconcile", help="run or resume a source transition")
    reconcile.add_argument("--repo", required=True)
    reconcile.add_argument("--from", dest="source_from", required=True)
    reconcile.add_argument("--to", dest="source_to", required=True)
    reconcile.add_argument("--target", default="modal")
    reconcile.add_argument("--dry-run", action="store_true")
    reconcile.add_argument("--resume", action="store_true")
    reconcile.add_argument("--through", choices=(
        "resolve", "pin", "build", "sense", "decide", "adapt", "verify", "record"
    ))
    _artifact_argument(reconcile)
    reconcile.set_defaults(handler=_cmd_reconcile)

    status = sub.add_parser("status", help="print a reconcile receipt")
    status.add_argument("run_id")
    _artifact_argument(status)
    status.set_defaults(handler=_cmd_status)

    approve = sub.add_parser("approve", help="approve a verified promotion decision")
    approve.add_argument("run_id")
    _artifact_argument(approve)
    approve.set_defaults(handler=_cmd_approve)

    rollback = sub.add_parser("rollback", help="restore a previously accepted state")
    rollback.add_argument("--state-id", required=True)
    rollback.add_argument("--reason", required=True)
    _artifact_argument(rollback)
    rollback.set_defaults(handler=_cmd_rollback)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except Exception as exc:
        print(f"delta: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
