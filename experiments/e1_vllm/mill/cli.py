#!/usr/bin/env python3
"""mill CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOTS = REPO_ROOT / "data" / "experiments" / "e1_vllm" / "snapshots"


def _snapshot_from_args(args: argparse.Namespace) -> Path:
    if getattr(args, "snapshot", None):
        return args.snapshot
    if getattr(args, "tag", None):
        return SNAPSHOTS / args.tag.replace("/", "_")
    raise SystemExit("provide --snapshot or --tag")


def cmd_pin(args: argparse.Namespace) -> int:
    from pin import pin_snapshot

    out = args.out or (SNAPSHOTS / args.tag.replace("/", "_"))
    print(json.dumps(pin_snapshot(repo=args.repo, tag=args.tag, out=out), indent=2))
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    from extract import extract

    print(json.dumps(extract(_snapshot_from_args(args)), indent=2))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from generate import generate

    print(json.dumps(generate(_snapshot_from_args(args), limit=args.limit), indent=2))
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    from gate import gate

    print(
        json.dumps(
            gate(_snapshot_from_args(args), eval_denylist=args.eval_denylist),
            indent=2,
        )
    )
    return 0


def cmd_emit(args: argparse.Namespace) -> int:
    from emit import emit

    print(json.dumps(emit(_snapshot_from_args(args), version=args.version), indent=2))
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from build import build

    snapshot = args.out or (SNAPSHOTS / args.tag.replace("/", "_"))
    summary = build(
        repo=args.repo,
        tag=args.tag,
        snapshot=snapshot,
        eval_denylist=args.eval_denylist,
        limit=args.limit,
        version=args.version,
    )
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mill")
    sub = p.add_subparsers(dest="cmd", required=True)

    pin = sub.add_parser("pin", help="repo + tag → snapshot_meta.json")
    pin.add_argument("--repo", required=True)
    pin.add_argument("--tag", required=True)
    pin.add_argument("--out", type=Path, default=None)
    pin.set_defaults(func=cmd_pin)

    for name, help_text, handler in (
        ("extract", "snapshot → corpus + structure_index", cmd_extract),
        ("generate", "structure → candidate rows (T1/T3/T4/T5)", cmd_generate),
        ("gate", "validate + dedupe + denylist", cmd_gate),
        ("emit", "gated → train JSONL + manifest", cmd_emit),
    ):
        s = sub.add_parser(name, help=help_text)
        s.add_argument("--snapshot", type=Path, default=None)
        s.add_argument("--tag", default=None)
        if name == "generate":
            s.add_argument("--limit", type=int, default=None)
        if name == "gate":
            s.add_argument("--eval-denylist", type=Path, default=None)
        if name == "emit":
            s.add_argument("--version", type=int, default=1)
        s.set_defaults(func=handler)

    b = sub.add_parser("build", help="pin → extract → generate → gate → emit")
    b.add_argument("--repo", required=True)
    b.add_argument("--tag", required=True)
    b.add_argument("--out", type=Path, default=None, help="snapshot dir")
    b.add_argument("--eval-denylist", type=Path, default=None)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--version", type=int, default=1)
    b.set_defaults(func=cmd_build)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
