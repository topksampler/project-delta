"""CLI for the version-delta eval factory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from . import PROTOCOL_ID
from .audit import build_audit
from .factory import build
from .finalize_surfaces import finalize
from .validate import seal, validate

REPO_ROOT = Path(__file__).resolve().parents[3]


def _cmd_build(args: argparse.Namespace) -> int:
    result = build(
        repo_root=REPO_ROOT,
        snapshots_path=args.snapshots,
        before_alias=args.before,
        after_alias=args.after,
        protocol_id=args.protocol,
    )
    print(json.dumps(result, indent=2))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    result = validate(args.artifact_dir)
    print(json.dumps(result, indent=2))
    return 0 if result["freeze_ready"] else 2


def _cmd_import_generated(args: argparse.Namespace) -> int:
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    destination = args.artifact_dir / "probes_eval.jsonl"
    shutil.copyfile(args.generated, destination)
    if args.metrics:
        shutil.copyfile(args.metrics, args.artifact_dir / "surface_generation.json")
    print(json.dumps({"probes_eval": str(destination)}, indent=2))
    return 0


def _cmd_build_audit(args: argparse.Namespace) -> int:
    print(json.dumps(build_audit(args.artifact_dir, size=args.size), indent=2))
    return 0


def _cmd_finalize_surfaces(args: argparse.Namespace) -> int:
    result = finalize(
        artifact_dir=args.artifact_dir,
        teacher_generated=args.generated,
        teacher_rejected=args.rejected,
        teacher_metrics=args.metrics,
    )
    print(json.dumps(result, indent=2))
    return 0


def _cmd_seal(args: argparse.Namespace) -> int:
    print(json.dumps(seal(args.artifact_dir), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="e1-eval-factory")
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("build", help="extract claims and emit split probes")
    build_parser.add_argument(
        "--snapshots",
        type=Path,
        default=REPO_ROOT / "experiments" / "e1_vllm" / "snapshots.yaml",
    )
    build_parser.add_argument("--before", required=True, help="snapshot alias")
    build_parser.add_argument("--after", required=True, help="snapshot alias")
    build_parser.add_argument("--protocol", default=PROTOCOL_ID)
    build_parser.set_defaults(func=_cmd_build)

    validate_parser = sub.add_parser("validate", help="run mandatory freeze gates")
    validate_parser.add_argument("--artifact-dir", type=Path, required=True)
    validate_parser.set_defaults(func=_cmd_validate)

    import_parser = sub.add_parser(
        "import-generated",
        help="attach teacher-generated eval surfaces to a pre-freeze artifact",
    )
    import_parser.add_argument("--artifact-dir", type=Path, required=True)
    import_parser.add_argument("--generated", type=Path, required=True)
    import_parser.add_argument("--metrics", type=Path)
    import_parser.set_defaults(func=_cmd_import_generated)

    audit_parser = sub.add_parser(
        "build-audit",
        help="create a deterministic delta-oversampled human audit packet",
    )
    audit_parser.add_argument("--artifact-dir", type=Path, required=True)
    audit_parser.add_argument("--size", type=int, default=50)
    audit_parser.set_defaults(func=_cmd_build_audit)

    finalize_parser = sub.add_parser(
        "finalize-surfaces",
        help="combine accepted teacher wording with neutral delta fallbacks",
    )
    finalize_parser.add_argument("--artifact-dir", type=Path, required=True)
    finalize_parser.add_argument("--generated", type=Path, required=True)
    finalize_parser.add_argument("--rejected", type=Path)
    finalize_parser.add_argument("--metrics", type=Path)
    finalize_parser.set_defaults(func=_cmd_finalize_surfaces)

    seal_parser = sub.add_parser(
        "seal",
        help="freeze a passing EvalEnvironment and hash all derived artifacts",
    )
    seal_parser.add_argument("--artifact-dir", type=Path, required=True)
    seal_parser.set_defaults(func=_cmd_seal)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
