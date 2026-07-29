#!/usr/bin/env python3
"""Build path sets that differ between two pinned code snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _py_hashes(snapshot: Path) -> dict[str, str]:
    root = snapshot / "vllm"
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for path in root.rglob("*.py"):
        rel = path.relative_to(snapshot).as_posix()
        out[rel] = hashlib.sha1(path.read_bytes()).hexdigest()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    a = _py_hashes(args.before)
    b = _py_hashes(args.after)
    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    changed = sorted(path for path in set(a) & set(b) if a[path] != b[path])
    payload = {
        "schema": "delta.eval_factory.version_diff_paths.v1",
        "before": str(args.before),
        "after": str(args.after),
        "n_added": len(added),
        "n_removed": len(removed),
        "n_changed": len(changed),
        "added": added,
        "removed": removed,
        "changed": changed,
        "diff_paths": sorted(set(added) | set(removed) | set(changed)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n_diff_paths": len(payload["diff_paths"]),
                "n_added": len(added),
                "n_removed": len(removed),
                "n_changed": len(changed),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
