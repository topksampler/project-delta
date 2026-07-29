#!/usr/bin/env python3
"""Chunk Python sources under a pinned vLLM snapshot into a retrieval corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def chunk_file(text: str, *, max_chars: int = 1200, overlap: int = 100) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            # prefer breaking on newline
            nl = text.rfind("\n", start + max_chars // 2, end)
            if nl > start:
                end = nl + 1
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_code_corpus(
    *,
    snapshot_dir: Path,
    alias: str,
    version: str,
    git_tag: str,
    max_chars: int = 1200,
) -> list[dict]:
    root = snapshot_dir / "vllm"
    if not root.is_dir():
        raise FileNotFoundError(f"missing package tree: {root}")
    rows: list[dict] = []
    files = sorted(p for p in root.rglob("*.py") if p.is_file())
    for path in files:
        rel = path.relative_to(snapshot_dir).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        parts = chunk_file(text, max_chars=max_chars)
        for i, chunk in enumerate(parts):
            rows.append(
                {
                    "chunk_id": f"{alias}:{rel}:{i}",
                    "text": chunk,
                    "source_path": rel,
                    "vllm_version": version,
                    "git_tag": git_tag,
                    "chunk_index": i,
                    "chunk_total": len(parts),
                    "modality": "code",
                }
            )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True, help="…/snapshots/v0.XX.0")
    p.add_argument("--alias", required=True, help="corpus alias e.g. code_8")
    p.add_argument("--version", required=True)
    p.add_argument("--git-tag", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-chars", type=int, default=1200)
    args = p.parse_args()
    snap = args.snapshot if args.snapshot.is_absolute() else REPO_ROOT / args.snapshot
    rows = build_code_corpus(
        snapshot_dir=snap,
        alias=args.alias,
        version=args.version,
        git_tag=args.git_tag,
        max_chars=args.max_chars,
    )
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    print(json.dumps({"out": str(out), "n_chunks": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
