#!/usr/bin/env python3
"""Build a BM25-ready corpus from unified-diff hunks between two snapshots.

Each chunk is one hunk (or a size-capped slice of a hunk). ``source_path`` is the
repo-relative file path so path-hit metrics stay comparable to code corpora.
Removed lines appear as ``-`` hunk content — searchable without the after-era tree.
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _read_py(snapshot: Path, rel: str) -> str | None:
    path = snapshot / rel
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _py_paths(snapshot: Path) -> set[str]:
    root = snapshot / "vllm"
    if not root.is_dir():
        return set()
    return {p.relative_to(snapshot).as_posix() for p in root.rglob("*.py") if p.is_file()}


def _split_hunks(unified: list[str]) -> list[tuple[str, list[str]]]:
    """Return (hunk_header, lines including header) for each @@ hunk."""
    hunks: list[tuple[str, list[str]]] = []
    cur_header = ""
    cur: list[str] = []
    for line in unified:
        if line.startswith("@@"):
            if cur:
                hunks.append((cur_header, cur))
            cur_header = line.rstrip("\n")
            cur = [line]
        elif cur:
            cur.append(line)
    if cur:
        hunks.append((cur_header, cur))
    return hunks


def _chunk_text(text: str, *, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            nl = text.rfind("\n", start + max_chars // 2, end)
            if nl > start:
                end = nl + 1
        parts.append(text[start:end])
        if end >= len(text):
            break
        start = end
    return parts


def build_diff_hunk_corpus(
    *,
    before: Path,
    after: Path,
    alias: str,
    before_tag: str,
    after_tag: str,
    max_chars: int = 1200,
) -> list[dict]:
    before_paths = _py_paths(before)
    after_paths = _py_paths(after)
    all_paths = sorted(before_paths | after_paths)
    rows: list[dict] = []
    for rel in all_paths:
        old = _read_py(before, rel)
        new = _read_py(after, rel)
        if old is None and new is None:
            continue
        if old is None:
            change_type = "added"
            old_lines: list[str] = []
            new_lines = (new or "").splitlines()
        elif new is None:
            change_type = "removed"
            old_lines = (old or "").splitlines()
            new_lines = []
        elif old == new:
            continue
        else:
            change_type = "changed"
            old_lines = old.splitlines()
            new_lines = new.splitlines()

        unified = [
            ln if ln.endswith("\n") else ln + "\n"
            for ln in difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                n=3,
                lineterm="",
            )
        ]
        hunks = _split_hunks(unified)
        if not hunks and change_type in {"added", "removed"}:
            # empty-ish file edge: synthesize one hunk from full content
            body = "".join(f"+{ln}" for ln in new_lines) if change_type == "added" else "".join(
                f"-{ln}" for ln in old_lines
            )
            hunks = [("@@ synthesized @@", [f"@@ synthesized @@\n", body])]

        for hi, (header, hunk_lines) in enumerate(hunks):
            body = "".join(hunk_lines)
            preamble = (
                f"diff --git a/{rel} b/{rel}\n"
                f"change_type: {change_type}\n"
                f"before: {before_tag} after: {after_tag}\n"
                f"{header}\n"
            )
            # avoid duplicating header if already first line
            if hunk_lines and hunk_lines[0].startswith("@@"):
                text = (
                    f"diff --git a/{rel} b/{rel}\n"
                    f"change_type: {change_type}\n"
                    f"before: {before_tag} after: {after_tag}\n"
                    + body
                )
            else:
                text = preamble + body
            parts = _chunk_text(text, max_chars=max_chars)
            for ci, chunk in enumerate(parts):
                rows.append(
                    {
                        "chunk_id": f"{alias}:{rel}:{hi}:{ci}",
                        "text": chunk,
                        "source_path": rel,
                        "vllm_version": after_tag.lstrip("v"),
                        "git_tag": after_tag,
                        "before_tag": before_tag,
                        "after_tag": after_tag,
                        "change_type": change_type,
                        "hunk_index": hi,
                        "chunk_index": ci,
                        "chunk_total": len(parts),
                        "modality": "diff_hunk",
                    }
                )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--alias", required=True, help="e.g. diff_hunk_022_023")
    p.add_argument("--before-tag", required=True)
    p.add_argument("--after-tag", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-chars", type=int, default=1200)
    args = p.parse_args()

    before = args.before if args.before.is_absolute() else REPO_ROOT / args.before
    after = args.after if args.after.is_absolute() else REPO_ROOT / args.after
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out

    rows = build_diff_hunk_corpus(
        before=before,
        after=after,
        alias=args.alias,
        before_tag=args.before_tag,
        after_tag=args.after_tag,
        max_chars=args.max_chars,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    by_ct: dict[str, int] = {}
    for r in rows:
        by_ct[r["change_type"]] = by_ct.get(r["change_type"], 0) + 1
    print(
        json.dumps(
            {"out": str(out), "n_chunks": len(rows), "by_change_type": by_ct},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
