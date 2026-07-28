"""Stage 0 — pin: repo + tag → checkout + snapshot_meta.json."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Trees we hash for content_sha (relative to repo root).
DOC_DIRS = ("docs", "examples")
DOC_FILES = ("README.md",)


def run(cmd: list[str], cwd: Path | None = None) -> str:
    out = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    return out.stdout.strip()


def ensure_checkout(repo: str, tag: str, out: Path) -> None:
    """Make `out` a shallow clone of `repo` at `tag`."""
    out.parent.mkdir(parents=True, exist_ok=True)

    if (out / ".git").exists():
        run(["git", "fetch", "origin", "tag", tag, "--depth", "1"], cwd=out)
        run(["git", "checkout", "--force", tag], cwd=out)
        return

    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"out exists but is not a git repo: {out}")

    run(["git", "clone", "--depth", "1", "--branch", tag, repo, str(out)])


def iter_doc_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for name in DOC_DIRS:
        d = root / name
        if d.is_dir():
            files.extend(p for p in d.rglob("*") if p.is_file())
    for name in DOC_FILES:
        p = root / name
        if p.is_file():
            files.append(p)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def docs_content_sha(root: Path) -> str:
    files = iter_doc_files(root)
    if not files:
        raise RuntimeError(f"no docs files under {root}")

    h = hashlib.sha256()
    for path in files:
        rel = path.relative_to(root).as_posix()
        h.update(rel.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def pin_snapshot(*, repo: str, tag: str, out: Path) -> dict:
    out = out.resolve()
    source = str(Path(repo).expanduser().resolve()) if Path(repo).expanduser().exists() else repo

    ensure_checkout(source, tag, out)

    meta = {
        "repo": source,
        "tag": tag,
        "commit_sha": run(["git", "rev-parse", "HEAD"], cwd=out),
        "docs_dirs": list(DOC_DIRS),
        "docs_files": list(DOC_FILES),
        "content_sha": docs_content_sha(out),
        "pinned_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out": str(out),
    }
    (out / "snapshot_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
