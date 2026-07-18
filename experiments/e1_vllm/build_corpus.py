#!/usr/bin/env python3
"""Build vLLM documentation corpora for e1_vllm (doc_0 / doc_8)."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
SNAPSHOTS_PATH = EXPERIMENT_DIR / "snapshots.yaml"
DATA_DIR = REPO_ROOT / "data" / "experiments" / "e1_vllm"
REPO_CACHE_DIR = DATA_DIR / "repos"

VLLM_GIT_URL = "https://github.com/vllm-project/vllm.git"
DOC_SUFFIXES = {".md", ".rst", ".txt"}
SKIP_DIR_NAMES = {".git", "__pycache__", "_build", "node_modules", ".venv"}

MAX_CHUNK_CHARS = 1_800
CHUNK_OVERLAP_CHARS = 200

B2_PREFIX = "datasets/experiments/e1_vllm"
OUTPUT_NAMES = {"doc_0": "corpus_doc_0.jsonl", "doc_8": "corpus_doc_8.jsonl"}


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocSnapshot:
    key: str
    version: str
    git_tag: str
    docs_base: str


def load_snapshots(path: Path = SNAPSHOTS_PATH) -> dict[str, DocSnapshot]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        key: DocSnapshot(
            key=key,
            version=spec["version"],
            git_tag=spec["git_tag"],
            docs_base=spec["docs_base"],
        )
        for key, spec in raw["vllm_docs"].items()
    }


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------


def repo_cache_path(snapshot: DocSnapshot) -> Path:
    safe_tag = snapshot.git_tag.replace("/", "_")
    return REPO_CACHE_DIR / safe_tag


def ensure_vllm_repo(snapshot: DocSnapshot) -> Path:
    dest = repo_cache_path(snapshot)
    if (dest / ".git").is_dir():
        subprocess.run(["git", "fetch", "origin", "tag", snapshot.git_tag, "--depth", "1"], cwd=dest, check=True)
        subprocess.run(["git", "checkout", snapshot.git_tag], cwd=dest, check=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            raise RuntimeError(f"cache path exists but is not a git repo: {dest}")
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                snapshot.git_tag,
                VLLM_GIT_URL,
                str(dest),
            ],
            check=True,
        )
    return dest


# ---------------------------------------------------------------------------
# discovery + text
# ---------------------------------------------------------------------------


def is_doc_file(path: Path) -> bool:
    return path.suffix.lower() in DOC_SUFFIXES and path.is_file()


def iter_doc_files(repo_root: Path) -> Iterator[Path]:
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        rel = path.relative_to(repo_root)
        rel_posix = rel.as_posix()
        if rel_posix == "README.md":
            yield path
            continue
        if not is_doc_file(path):
            continue
        if rel_posix.startswith("docs/") or rel_posix.startswith("examples/"):
            yield path


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_doc_text(path: Path) -> str:
    return normalize_text(path.read_text(encoding="utf-8", errors="replace"))


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------

SECTION_BREAK_RE = re.compile(
    r"(?:\n(?=#{1,6}\s)|\n(?:={3,}|-{3,})\s*\n)",
    re.MULTILINE,
)


def split_sections(text: str) -> list[str]:
    parts = [p.strip() for p in SECTION_BREAK_RE.split(text) if p.strip()]
    return parts or ([text] if text else [])


def split_long_section(section: str, max_chars: int, overlap: int) -> list[str]:
    if len(section) <= max_chars:
        return [section]

    paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]
    if not paragraphs:
        return [section[:max_chars]]

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        chunks.append("\n\n".join(buf))
        buf = []
        buf_len = 0

    for para in paragraphs:
        if len(para) > max_chars:
            flush()
            start = 0
            while start < len(para):
                end = min(start + max_chars, len(para))
                chunks.append(para[start:end])
                if end >= len(para):
                    break
                start = max(end - overlap, start + 1)
            continue

        extra = len(para) + (2 if buf else 0)
        if buf and buf_len + extra > max_chars:
            flush()
        buf.append(para)
        buf_len += extra

    flush()

    if len(chunks) <= 1:
        return chunks

    overlapped: list[str] = [chunks[0]]
    for chunk in chunks[1:]:
        prev = overlapped[-1]
        tail = prev[-overlap:] if overlap else ""
        overlapped.append((tail + "\n\n" + chunk).strip() if tail else chunk)
    return overlapped


def chunk_text(
    text: str,
    *,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    sections = split_sections(text)
    out: list[str] = []
    for section in sections:
        out.extend(split_long_section(section, max_chars, overlap))
    return [c for c in out if c.strip()]


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


def make_chunk_id(snapshot: DocSnapshot, source_path: str, chunk_index: int) -> str:
    return f"{snapshot.key}:{source_path}:{chunk_index}"


def records_for_file(
    snapshot: DocSnapshot,
    repo_root: Path,
    doc_path: Path,
    *,
    max_chars: int,
    overlap: int,
) -> list[dict]:
    rel = doc_path.relative_to(repo_root).as_posix()
    chunks = chunk_text(read_doc_text(doc_path), max_chars=max_chars, overlap=overlap)
    total = len(chunks)
    return [
        {
            "chunk_id": make_chunk_id(snapshot, rel, i),
            "text": chunk,
            "source_path": rel,
            "vllm_version": snapshot.version,
            "git_tag": snapshot.git_tag,
            "docs_base": snapshot.docs_base,
            "chunk_index": i,
            "chunk_total": total,
        }
        for i, chunk in enumerate(chunks)
    ]


def build_corpus_records(
    snapshot: DocSnapshot,
    repo_root: Path,
    *,
    max_chars: int,
    overlap: int,
) -> list[dict]:
    records: list[dict] = []
    for doc_path in iter_doc_files(repo_root):
        records.extend(
            records_for_file(snapshot, repo_root, doc_path, max_chars=max_chars, overlap=overlap)
        )
    records.sort(key=lambda r: (r["source_path"], r["chunk_index"]))
    return records


# ---------------------------------------------------------------------------
# I/O + upload
# ---------------------------------------------------------------------------


def output_path(doc_key: str) -> Path:
    return DATA_DIR / OUTPUT_NAMES[doc_key]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def upload_corpus(local_path: Path, doc_key: str) -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from lab.dispatch import b2  # noqa: WPS433

    remote_key = f"{B2_PREFIX}/{OUTPUT_NAMES[doc_key]}"
    b2.upload_file(local_path, remote_key)
    print(f"uploaded {local_path} -> s3://{os.environ['S3_BUCKET']}/{remote_key}")


def build_one(
    doc_key: str,
    snapshot: DocSnapshot,
    *,
    upload: bool,
    max_chars: int,
    overlap: int,
) -> Path:
    print(f"[{doc_key}] clone/checkout {snapshot.git_tag} ...")
    repo_root = ensure_vllm_repo(snapshot)

    print(f"[{doc_key}] chunking docs under {repo_root} ...")
    records = build_corpus_records(snapshot, repo_root, max_chars=max_chars, overlap=overlap)
    if not records:
        raise RuntimeError(f"no chunks produced for {doc_key} — check repo layout at {snapshot.git_tag}")

    out = output_path(doc_key)
    write_jsonl(records, out)

    sources = {r["source_path"] for r in records}
    print(
        f"[{doc_key}] wrote {len(records)} chunks from {len(sources)} files -> {out} "
        f"(vllm {snapshot.version})"
    )

    if upload:
        upload_corpus(out, doc_key)

    return out


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build e1_vllm documentation corpora.")
    parser.add_argument(
        "--doc",
        choices=["doc_0", "doc_8", "all"],
        default="all",
        help="which snapshot to build (default: all)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="upload JSONL to B2 (requires .env with S3_* vars)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=MAX_CHUNK_CHARS,
        help=f"max characters per chunk (default: {MAX_CHUNK_CHARS})",
    )
    return parser.parse_args()


def selected_doc_keys(args: argparse.Namespace) -> list[str]:
    if args.doc == "all":
        return ["doc_0", "doc_8"]
    return [args.doc]


def main() -> None:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    snapshots = load_snapshots()

    for doc_key in selected_doc_keys(args):
        if doc_key not in snapshots:
            raise SystemExit(f"unknown doc key: {doc_key}")
        build_one(
            doc_key,
            snapshots[doc_key],
            upload=args.upload,
            max_chars=args.max_chars,
            overlap=CHUNK_OVERLAP_CHARS,
        )


if __name__ == "__main__":
    main()
