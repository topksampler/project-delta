"""Stage 1 — extract: snapshot → corpus.jsonl + structure_index.json."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pin import DOC_DIRS, DOC_FILES

MAX_CHARS = 1800
OVERLAP = 200
DOC_SUFFIXES = {".md", ".rst", ".txt"}

HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
FLAG_RE = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")
ENV_RE = re.compile(r"\b(VLLM_[A-Z0-9_]+)\b")
SECTION_RE = re.compile(r"(?:\n(?=#{1,6}\s)|\n(?:={3,}|-{3,})\s*\n)", re.MULTILINE)


def load_meta(snapshot: Path) -> dict:
    path = snapshot / "snapshot_meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path} — run mill pin first")
    return json.loads(path.read_text())


def iter_doc_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for name in DOC_DIRS:
        d = root / name
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in DOC_SUFFIXES:
                paths.append(p)
    for name in DOC_FILES:
        p = root / name
        if p.is_file():
            paths.append(p)
    return paths


def chunk_text(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    sections = [s.strip() for s in SECTION_RE.split(text) if s.strip()] or [text]
    out: list[str] = []
    for section in sections:
        if len(section) <= MAX_CHARS:
            out.append(section)
            continue
        # pack paragraphs
        buf: list[str] = []
        size = 0
        for para in [p.strip() for p in section.split("\n\n") if p.strip()]:
            extra = len(para) + (2 if buf else 0)
            if buf and size + extra > MAX_CHARS:
                out.append("\n\n".join(buf))
                # overlap from previous chunk
                tail = out[-1][-OVERLAP:] if OVERLAP else ""
                buf = [tail, para] if tail else [para]
                size = sum(len(x) for x in buf) + 2 * (len(buf) - 1)
            else:
                buf.append(para)
                size += extra
        if buf:
            out.append("\n\n".join(buf))
    return out


def build_corpus(snapshot: Path, meta: dict) -> list[dict]:
    tag = meta["tag"]
    rows: list[dict] = []
    for path in iter_doc_paths(snapshot):
        rel = path.relative_to(snapshot).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            rows.append(
                {
                    "chunk_id": f"{tag}:{rel}:{i}",
                    "text": chunk,
                    "source_path": rel,
                    "git_tag": tag,
                    "commit_sha": meta["commit_sha"],
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                }
            )
    rows.sort(key=lambda r: (r["source_path"], r["chunk_index"]))
    return rows


def build_structure(corpus: list[dict], meta: dict) -> dict:
    """Headers + CLI flags + env vars mined from chunk text (no LLM)."""
    paths: dict[str, dict] = {}
    symbols: list[dict] = []

    for row in corpus:
        src = row["source_path"]
        text = row["text"]
        chunk_id = row["chunk_id"]

        headers = [m.group(2).strip() for m in HEADER_RE.finditer(text)]
        if src not in paths:
            paths[src] = {"source_path": src, "headers": []}
        for h in headers:
            if h not in paths[src]["headers"]:
                paths[src]["headers"].append(h)
            symbols.append(
                {
                    "kind": "heading",
                    "name": h,
                    "source_path": src,
                    "chunk_id": chunk_id,
                }
            )

        for flag in sorted(set(FLAG_RE.findall(text))):
            symbols.append(
                {
                    "kind": "flag",
                    "name": flag,
                    "source_path": src,
                    "chunk_id": chunk_id,
                }
            )

        for env in sorted(set(ENV_RE.findall(text))):
            symbols.append(
                {
                    "kind": "env",
                    "name": env,
                    "source_path": src,
                    "chunk_id": chunk_id,
                }
            )

    by_kind: dict[str, int] = {}
    for s in symbols:
        by_kind[s["kind"]] = by_kind.get(s["kind"], 0) + 1

    return {
        "tag": meta["tag"],
        "commit_sha": meta["commit_sha"],
        "content_sha": meta["content_sha"],
        "paths": sorted(paths.values(), key=lambda p: p["source_path"]),
        "symbols": symbols,
        "stats": {
            "n_paths": len(paths),
            "n_chunks": len(corpus),
            "n_symbols": len(symbols),
            "by_kind": by_kind,
        },
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract(snapshot: Path) -> dict:
    """Write corpus.jsonl + structure_index.json into the snapshot dir."""
    snapshot = snapshot.resolve()
    meta = load_meta(snapshot)

    corpus = build_corpus(snapshot, meta)
    structure = build_structure(corpus, meta)

    corpus_path = snapshot / "corpus.jsonl"
    structure_path = snapshot / "structure_index.json"
    write_jsonl(corpus, corpus_path)
    structure_path.write_text(json.dumps(structure, indent=2) + "\n")

    return {
        "snapshot": str(snapshot),
        "corpus": str(corpus_path),
        "structure_index": str(structure_path),
        "n_chunks": len(corpus),
        "n_paths": structure["stats"]["n_paths"],
        "n_symbols": structure["stats"]["n_symbols"],
        "by_kind": structure["stats"]["by_kind"],
    }
