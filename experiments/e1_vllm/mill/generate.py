"""Stage 2 — generate candidate train rows (T1–T7)."""

from __future__ import annotations

import json
from pathlib import Path

from generators.t1 import generate_t1
from generators.t2 import generate_t2
from generators.t3 import generate_t3
from generators.t4 import generate_t4
from generators.t5 import generate_t5
from generators.t6 import generate_t6
from generators.t7 import generate_t7


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate(snapshot: Path, *, limit: int | None = None) -> dict:
    snapshot = snapshot.resolve()
    corpus_path = snapshot / "corpus.jsonl"
    structure_path = snapshot / "structure_index.json"
    if not corpus_path.is_file() or not structure_path.is_file():
        raise FileNotFoundError("missing corpus/structure — run mill extract first")

    corpus = _load_jsonl(corpus_path)
    structure = json.loads(structure_path.read_text())
    chunks = {row["chunk_id"]: row["text"] for row in corpus}
    symbols = structure["symbols"]
    tag = structure["tag"]

    t7_limit = limit if limit is not None else 200
    by_class = {
        "T1": generate_t1(symbols=symbols, chunks=chunks, tag=tag, limit=limit),
        "T2": generate_t2(symbols=symbols, chunks=chunks, tag=tag, limit=limit),
        "T3": generate_t3(symbols=symbols, chunks=chunks, tag=tag, limit=limit),
        "T4": generate_t4(symbols=symbols, tag=tag, limit=limit),
        "T5": generate_t5(symbols=symbols, chunks=chunks, tag=tag, limit=limit),
        "T6": generate_t6(symbols=symbols, chunks=chunks, tag=tag, limit=limit),
        "T7": generate_t7(symbols=symbols, chunks=chunks, tag=tag, limit=t7_limit),
    }

    out_dir = snapshot / "candidates"
    counts = {}
    for cls, rows in by_class.items():
        _write_jsonl(rows, out_dir / f"{cls}.jsonl")
        counts[f"n_{cls}"] = len(rows)

    return {"snapshot": str(snapshot), "candidates_dir": str(out_dir), **counts}
