#!/usr/bin/env python3
"""Build SFT messages JSONL from e1_vllm doc_0 corpus for LoRA training."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
SNAPSHOTS_PATH = EXPERIMENT_DIR / "snapshots.yaml"
DATA_DIR = REPO_ROOT / "data" / "experiments" / "e1_vllm"
CORPUS_PATH = DATA_DIR / "corpus_doc_0.jsonl"
B2_PREFIX = "datasets/experiments/e1_vllm"

TRAIN_OUT = DATA_DIR / "train_doc_0_messages.jsonl"
EVAL_OUT = DATA_DIR / "eval_doc_0_messages.jsonl"


def load_snapshots() -> dict:
    return yaml.safe_load(SNAPSHOTS_PATH.read_text(encoding="utf-8"))


def load_corpus(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path} — run build_corpus.py first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def chunk_to_messages(row: dict, *, version: str) -> dict:
    source = row["source_path"]
    text = row["text"].strip()
    user = (
        f"You are answering from vLLM v{version} documentation.\n"
        f"Source file: {source}\n"
        f"Summarize the key technical content in 1–3 sentences."
    )
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": text}]}


def split_rows(rows: list[dict], *, eval_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = {}
    for row in rows:
        by_source.setdefault(row["source_path"], []).append(row)

    sources = list(by_source.keys())
    rng.shuffle(sources)
    eval_source_count = max(1, int(len(sources) * eval_ratio))
    eval_sources = set(sources[:eval_source_count])

    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    for source, chunks in by_source.items():
        (eval_rows if source in eval_sources else train_rows).extend(chunks)
    return train_rows, eval_rows


def subsample(rows: list[dict], limit: int | None, seed: int) -> list[dict]:
    if limit is None or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, limit)


def write_messages(path: Path, rows: list[dict], *, version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(chunk_to_messages(row, version=version), ensure_ascii=False) + "\n")


def upload(path: Path, remote_name: str) -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from lab.dispatch import b2  # noqa: WPS433

    b2.upload_file(path, f"{B2_PREFIX}/{remote_name}")
    print(f"uploaded {path} -> s3://{os.environ['S3_BUCKET']}/{B2_PREFIX}/{remote_name}")


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LoRA SFT messages from doc_0 corpus.")
    parser.add_argument("--train-limit", type=int, default=800, help="max train chunks (default 800)")
    parser.add_argument("--eval-limit", type=int, default=100, help="max eval chunks (default 100)")
    parser.add_argument("--eval-ratio", type=float, default=0.08, help="hold out by source file fraction")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    version = load_snapshots()["vllm_docs"]["doc_0"]["version"]
    rows = load_corpus(CORPUS_PATH)
    train_rows, eval_rows = split_rows(rows, eval_ratio=args.eval_ratio, seed=args.seed)
    train_rows = subsample(train_rows, args.train_limit, args.seed)
    eval_rows = subsample(eval_rows, args.eval_limit, args.seed + 1)

    write_messages(TRAIN_OUT, train_rows, version=version)
    write_messages(EVAL_OUT, eval_rows, version=version)
    print(f"train: {len(train_rows)} examples -> {TRAIN_OUT}")
    print(f"eval:  {len(eval_rows)} examples -> {EVAL_OUT}")

    if args.upload:
        upload(TRAIN_OUT, "train_doc_0_messages.jsonl")
        upload(EVAL_OUT, "eval_doc_0_messages.jsonl")


if __name__ == "__main__":
    main()
