"""Build sequential stage-2 mix: eval_v3 hole paraphrases + honesty retention."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .build_blend_v6 import build_eval_hole_paraphrases, load_jsonl, write_jsonl

VERSION = 7
SEED = 20260728


def build_v7(
    *,
    v5_full_path: Path,
    eval_path: Path,
    base_samples_path: Path,
    out_train: Path,
    out_eval: Path,
    out_manifest: Path,
    n_retain_false: int = 120,
    n_retain_true: int = 60,
    n_retain_recog: int = 60,
) -> dict:
    holes, rejected = build_eval_hole_paraphrases(
        eval_path=eval_path, base_samples_path=base_samples_path
    )
    v5 = load_jsonl(v5_full_path)
    by_form: dict[str, list[dict]] = {"meaning_false": [], "meaning_true": [], "meaning_recognition": []}
    for row in v5:
        form = (row.get("provenance") or {}).get("form")
        if form in by_form:
            by_form[form].append({"messages": row["messages"]})

    rng = random.Random(SEED)
    retain: list[dict] = []
    for form, n in (
        ("meaning_false", n_retain_false),
        ("meaning_true", n_retain_true),
        ("meaning_recognition", n_retain_recog),
    ):
        pool = by_form[form]
        rng.shuffle(pool)
        retain.extend(pool[:n])

    hole_msgs = [{"messages": r["messages"]} for r in holes]
    # Upsample holes so stage-2 sees them often in short runs.
    hole_msgs = hole_msgs * 4

    merged = hole_msgs + retain
    rng.shuffle(merged)

    eval_rows = merged[::8][:40]
    eval_users = {r["messages"][0]["content"] for r in eval_rows}
    train_rows = [r for r in merged if r["messages"][0]["content"] not in eval_users]

    write_jsonl(out_train, train_rows)
    write_jsonl(out_eval, eval_rows)

    sources = Counter(
        {
            "eval_holes_x4": len(hole_msgs),
            "retain_false": min(n_retain_false, len(by_form["meaning_false"])),
            "retain_true": min(n_retain_true, len(by_form["meaning_true"])),
            "retain_recog": min(n_retain_recog, len(by_form["meaning_recognition"])),
        }
    )
    manifest = {
        "schema": "delta.mill_manifest.v1",
        "tag": "v0.22.0",
        "version": VERSION,
        "generator": "sequential_v5_then_eval_holes_v1",
        "emitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "continue_from": "e1-vllm-c3-ft-mill-v5-qwen35-08b-modal",
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "n_rejected_eval_contam": len(rejected),
        "sources": dict(sources),
        "role": "sequential FT stage-2: light eval-hole tune on v5 with honesty retain",
        "train": str(out_train),
        "eval": str(out_eval),
        "train_sha256": hashlib.sha256(out_train.read_bytes()).hexdigest(),
    }
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    data = root / "data/experiments/e1_vllm"
    print(
        json.dumps(
            build_v7(
                v5_full_path=data / "train_v0.22.0_v5_full.jsonl",
                eval_path=data / "eval_v3.jsonl",
                base_samples_path=root
                / "runs/e1-vllm-c0-base-eval-v3-qwen35-08b-modal/samples.jsonl",
                out_train=data / "train_v0.22.0_v7.jsonl",
                out_eval=data / "eval_v0.22.0_v7.jsonl",
                out_manifest=data / "manifest_v0.22.0_v7.json",
            ),
            indent=2,
        )
    )
