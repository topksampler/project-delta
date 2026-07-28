"""Stage 4 — emit: gated rows → train JSONL + manifest."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = REPO_ROOT / "data" / "experiments" / "e1_vllm"

GENERATOR_VERSIONS = {
    "T1": "t1_flag_v2",
    "T2": "t2_howto_v1",
    "T3": "t3_version_v2",
    "T4": "t4_abstain_v1",
    "T5": "t5_contrast_v1",
    "T6": "t6_cite_v1",
    "T7": "t7_multihop_v1",
}


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


def emit(snapshot: Path, *, version: int = 1) -> dict:
    snapshot = snapshot.resolve()
    accepted_path = snapshot / "gated" / "accepted.jsonl"
    meta_path = snapshot / "snapshot_meta.json"
    if not accepted_path.is_file():
        raise FileNotFoundError("missing gated/accepted.jsonl — run mill gate first")
    if not meta_path.is_file():
        raise FileNotFoundError("missing snapshot_meta.json")

    meta = json.loads(meta_path.read_text())
    accepted = _load_jsonl(accepted_path)
    tag = meta["tag"]
    safe = tag.replace("/", "_")

    train_rows = [{"messages": row["messages"]} for row in accepted]
    by_class = Counter(row.get("train_class", "?") for row in accepted)

    train_path = DATA / f"train_{safe}_v{version}.jsonl"
    manifest_path = DATA / f"manifest_{safe}_v{version}.json"

    _write_jsonl(train_rows, train_path)

    manifest = {
        "tag": tag,
        "commit_sha": meta["commit_sha"],
        "content_sha": meta["content_sha"],
        "version": version,
        "emitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_train": len(train_rows),
        "by_class": dict(by_class),
        "generators": GENERATOR_VERSIONS,
        "snapshot": str(snapshot),
        "train": str(train_path),
        "accepted": str(accepted_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return {
        "train": str(train_path),
        "manifest": str(manifest_path),
        "n_train": len(train_rows),
        "by_class": dict(by_class),
    }
