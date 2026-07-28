"""One-shot: pin → extract → generate → gate → emit."""

from __future__ import annotations

from pathlib import Path

from emit import emit
from extract import extract
from gate import gate
from generate import generate
from pin import pin_snapshot


def build(
    *,
    repo: str,
    tag: str,
    snapshot: Path,
    eval_denylist: Path | None = None,
    limit: int | None = None,
    version: int = 1,
) -> dict:
    pin_meta = pin_snapshot(repo=repo, tag=tag, out=snapshot)
    extract_summary = extract(snapshot)
    gen_summary = generate(snapshot, limit=limit)
    gate_summary = gate(snapshot, eval_denylist=eval_denylist)
    emit_summary = emit(snapshot, version=version)
    return {
        "pin": {"tag": pin_meta["tag"], "commit_sha": pin_meta["commit_sha"]},
        "extract": {
            "n_chunks": extract_summary["n_chunks"],
            "n_symbols": extract_summary["n_symbols"],
        },
        "generate": {k: v for k, v in gen_summary.items() if k.startswith("n_")},
        "gate": {
            "n_accepted": gate_summary["n_accepted"],
            "n_rejected": gate_summary["n_rejected"],
            "by_class": gate_summary.get("by_class", {}),
        },
        "emit": emit_summary,
    }
