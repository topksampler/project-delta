#!/usr/bin/env python3
"""Convert factory probes JSONL → chat messages for LoRA SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def gold_answer(gold: dict) -> str:
    gold = gold or {}
    if gold.get("boolean"):
        return f"{str(gold['boolean']).strip().capitalize()}."
    must = gold.get("must_contain") or []
    any_groups = gold.get("must_contain_any") or []
    parts: list[str] = []
    if must:
        parts.append(" ".join(str(x) for x in must))
    for group in any_groups:
        if group:
            parts.append(str(group[0]))
    if parts:
        return " ".join(parts) + "."
    return "Unknown."


def convert(probes_path: Path, out_path: Path) -> dict:
    rows_out = []
    with probes_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows_out.append(
                {
                    "messages": [
                        {"role": "user", "content": row["question"]},
                        {"role": "assistant", "content": gold_answer(row.get("gold") or {})},
                    ],
                    "id": row.get("id"),
                    "claim_id": row.get("claim_id"),
                    "drift_type": row.get("drift_type"),
                }
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Trainer only needs messages; keep id fields stripped for Dataset.select_columns
    slim = [{"messages": r["messages"]} for r in rows_out]
    out_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in slim),
        encoding="utf-8",
    )
    meta = out_path.with_suffix(".meta.json")
    meta.write_text(
        json.dumps(
            {
                "schema": "delta.eval_factory.sft_export.v1",
                "source": str(probes_path),
                "n": len(slim),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"out": str(out_path), "n": len(slim), "meta": str(meta)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probes", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(convert(args.probes, args.out), indent=2))


if __name__ == "__main__":
    main()
