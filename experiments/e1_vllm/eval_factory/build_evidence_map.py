#!/usr/bin/env python3
"""Build probe→evidence-span context map from pinned code snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_span(root: Path, source_path: str, line_start: int, line_end: int) -> str:
    path = root / source_path
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, int(line_start)) - 1
    end = max(start + 1, int(line_end))
    return "\n".join(lines[start:end]).strip()


def build_evidence_map(
    *,
    probes_path: Path,
    snapshots_root: Path,
    version_to_tag: dict[str, str],
    max_chars: int = 3500,
    era: str = "fresh",
    force_tag: str | None = None,
    stale_pair: tuple[str, str] | None = None,
) -> dict:
    """era=fresh uses requires_doc; era=stale flips within stale_pair; force_tag wins."""
    probes = _load_jsonl(probes_path)
    out: dict[str, dict] = {}
    missing = 0
    pair = stale_pair or ("0.22.0", "0.23.0")
    flip = {pair[0]: pair[1], pair[1]: pair[0]}
    for probe in probes:
        version = str(probe.get("requires_doc") or "")
        if force_tag:
            tag = force_tag
        elif era == "stale":
            flipped = flip.get(version, version)
            tag = version_to_tag.get(flipped) or f"v{flipped}"
        else:
            tag = version_to_tag.get(version) or version_to_tag.get(f"v{version}")
            if not tag:
                tag = f"v{version}" if version and not version.startswith("v") else version
        snap = snapshots_root / tag
        parts: list[str] = []
        ids: list[str] = []
        used = 0
        for ev in probe.get("evidence") or []:
            source_path = ev["source_path"]
            text = _read_span(
                snap, source_path, ev.get("line_start", 1), ev.get("line_end", 1)
            )
            if not text:
                missing += 1
                continue
            cid = f"evidence:{tag}:{source_path}:{ev.get('line_start')}-{ev.get('line_end')}"
            block = f"[{cid}]\n{text}"
            if used + len(block) + 2 > max_chars and parts:
                break
            parts.append(block)
            ids.append(cid)
            used += len(block) + 2
        context = (
            "Retrieved source evidence:\n\n" + "\n\n".join(parts) if parts else ""
        )
        out[probe["id"]] = {
            "context": context,
            "chunk_ids": ids,
            "snapshot_tag": tag,
            "n_spans": len(ids),
        }
    return {
        "schema": "delta.eval_factory.evidence_map.v1",
        "probes_path": str(probes_path),
        "era": era if not force_tag else f"force:{force_tag}",
        "n_probes": len(probes),
        "n_with_context": sum(1 for v in out.values() if v["context"]),
        "n_missing_spans": missing,
        "by_probe": out,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probes", type=Path, required=True)
    p.add_argument(
        "--snapshots-root",
        type=Path,
        default=Path("data/experiments/e1_vllm/snapshots"),
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-chars", type=int, default=3500)
    p.add_argument("--era", choices=("fresh", "stale"), default="fresh")
    p.add_argument("--force-tag", default=None)
    p.add_argument(
        "--stale-pair",
        default="0.22.0,0.23.0",
        help="comma pair of versions to flip under --era stale",
    )
    args = p.parse_args()
    version_to_tag = {
        "0.20.0": "v0.20.0",
        "0.21.0": "v0.21.0",
        "0.22.0": "v0.22.0",
        "0.23.0": "v0.23.0",
        "0.24.0": "v0.24.0",
        "0.26.0": "v0.26.0",
    }
    left, right = [x.strip() for x in args.stale_pair.split(",", 1)]
    payload = build_evidence_map(
        probes_path=args.probes,
        snapshots_root=args.snapshots_root,
        version_to_tag=version_to_tag,
        max_chars=args.max_chars,
        era=args.era,
        force_tag=args.force_tag,
        stale_pair=(left, right),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "era": payload.get("era"),
                "n_probes": payload["n_probes"],
                "n_with_context": payload["n_with_context"],
                "n_missing_spans": payload["n_missing_spans"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
