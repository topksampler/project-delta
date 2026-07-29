#!/usr/bin/env python3
"""Build claim-local DPO micro-edit shards + stable canary (Experiment B).

Clusters sealed train delta probes by claim_id, orders changed → removed →
added, caps to a forget/remember-critical subset, and emits per-claim DPO
JSONL (claim pairs × copies + fixed stable replay) plus a fixed canary set.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .export_dpo_pairs import to_dpo_row
except ImportError:  # script: python build_claim_local_edits.py
    from export_dpo_pairs import to_dpo_row

SCHEMA = "delta.eval_factory.claim_local_edits.v1"
DRIFT_ORDER = ("changed", "removed", "added")


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _slug(claim_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", claim_id)[:80]


def _primary_drift(probes: list[dict]) -> str:
    counts = Counter(p.get("drift_type") or "stable" for p in probes)
    for drift in DRIFT_ORDER:
        if counts.get(drift):
            return drift
    return "stable"


def order_claims(
    by_claim: dict[str, list[dict]],
    *,
    max_claims: int,
) -> list[str]:
    """Order claim_ids: changed, removed, added; alphabetical within band."""
    buckets: dict[str, list[str]] = {d: [] for d in DRIFT_ORDER}
    for claim_id, probes in by_claim.items():
        buckets[_primary_drift(probes)].append(claim_id)
    ordered: list[str] = []
    for drift in DRIFT_ORDER:
        ordered.extend(sorted(buckets[drift]))
    return ordered[:max_claims]


def slim_dpo(row: dict) -> dict:
    return {
        "prompt": row["prompt"],
        "chosen": row["chosen"],
        "rejected": row["rejected"],
    }


def build(
    *,
    probes_path: Path,
    out_dir: Path,
    max_claims: int = 10,
    claim_copies: int = 8,
    stable_replay_n: int = 12,
    canary_n: int = 24,
    seed: int = 20260729,
    canary_budget: float = 0.02,
) -> dict:
    rows = _load_jsonl(probes_path)
    by_claim: dict[str, list[dict]] = defaultdict(list)
    stable: list[dict] = []
    for row in rows:
        drift = row.get("drift_type") or "stable"
        if drift in DRIFT_ORDER:
            cid = row.get("claim_id")
            if not cid:
                continue
            by_claim[str(cid)].append(row)
        elif drift == "stable":
            stable.append(row)

    available = len(by_claim)
    claim_ids = order_claims(by_claim, max_claims=max_claims)
    deferred = sorted(set(by_claim) - set(claim_ids))

    rng = random.Random(seed)
    stable_shuffled = list(stable)
    rng.shuffle(stable_shuffled)
    canary_probes = stable_shuffled[:canary_n]
    replay_pool = stable_shuffled[canary_n : canary_n + max(stable_replay_n * 4, 64)]
    if len(replay_pool) < stable_replay_n:
        replay_pool = list(stable_shuffled)
        rng.shuffle(replay_pool)

    out_dir.mkdir(parents=True, exist_ok=True)
    canary_path = out_dir / "canary_stable.jsonl"
    _write_jsonl(canary_path, canary_probes)

    edits: list[dict] = []
    for index, claim_id in enumerate(claim_ids):
        probes = by_claim[claim_id]
        primary = _primary_drift(probes)
        claim_rng = random.Random(seed + index * 997)
        pairs: list[dict] = []
        for _ in range(claim_copies):
            for probe in probes:
                row = to_dpo_row(probe, claim_rng)
                if row is not None:
                    pairs.append(slim_dpo(row))
        replay = list(replay_pool)
        claim_rng.shuffle(replay)
        for probe in replay[:stable_replay_n]:
            row = to_dpo_row(probe, claim_rng)
            if row is not None:
                pairs.append(slim_dpo(row))
        claim_rng.shuffle(pairs)

        edit_dir = out_dir / "edits" / f"{index:02d}_{_slug(claim_id)}"
        train_path = edit_dir / "train.jsonl"
        _write_jsonl(train_path, pairs)
        edits.append(
            {
                "index": index,
                "claim_id": claim_id,
                "primary_drift": primary,
                "drift_types": sorted(
                    {p.get("drift_type") or "stable" for p in probes}
                ),
                "n_probes": len(probes),
                "n_pairs": len(pairs),
                "train_path": str(train_path),
            }
        )

    cap_reason = (
        f"capped at {len(claim_ids)}/{available} delta claims; "
        "order=changed→removed→added; deferred mostly added when max_claims "
        "covers all changed+removed first"
    )
    manifest = {
        "schema": SCHEMA,
        "source": str(probes_path),
        "seed": seed,
        "max_claims": max_claims,
        "n_delta_claims_available": available,
        "n_edits": len(edits),
        "claim_copies": claim_copies,
        "stable_replay_n": stable_replay_n,
        "canary_n": len(canary_probes),
        "canary_budget": canary_budget,
        "cap_reason": cap_reason,
        "deferred_claim_ids": deferred,
        "canary_path": str(canary_path),
        "edits": edits,
        "by_primary_drift": dict(
            Counter(e["primary_drift"] for e in edits)
        ),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"manifest": str(manifest_path), **manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-claims", type=int, default=10)
    parser.add_argument("--claim-copies", type=int, default=8)
    parser.add_argument("--stable-replay-n", type=int, default=12)
    parser.add_argument("--canary-n", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--canary-budget", type=float, default=0.02)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                probes_path=args.probes,
                out_dir=args.out_dir,
                max_claims=args.max_claims,
                claim_copies=args.claim_copies,
                stable_replay_n=args.stable_replay_n,
                canary_n=args.canary_n,
                seed=args.seed,
                canary_budget=args.canary_budget,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
