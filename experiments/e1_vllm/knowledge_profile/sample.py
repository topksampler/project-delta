"""Stratified claim sampling for pilot closed-book runs."""

from __future__ import annotations

import random
from collections import defaultdict


def sample_claims(
    claims: list[dict],
    *,
    seed: int = 20260718,
    per_type: dict[str, int] | None = None,
) -> list[dict]:
    """Sample a stratified pilot bank.

    Default budget (~150 claims) fits an A10G closed-book pass without drowning
    the first profile in low-centrality path noise.
    """
    per_type = per_type or {
        "flag_exists": 40,
        "flag_meaning": 30,
        "env_exists": 25,
        "path_exists": 40,
    }
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for c in claims:
        by_type[c["claim_type"]].append(c)

    picked: list[dict] = []
    for ctype, limit in per_type.items():
        pool = by_type.get(ctype, [])
        # prefer high centrality, then shuffle within band
        high = [c for c in pool if c.get("centrality") == "high"]
        rest = [c for c in pool if c.get("centrality") != "high"]
        rng.shuffle(high)
        rng.shuffle(rest)
        ordered = high + rest
        picked.extend(ordered[:limit])

    picked.sort(key=lambda c: c["claim_id"])
    return picked
