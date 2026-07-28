"""Compare TruthSource arm profiles on the bridge set (pre-registered margins)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


GAP = {"unknown", "partial", "unstable"}


def _load_profile(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _bridge_status(profile: dict) -> dict[str, str]:
    out = {}
    for c in profile.get("claims", []):
        cid = c["claim_id"]
        if cid.startswith("bridge."):
            out[cid] = c["status"]
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _honesty_from_samples(samples: list[dict], bridge_ids: set[str]) -> float:
    rows = [
        s
        for s in samples
        if s.get("claim_id") in bridge_ids and s.get("probe_form") == "negation"
    ]
    if not rows:
        return float("nan")
    return sum(1 for r in rows if r.get("failure_mode") == "correct") / len(rows)


def compare(
    *,
    profiles: dict[str, Path],
    samples: dict[str, Path],
    out_path: Path,
) -> dict:
    loaded = {arm: _load_profile(p) for arm, p in profiles.items()}
    statuses = {arm: _bridge_status(p) for arm, p in loaded.items()}
    bridge_ids = set.intersection(*(set(s) for s in statuses.values())) if statuses else set()

    gaps = {
        arm: {cid for cid, st in stmap.items() if cid in bridge_ids and st in GAP}
        for arm, stmap in statuses.items()
    }

    pairs = [("D", "C_distill"), ("C_distill", "C_full"), ("D", "C_full")]
    jaccard = {}
    flip_rate = {}
    for a, b in pairs:
        jaccard[f"{a}_vs_{b}"] = round(_jaccard(gaps[a], gaps[b]), 4)
        flips = 0
        n = 0
        for cid in bridge_ids:
            sa, sb = statuses[a].get(cid), statuses[b].get(cid)
            if sa is None or sb is None:
                continue
            n += 1
            if sa != sb:
                flips += 1
        flip_rate[f"{a}_vs_{b}"] = round(flips / max(1, n), 4)

    honesty = {}
    for arm, sp in samples.items():
        rows = [json.loads(l) for l in sp.read_text().splitlines() if l.strip()]
        honesty[arm] = round(_honesty_from_samples(rows, bridge_ids), 4)

    same = (
        jaccard["D_vs_C_distill"] >= 0.80
        and jaccard["C_distill_vs_C_full"] >= 0.80
        and flip_rate["D_vs_C_distill"] < 0.15
        and flip_rate["C_distill_vs_C_full"] < 0.15
    )

    # decision table
    d_cd = jaccard["D_vs_C_distill"] >= 0.80 and flip_rate["D_vs_C_distill"] < 0.15
    cd_cf = jaccard["C_distill_vs_C_full"] >= 0.80 and flip_rate["C_distill_vs_C_full"] < 0.15
    if d_cd and cd_cf:
        decision = "docs_only_for_now"
        reading = "D ≈ C_distill ≈ C_full — don't add code yet; exquisite docs+honesty first"
    elif (not d_cd) and cd_cf:
        decision = "distill_enough"
        reading = "D ≠ C_distill ≈ C_full — distill earns its keep; thorough wasted"
    elif d_cd and (not cd_cf):
        decision = "need_thorough_or_better_distill"
        reading = "D ≈ C_distill ≠ C_full — distill failed the test"
    else:
        decision = "code_and_coverage_matter"
        reading = "D ≠ C_distill ≠ C_full — code + coverage both matter"

    honesty_material = {}
    arms = list(honesty)
    for i, a in enumerate(arms):
        for b in arms[i + 1 :]:
            if honesty[a] == honesty[a] and honesty[b] == honesty[b]:  # not nan
                honesty_material[f"{a}_vs_{b}"] = abs(honesty[a] - honesty[b]) >= 0.10

    report = {
        "schema": "delta.truthsource_compare.v0",
        "n_bridge_claims": len(bridge_ids),
        "gap_sizes": {k: len(v) for k, v in gaps.items()},
        "jaccard": jaccard,
        "flip_rate": flip_rate,
        "honesty_negation_acc": honesty,
        "honesty_material_ge_0.10": honesty_material,
        "pre_registered_same_map": same,
        "decision": decision,
        "reading": reading,
        "status_counts_bridge": {
            arm: dict(Counter(st for cid, st in stmap.items() if cid in bridge_ids))
            for arm, stmap in statuses.items()
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--profile-D", required=True)
    p.add_argument("--profile-C_distill", required=True)
    p.add_argument("--profile-C_full", required=True)
    p.add_argument("--samples-D", required=True)
    p.add_argument("--samples-C_distill", required=True)
    p.add_argument("--samples-C_full", required=True)
    args = p.parse_args()
    report = compare(
        profiles={
            "D": Path(args.profile_D),
            "C_distill": Path(args.profile_C_distill),
            "C_full": Path(args.profile_C_full),
        },
        samples={
            "D": Path(args.samples_D),
            "C_distill": Path(args.samples_C_distill),
            "C_full": Path(args.samples_C_full),
        },
        out_path=Path(args.out),
    )
    print(json.dumps({k: report[k] for k in ("decision", "reading", "jaccard", "flip_rate", "honesty_negation_acc")}, indent=2))
