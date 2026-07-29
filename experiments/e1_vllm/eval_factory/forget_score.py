#!/usr/bin/env python3
"""Stale-answer (gold_before) bank + forget/remember scoring for factory eval.

Invariant:
  forget_rate   = P(model matches gold_before) on changed/removed probes
  remember_rate = P(model matches gold_after)  on the same probe set
  Stable probes are never in the forget denominator.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lab.qa_score import score_gold  # noqa: E402

FORGET_DRIFTS = frozenset({"changed", "removed"})
STABLE_STATUS_WORDS = [["unchanged", "stable", "same", "both"]]
SCHEMA_BANK = "delta.eval_factory.gold_before_bank.v1"
SCHEMA_FORGET = "delta.eval_factory.forget_summary.v1"


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _value_forms(value: object) -> list[str]:
    if value is None:
        return ["none", "null", "unspecified"]
    if isinstance(value, bool):
        return [str(value).lower()]
    return [str(value)]


def _constraint_forms(name: str, value: object) -> list[str]:
    symbol = {"ge": ">=", "gt": ">", "le": "<=", "lt": "<"}.get(name)
    words = {
        "ge": "greater than or equal to",
        "gt": "greater than",
        "le": "less than or equal to",
        "lt": "less than",
    }.get(name)
    forms = [f"{name}={value}", f"{name} {value}"]
    if symbol:
        forms.append(f"{symbol} {value}")
    if words:
        forms.append(f"{words} {value}")
    return forms


def _stable_status_gold(display_entity: str | None) -> dict:
    gold: dict = {"must_contain_any": [list(STABLE_STATUS_WORDS[0])]}
    if display_entity:
        gold["must_contain"] = [display_entity]
    return gold


def gold_before_for_probe(probe: dict) -> dict | None:
    """Derive before-revision gold for a changed/removed probe.

    Returns None when forget scoring does not apply (stable/added, or
    existence probes whose gold already equals the before-world answer).
    """
    drift = probe.get("drift_type")
    if drift not in FORGET_DRIFTS:
        return None

    form = probe.get("probe_form") or ""
    gold_after = probe.get("gold") or {}
    display = probe.get("display_entity")

    if form == "versioned_existence":
        # After-revision "does it exist?" for a removed claim: stale = yes.
        if drift == "removed" and gold_after.get("boolean") == "no":
            return {"boolean": "yes"}
        return None

    if form != "version_delta":
        return None

    if drift == "removed":
        return _stable_status_gold(display)

    # changed
    expected = probe.get("expected_change") or {}
    dimension = expected.get("dimension")

    if dimension == "constraints":
        groups: list[list[str]] = []
        for key, values in sorted((expected.get("changes") or {}).items()):
            before = values.get("before")
            if before is not None:
                groups.append(_constraint_forms(key, before))
        if groups:
            return {"must_contain_any": groups}
        return _stable_status_gold(display)

    if dimension == "choices":
        removed = list(expected.get("removed") or [])
        if removed and not (expected.get("added") or []):
            return {"must_contain_any": [[value] for value in removed]}
        return _stable_status_gold(display)

    if dimension == "default":
        return {"must_contain_any": [_value_forms(expected.get("before"))]}

    if dimension == "annotation":
        return {"must_contain_any": [[str(expected.get("before"))]]}

    return _stable_status_gold(display)


def attach_gold_before(probes: list[dict]) -> list[dict]:
    """Return probe copies with gold_before / gold_after fields set when applicable."""
    out: list[dict] = []
    for probe in probes:
        row = dict(probe)
        gold_after = row.get("gold")
        if gold_after is not None and "gold_after" not in row:
            row["gold_after"] = gold_after
        before = gold_before_for_probe(probe)
        if before is not None:
            row["gold_before"] = before
        out.append(row)
    return out


def write_gold_before_bank(*, probes_path: Path, out_path: Path) -> dict:
    probes = attach_gold_before(_load_jsonl(probes_path))
    with_before = [row for row in probes if "gold_before" in row]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in probes),
        encoding="utf-8",
    )
    meta = {
        "schema": SCHEMA_BANK,
        "source": str(probes_path),
        "out": str(out_path),
        "n_probes": len(probes),
        "n_with_gold_before": len(with_before),
        "by_drift_type": {
            drift: sum(1 for row in with_before if row.get("drift_type") == drift)
            for drift in sorted(FORGET_DRIFTS)
        },
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**meta, "meta": str(meta_path)}


def matches_gold(output: str, gold: dict | None, *, threshold: float = 1.0) -> bool:
    if not gold:
        return False
    score, _, _ = score_gold(output, gold)
    return score >= threshold


def summarize_forget(
    *,
    samples: list[dict],
    probes: list[dict],
    threshold: float = 1.0,
) -> dict:
    """Compute forget/remember rates over changed/removed probes with gold_before."""
    probe_by_id = {row["id"]: row for row in attach_gold_before(probes)}
    forget_hits: list[float] = []
    remember_hits: list[float] = []
    by_drift: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"forget": [], "remember": []}
    )
    n_scored = 0
    for sample in samples:
        probe = probe_by_id.get(sample["id"])
        if not probe:
            continue
        gold_before = probe.get("gold_before")
        if gold_before is None:
            continue
        drift = probe.get("drift_type") or sample.get("drift_type") or "?"
        if drift not in FORGET_DRIFTS:
            continue
        output = sample.get("output") or ""
        gold_after = probe.get("gold_after") or probe.get("gold") or {}
        forgot = 1.0 if matches_gold(output, gold_before, threshold=threshold) else 0.0
        remembered = 1.0 if matches_gold(output, gold_after, threshold=threshold) else 0.0
        forget_hits.append(forgot)
        remember_hits.append(remembered)
        by_drift[drift]["forget"].append(forgot)
        by_drift[drift]["remember"].append(remembered)
        n_scored += 1

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "schema": SCHEMA_FORGET,
        "n_scored": n_scored,
        "forget_rate": round(_mean(forget_hits), 4),
        "remember_rate": round(_mean(remember_hits), 4),
        "threshold": threshold,
        "by_drift_type": {
            drift: {
                "n": len(vals["forget"]),
                "forget_rate": round(_mean(vals["forget"]), 4),
                "remember_rate": round(_mean(vals["remember"]), 4),
            }
            for drift, vals in sorted(by_drift.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="gold_before bank JSONL")
    parser.add_argument("--samples", type=Path, help="optional samples.jsonl for forget summary")
    parser.add_argument("--summary-out", type=Path, help="optional forget summary JSON")
    args = parser.parse_args()
    bank = write_gold_before_bank(probes_path=args.probes, out_path=args.out)
    result: dict = {"bank": bank}
    if args.samples:
        summary = summarize_forget(
            samples=_load_jsonl(args.samples),
            probes=_load_jsonl(args.out),
        )
        if args.summary_out:
            args.summary_out.parent.mkdir(parents=True, exist_ok=True)
            args.summary_out.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            summary = {**summary, "out": str(args.summary_out)}
        result["forget"] = summary
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
