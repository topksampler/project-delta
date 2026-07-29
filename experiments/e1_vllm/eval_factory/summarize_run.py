"""Aggregate frozen eval-factory run samples by drift type and family.

Optionally attaches forget/remember rates (changed/removed vs gold_before).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from .forget_score import summarize_forget
except ImportError:  # script: python summarize_run.py
    from forget_score import summarize_forget


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(
    *,
    samples_path: Path,
    probes_path: Path,
    out_path: Path,
    include_forget: bool = True,
) -> dict:
    probes = _load_jsonl(probes_path)
    samples = _load_jsonl(samples_path)
    by_drift: dict[str, list[float]] = defaultdict(list)
    by_family: dict[str, list[float]] = defaultdict(list)
    by_form: dict[str, list[float]] = defaultdict(list)
    by_family_drift: dict[str, list[float]] = defaultdict(list)
    probe_by_id = {row["id"]: row for row in probes}
    for row in samples:
        probe = probe_by_id.get(row["id"], {})
        score = float(row["content_score"])
        drift = probe.get("drift_type") or row.get("drift_type") or "?"
        family = probe.get("entity_type") or "?"
        form = probe.get("probe_form") or "?"
        by_drift[drift].append(score)
        by_family[family].append(score)
        by_form[form].append(score)
        by_family_drift[f"{family}:{drift}"].append(score)
    scores = [float(row["content_score"]) for row in samples]
    result = {
        "schema": "delta.eval_factory.run_summary.v2",
        "n_samples": len(samples),
        "avg_content_score": round(_mean(scores), 4),
        "exact_accuracy": round(_mean([1.0 if s >= 1.0 else 0.0 for s in scores]), 4),
        "by_drift_type": {
            key: round(_mean(values), 4) for key, values in sorted(by_drift.items())
        },
        "by_entity_type": {
            key: round(_mean(values), 4) for key, values in sorted(by_family.items())
        },
        "by_probe_form": {
            key: round(_mean(values), 4) for key, values in sorted(by_form.items())
        },
        "by_family_drift": {
            key: {
                "n": len(values),
                "accuracy": round(_mean(values), 4),
            }
            for key, values in sorted(by_family_drift.items())
        },
        "counts": {
            "by_drift_type": {key: len(values) for key, values in sorted(by_drift.items())},
            "by_entity_type": {
                key: len(values) for key, values in sorted(by_family.items())
            },
        },
    }
    if include_forget:
        result["forget"] = summarize_forget(samples=samples, probes=probes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--no-forget",
        action="store_true",
        help="skip forget/remember metrics (legacy summary fields only)",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            aggregate(
                samples_path=args.samples,
                probes_path=args.probes,
                out_path=args.out,
                include_forget=not args.no_forget,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
