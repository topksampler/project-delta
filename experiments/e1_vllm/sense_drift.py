#!/usr/bin/env python3
"""SENSE v1 — emit a DriftEvent from two frozen eval_v3 run metrics (+ samples).

Compares a baseline condition (usually c0) to a probe condition (c3, c1, …).
Re-scores samples with the current failure-mode classifier when samples are present.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from lab.qa_score import (  # noqa: E402
    classify_failure_mode,
    gold_wants_unknown,
    is_abstain,
)

VLLM_REPO = "https://github.com/vllm-project/vllm.git"
SCHEMA = "delta.drift_event.v1"
CORPUS_SIGNAL_SCHEMA = "delta.corpus_signal.eval_factory.v1"
DEFAULT_CORPUS_MANIFEST = (
    REPO
    / "data"
    / "experiments"
    / "e1_vllm"
    / "eval_factory"
    / "v0.22.0_to_v0.23.0"
    / "e1_eval_factory_v1"
    / "manifest.json"
)
DEFAULT_CORPUS_B2 = (
    "datasets/experiments/e1_vllm/eval_factory/"
    "v0.22.0_to_v0.23.0/e1_eval_factory_v1/"
)

DEFAULT_PAIRS = [
    (
        "e1-vllm-c0-base-eval-v3-qwen35-08b-modal",
        "e1-vllm-c3-ft-mill-v3-eval-v3-qwen35-08b-modal",
        "c0_base",
        "c3_ft",
    ),
    (
        "e1-vllm-c0-base-eval-v3-qwen35-08b-modal",
        "e1-vllm-c1-rag-fresh-eval-v3-qwen35-08b-modal",
        "c0_base",
        "c1_rag_fresh",
    ),
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _find_metrics(metrics_dir: Path, run_id: str) -> Path:
    for cand in (metrics_dir / f"{run_id}.json", metrics_dir / run_id / "metrics.json"):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"metrics for {run_id} not under {metrics_dir}")


def _find_samples(metrics_dir: Path, run_id: str) -> Path | None:
    for cand in (metrics_dir / f"{run_id}.samples.jsonl", metrics_dir / run_id / "samples.jsonl"):
        if cand.is_file():
            return cand
    return None


def _class_means(rows: list[dict]) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        buckets[r.get("eval_class", "?")].append(float(r["content_score"]))
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in buckets.items()}


def _failure_counts(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in rows:
        out[r["failure_mode"]] += 1
    return dict(out)


def build_corpus_signal(manifest_path: Path) -> dict:
    """Attach sealed eval-factory claim census as structural corpus_signal."""
    if not manifest_path.is_file():
        return {
            "status": "deferred",
            "note": f"corpus manifest missing: {manifest_path}",
        }
    manifest = _load_json(manifest_path)
    counts = manifest.get("counts") or {}
    by_status = counts.get("by_status") or {}
    deltas = int(
        by_status.get("added", 0)
        + by_status.get("changed", 0)
        + by_status.get("removed", 0)
    )
    try:
        local = str(manifest_path.resolve().relative_to(REPO))
    except ValueError:
        local = str(manifest_path)
    return {
        "status": "attached",
        "schema": CORPUS_SIGNAL_SCHEMA,
        "protocol_id": manifest.get("protocol_id"),
        "transition": manifest.get("transition"),
        "source_before": manifest.get("source_before"),
        "source_after": manifest.get("source_after"),
        "local_manifest": local,
        "b2_prefix": DEFAULT_CORPUS_B2,
        "claims": counts.get("claims"),
        "deltas": deltas,
        "by_status": by_status,
        "by_family": counts.get("by_family") or {},
        "by_family_status": counts.get("by_family_status") or {},
        "by_split": counts.get("by_split") or {},
        "note": (
            "Executable claim census from sealed EvalEnvironment; "
            "structural signal, not a behavioral score."
        ),
    }


def rescore_samples(rows: list[dict]) -> list[dict]:
    """Apply current classifier; keep content_score from the run."""
    out = []
    for r in rows:
        needles = r.get("gold_must_contain") or []
        score = float(r["content_score"])
        # if unknown-gold and model abstained but score was 0 due to phrasing, treat as hit
        if gold_wants_unknown(needles) and is_abstain(r.get("output") or ""):
            score = 1.0
        mode = classify_failure_mode(
            score=score,
            output=r.get("output") or "",
            requires_doc=r.get("requires_doc", "?"),
            misses=r.get("misses") or [],
            needles=needles,
        )
        out.append({**r, "content_score": score, "failure_mode": mode, "rescored": True})
    return out


def rows_from_metrics(metrics: dict) -> list[dict]:
    """Fallback when samples missing: synthetic rows from class means only."""
    # cannot rebuild probe_failures without samples
    return []


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    by_class = _class_means(rows)
    fm = _failure_counts(rows)
    acc = sum(1 for r in rows if r["failure_mode"] == "correct") / max(1, n)
    avg = sum(float(r["content_score"]) for r in rows) / max(1, n)
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "avg_content_score": round(avg, 4),
        "by_eval_class": {k: round(v, 4) for k, v in sorted(by_class.items())},
        "by_failure_mode": fm,
    }


def build_event(
    *,
    baseline_id: str,
    probe_id: str,
    baseline_cond: str,
    probe_cond: str,
    baseline_rows: list[dict],
    probe_rows: list[dict],
    baseline_metrics: dict,
    probe_metrics: dict,
    corpus_signal: dict | None = None,
) -> dict:
    b_sum = summarize(baseline_rows) if baseline_rows else {
        "n": baseline_metrics.get("num_examples"),
        "accuracy": baseline_metrics.get("accuracy"),
        "avg_content_score": baseline_metrics.get("avg_content_score"),
        "by_eval_class": {
            k: round(v, 4)
            for k, v in (baseline_metrics.get("breakdown") or {}).get("by_eval_class", {}).items()
        },
        "by_failure_mode": (baseline_metrics.get("breakdown") or {}).get("by_failure_mode", {}),
    }
    p_sum = summarize(probe_rows) if probe_rows else {
        "n": probe_metrics.get("num_examples"),
        "accuracy": probe_metrics.get("accuracy"),
        "avg_content_score": probe_metrics.get("avg_content_score"),
        "by_eval_class": {
            k: round(v, 4)
            for k, v in (probe_metrics.get("breakdown") or {}).get("by_eval_class", {}).items()
        },
        "by_failure_mode": (probe_metrics.get("breakdown") or {}).get("by_failure_mode", {}),
    }

    def delta(cls: str) -> float | None:
        if cls not in b_sum["by_eval_class"] or cls not in p_sum["by_eval_class"]:
            return None
        return round(p_sum["by_eval_class"][cls] - b_sum["by_eval_class"][cls], 4)

    probe_failures = []
    if baseline_rows and probe_rows:
        b_by_id = {r["id"]: r for r in baseline_rows}
        for r in probe_rows:
            br = b_by_id.get(r["id"])
            if not br:
                continue
            if r["failure_mode"] != "correct":
                probe_failures.append(
                    {
                        "id": r["id"],
                        "eval_class": r.get("eval_class"),
                        "requires_doc": r.get("requires_doc"),
                        "baseline_mode": br["failure_mode"],
                        "probe_mode": r["failure_mode"],
                        "baseline_score": br["content_score"],
                        "probe_score": r["content_score"],
                    }
                )

    overall_delta = None
    if b_sum.get("accuracy") is not None and p_sum.get("accuracy") is not None:
        overall_delta = round(float(p_sum["accuracy"]) - float(b_sum["accuracy"]), 4)

    return {
        "schema": SCHEMA,
        "experiment_id": "e1_vllm",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_before": {
            "repo": VLLM_REPO,
            "revision": "v0.22.0",
            "alias": "doc_0",
        },
        "source_after": {
            "repo": VLLM_REPO,
            "revision": "v0.23.0",
            "alias": "doc_8",
        },
        "model": {"id": baseline_metrics.get("model") or probe_metrics.get("model")},
        "eval": {
            "fixture": "eval_v3.jsonl",
            "n": b_sum.get("n") or p_sum.get("n"),
        },
        "baseline": {
            "condition_id": baseline_cond,
            "run_id": baseline_id,
            "metrics": b_sum,
        },
        "probe": {
            "condition_id": probe_cond,
            "run_id": probe_id,
            "metrics": p_sum,
        },
        "drift_score": {
            "accuracy_delta": overall_delta,
            "by_eval_class_delta": {c: delta(c) for c in ("A", "B", "C", "D")},
        },
        "stable_knowledge": {
            "class": "A",
            "baseline": b_sum["by_eval_class"].get("A"),
            "probe": p_sum["by_eval_class"].get("A"),
            "delta": delta("A"),
            "note": "stable facts; drop here is regression",
        },
        "changed_knowledge": {
            "B": {
                "baseline": b_sum["by_eval_class"].get("B"),
                "probe": p_sum["by_eval_class"].get("B"),
                "delta": delta("B"),
            },
            "D": {
                "baseline": b_sum["by_eval_class"].get("D"),
                "probe": p_sum["by_eval_class"].get("D"),
                "delta": delta("D"),
            },
            "note": "version-sensitive / new-fact classes",
        },
        "failure_mode_delta": {
            k: int(p_sum["by_failure_mode"].get(k, 0) - b_sum["by_failure_mode"].get(k, 0))
            for k in sorted(
                set(b_sum["by_failure_mode"]) | set(p_sum["by_failure_mode"])
            )
        },
        "probe_failures": probe_failures,
        "corpus_signal": corpus_signal
        or {
            "status": "deferred",
            "note": "behavioral probe only; pass --corpus-manifest to attach",
        },
        "drift_type": "changed",
        "affected_zones": ["eval_class:B", "eval_class:D", "eval_class:A", "eval_class:C"],
    }


def render_baseline_report(events: list[dict]) -> str:
    lines = [
        "# e1_vllm — Phase B drift baseline",
        "",
        "## invariant",
        "",
        "```text",
        "DriftEvent compares the same eval_v3 probes under two conditions.",
        "Stable knowledge = class A. Changed knowledge = classes B and D.",
        "Behavioral scoreboard + structural corpus_signal (eval-factory census).",
        "```",
        "",
        "## what goes where",
        "",
        "- DriftEvent JSON: `artifacts/reports/drift_events/`",
        "- this report: `artifacts/reports/e1_vllm_drift_baseline.md`",
        "- generator: `experiments/e1_vllm/sense_drift.py`",
        "- corpus census: sealed `e1_eval_factory_v1` manifest",
        "",
        "## what can die",
        "",
        "- local metric/sample caches used to rebuild events",
        "",
        "## what must survive",
        "",
        "- frozen eval_v3 + cited run_ids",
        "- DriftEvent JSON + this baseline report",
        "- attached corpus_signal provenance (protocol_id + b2_prefix)",
        "",
        "## events",
        "",
    ]
    for ev in events:
        d = ev["drift_score"]
        cs = ev.get("corpus_signal") or {}
        lines.append(f"### `{ev['baseline']['condition_id']}` → `{ev['probe']['condition_id']}`")
        lines.append("")
        lines.append(f"- baseline run: `{ev['baseline']['run_id']}`")
        lines.append(f"- probe run: `{ev['probe']['run_id']}`")
        lines.append(f"- accuracy_delta: **{d['accuracy_delta']}**")
        lines.append(
            f"- class deltas A/B/C/D: "
            f"{d['by_eval_class_delta']}"
        )
        lines.append(
            f"- stable (A) delta: {ev['stable_knowledge']['delta']} "
            f"(regression if negative)"
        )
        lines.append(
            f"- changed B/D deltas: "
            f"{ev['changed_knowledge']['B']['delta']} / "
            f"{ev['changed_knowledge']['D']['delta']}"
        )
        lines.append(f"- probe_failures listed: {len(ev.get('probe_failures') or [])}")
        if cs.get("status") == "attached":
            lines.append(
                f"- corpus_signal: attached "
                f"({cs.get('claims')} claims / {cs.get('deltas')} deltas; "
                f"`{cs.get('protocol_id')}`)"
            )
        else:
            lines.append(f"- corpus_signal: {cs.get('status', 'missing')}")
        lines.append("")

    lines.extend(
        [
            "## reading",
            "",
            "```text",
            "c0→c3: LoRA@doc_0. Expect A/C up or flat; B/D not recovered (stale diet).",
            "c0→c1: fresh RAG. Expect B up; D mixed (negative-existence contamination).",
            "```",
            "",
            "## command",
            "",
            "```bash",
            "python experiments/e1_vllm/sense_drift.py \\",
            "  --metrics-dir /tmp/e1_compare \\",
            "  --out-dir artifacts/reports/drift_events \\",
            "  --report artifacts/reports/e1_vllm_drift_baseline.md",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def attach_corpus_to_events(
    *,
    out_dir: Path,
    corpus_manifest: Path,
    report: Path,
) -> list[dict]:
    """Patch existing DriftEvent JSONs with factory census; rewrite baseline report."""
    signal = build_corpus_signal(corpus_manifest)
    events: list[dict] = []
    for path in sorted(out_dir.glob("drift_*.json")):
        ev = _load_json(path)
        ev["corpus_signal"] = signal
        path.write_text(json.dumps(ev, indent=2) + "\n", encoding="utf-8")
        print(f"attached corpus_signal → {path}")
        events.append(ev)
    if not events:
        raise FileNotFoundError(f"no drift_*.json under {out_dir}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_baseline_report(events), encoding="utf-8")
    print(f"wrote {report}")
    return events


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metrics-dir",
        type=Path,
        help="Required unless --attach-corpus-only",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "artifacts" / "reports" / "drift_events",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=REPO / "artifacts" / "reports" / "e1_vllm_drift_baseline.md",
    )
    p.add_argument(
        "--corpus-manifest",
        type=Path,
        default=DEFAULT_CORPUS_MANIFEST,
        help="Sealed eval-factory manifest for corpus_signal",
    )
    p.add_argument(
        "--attach-corpus-only",
        action="store_true",
        help="Only attach corpus_signal to existing DriftEvent JSON files",
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.attach_corpus_only:
        attach_corpus_to_events(
            out_dir=args.out_dir,
            corpus_manifest=args.corpus_manifest,
            report=args.report,
        )
        return

    if args.metrics_dir is None:
        p.error("--metrics-dir is required unless --attach-corpus-only")

    corpus_signal = build_corpus_signal(args.corpus_manifest)
    events: list[dict] = []
    for base_id, probe_id, base_cond, probe_cond in DEFAULT_PAIRS:
        b_metrics = _load_json(_find_metrics(args.metrics_dir, base_id))
        p_metrics = _load_json(_find_metrics(args.metrics_dir, probe_id))
        b_samp = _find_samples(args.metrics_dir, base_id)
        p_samp = _find_samples(args.metrics_dir, probe_id)
        b_rows = rescore_samples(_load_jsonl(b_samp)) if b_samp else []
        p_rows = rescore_samples(_load_jsonl(p_samp)) if p_samp else []
        ev = build_event(
            baseline_id=base_id,
            probe_id=probe_id,
            baseline_cond=base_cond,
            probe_cond=probe_cond,
            baseline_rows=b_rows,
            probe_rows=p_rows,
            baseline_metrics=b_metrics,
            probe_metrics=p_metrics,
            corpus_signal=corpus_signal,
        )
        out = args.out_dir / f"drift_{base_cond}_to_{probe_cond}.json"
        out.write_text(json.dumps(ev, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
        events.append(ev)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_baseline_report(events), encoding="utf-8")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
