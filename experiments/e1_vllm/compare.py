#!/usr/bin/env python3
"""Build e1_vllm comparison tables from frozen eval_v3 metrics JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "artifacts" / "reports" / "e1_vllm_failure_modes.md"

# condition_id → (run_id, role)
RUNS: list[tuple[str, str, str]] = [
    ("c0_base", "e1-vllm-c0-base-eval-v3-qwen35-08b-modal", "base, no retrieval"),
    ("c6_shuffle", "e1-vllm-c6-shuffle-eval-v3-qwen35-08b-modal", "random doc_8 chunks"),
    ("c3_mill_v3", "e1-vllm-c3-ft-mill-v3-eval-v3-qwen35-08b-modal", "LoRA@doc_0 mill v3"),
    ("c3_mill_v4", "e1-vllm-c3-ft-mill-v4-eval-v3-qwen35-08b-modal", "LoRA@doc_0 mill v4 T1–T7"),
    ("c4_ft_rag", "e1-vllm-c4-ft-rag-fresh-eval-v3-qwen35-08b-modal", "mill-v3 LoRA + doc_8 BM25"),
    ("c2_rag_stale", "e1-vllm-c2-rag-stale-eval-v3-qwen35-08b-modal", "BM25 doc_0"),
    ("c1_rag_fresh", "e1-vllm-c1-rag-fresh-eval-v3-qwen35-08b-modal", "BM25 doc_8"),
]

TRAIN_COST = [
    ("c3_train_v3", "e1-vllm-c3-ft-mill-v3-qwen35-08b-modal", "H100", "~200 steps"),
    ("c3_train_v4", "e1-vllm-c3-ft-mill-v4-qwen35-08b-modal", "H100", "~200 steps / ~460s train_runtime"),
]


def _load(metrics_dir: Path, run_id: str) -> dict:
    path = metrics_dir / f"{run_id}.json"
    if not path.is_file():
        path = metrics_dir / run_id / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def render(metrics_dir: Path) -> str:
    rows = []
    for cid, run_id, role in RUNS:
        m = _load(metrics_dir, run_id)
        b = m["breakdown"]["by_eval_class"]
        fm = m["breakdown"]["by_failure_mode"]
        rows.append(
            {
                "cid": cid,
                "run_id": run_id,
                "role": role,
                "acc": m["accuracy"],
                "A": b.get("A", 0.0),
                "B": b.get("B", 0.0),
                "C": b.get("C", 0.0),
                "D": b.get("D", 0.0),
                "fm": fm,
                "retrieval": m.get("retrieval"),
            }
        )

    lines: list[str] = []
    lines.append("# e1_vllm — failure modes & intervention comparison")
    lines.append("")
    lines.append("## invariant")
    lines.append("")
    lines.append("```text")
    lines.append("Same eval_v3 (46 items, A20/B10/C8/D8), same model Qwen/Qwen3.5-0.8B.")
    lines.append("Primary signal = by_eval_class + failure_mode, not headline accuracy alone.")
    lines.append("Index freshness and retrieval relevance are separate levers from LoRA@doc_0.")
    lines.append("```")
    lines.append("")
    lines.append("## legend")
    lines.append("")
    lines.append("### snapshots")
    lines.append("")
    lines.append("| name | meaning |")
    lines.append("|---|---|")
    lines.append("| `doc_0` | vLLM docs @ **v0.22.0** — train corpus and stale RAG index |")
    lines.append("| `doc_8` | vLLM docs @ **v0.23.0** — fresh RAG index; truth for new facts |")
    lines.append("")
    lines.append("### conditions")
    lines.append("")
    lines.append("| id | recipe |")
    lines.append("|---|---|")
    lines.append("| `c0_base` | base model only — no LoRA, no retrieval |")
    lines.append("| `c1_rag_fresh` | base + BM25 top-k from **doc_8** |")
    lines.append("| `c2_rag_stale` | base + BM25 top-k from **doc_0** |")
    lines.append("| `c3_ft` / `c3_mill_v*` | LoRA trained on doc_0 mill data; no RAG at eval |")
    lines.append("| `c4_ft_rag` | c3 LoRA + doc_8 BM25 at eval |")
    lines.append("| `c6_shuffle` | base + **random** doc_8 chunks (retrieval placebo) |")
    lines.append("")
    lines.append("No `c5` in this matrix.")
    lines.append("")
    lines.append("### eval classes (eval_v3 columns A–D)")
    lines.append("")
    lines.append("| class | tests |")
    lines.append("|---|---|")
    lines.append("| **A** | stable facts (true in both versions) |")
    lines.append("| **B** | what changed / only-in-0.23 |")
    lines.append("| **C** | how-to / procedural |")
    lines.append("| **D** | version-binary (“does X exist in v0.22?”) |")
    lines.append("")
    lines.append("### mill train classes (LoRA diet)")
    lines.append("")
    lines.append("| id | skill |")
    lines.append("|---|---|")
    lines.append("| T1 | declarative flag Q→A |")
    lines.append("| T2 | procedural how-to |")
    lines.append("| T3 | version-conditioned (“In v0.22, …”) |")
    lines.append("| T4 | abstain / unknown |")
    lines.append("| T5 | contrast (“not this flag”) |")
    lines.append("| T6 | answer + cite source path |")
    lines.append("| T7 | multi-hop (two chunks) |")
    lines.append("| T8 | cross-version delta — not used in c3@doc_0 |")
    lines.append("")
    lines.append("`mill v3` / `mill v4` = successive train-set builds (v4 adds T2/T6/T7).")
    lines.append("")
    lines.append("### other")
    lines.append("")
    lines.append("| term | meaning |")
    lines.append("|---|---|")
    lines.append("| `e1_vllm` | this experiment (first DELTA vertical slice) |")
    lines.append("| BM25 | lexical retrieval over chunk text (no embeddings) |")
    lines.append("| `eval_v3` | frozen 46-item Q&A set used for all conditions here |")
    lines.append("| `run_id` | one Modal job; evidence under B2 `runs/{run_id}/` |")
    lines.append("")
    lines.append("## what goes where")
    lines.append("")
    lines.append("- this report: `artifacts/reports/e1_vllm_failure_modes.md` (also B2)")
    lines.append("- regenerator: `experiments/e1_vllm/compare.py`")
    lines.append("- per-run evidence: B2 `runs/{run_id}/metrics.json` + `samples.jsonl`")
    lines.append("- condition definitions: `docs/experiments/e1_vllm.md`")
    lines.append("")
    lines.append("## what can die")
    lines.append("")
    lines.append("- superseded draft tables in chat logs")
    lines.append("- local `/tmp` metric caches used to rebuild this file")
    lines.append("")
    lines.append("## what must survive")
    lines.append("")
    lines.append("- frozen `eval_v3.jsonl` + run_ids listed below")
    lines.append("- this report next to its generator command")
    lines.append("- B2 copies of metrics/samples for each run_id")
    lines.append("")
    lines.append("## scoreboard (eval_v3)")
    lines.append("")
    lines.append("| condition | acc | A | B | C | D | role |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for r in sorted(rows, key=lambda x: x["acc"]):
        lines.append(
            f"| `{r['cid']}` | {_fmt(r['acc'])} | {_fmt(r['A'])} | {_fmt(r['B'])} | "
            f"{_fmt(r['C'])} | {_fmt(r['D'])} | {r['role']} |"
        )
    lines.append("")
    lines.append("## failure modes (counts / 46)")
    lines.append("")
    lines.append("| condition | correct | wrong | stale_version | abstain_wrong |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: x["acc"]):
        fm = r["fm"]
        lines.append(
            f"| `{r['cid']}` | {fm.get('correct', 0)} | {fm.get('wrong', 0)} | "
            f"{fm.get('stale_version', 0)} | {fm.get('abstain_wrong', 0)} |"
        )
    lines.append("")
    lines.append("## findings (locked to this matrix)")
    lines.append("")
    lines.append("```text")
    lines.append("1. Fresh BM25 (c1) wins overall and on B. Index freshness > doc_0 LoRA for drift.")
    lines.append("2. Shuffle (c6) < base → c1 lift is relevance, not “any context”.")
    lines.append("3. Stale BM25 (c2) beats base but hurts D vs c0 (wrong-era contamination).")
    lines.append("4. LoRA@doc_0 (c3) helps A/C; does not recover B; D weak/worse on v4.")
    lines.append("5. c4 (LoRA + fresh RAG) interferes: abstain_wrong spikes; D collapses.")
    lines.append("6. c1 D flat vs c0: many D items are negative existence; fresh chunks bias YES.")
    lines.append("```")
    lines.append("")
    lines.append("## cost proxy")
    lines.append("")
    lines.append("```text")
    lines.append("Eval conditions: Modal A10G, ~100–140s compute each (46 generations).")
    lines.append("No LoRA train cost for c0/c1/c2/c6.")
    lines.append("c3/c4 require a prior H100 LoRA train (~200 steps; mill v4 ~460s train_runtime).")
    lines.append("Dollar cost not frozen here — GPU class + wall seconds are the durable proxy.")
    lines.append("```")
    lines.append("")
    lines.append("| train run | gpu | note |")
    lines.append("|---|---|---|")
    for cid, run_id, gpu, note in TRAIN_COST:
        lines.append(f"| `{cid}` (`{run_id}`) | {gpu} | {note} |")
    lines.append("")
    lines.append("## run_ids")
    lines.append("")
    lines.append("| condition | run_id |")
    lines.append("|---|---|")
    for r in rows:
        lines.append(f"| `{r['cid']}` | `{r['run_id']}` |")
    lines.append("")
    lines.append("## command")
    lines.append("")
    lines.append("```bash")
    lines.append("# after downloading metrics into a dir as {run_id}.json")
    lines.append("python experiments/e1_vllm/compare.py --metrics-dir /tmp/e1_compare \\")
    lines.append("  --out artifacts/reports/e1_vllm_failure_modes.md")
    lines.append("```")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    text = render(args.metrics_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
