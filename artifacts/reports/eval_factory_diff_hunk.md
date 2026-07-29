# Diff-hunk corpus — path-hit + QA

**Date:** 2026-07-29
**Bank:** sealed factory-v1 `v0.22.0→v0.23.0`
**FT:** stopped (reuse frozen delta-diet adapter only)

## Invariant

```text
A version-diff is a searchable corpus, not a path allowlist.
Diff-hunk BM25 is a transition specialist: lifts changed/removed path-hit and QA,
collapses stable / pooled exact.
Stacking it on delta-diet without a router regresses changed vs closed-book diet.
```

## Path-hit (top-4 vs gold evidence files)

| corpus | overall | added | changed | removed | stable |
|---|---:|---:|---:|---:|---:|
| code_8 BM25 | 0.311 | 0.500 | 0.375 | **0.000** | 0.309 |
| symbol (prior) | 0.684 | 1.000 | 0.375 | **0.000** | 0.688 |
| **diff_hunk BM25** | 0.230 | **0.750** | **0.875** | **1.000** | 0.211 |

n: stable 602 / added 8 / changed 8 / removed 3.

## QA (Qwen3.5-0.8B, greedy)

| condition | exact | added | changed | removed | stable |
|---|---:|---:|---:|---:|---:|
| base closed-book (prior) | 0.51 | — | **0.00** | — | — |
| base × evidence oracle (prior) | 0.72 | — | 0.38 | — | — |
| **base × diff_hunk** | 0.098 | **0.78** | **0.50** | **0.83** | 0.22 |
| delta-diet closed-book (prior) | 0.82 | — | **0.38** | — | — |
| delta-diet × symbol (prior) | 0.84 | — | 0.25 | — | — |
| **delta-diet × diff_hunk** | 0.652 | 0.45 | **0.13** | 0.33 | 0.67 |

Runs:
- `e1-vllm-eval-factory-v1-c1-diff-hunk-qwen35-08b-modal`
- `e1-vllm-c3-ft-factory-v1-delta-diet-diff-hunk-eval-qwen35-08b-modal`

## Verdict

1. Step-1 bet **held**: removed path-hit 0→1.0; changed 0.375→0.875.
2. Step-2: on **base**, diff-hunk is the first *deployable* retriever that beats oracle on `changed` (0.50 vs 0.38) and unlocks `removed` — at the cost of pooled exact.
3. On **delta-diet**, stacking diff-hunk **hurts** `changed` (0.38→0.13). Same composition-interference pattern as existence-LoRA + oracle.
4. DECIDE implication: route by failure class — do not one-retriever-fits-all.

## What can die

- Diff-as-allowlist defaults (`diff_bm25`, `symbol_diff`)
- Blind diet+diff_hunk stacking

## What must survive

- Diff-hunk builder + BM25 index
- Path-hit + QA tables above

## Command

```bash
.venv/bin/python experiments/e1_vllm/eval_factory/build_diff_hunk_corpus.py \
  --before data/experiments/e1_vllm/snapshots/v0.22.0 \
  --after data/experiments/e1_vllm/snapshots/v0.23.0 \
  --alias diff_hunk_022_023 --before-tag v0.22.0 --after-tag v0.23.0 \
  --out data/experiments/e1_vllm/corpus_diff_hunk_022_023.jsonl
```
