# Diff-hunk transfer — v0.21→v0.22 (held-out)

**Date:** 2026-07-29
**Bank:** sealed factory-v1 `v0.21.0→v0.22.0` (t2122)
**Recipe:** same as v022→v023 — no retuning

## Invariant

```text
Diff-hunk BM25 on base is a transition specialist across both sealed banks.
Pooled exact drops; changed lifts. Diet+diff_hunk still not a free stack.
```

## Path-hit (top-4)

| corpus | overall | added | changed | removed | stable |
|---|---:|---:|---:|---:|---:|
| **diff_hunk_021_022 BM25** | 0.157 | 0.500 | 0.444 | **0.667** | 0.124 |

n: stable 596 / added 32 / changed 9 / removed 12.

## QA

| condition | exact | added | changed | removed | stable |
|---|---:|---:|---:|---:|---:|
| base closed-book | 0.245 | 0.589 | 0.278 | 0.536 | 0.305 |
| base × evidence | 0.642 | 0.729 | 0.667 | 0.714 | 0.750 |
| **base × diff_hunk** | 0.134 | **0.755** | **0.500** | 0.548 | 0.227 |
| delta-diet closed-book | 0.721 | 0.458 | **0.778** | 0.524 | 0.764 |
| delta-diet × symbol | 0.608 | 0.354 | 0.667 | 0.429 | 0.638 |
| delta-diet × diff_hunk | 0.240 | 0.417 | 0.667 | 0.357 | 0.219 |

## Cross-transition compare (base × diff_hunk)

| transition | exact | changed | removed |
|---|---:|---:|---:|
| v0.22→v0.23 | 0.098 | **0.50** | **0.83** |
| v0.21→v0.22 | 0.134 | **0.50** | 0.55 |

## Verdict

Transfer **holds on the core claim**: deployable diff-hunk retrieval lifts `changed` on a held-out bank without FT (0.28→0.50), matching the 0.50 `changed` on v022→v023. `removed` lift is bank-dependent (strong on v023, flat on t2122 where base already scored 0.54). Delta-diet closed-book remains the best *balanced* recipe; stacking diff-hunk on diet collapses pooled exact on both banks.

## What must survive

- Both corpora + this transfer table
- DECIDE: route transition mass → diff-hunk; balanced → diet; never pooled-only promote

## Command

```bash
.venv/bin/python experiments/e1_vllm/eval_factory/build_diff_hunk_corpus.py \
  --before data/experiments/e1_vllm/snapshots/v0.21.0 \
  --after data/experiments/e1_vllm/snapshots/v0.22.0 \
  --alias diff_hunk_021_022 --before-tag v0.21.0 --after-tag v0.22.0 \
  --out data/experiments/e1_vllm/corpus_diff_hunk_021_022.jsonl
```
