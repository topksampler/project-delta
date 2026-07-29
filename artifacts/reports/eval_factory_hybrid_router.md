# Hybrid router (entity-touched → hunks, else symbol/code)

**Date:** 2026-07-29
**Mode:** `hybrid_diff` in `src/lab/eval_vllm_qa.py`
**FT:** stopped — retrieval-only intervention

## Invariant

```text
Route per-probe with a label-free signal: entity appears on a +/- hunk line
between the two pinned tags → search diff hunks; else symbol/BM25 on after-era code.
Computable from two git tags. No drift labels, no oracle.
```

## Routing quality (offline, label-free signal vs true drift)

v0.22→v0.26: routes 100% of added/changed/removed probes to hunks; pulls 36%
of stable along (four releases genuinely touch that much). v0.22→v0.23: 131/621
routed to hunks; t2122: 196/649.

## Path-hit (top-4 vs gold evidence files)

| bank | mode | overall | changed | removed | stable |
|---|---|---:|---:|---:|---:|
| v022→v023 | symbol | 0.68 | 0.38 | 0.00 | 0.69 |
| v022→v023 | hunks only | 0.23 | 0.88 | 1.00 | 0.21 |
| v022→v023 | **hybrid** | 0.64 | **0.88** | **1.00** | 0.64 |
| v021→v022 | **hybrid** | 0.61 | **0.78** | 0.50 | 0.61 |
| v022→v026 | **hybrid** | 0.65 | **0.72** | 0.62 | 0.63 |

## QA (Qwen3.5-0.8B, greedy, base model — no adapter)

| bank | condition | exact | changed | removed | stable |
|---|---|---:|---:|---:|---:|
| v022→v023 | closed-book | 0.51 | 0.00 | — | — |
| v022→v023 | hunks only | 0.10 | 0.50 | 0.83 | 0.22 |
| v022→v023 | **hybrid** | 0.42 | **0.50** | **0.75** | 0.52 |
| v021→v022 | closed-book | 0.245 | 0.278 | 0.536 | 0.305 |
| v021→v022 | hunks only | 0.134 | 0.50 | 0.548 | 0.227 |
| v021→v022 | **hybrid** | **0.356** | **0.50** | 0.476 | 0.456 |
| v022→v026 | closed-book | 0.118 | 0.069 | 0.667 | 0.209 |
| v022→v026 | hunks only (fixed text) | 0.172 | 0.347 | 0.563 | 0.208 |
| v022→v026 | **hybrid** | **0.335** | **0.347** | 0.573 | 0.406 |
| v022→v026 | oracle ceiling | 0.569 | 0.708 | 0.896 | 0.654 |

## Findings

1. **Bet held on all three banks:** hybrid keeps hunks' transition gains and
   recovers most of the pooled/stable loss. First deployable retriever that
   lifts `changed` and pooled together (t2122, v026).
2. On v022→v023 pooled hybrid (0.42) still sits below closed-book (0.51):
   when the base model already knows the era, retrieval taxes stable
   existence answers. DECIDE cost row: retrieval pays when closed-book is
   weak, not when it is strong.
3. Oracle gap remains (v026 changed 0.35 vs 0.71): span selection inside the
   right file is the next frontier, not file routing.
4. Bug note: first hunk corpora had run-on lines (no newline normalization);
   fixed 2026-07-29. QA delta from the fix was small (v026 hunks-only
   changed 0.39→0.35); path-hit unchanged. Fixed corpora are canonical.

## What can die

- Broken-text hunk corpora (rebuilt)
- Hunks-only as a deploy default

## What must survive

- `hybrid_diff` mode + routing-signal definition (+/- line containment)
- This three-bank table; run IDs `*-c1-hybrid-qwen35-08b-modal`

## Command

```bash
./scripts/lab run --target modal --gpu H100 \
  --config configs/experiments/e1_vllm/eval_factory_v022_v026_c1_hybrid_qwen35_08b_modal.yaml
```
