# Eval Factory FT arc — freeze + stop FT

**Date:** 2026-07-29 (updated; original stop 2026-07-28)
**Status:** factory LoRA **STOPPED** after remember/forget A–D closeout. Profile wheel past v7 remains paused. Do not restart FT without explicit go.

## Invariant

```text
Report by drift_type. Pooled exact lies.
Existence diet ≠ delta diet.
Oracle evidence ≠ symbol/BM25 ≠ matched context-SFT.
Factory ≠ eval_v3 ≠ TopicKnowledgeProfile.
Same weights must remember + forget closed-book (A–D scoreboard).
```

## What goes where

| artifact | role |
|---|---|
| sealed banks `v0.22→0.23`, `v0.21→0.22` | frozen eval |
| `train_factory_v1_*_sft.jsonl` | diet / context variants |
| `artifacts/reports/eval_factory_diet_lever_freeze.md` | diet×evidence |
| `artifacts/reports/eval_factory_nonoracle_transfer.md` | non-oracle eval + t2122 |
| `artifacts/reports/eval_factory_remember_forget_abcd.md` | A–D remember/forget compare |
| this file | **stop-FT seal** |

## Durable scoreboard (22→23, 0.8B)

| condition | exact | changed |
|---|---:|---:|
| base CB | 0.512 | 0.000 |
| evidence only | 0.724 | 0.375 |
| LoRA balanced CB | 0.884 | 0.000 |
| delta-diet CB | 0.815 | 0.375 |
| delta-diet + symbol / BM25 | 0.841 / 0.753 | 0.250 |
| delta-diet + evidence | 0.815 | **0.625** |
| oracle-RAG-SFT + evidence | **0.925** | **0.750** |
| oracle-RAG-SFT + symbol | 0.836 | 0.000 |
| **symbol-SFT × CB** | 0.840 | 0.000 |
| **symbol-SFT × symbol** | 0.745 | **0.125** |
| **symbol-SFT × evidence** | 0.722 | 0.000 |

## Transfer (21→22)

| condition | exact | changed |
|---|---:|---:|
| base CB | 0.245 | 0.278 |
| delta-diet CB | **0.721** | **0.778** |
| delta-diet symbol / evidence | 0.608 / 0.690 | 0.667 / 0.778 |

## Closeout findings (2026-07-28)

1. **Diet** is the closed-book lever for `changed`.
2. **Matched context** (oracle train + oracle eval) is the ceiling.
3. **Symbol-SFT** partially unlocks symbol-eval (`changed` 0.125 vs 0 for oracle-SFT×symbol) but is weaker than delta-diet×symbol (0.250) and **does not** transfer to evidence (`changed` 0).
4. Next research bet after symbol-SFT (no FT): **deployable retriever quality** on a delta-diet backbone — not more adapters.

## Remember/forget reopen (2026-07-29)

FT was **explicitly reopened** for the A–D remember/forget arc only (`ft_reopened: true`). Shared compare: `artifacts/reports/eval_factory_remember_forget_abcd.md`.

| arm | primary `changed` | forget | remember | pass bar |
|---|---:|---:|---:|---|
| A DPO | 0.000 | 0.546 | 0.000 | fail |
| B claim-local | 0.000 | 0.182 | 0.000 | fail (forget only) |
| C teacher+consolidate | 0.000 | 0.909 | 0.000 | fail |
| D1 attach / D2 merge | 0.000 | 0.546 / 0.455 | 0.000 | fail |

**Honest status:** numbers do **not** warrant continuing FT. No arm cleared `changed ≥ 0.50` or `remember > 0` on the sealed 22→23 bank. Delta-diet CB (`changed=0.375`) still dominates A–D on primary remember. **Pause FT again** unless explicit go.

## Ops

B2 download caps: raise **Download Bandwidth** (API text says “Class B”) under
Caps & Alerts if pulls 403. Closeout samples archived locally after cap lift.

## What can die

- Rejected adapters after B2 push
- Local run caches
- `/tmp` Modal logs

## What must survive

- Full drift tables
- Diet recipe (`delta_copies=40`, `stable_cap=200`, seed `20260728`)
- Negatives + positives above
- A–D compare + per-arm forget tables
- **Stop FT** until explicit reopen

## Command

```bash
.venv/bin/python experiments/e1_vllm/eval_factory/summarize_run.py \
  --samples runs/<run_id>/samples.jsonl \
  --probes data/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1/probes_eval.jsonl \
  --out artifacts/reports/<name>.json
```
