# Remember/Forget A–D — shared compare freeze

**Date:** 2026-07-29
**Status:** all four arms closed-book complete. FT reopened for this arc only; **pause again** (no arm cleared the pass bar).
**Model:** `Qwen/Qwen3.5-0.8B`

## Invariant

```text
Same weights must remember + forget closed-book.
Remember = changed/added rise.
Forget   = gold_before dies on changed/removed.
Stable   loss ≤ 0.02 vs base on the same bank.
Retrieval may teach or diagnose; closed-book is the only promotion scoreboard.
```

## Shared frame

| knob | value |
|---|---|
| primary bank | sealed `v0.22.0→v0.23.0` / `e1_eval_factory_v1` |
| transfer bank | sealed `v0.21.0→v0.22.0` closed-book only |
| baselines | base CB + delta-diet CB from `eval_factory_ft_stop_freeze.md` / diet-lever freeze |
| pass bar | CB `changed ≥ 0.50`, `removed ≥ 0.50`, forget ≤ 0.25, stable loss ≤ 0.02, transfer `changed` ≉ 0 |
| FT policy | `ft_reopened: true` for this arc; closed again after this freeze |

Arm notes (sources of truth for numbers below):

- A: `artifacts/reports/remember_forget_a_dpo.md`
- B: `artifacts/reports/eval_factory_remember_forget_b.md`
- C: `artifacts/reports/eval_factory_remember_forget_c.md`
- D: `artifacts/reports/eval_factory_remember_forget_d.md`

## Shared scoreboard

Primary = v0.22→0.23 closed-book content means by drift + forget. Transfer column = v0.21→0.22 `changed` only.

| condition | exact | changed | removed | added | stable | forget | remember | transfer changed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base CB (frozen) | 0.512 | 0.000 | 0.750 | 0.725 | 0.581 | — | — | 0.278 |
| delta-diet CB (frozen) | 0.815 | **0.375** | 0.500 | 0.450 | 0.841 | — | — | **0.778** |
| A DPO CB | 0.669 | 0.000 | 0.750 | 0.750 | 0.816 | 0.546 | 0.000 | 0.722 |
| B claim-local CB | 0.504 | 0.000 | 0.500 | 0.425 | 0.569 | **0.182** | 0.000 | 0.722 |
| C CB (win metric) | 0.947 | 0.000 | 0.500 | 0.600 | 0.982 | 0.909 | 0.000 | 0.778 |
| D1 attach | 0.669 | 0.000 | 0.750 | 0.750 | 0.816 | 0.546 | 0.000 | 0.722 |
| D2 merge | 0.673 | 0.000 | 0.750 | 0.750 | 0.816 | 0.455 | 0.000 | 0.722 |

Notes:

- Delta-diet has no forget scorer (pre-dates this arc); `changed=0.375` remains the best closed-book weight `changed` on primary among rows above.
- D1 matches A bit-for-bit (same adapter, attach path). D2 ≈ D1.
- C teacher-diag hybrid (not win): `changed=0.750` / forget=0.091 — diagnostic ceiling only.

## Per-arm verdict / falsifier

| arm | one-line verdict |
|---|---|
| **A** | Falsifier hit: remember dead (`changed=0`, `remember=0`); forget stayed high (~0.55). Transfer `changed` held. |
| **B** | Forget fell (0.18) without canary abort, but remember stayed dead (`changed=0`). Surgical patches ≠ global `changed`. |
| **C** | Falsifier hit: teacher-with-context high, CB `changed=0` / forget=0.91 — consolidation did not close the matched-context trap. |
| **D** | Attach/merge protocol works (detach restores base; merge ≈ attach); inherits A's primary failure. Lineage/swap answered; remember verb not. |

**Headline across A–D:** none cleared the remember/forget pass bar. Primary `changed` stayed **0.000** on every arm; `remember_rate` stayed **0.000**. Only B met forget ≤ 0.25. Transfer `changed` never collapsed (~0.72–0.78). Prior **delta-diet CB** (`changed=0.375`) still beats every A–D primary weight row on remember.

## What can die

- Failed / intermediate adapters after B2 push (keep one composed B + A world adapter if needed for D replay)
- Local `/tmp` Modal logs
- Teacher-context diagnostic samples (C) if space-constrained
- Per-edit B adapter snapshots after interference curve is archived

## What must survive

- Sealed banks + forget-scorer definition
- Per-arm notes + primary/transfer summaries with forget blocks
- This compare table (negative on remember across A–D)
- Transfer numbers (adapter copy, not retrain)
- D attach vs merge ≈ attach result; detach = base rollback
- FT pause: do not reopen without explicit go

## Command

Summarize configs (no GPU required if `samples.jsonl` already local):

```bash
# A
.venv/bin/python experiments/e1_vllm/eval_factory/summarize_run.py \
  --samples runs/e1-vllm-remember-forget-a-dpo-eval-qwen35-08b-modal/samples.jsonl \
  --probes data/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1/probes_eval.jsonl \
  --out artifacts/reports/remember_forget_a_primary_summary.json

# B
.venv/bin/python experiments/e1_vllm/eval_factory/summarize_run.py \
  --samples runs/e1-vllm-remember-forget-b-cb-eval-qwen35-08b-modal/samples.jsonl \
  --probes data/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1/probes_eval.jsonl \
  --out artifacts/reports/remember_forget_b_primary_summary.json

# C
.venv/bin/python experiments/e1_vllm/eval_factory/summarize_run.py \
  --samples runs/e1-vllm-remember-forget-c-cb-eval-qwen35-08b-modal/samples.jsonl \
  --probes data/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1/probes_eval.jsonl \
  --out runs/e1-vllm-remember-forget-c-cb-eval-qwen35-08b-modal/summary.json

# D1 / D2
.venv/bin/python experiments/e1_vllm/eval_factory/summarize_run.py \
  --samples runs/e1-vllm-remember-forget-d1-attach-eval-qwen35-08b-modal/samples.jsonl \
  --probes data/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1/probes_eval.jsonl \
  --out artifacts/reports/eval_factory_remember_forget_d1_attach_primary_summary.json
```

Train/eval configs live under `configs/experiments/e1_vllm/remember_forget_{a,b,c,d*}*.yaml`. Replay commands are in each arm note.
