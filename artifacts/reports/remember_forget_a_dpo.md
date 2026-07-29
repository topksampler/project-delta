# Remember/Forget Experiment A — DPO preference (arm note)

**Date:** 2026-07-29
**Status:** closed-book primary + transfer complete. Not an ABCD compare.
**FT:** reopened for this arc only (`ft_reopened: true`).

## Invariant

```text
Closed-book is the scoreboard.
Remember = changed/added rise.
Forget   = stale gold_before dies on changed/removed.
Stable   budget ≤ 0.02 absolute loss vs base on the same bank.
```

## Bet

Paired preference teaches forget: same question, reject old world, prefer new world.

## What goes where

| artifact | role |
|---|---|
| `data/experiments/e1_vllm/train_factory_v1_dpo_remember_forget.jsonl` | DPO train (diet: delta×40, stable_cap 200, seed 20260728) |
| `data/experiments/e1_vllm/eval_factory_v1_dpo_dev.jsonl` | DPO train-time eval slice |
| `configs/experiments/e1_vllm/remember_forget_a_dpo_*.yaml` | train + primary CB + t2122 CB |
| `runs/e1-vllm-remember-forget-a-dpo-qwen35-08b-modal/` | LoRA adapter (B2) |
| `artifacts/reports/remember_forget_a_primary_summary.json` | v0.22→0.23 CB + forget |
| `artifacts/reports/remember_forget_a_t2122_summary.json` | v0.21→0.22 CB + forget |
| this file | arm verdict only |

## Run IDs

| role | run_id |
|---|---|
| train (DPO LoRA) | `e1-vllm-remember-forget-a-dpo-qwen35-08b-modal` |
| primary CB eval | `e1-vllm-remember-forget-a-dpo-eval-qwen35-08b-modal` |
| transfer CB eval | `e1-vllm-remember-forget-a-dpo-t2122-eval-qwen35-08b-modal` |

## Configs

- `configs/experiments/e1_vllm/remember_forget_a_dpo_qwen35_08b_modal.yaml`
- `configs/experiments/e1_vllm/remember_forget_a_dpo_eval_qwen35_08b_modal.yaml`
- `configs/experiments/e1_vllm/remember_forget_a_dpo_t2122_eval_qwen35_08b_modal.yaml`

LoRA band matches prior factory LoRA: r=8, α=16, dropout=0.05, max_steps=300, lr=2e-4, bs=8. Trainer: `training.module: lab.train_dpo_lora`, β=0.1.

## Primary closed-book (v0.22.0→v0.23.0)

Content means by drift (same scoreboard as factory freeze tables):

| condition | exact | changed | removed | added | stable | forget_rate | remember_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| base CB (frozen) | 0.512 | 0.000 | 0.750 | 0.725 | 0.581 | — | — |
| delta-diet CB (frozen) | 0.815 | 0.375 | — | — | — | — | — |
| **A DPO CB** | **0.669** | **0.000** | **0.750** | **0.750** | **0.816** | **0.546** | **0.000** |

Forget detail (n_scored=11): changed forget 0.50 / remember 0.00; removed forget 0.667 / remember 0.00.

Stable vs base: **+0.235** (budget is loss ≤0.02 — no stable collapse).

## Transfer closed-book (v0.21.0→v0.22.0)

| condition | exact | changed | removed | added | stable | forget_rate | remember_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| base CB (frozen) | 0.245 | 0.278 | 0.536 | 0.589 | 0.305 | — | — |
| delta-diet CB (frozen) | 0.721 | 0.778 | — | — | — | — | — |
| **A DPO CB** | **0.666** | **0.722** | **0.583** | **0.635** | **0.820** | **0.529** | **0.235** |

Forget detail (n_scored=17): changed forget 0.667 / remember 0.00; removed forget 0.50 / remember 0.286.

Transfer `changed` does **not** collapse (~0.72).

## Pass bar (candidate interesting)

Plan bar: CB `changed ≥ 0.50`, `removed ≥ 0.50`, forget ≤ 0.25, stable loss ≤ 0.02, transfer changed ≉ 0.

| gate | primary A | transfer A |
|---|---|---|
| changed ≥ 0.50 | **fail** (0.00) | pass (0.72) |
| removed ≥ 0.50 | pass (0.75 content) | pass (0.58) |
| forget ≤ 0.25 | **fail** (0.55) | **fail** (0.53) |
| stable loss ≤ 0.02 | pass (gain) | pass (gain) |
| transfer changed ≉ 0 | — | pass |

**Arm verdict:** falsifier hit on primary — remember did not rise (`changed=0`, `remember_rate=0`); forget stayed high (~0.55). Transfer `changed` held, so the adapter is not inert, but the primary forget/remember bet failed.

## What can die

- Local `/tmp` Modal logs
- Smoke `/tmp/dpo_smoke`

## What must survive

- DPO JSONL + metas
- Three configs
- Primary + transfer summaries with forget blocks
- This arm note (negative primary result)

## Command

```bash
set -a && source .env && set +a
.venv/bin/modal run infra/modal/app.py \
  --run-id e1-vllm-remember-forget-a-dpo-qwen35-08b-modal \
  --config configs/experiments/e1_vllm/remember_forget_a_dpo_qwen35_08b_modal.yaml \
  --task train_then_eval --gpu H100 \
  --eval-configs 'configs/experiments/e1_vllm/remember_forget_a_dpo_eval_qwen35_08b_modal.yaml,configs/experiments/e1_vllm/remember_forget_a_dpo_t2122_eval_qwen35_08b_modal.yaml'
```
