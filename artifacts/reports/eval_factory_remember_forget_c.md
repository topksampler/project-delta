# Remember/Forget Experiment C — teacher distill + CB consolidation

**Date:** 2026-07-29
**Arm:** C only (not ABCD compare)
**Model:** `Qwen/Qwen3.5-0.8B`
**FT:** reopened for this arc (`ft_reopened: true`); default `lab.train_sft_lora` (not DPO)

## Invariant

```text
Teacher context is train-time only.
Closed-book is the only promotion scoreboard.
Forget = still emitting gold_before on changed/removed.
Consolidation = context dropped on the same claims after teacher phase.
```

## Bet

Hybrid/oracle context teaches the new world; a second closed-book consolidation phase writes it into weights so CB `changed` rises and forget-rate falls.

## What goes where

| path | role |
|---|---|
| `experiments/e1_vllm/eval_factory/export_sft_teacher_context.py` | teacher + consolidate exporters |
| `data/experiments/e1_vllm/train_factory_v1_teacher_hybrid_sft.jsonl` | phase-1 hybrid_diff + bare paraphrases (n=2834) |
| `data/experiments/e1_vllm/train_factory_v1_teacher_consolidate_sft.jsonl` | phase-2 CB consolidate (n=2240, same seed/diet) |
| `configs/experiments/e1_vllm/remember_forget_c_*.yaml` | train/eval matrix |
| `runs/e1-vllm-remember-forget-c-*/summary.json` | forget-aware summaries |

## Run IDs

| stage | run_id |
|---|---|
| teacher train | `e1-vllm-remember-forget-c-teacher-hybrid-qwen35-08b-modal` |
| consolidate train (continues teacher) | `e1-vllm-remember-forget-c-consolidate-qwen35-08b-modal` |
| primary CB eval (22→23) | `e1-vllm-remember-forget-c-cb-eval-qwen35-08b-modal` |
| transfer CB (21→22) | `e1-vllm-remember-forget-c-t2122-cb-eval-qwen35-08b-modal` |
| teacher-context diagnostic | `e1-vllm-remember-forget-c-teacher-diag-eval-qwen35-08b-modal` |

## Scoreboard

Primary bank sealed `v0.22.0→v0.23.0`. Base CB reference: exact 0.512 / stable 0.5814 / changed 0.000 (`eval_factory_v1_base_08b_summary.json`).

| condition | exact | changed | removed | added | stable | forget | remember |
|---|---:|---:|---:|---:|---:|---:|---:|
| **C CB (win metric)** | 0.947 | **0.000** | 0.500 | 0.600 | 0.982 | **0.909** | **0.000** |
| C transfer CB t2122 | 0.902 | 0.778 | 0.500 | 0.500 | 0.977 | 0.588 | 0.059 |
| C teacher-diag hybrid (not win) | 0.844 | 0.750 | 0.500 | 0.800 | 0.875 | 0.091 | 0.636 |

Forget by drift (primary CB): changed forget=1.000 remember=0.000 (n=8); removed forget=0.667 remember=0.000 (n=3).

Pass bar (`changed≥0.50`, `removed≥0.50`, `forget≤0.25`, stable loss≤0.02, transfer not ~0): **FAIL** on primary `changed` and forget. Stable did not regress (gain vs base). Transfer `changed` did not collapse (0.778).

## Verdict

Falsifier hit: teacher-with-context `changed=0.75` / forget=0.09, closed-book `changed=0.0` / forget=0.91 — consolidation did not close the matched-context trap on the sealed 22→23 bank. Same shape as prior oracle-RAG-SFT×CB.

## What can die

- Local Modal logs (`/tmp/remember_forget_c_modal.log`)
- Teacher-context diagnostic samples if space-constrained
- Intermediate teacher adapter after consolidate is archived

## What must survive

- Exporter + configs above
- Per-run `summary.json` with forget block
- This negative: consolidation alone ≠ CB remember on 22→23
- Transfer numbers (adapter copy, not retrain)

## Command

```bash
set -a; source .env; set +a
.venv/bin/modal run infra/modal/app.py \
  --run-id e1-vllm-remember-forget-c-consolidate-qwen35-08b-modal \
  --config configs/experiments/e1_vllm/remember_forget_c_consolidate_qwen35_08b_modal.yaml \
  --task train_then_eval --gpu H100 \
  --train-chain configs/experiments/e1_vllm/remember_forget_c_teacher_hybrid_qwen35_08b_modal.yaml \
  --eval-configs 'configs/experiments/e1_vllm/remember_forget_c_cb_eval_qwen35_08b_modal.yaml,configs/experiments/e1_vllm/remember_forget_c_t2122_cb_eval_qwen35_08b_modal.yaml,configs/experiments/e1_vllm/remember_forget_c_teacher_diag_eval_qwen35_08b_modal.yaml'

.venv/bin/python experiments/e1_vllm/eval_factory/summarize_run.py \
  --samples runs/e1-vllm-remember-forget-c-cb-eval-qwen35-08b-modal/samples.jsonl \
  --probes data/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1/probes_eval.jsonl \
  --out runs/e1-vllm-remember-forget-c-cb-eval-qwen35-08b-modal/summary.json
```
