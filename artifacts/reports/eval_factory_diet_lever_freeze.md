# Eval Factory v1 — diet is the intervention lever (freeze writeup)

**Date:** 2026-07-28
**Bank:** sealed `v0.22.0→v0.23.0` / `e1_eval_factory_v1` (623 eval probes)
**Model:** Qwen3.5-0.8B

## Invariant

```text
Pooled exact can lie. Report by drift_type.
Existence LoRA ≠ delta LoRA.
Oracle evidence ≠ deployable retrieval.
Train-with-context adapters are context-conditioned at eval.
Factory scoreboard ≠ eval_v3 / TopicKnowledgeProfile.
```

## What goes where

| artifact | role |
|---|---|
| sealed probes_eval / probes_train | frozen bank |
| `train_factory_v1_sft.jsonl` | balanced existence diet (~97% stable) |
| `train_factory_v1_delta_diet_sft.jsonl` | delta-overweight diet (40× deltas, stable cap 200) |
| `train_factory_v1_delta_evidence_sft.jsonl` | same diet + evidence in user turn |
| `indexes/factory_v1_evidence_map_v022_v023.json` | oracle spans @ eval |
| adapters under `runs/e1-vllm-c3-ft-factory-v1-*` | condition checkpoints |

## Scoreboard

| condition | exact | stable | added | changed | removed |
|---|---:|---:|---:|---:|---:|
| base closed-book | 0.512 | 0.581 | 0.725 | 0.000 | 0.750 |
| evidence only | 0.724 | 0.833 | 0.775 | 0.375 | 0.667 |
| LoRA balanced CB | 0.884 | **0.952** | 0.350 | 0.000 | 0.333 |
| LoRA balanced + evidence | 0.860 | 0.940 | 0.350 | 0.000 | 0.333 |
| LoRA delta-diet CB | 0.815 | 0.841 | 0.450 | 0.375 | 0.500 |
| LoRA delta-diet + evidence | 0.815 | 0.896 | 0.500 | **0.625** | 0.500 |
| RAG-SFT CB (no context @ eval) | 0.461 | 0.474 | 0.400 | 0.000 | 0.500 |
| **RAG-SFT + evidence @ eval** | **0.925** | 0.944 | **0.700** | **0.750** | 0.500 |

## Durable findings

1. **Diet is first-class.** Existence diet → high pooled, `changed=0`. Delta diet → first closed-book non-zero `changed` (0.375).
2. **Diet unlocks the combo.** Existence LoRA + evidence still `changed=0`. Delta-diet + evidence → `changed=0.625` (above evidence-only and delta CB).
3. **RAG-aware FT is context-conditioned.** Closed-book collapses (0.461). Matched evidence eval is best-so-far exact **0.925** / `changed` **0.750**.
4. Docs/naive code BM25 are not the factory intervention. Evidence remains oracle.

## What can die

- Rejected adapters after B2 push
- Local run caches
- Intermediate surface-gen drafts

## What must survive

- Full drift table above
- Delta-diet recipe (`delta_copies=40`, `stable_cap=200`, seed `20260728`)
- Negatives: existence+evidence `changed=0`; RAG-SFT without context @ eval collapses
- Positives: delta-diet CB `changed=0.375`; delta+evidence `changed=0.625`; RAG-SFT+evidence `0.925` / `0.750`
- Two-scoreboard split; profile wheel past v7 still paused

## Command

```bash
.venv/bin/python experiments/e1_vllm/eval_factory/build_delta_diet.py \
  --probes data/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1/probes_train.jsonl \
  --out data/experiments/e1_vllm/train_factory_v1_delta_diet_sft.jsonl

.venv/bin/python experiments/e1_vllm/eval_factory/export_sft_with_evidence.py \
  --probes data/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1/probes_train.jsonl \
  --out data/experiments/e1_vllm/train_factory_v1_delta_evidence_sft.jsonl \
  --evidence-map-out data/experiments/e1_vllm/indexes/factory_v1_evidence_map_train_v022_v023.json

.venv/bin/python experiments/e1_vllm/eval_factory/summarize_run.py \
  --samples runs/<run_id>/samples.jsonl \
  --probes data/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1/probes_eval.jsonl \
  --out artifacts/reports/<name>.json
```

## FT status

Factory-v1 LoRA arc in this writeup is **frozen for interpretation**. Profile wheel past v7 remains **paused** unless explicitly reopened.
