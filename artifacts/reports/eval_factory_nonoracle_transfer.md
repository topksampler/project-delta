# Non-oracle under adapters (A) + t2122 transfer (B)

**Date:** 2026-07-28
**A bank:** sealed `v0.22.0→v0.23.0` (623)
**B bank:** sealed `v0.21.0→v0.22.0` / t2122 (674)

## Invariant

```text
Oracle gain ≠ non-oracle gain.
RAG-SFT is train/eval context-matched; symbol/BM25 are distribution shift.
Diet transfer = retrain per bank, not adapter copy.
Report by drift_type.
```

## A — non-oracle under delta-diet / RAG-SFT (22→23)

| adapter × retrieval | exact | changed |
|---|---:|---:|
| delta-diet × CB | 0.815 | 0.375 |
| delta-diet × symbol | 0.841 | **0.250** |
| delta-diet × code-BM25 | 0.753 | **0.250** |
| delta-diet × evidence | 0.815 | **0.625** |
| RAG-SFT × CB | 0.461 | 0.000 |
| RAG-SFT × symbol | 0.836 | **0.000** |
| RAG-SFT × code-BM25 | 0.356 | **0.000** |
| RAG-SFT × evidence | 0.925 | **0.750** |

Non-oracle keeps a **slice** of delta-diet `changed` (0.25). RAG-SFT keeps **none** under symbol/BM25 — symbol can still look strong on pooled (0.836) with dead transitions.

## B — same ladder on v0.21→v0.22

Base (prior) vs delta-diet (chain retrain, same adapter for all three):

| condition | exact | changed |
|---|---:|---:|
| base CB | 0.245 | 0.278 |
| base symbol | 0.355 | 0.333 |
| base evidence | 0.642 | 0.667 |
| **delta-diet CB** | **0.721** | **0.778** |
| delta-diet symbol | 0.608 | 0.667 |
| delta-diet evidence | 0.690 | 0.778 |

Diet transfers: closed-book `changed` jumps past base evidence. Symbol/evidence do **not** beat delta CB on this bank (evidence ties on `changed`, loses pooled).

Note: an earlier CB-only attempt logged exact **0.828** before B2 Class-B capped sample download; chain numbers above are the complete same-adapter ladder.

## Ops landmine

B2 **Class B download cap** blocked pulls mid-ladder. Workaround: mount `data/experiments/e1_vllm` + `train_then_eval` in `infra/modal/app.py` (local adapter reuse, print summarize to logs).

## What can die

- `/tmp/*` Modal logs
- Duplicate t2122 adapter from chain retrain
- First CB-only 0.828 run without drift artifact

## What must survive

- A table (RAG-SFT needs matched context; delta-diet keeps partial non-oracle `changed`)
- B table (diet transfers; oracle less necessary once diet works)
- Platform: `train_then_eval` + local data mount for cap resilience

## Command

```bash
.venv/bin/modal run infra/modal/app.py \
  --run-id e1-vllm-c3-ft-factory-t2122-delta-diet-qwen35-08b-modal \
  --config configs/experiments/e1_vllm/c3_ft_factory_t2122_delta_diet_qwen35_08b_modal.yaml \
  --task train_then_eval --gpu H100 \
  --eval-configs 'configs/experiments/e1_vllm/c3_ft_factory_t2122_delta_diet_eval_qwen35_08b_modal.yaml,configs/experiments/e1_vllm/c3_ft_factory_t2122_delta_diet_symbol_eval_qwen35_08b_modal.yaml,configs/experiments/e1_vllm/c3_ft_factory_t2122_delta_diet_evidence_eval_qwen35_08b_modal.yaml'
```
