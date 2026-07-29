# Eval Factory — LoRA×evidence + delta diet (2026-07-28)

Sealed bank: `v0.22.0→v0.23.0` / `e1_eval_factory_v1` (623 eval probes).

## A — LoRA + evidence RAG combo

Adapter: `e1-vllm-c3-ft-factory-v1-qwen35-08b-modal`
Eval: `e1-vllm-c3-ft-factory-v1-evidence-eval-qwen35-08b-modal`
Config: `configs/experiments/e1_vllm/c3_ft_factory_v1_evidence_eval_qwen35_08b_modal.yaml`

| condition | exact | stable | added | changed | removed |
|---|---:|---:|---:|---:|---:|
| base closed-book | 0.512 | — | — | 0.000 | — |
| evidence only (base) | 0.724 | 0.833 | 0.775 | **0.375** | 0.667 |
| LoRA closed-book | **0.884** | 0.952 | 0.350 | **0.000** | 0.333 |
| LoRA + evidence | 0.860 | 0.940 | 0.350 | **0.000** | 0.333 |

Combo does **not** recover `changed`. It underperforms LoRA-CB on pooled exact and keeps the LoRA delta profile (existence up, transition flat). Evidence alone still owns the only non-zero `changed`.

## B — delta-heavy train diet

Diet: `train_factory_v1_delta_diet_sft.jsonl` (n=2240; added 840 / changed 840 / removed 360 / stable 200)
Builder: `experiments/e1_vllm/eval_factory/build_delta_diet.py`
Train: `e1-vllm-c3-ft-factory-v1-delta-diet-qwen35-08b-modal` (~$0.37 H100)
Eval: `e1-vllm-c3-ft-factory-v1-delta-diet-eval-qwen35-08b-modal`

| condition | exact | stable | added | changed | removed |
|---|---:|---:|---:|---:|---:|
| LoRA (balanced diet) | **0.884** | **0.952** | 0.350 | 0.000 | 0.333 |
| LoRA (delta diet) | 0.815 | 0.841 | **0.450** | **0.375** | **0.500** |
| evidence only (base) | 0.724 | 0.833 | 0.775 | 0.375 | 0.667 |

Delta diet is the first **closed-book** adapter with non-zero `changed` on this bank (matches evidence-only `changed`). Cost: ~7pt pooled exact and ~11pt stable.

## What must survive

- Two scoreboards: factory ≠ eval_v3.
- Headline `changed` / drift breakdown, not pooled alone.
- Evidence = oracle spans; not deployable retriever.
- Diet mix is a first-class intervention knob (existence LoRA ≠ delta LoRA).
