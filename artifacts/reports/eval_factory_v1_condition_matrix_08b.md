# Eval factory v1 — 0.8B condition matrix (boolhint + RAG ladder)

Sealed bank: `e1_eval_factory_v1` / 623 probes
Model: `Qwen/Qwen3.5-0.8B`
Instrument: `eval.boolean_answer_hint: true` on all new runs (A)

## invariant

```text
Report by drift_type. changed=0 is the version-delta hole.
eval_v3 intervention ranking is a different scoreboard.
Boolhint is eval-time only — sealed probes are not rewritten.
```

## A — boolean prompt hygiene

`eval.boolean_answer_hint` appends `Answer yes or no.` when gold is boolean
(and the question lacks that phrase). Finalize also re-attaches the phrase for
future surface builds.

| condition | exact | avg | stable | changed | $ |
|-----------|------:|----:|-------:|--------:|--:|
| c0 no hint (prior) | **0.512** | 0.577 | 0.581 | 0.000 | 0.04 |
| c0 + boolhint | 0.265 | 0.330 | 0.326 | 0.000 | 0.07 |

**Boolhint hurt this 0.8B baseline** (exact −0.25). Mechanism: on 154 boolean
probes, the model flipped `Yes…` → `No…` when the hint was appended (helped
only 11). Do not promote boolhint as default for 0.8B on this bank.

## B/C — retrieval ladder on factory eval (with boolhint)

| condition | exact | avg | stable | changed | retrieval | $ |
|-----------|------:|----:|-------:|--------:|-----------|--:|
| c0 boolhint | 0.265 | 0.330 | 0.326 | 0.000 | none | 0.07 |
| c1 fresh BM25 doc_8 | 0.080 | 0.211 | 0.204 | 0.000 | hit_rate 1.0 | 0.10 |
| c2 stale BM25 doc_0 | 0.072 | 0.205 | 0.198 | 0.000 | hit_rate 1.0 | 0.09 |
| c6 shuffle doc_8 | 0.027 | 0.183 | 0.174 | 0.000 | hit_rate 1.0 | 0.11 |

Ordering on this bank: **closed-book ≫ fresh ≈ stale ≫ shuffle**.
That is **not** the eval_v3 story (where fresh RAG won). Changed stays **0**
everywhere. Fresh RAG also doubles hallucinated-churn rate on stable deltas
(36/202 → 93/202 invent rename/deprecate/remove language).

## paths

- matrix JSON: `artifacts/reports/eval_factory_v1_condition_matrix_08b.json`
- runs under `runs/e1-vllm-eval-factory-v1-{base-boolhint,c1-rag-fresh,c2-rag-stale,c6-shuffle}-qwen35-08b-modal/`

## what can die

- boolhint as a default if follow-up confirms it only taxes this model/bank

## what must survive

- sealed factory bank unchanged
- this matrix + the contradiction vs eval_v3 ranking
- `changed=0` across all four conditions

## command

```bash
./scripts/lab run --target modal --gpu A10G \
  --config configs/experiments/e1_vllm/eval_factory_v1_base_boolhint_qwen35_08b_modal.yaml
./scripts/lab run --target modal --gpu A10G \
  --config configs/experiments/e1_vllm/eval_factory_v1_c1_rag_fresh_qwen35_08b_modal.yaml
./scripts/lab run --target modal --gpu A10G \
  --config configs/experiments/e1_vllm/eval_factory_v1_c2_rag_stale_qwen35_08b_modal.yaml
./scripts/lab run --target modal --gpu A10G \
  --config configs/experiments/e1_vllm/eval_factory_v1_c6_shuffle_qwen35_08b_modal.yaml
```

Note: PROTOCOL “four-transition” (more version *pairs* for factory BUILD) is
still open — this matrix is the four *conditions* on the sealed v0.22→v0.23 eval.
