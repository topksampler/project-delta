# Eval factory — fair RAG ladder (no boolhint)

Same sealed `e1_eval_factory_v1` bank, Qwen3.5-0.8B, **no** `boolean_answer_hint`.

## invariant

```text
Compare retrieval conditions only against the no-hint closed-book baseline.
eval_v3 ranking is a different scoreboard.
```

## results

| condition | exact | avg | stable | changed | $ |
|-----------|------:|----:|-------:|--------:|--:|
| c0 closed-book | **0.512** | 0.577 | 0.581 | 0.000 | 0.04 |
| c1 fresh BM25 doc_8 | 0.080 | 0.212 | 0.204 | 0.000 | 0.09 |
| c2 stale BM25 doc_0 | 0.084 | 0.214 | 0.209 | 0.000 | 0.10 |

Retrieval hit_rate_nonempty = 1.0 on both RAG runs. Fresh ≈ stale ≪ closed-book.
`changed` stays 0. Boolhint was not the confounder — the RAG inversion vs eval_v3 remains.

## paths

- `artifacts/reports/eval_factory_v1_rag_ladder_nohint.json`
- runs: `…-c1-rag-fresh-nohint-…`, `…-c2-rag-stale-nohint-…`

## command

```bash
./scripts/lab run --target modal --gpu A10G \
  --config configs/experiments/e1_vllm/eval_factory_v1_c1_rag_fresh_nohint_qwen35_08b_modal.yaml
./scripts/lab run --target modal --gpu A10G \
  --config configs/experiments/e1_vllm/eval_factory_v1_c2_rag_stale_nohint_qwen35_08b_modal.yaml
```
