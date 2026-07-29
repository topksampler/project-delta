# Evidence-span RAG on factory eval (vs docs BM25 / closed-book)

Model: Qwen3.5-0.8B · sealed `e1_eval_factory_v1` (623) · no boolhint

## invariant

```text
Retrieval modality must match claim modality.
Factory gold is code contracts → retrieve AST/source spans, not docs markdown.
```

## results

| condition | exact | avg | stable | **changed** | hit_rate |
|-----------|------:|----:|-------:|------------:|---------:|
| c0 closed-book | 0.512 | 0.577 | 0.581 | **0.000** | — |
| c1 docs BM25 | 0.080 | 0.212 | 0.204 | **0.000** | 1.0 |
| **c1 evidence spans** | **0.724** | **0.825** | **0.833** | **0.375** | 0.997 |

Run: `e1-vllm-eval-factory-v1-c1-evidence-qwen35-08b-modal` (~$0.09)

First non-zero **changed** score on this bank. Existence probes nearly solved
(versioned_existence 0.965); version_delta still harder (0.567).

## what goes where

- evidence map: `data/experiments/e1_vllm/indexes/factory_v1_evidence_map_v022_v023.json`
- builder: `experiments/e1_vllm/eval_factory/build_evidence_map.py`
- summary: `artifacts/reports/eval_factory_v1_c1_evidence_summary.json`

## command

```bash
python experiments/e1_vllm/eval_factory/build_evidence_map.py \
  --probes data/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1/probes_eval.jsonl \
  --out data/experiments/e1_vllm/indexes/factory_v1_evidence_map_v022_v023.json

./scripts/lab run --target modal --gpu A10G \
  --config configs/experiments/e1_vllm/eval_factory_v1_c1_evidence_qwen35_08b_modal.yaml
```
