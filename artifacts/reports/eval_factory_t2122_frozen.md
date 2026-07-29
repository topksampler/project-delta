# Eval factory — frozen EvalEnvironment v0.21.0→v0.22.0

Protocol: `e1_eval_factory_v1`
Status: **frozen** (`freeze_ready=true`, sealed)

## invariant

```text
Same freeze gates as v0.22→v0.23.
Audit provenance for this transition: executable_reverify_v1
(claims already AST-verified; not a human/agent panel).
```

## frozen counts

| measure | value |
|---------|------:|
| verified claims | 1,071 |
| deltas | **62** |
| train / dev / eval claims | 630 / 216 / 225 |
| final eval probes | 674 |
| surface coverage | 99.85% |
| max train/eval Jaccard | 0.75 |
| audit agreement | 50/50 (`executable_reverify_v1`) |

## what goes where

- local: `data/experiments/e1_vllm/eval_factory/v0.21.0_to_v0.22.0/e1_eval_factory_v1/`
- B2: `datasets/experiments/e1_vllm/eval_factory/v0.21.0_to_v0.22.0/e1_eval_factory_v1/`
- teacher: `e1-vllm-eval-factory-t2122-surface-gen-qwen35-4b-modal` (accept 97.8%)

## what can die

- census-only sibling under `e1_eval_factory_census_v1` for this transition

## what must survive

- sealed manifest + hashed artifacts
- audit provenance label
- this freeze note

## command

```bash
python -m experiments.e1_vllm.eval_factory.cli validate \
  --artifact-dir \
  data/experiments/e1_vllm/eval_factory/v0.21.0_to_v0.22.0/e1_eval_factory_v1
```
