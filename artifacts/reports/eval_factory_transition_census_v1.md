# Eval factory — four-transition structural census

PROTOCOL alternative when one transition is delta-thin: accumulate executable
deltas across adjacent version pairs.

## invariant

```text
Census is structural (claims + verifiers). Surfaces/audit/seal are separate.
Only v0.22.0→v0.23.0 is a frozen EvalEnvironment today.
```

## pins

| alias | tag | role |
|-------|-----|------|
| doc_m2 | v0.20.0 | matrix |
| doc_m1 | v0.21.0 | matrix |
| doc_0 | v0.22.0 | core |
| doc_8 | v0.23.0 | core / sealed |
| doc_9 | v0.24.0 | matrix |

## census

| transition | claims | deltas | ≥40 gate |
|------------|-------:|-------:|:--------:|
| v0.20→v0.21 | 1034 | 36 | no |
| v0.21→v0.22 | 1071 | **62** | yes |
| v0.22→v0.23 (sealed) | 1070 | **42** | yes |
| v0.23→v0.24 | 1091 | 37 | no |
| **sum** | — | **177** | — |

Two of four adjacent pairs clear the 40-delta gate alone; the sum is 177.

## what goes where

- local: `data/experiments/e1_vllm/eval_factory/{transition}/e1_eval_factory_census_v1/`
- B2: `datasets/experiments/e1_vllm/eval_factory/{transition}/e1_eval_factory_census_v1/`
- sealed instrument remains: `…/v0.22.0_to_v0.23.0/e1_eval_factory_v1/`
- JSON: `artifacts/reports/eval_factory_transition_census_v1.json`

## what can die

- local snapshot clones after B2 census upload
- census seed probes if a later freeze rebuilds surfaces

## what must survive

- snapshot aliases in `experiments/e1_vllm/snapshots.yaml`
- census manifests + claim banks on B2
- sealed v0.22→v0.23 EvalEnvironment

## command

```bash
python -m experiments.e1_vllm.eval_factory.cli build \
  --before doc_m2 --after doc_m1 --protocol e1_eval_factory_census_v1
python -m experiments.e1_vllm.eval_factory.cli build \
  --before doc_m1 --after doc_0 --protocol e1_eval_factory_census_v1
python -m experiments.e1_vllm.eval_factory.cli build \
  --before doc_8 --after doc_9 --protocol e1_eval_factory_census_v1
```
