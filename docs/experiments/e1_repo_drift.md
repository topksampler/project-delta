# e1_repo_drift — archived

**Status:** abandoned before implementation.

This charter proposed measuring knowledge decay across the history of
`lalith-ai-lab`. It was superseded by [`e1_vllm`](./e1_vllm.md), which uses
pinned upstream vLLM documentation releases as a cleaner controlled change.

## invariant retained

```text
Train and stale retrieval see only the old snapshot.
Evaluation is sourced from later changes and held out from training.
Every condition receives the same questions.
```

## idea that survived

Corpus changes can reveal likely knowledge drift before users report behavioral
failures. That idea now belongs to the
[Project DELTA SENSE module](../project-delta.md#sense), where it is combined with
behavioral probes. This archived experiment is not a second DELTA roadmap.

## what goes where

The historical decision stays here. The surviving system requirement lives in
`docs/project-delta.md`; active experimental work lives under `e1_vllm`.

## what can die

- the unimplemented directory, config matrix, example run IDs, and B2 paths;
- the old claim that a time split on this lab should be the primary benchmark.

## what must survive

- the contamination invariant above;
- the distinction between corpus change and measured model drift.

## command

None. Do not dispatch this experiment.
