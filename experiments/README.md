# experiments/

**You write code here.** Platform code stays in `src/lab/dispatch/`.

Experiments are vertical slices of [Project DELTA](../docs/project-delta.md).
The system vision and roadmap do not live in harness READMEs.

## invariant

```text
experiments/{id}/  = harness for one study
configs/experiments/{id}/  = one YAML per dispatched run
docs/experiments/{id}.md  = charter (hypothesis, matrix, metrics)
```

## what goes where

```text
experiments/
  e1_vllm/           # active — see docs/experiments/e1_vllm.md
  e2_.../            # your next study
```

## active

[e1_vllm](../docs/experiments/e1_vllm.md) — measure version fidelity across
vLLM v0.22.0 and v0.23.0 using the model pinned in `e1_vllm/snapshots.yaml`.

Current status: corpora and `eval_v2` exist; grounded dataset generation and
trustworthy failure scoring remain DELTA Phase A work.

## what can die

Local corpus clones, candidate datasets, and failed run caches.

## what must survive

Harness code, recipes, frozen fixture versions, and manifests binding outputs to
source revisions.

## command

```bash
python experiments/e1_vllm/inspect_data.py validate-eval eval_v2.jsonl
```
