# docs

## what goes where

- [Project DELTA](./project-delta.md) — system invariant and module contracts
- [DELTA roadmap](./delta-roadmap.md) — current status and phase exit artifacts
- [developer workflow](./developer-workflow.md) — platform vs experiment, paths, naming
- [experiments registry](./experiments/README.md) — studies and vertical slices

## platform

- [the split](./the-split.md)
- [b2 tree](./b2-tree.md)

## active vertical slice

- [e1_vllm charter](./experiments/e1_vllm.md) — conditions and operational state
- [e1_vllm thesis](./experiments/e1_vllm_thesis.md) — version-fidelity question and metrics
- [e1_vllm data](./experiments/e1_vllm_data.md) — corpus facts and eval validation

## invariant

```text
System vision       → project-delta.md
Delivery status     → delta-roadmap.md
Platform mechanics  → developer-workflow.md + b2-tree.md
Experiment claims   → docs/experiments/{id}*.md
```

Leaf documents link to these owners. They do not redefine the DELTA loop, current
phase, model ID, or storage contract.

## what can die

Duplicated summaries and stale roadmaps after their surviving facts move to the
canonical owner.

## what must survive

System contracts, current status, platform mechanics, experiment evidence, and
working links between their owning documents.

## command

```bash
python experiments/e1_vllm/inspect_data.py validate-eval eval_v2.jsonl
```
