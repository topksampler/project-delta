# Project DELTA

CI/CD for model knowledge: when a model's source of truth changes, detect the
drift, build grounded evidence, choose the least-cost intervention, and promote or
rollback the resulting model state.

```text
SENSE → BUILD → DECIDE → VERIFY
Mac = cockpit | GitHub = code | B2 = data | Lambda/Modal = workers
```

Workers are disposable. Code and data are not.

## what goes where

Start with:

- [Project DELTA system contract](docs/project-delta.md)
- [DELTA roadmap and current status](docs/delta-roadmap.md)
- [Developer workflow](docs/developer-workflow.md)
- [Experiments registry](docs/experiments/README.md)

The active vertical slice is [`e1_vllm`](docs/experiments/e1_vllm.md): version
fidelity and failure modes across vLLM documentation revisions.

## command

```bash
./scripts/lab run --target modal \
  --config configs/experiments/e1_vllm/c0_base_eval_qwen35_08b_modal.yaml
```

## invariant

```text
World changes produce versioned evidence.
Interventions are promoted only after recovery and regression checks.
The cheapest intervention satisfying the quality constraint wins.
```

## what can die

Workers, local caches, failed candidates, and rejected interventions.

## what must survive

Pinned source revisions, dataset manifests, run evidence, promotion decisions, and
model-state lineage.
