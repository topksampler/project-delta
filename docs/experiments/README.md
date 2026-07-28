# experiments registry

Each experiment is a **study** (hypothesis + factorial design). The **platform** (`./scripts/lab`, `src/lab/dispatch/`) runs individual jobs; experiments orchestrate many jobs and compare them.

Experiments are vertical slices of [Project DELTA](../project-delta.md). They
produce evidence for the closed loop; no individual experiment redefines it.

## invariant

```text
one experiment  →  many run_ids  →  one comparison report
platform code   →  shared across all experiments
experiment code →  lives under experiments/{experiment_id}/ only
program vision  →  docs/project-delta.md
```

## registry

| ID | status | charter | question |
|----|--------|---------|----------|
| `e1_vllm` | **A+C done; B partial (DriftEvent v1)** | [thesis](./e1_vllm_thesis.md) · [charter](./e1_vllm.md) | First slice: version fidelity and failure modes on a vLLM doc delta |
| `e1_repo_drift` | archived | [e1_repo_drift.md](./e1_repo_drift.md) | Early time-split design; surviving drift idea moved into DELTA SENSE |

Add a row when you start `e2_*`. Do not delete old rows — mark `done` or `abandoned`.

## what goes where

| artifact | location |
|----------|----------|
| charter (hypothesis, matrix, metrics) | `docs/experiments/{id}.md` |
| harness code | `experiments/{id}/` |
| run configs (one YAML per run) | `configs/experiments/{id}/` |
| frozen eval data | B2 `datasets/experiments/{id}/` |
| indexes (if RAG) | B2 `indexes/{id}/{sha}/` |
| per-run outputs | B2 `runs/{run_id}/` |
| cross-run report | `artifacts/reports/{id}.md` |

## run matrix template

```text
experiment_id: e1_vllm
conditions:    c0, c1, c2, c3, c4, c6
models:        qwen35-08b, ...      (parallel = separate run_id each)
targets:       modal, lambda
snapshots:     doc_0 (v0.22.0), doc_8 (v0.23.0)   # vLLM docs, not lab git
```

You dispatch each cell manually or with your own sweep shell loop. Platform has no sweep command yet.

## naming

See [developer-workflow.md](../developer-workflow.md#naming-multi-experiment-multi-model-multi-condition).

## what can die

- failed smoke `run_id` directories on Mac
- candidate eval JSONL before a version is frozen

## what must survive

- charter + harness in git
- frozen eval and dataset manifest on B2 with SHA documented
- every completed run's `config.yaml` + `metrics.json` on B2

## command

```bash
./scripts/lab run --target modal \
  --config configs/experiments/e1_vllm/c0_base_eval_qwen35_08b_modal.yaml
```
