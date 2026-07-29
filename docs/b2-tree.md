# b2 tree

```text
GitHub = code
B2 = durable data
compute = disposable
```

## invariant

GitHub for code. B2 for data. Everything else is cache or scratch.

## what goes where

**Current dispatch reality:** flat `runs/{run_id}/` per job (see `docs/developer-workflow.md`).

```text
datasets/experiments/{experiment_id}/...
indexes/{experiment_id}/{source_revision}/...
runs/{run_id}/
  manifest.yaml
  config.yaml
  metrics.json
  samples.jsonl
  ledger.yaml
  adapter/                  # training runs
artifacts/reports/{experiment_id}/...
states/{state_id}/state.json
states/active.json          # only mutable DELTA object
```

DELTA promotion/rollback semantics are implemented locally in
`src/lab/delta/memory.py`; `src/lab/dispatch/b2.py` maps immutable states and
the active pointer to these B2 keys.

## lambda scratch

`/data/models`, `/data/datasets`, `/runs/<run-name>`.

Lambda filesystems = regional cache only. Never the only copy.

## sync

Write locally under `/runs/<run-name>/`, upload periodically to `s3://lalith-ai-lab/runs/<run-name>/`.

Don't wait until the end of a multi-hour job.

## if machine dies

Acceptable loss: compute time.

Unacceptable: only copy of checkpoint, dataset, or code change.

## what can die

Worker-local datasets, model caches, logs already uploaded, and failed smoke runs.

## what must survive

Frozen datasets, manifests, run configs, metrics, accepted adapters, promotion
decisions, immutable states, and active-state lineage.

## command

```bash
./scripts/lab status --run-id <run_id>
./scripts/delta status <reconcile_id>
```
