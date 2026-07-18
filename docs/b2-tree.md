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
```

Project DELTA will add `states/{state_id}/` and an active-state pointer only when
promotion/rollback semantics are implemented. Do not create an aspirational tree
and call it a registry.

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

Frozen datasets, manifests, run configs, metrics, accepted adapters, and—when
implemented—promotion decisions and active-state lineage.

## command

```bash
./scripts/lab status --run-id <run_id>
```
