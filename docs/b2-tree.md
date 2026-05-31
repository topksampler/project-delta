# b2 tree

```text
GitHub = code
B2 = durable data
compute = disposable
```

## source of truth

GitHub for code. B2 for data. Everything else is cache or scratch.

## layout

```text
lalith-ai-lab/
  datasets/
    raw/
    processed/
    evals/
  models/
    base/
    adapters/
    merged/
  checkpoints/
    sft/
    dpo/
    rl/
  runs/
    sft/
    dpo/
    rl/
    evals/
  artifacts/
    logs/
    reports/
    samples/
```

## lambda scratch

`/data/models`, `/data/datasets`, `/runs/<run-name>`.

Lambda filesystems = regional cache only. Never the only copy.

## sync

Write locally under `/runs/<run-name>/`, upload periodically to `s3://lalith-ai-lab/runs/<run-name>/`.

Don't wait until the end of a multi-hour job.

## if machine dies

Acceptable loss: compute time.

Unacceptable: only copy of checkpoint, dataset, or code change.
