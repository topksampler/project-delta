# archived lab capability phases

This was the original learning sequence for the general-purpose lab. It is not the
Project DELTA delivery roadmap.

The active roadmap is [delta-roadmap.md](./delta-roadmap.md).

## invariant retained from this document

```text
Reproducible storage and single-node train → reload → eval must work before
distributed training or post-training is credible.
```

## status

- reproducible cockpit, B2 storage, and disposable workers: implemented;
- single-node LoRA train, upload, reload, and eval: implemented;
- trustworthy DELTA eval and comparison: in progress;
- post-training and distributed optimization: deferred until the verifier is trusted.

## what goes where

The active sequence lives in `docs/delta-roadmap.md`. Platform prerequisites live
in `docs/developer-workflow.md`. This file remains only as a historical note.

## what can die

The old numeric phase labels and capability checklist.

## what must survive

The dependency ordering: storage → reproducible training → trustworthy evaluation
→ post-training → distributed optimization.

## command

See [developer-workflow.md](./developer-workflow.md) for platform commands and
[delta-roadmap.md](./delta-roadmap.md) for the active DELTA milestone.
