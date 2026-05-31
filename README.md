Personal AI lab.

Fine-tuning, evals, post-training, distributed training.

```text
Mac = cockpit | GitHub = code | B2 = data | Lambda/Modal = workers
```

Workers are disposable. Code and data are not.

Phase 1 — clone, `.env`, B2, run job, push artifacts, kill machine. Details in [phases](./docs/phases.md), [the split](./docs/the-split.md), [b2 tree](./docs/b2-tree.md).

```bash
bash scripts/check_storage.sh && bash scripts/test_storage_roundtrip.sh
```

Fix storage before Lambda.
