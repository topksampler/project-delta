# compute infrastructure

Provider-specific worker execution lives under `infra/`. The Mac-side control
plane remains in `src/lab/dispatch/`.

## invariant

```text
scripts/lab coordinates.
src/lab/dispatch orchestrates providers.
infra/{provider} executes on disposable workers.
GitHub and B2 preserve reproducible state.
```

## what goes where

| path | responsibility |
|---|---|
| `infra/modal/` | Modal image, secrets binding, and remote function entrypoints |
| `infra/lambda/` | Lambda VM environment bootstrap and one-shot job entrypoint |
| `src/lab/dispatch/` | provider API calls, SSH/rsync, manifests, timing, and costs |
| `configs/runtime/` | cockpit-side provider defaults and pricing |
| `src/lab/*.py` | train/eval modules shared by providers |

Experiment hypotheses and dataset logic do not belong here.

## what can die

Workers, virtual environments, downloaded model caches, and worker-local input
copies after outputs are durable.

## what must survive

Worker recipes in git; run manifests, configs, receipts, metrics, samples, and
accepted adapters in B2.

## command

The public interface does not expose provider file paths:

```bash
./scripts/lab run --target modal \
  --config configs/experiments/e1_vllm/c0_base_eval_qwen35_08b_modal.yaml
```
