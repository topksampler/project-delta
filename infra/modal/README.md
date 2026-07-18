# Modal execution plane

`app.py` defines the Modal image and remote train/eval functions. The cockpit
invokes them through `src/lab/dispatch/modal_driver.py`.

## invariant

```text
The Modal worker is disposable.
Secrets come from the `lalith-lab` Modal secret.
Inputs come from git/B2; run outputs return to B2.
```

## what goes where

| location | responsibility |
|---|---|
| `infra/modal/app.py` | image, mounted code/config, remote functions, B2 worker I/O |
| `src/lab/dispatch/modal_driver.py` | local `modal run` invocation |
| `configs/runtime/modal.yaml` | cockpit-side defaults; currently not consumed by `app.py` |

The hardcoded Modal GPU/secret settings and experiment-specific eval routing are
known follow-up work, not part of the compute-directory move.

## what can die

Modal containers, image build intermediates, model caches, and worker-local files
after upload.

## what must survive

The image recipe, secret-name contract, run manifest/config, worker receipts, and
outputs stored under B2 `runs/{run_id}/`.

## command

```bash
modal setup
modal secret create lalith-lab \
  S3_BUCKET=... S3_ENDPOINT_URL=... \
  AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
  AWS_REGION=... AWS_DEFAULT_REGION=...
```

Dispatch runs through `./scripts/lab run --target modal --config <yaml>`.
