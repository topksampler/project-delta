# Lambda execution plane

These scripts run inside a disposable Lambda GPU VM after the cockpit syncs the
repository over rsync.

## invariant

```text
The cockpit selects and syncs code.
run_job.sh executes exactly one run.
bootstrap_env.sh fails GPU jobs when CUDA is unavailable.
All durable receipts and outputs return to B2.
```

## what goes where

| file | responsibility |
|---|---|
| `run_job.sh` | install storage client, pull inputs, invoke the train/eval module, upload outputs |
| `bootstrap_env.sh` | probe hardware, select an explicit torch profile, build `.venv`, write receipts |
| `src/lab/dispatch/lambda_driver.py` | Lambda API, SSH, rsync, launch/terminate orchestration |
| `configs/runtime/lambda.yaml` | cockpit-side Lambda defaults |

## what can die

The VM, `.venv`, model cache, source checkout, and local run directory after
successful upload.

## what must survive

`hardware.json`, `env.json`, run config, manifest, metrics, samples, and training
artifacts under B2 `runs/{run_id}/`.

## command

```bash
./scripts/lab env build --target lambda \
  --instance-ip <ip> --run-id env-smoke
```

Do not execute `run_job.sh` manually from the Mac; the Lambda driver injects its
required environment.
