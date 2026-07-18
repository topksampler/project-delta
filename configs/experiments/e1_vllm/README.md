# e1_vllm run configs

One YAML = one `./scripts/lab run`.

The experiment contract is
[`docs/experiments/e1_vllm.md`](../../../docs/experiments/e1_vllm.md). The
authoritative model ID is in
[`experiments/e1_vllm/snapshots.yaml`](../../../experiments/e1_vllm/snapshots.yaml).

## invariant

```text
One immutable YAML config → one globally unique run_id → one B2 run prefix.
```

## naming

```text
run_id: e1-vllm-{condition}-{model_slug}-{target}
```

## current configs

| file | role |
|---|---|
| `c0_base_eval_qwen35_08b_modal.yaml` | evaluate frozen base model |
| `c3_ft_qwen35_08b_modal.yaml` | train doc_0 LoRA adapter |
| `c3_ft_eval_qwen35_08b_modal.yaml` | reload and evaluate c3 adapter |

Retrieval configs do not exist because the index and retrieval path are not
implemented.

## what goes where

- condition definitions: experiment charter;
- source/model pins: `snapshots.yaml`;
- dispatch parameters: YAML files in this directory;
- immutable config snapshots: B2 `runs/{run_id}/config.yaml`.

## what can die

Superseded local config drafts and failed smoke output directories.

## what must survive

Every dispatched config snapshot and its immutable `run_id`; c3 eval must retain
the `adapter_run_id` of the c3 training run.

## command

```bash
./scripts/lab run --target modal \
  --config configs/experiments/e1_vllm/c0_base_eval_qwen35_08b_modal.yaml
```
