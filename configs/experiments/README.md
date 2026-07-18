# configs/experiments/

One YAML file = one `./scripts/lab run` invocation.

## invariant

```text
run_id in YAML must be globally unique.
experiment_id + condition_id are conventions (documented in charter) — platform only requires run_id today.
```

## what goes where

```text
configs/experiments/
  e1_vllm/
    c0_base_eval_qwen35_08b_modal.yaml
    c3_ft_qwen35_08b_modal.yaml
    c3_ft_eval_qwen35_08b_modal.yaml
```

Legacy smoke configs remain in `configs/evals/` and `configs/sft/` — do not delete; new work goes here.

Planned conditions do not get example configs here. Their required fields belong
in the experiment charter until the corresponding harness exists.

## what can die

Undispatched config drafts.

## what must survive

The exact config snapshot for every dispatched `run_id`, stored with its run
artifacts.

## command

```bash
./scripts/lab run --target modal \
  --config configs/experiments/e1_vllm/c0_base_eval_qwen35_08b_modal.yaml
```
