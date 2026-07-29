# Eval factory v1 — base instrument check (0.8B)

Model: `Qwen/Qwen3.5-0.8B` closed-book
Eval: frozen `e1_eval_factory_v1` probes_eval (623)
Run: `e1-vllm-eval-factory-v1-base-qwen35-08b-modal` (~$0.04, A10G)

## invariant

```text
Report by drift_type / entity_type / probe_form.
Pooled accuracy alone is not the instrument readout.
eval_v3 is a different scoreboard.
```

## headline

| measure | value |
|---------|------:|
| n | 623 |
| avg content score | 0.577 |
| exact accuracy (score==1) | 0.512 |
| failure modes | correct 319 / wrong 139 / stale_version 117 / abstain_wrong 48 |

## by drift_type (mean content score)

| drift | n | score |
|-------|--:|------:|
| stable | 602 | 0.581 |
| added | 10 | 0.725 |
| removed | 3 | 0.750 |
| changed | 8 | **0.000** |

## by entity_type

| family | n | score |
|--------|--:|------:|
| config_field | 223 | 0.821 |
| env_var | 124 | 0.486 |
| cli_flag | 255 | 0.434 |
| public_export | 21 | 0.262 |

## by probe_form

| form | score |
|------|------:|
| version_delta | 0.760 |
| versioned_existence | 0.478 |

## what that suggests (for the learning beat)

Base 0.8B is not blank on the bank (~half exact), but **every changed-contract probe scored 0**. That is the version-delta signal the factory was built to expose. Stable config fields look easy; CLI/exports look hard. Do not collapse to one number.

## paths

- run: `runs/e1-vllm-eval-factory-v1-base-qwen35-08b-modal/`
- B2: `s3://lalith-ai-lab/runs/e1-vllm-eval-factory-v1-base-qwen35-08b-modal/`
- summary JSON: `artifacts/reports/eval_factory_v1_base_08b_summary.json`

## command

```bash
./scripts/lab run --target modal --gpu A10G \
  --config configs/experiments/e1_vllm/eval_factory_v1_base_qwen35_08b_modal.yaml

python experiments/e1_vllm/eval_factory/summarize_run.py \
  --samples runs/e1-vllm-eval-factory-v1-base-qwen35-08b-modal/samples.jsonl \
  --probes data/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1/probes_eval.jsonl \
  --out artifacts/reports/eval_factory_v1_base_08b_summary.json
```
