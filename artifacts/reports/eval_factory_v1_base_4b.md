# Eval factory v1 — base instrument check (4B)

Model: `Qwen/Qwen3.5-4B` closed-book
Eval: frozen `e1_eval_factory_v1` probes_eval (623)
Run: `e1-vllm-eval-factory-v1-base-qwen35-4b-modal` (~$0.15, A10G)

## invariant

```text
Report by drift_type / entity_type / probe_form.
Compare to 0.8B on the same frozen bank — size is not assumed to help.
eval_v3 is a different scoreboard.
```

## headline vs 0.8B

| measure | 0.8B | 4B |
|---------|-----:|---:|
| avg content score | 0.577 | 0.309 |
| exact accuracy | 0.512 | 0.148 |
| correct / wrong / stale / abstain_wrong | 319/139/117/48 | 92/318/119/94 |
| changed (n=8) | **0.000** | **0.000** |
| stable | 0.581 | 0.307 |
| added | 0.725 | 0.475 |
| removed | 0.750 | 0.833 |

## by entity_type (4B)

| family | n | score |
|--------|--:|------:|
| public_export | 21 | 0.476 |
| config_field | 223 | 0.359 |
| cli_flag | 255 | 0.298 |
| env_var | 124 | 0.212 |

## by probe_form (4B)

| form | score |
|------|------:|
| version_delta | 0.481 |
| versioned_existence | 0.215 |

## paths

- run: `runs/e1-vllm-eval-factory-v1-base-qwen35-4b-modal/`
- B2: `s3://lalith-ai-lab/runs/e1-vllm-eval-factory-v1-base-qwen35-4b-modal/`
- summary: `artifacts/reports/eval_factory_v1_base_4b_summary.json`
- 0.8B twin: `artifacts/reports/eval_factory_v1_base_08b.md`

## command

```bash
./scripts/lab run --target modal --gpu A10G \
  --config configs/experiments/e1_vllm/eval_factory_v1_base_qwen35_4b_modal.yaml
```
