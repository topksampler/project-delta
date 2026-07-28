# Adapter roles — profile wheel vs intervention eval

## frozen checkpoint (2026-07-28)

```text
FT paused. Promote seq v7 as the intervention LoRA checkpoint.
v5 remains the SENSE/profile specialist readout.
v6 is a negative result (eval_v3 up, honesty down) — keep for contrast, do not deploy.
Full-weight FT deferred.
```

## what goes where

| artifact | role |
|----------|------|
| `train_v0.22.0_v5` + `e1-vllm-c3-ft-mill-v5-*` | profile manifold (meanings / honesty) |
| dense / paraphrase profile runs | SENSE readout for that manifold |
| `eval_v3` | hand intervention comparison (A/B/C/D classes) |
| `train_v0.22.0_v6` (blend) | intervention attempt: mill v4 + profile v5 + eval-hole paraphrases |
| `train_v0.22.0_v7` + seq from v5 | preferred intervention path: keep v5 honesty, light eval-hole tune |

## what can die

- local copies of discarded failed adapters after B2 durability
- expectation that profile-only LoRA moves `eval_v3`

## what must survive

- this role split
- denylist: never train on exact `eval_v3` questions (jaccard gate)
- separate reports for profile metrics vs `eval_v3` accuracy

## command

```bash
# profile readout (SENSE)
./scripts/lab run --target modal --gpu H100 \
  --config configs/experiments/e1_vllm/profile_dense_v1_para_ft_v5_modal.yaml

# intervention readout (eval_v3) — frozen checkpoint is seq v7, not v6
./scripts/lab run --target modal --gpu H100 \
  --config configs/experiments/e1_vllm/c3_ft_seq_v7_eval_v3_qwen35_08b_modal.yaml
```
