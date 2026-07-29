# Remember/Forget Experiment B — claim-local sequential micro-edits

**Date:** 2026-07-29
**Arm:** B only (not ABCD compare)
**Model:** `Qwen/Qwen3.5-0.8B`
**FT:** reopened for this arc (`ft_reopened: true`)

## Invariant

```text
Sparse transitions want surgical patches, not one soup adapter.
Closed-book is the scoreboard.
Per-edit abort if canary stable drop > 0.02 vs baseline.
Remember = changed/added rise; forget = gold_before dies on changed/removed.
```

## Bet

Claim-local sequential DPO micro-edits + tiny stable replay compose a durable adapter without destroying stable.

## Caps

| knob | value |
|---|---|
| delta claims available | 17 |
| max edits (this run) | **10** |
| included | all 7 `changed` + all 3 `removed` |
| deferred | 7 `added` claim_ids |
| claim_copies | 8 (→ 24 delta pairs/claim) |
| stable_replay_n | 12 |
| canary_n | 24 (train-stable, disjoint from replay head) |
| micro_steps / edit | 20 |
| canary_budget | 0.02 absolute exact drop vs baseline |
| LoRA | r=8 α=16 dropout=0.05 β=0.1 lr=2e-4 bs=4 |

Cap reason: forget/remember-critical subset first; full 17 deferred for cost.

## What goes where

| path | role |
|---|---|
| `experiments/e1_vllm/eval_factory/build_claim_local_edits.py` | cluster + shard exporter |
| `src/lab/train_claim_local_edits.py` | sequential DPO + canary abort + interference curve |
| `data/experiments/e1_vllm/claim_local_edits_v1/` | manifest, canary, per-claim train JSONL |
| `configs/experiments/e1_vllm/remember_forget_b_*.yaml` | train + primary CB + t2122 CB |
| `runs/e1-vllm-remember-forget-b-claim-local-qwen35-08b-modal/` | composed adapter + `interference_curve.json` |
| `artifacts/reports/remember_forget_b_*_summary.json` | CB + forget tables |
| `artifacts/reports/remember_forget_b_interference_curve.json` | edit curve copy |
| this file | arm verdict only |

## Run IDs

| role | run_id |
|---|---|
| train (claim-local loop) | `e1-vllm-remember-forget-b-claim-local-qwen35-08b-modal` |
| primary CB eval (22→23) | `e1-vllm-remember-forget-b-cb-eval-qwen35-08b-modal` |
| transfer CB eval (21→22) | `e1-vllm-remember-forget-b-t2122-cb-eval-qwen35-08b-modal` |

## Interference curve

Baseline canary exact **0.0417** (near-floor on HF greedy before edits). All **10/10** edits accepted; **0** aborted.

| edit | drift | canary exact | drop vs baseline |
|---:|---|---:|---:|
| -1 | baseline | 0.0417 | 0.000 |
| 0–1 | changed | 0.625 | −0.583 |
| 2 | changed | 0.917 | −0.875 |
| 3–4 | changed | 0.625 | −0.583 |
| 5–6 | changed | 0.958 | −0.917 |
| 7–9 | removed | 0.625 | −0.583 |

Final canary **0.625** (stable replay lifted canary; no interference abort). Curve artifact: `artifacts/reports/remember_forget_b_interference_curve.json`.

## Primary closed-book (v0.22.0→v0.23.0)

Content means by drift (base from `eval_factory_v1_base_08b_summary.json`):

| condition | exact | changed | removed | added | stable | forget_rate | remember_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| base CB (frozen) | 0.512 | 0.000 | 0.750 | 0.725 | 0.581 | — | — |
| **B claim-local CB** | **0.504** | **0.000** | **0.500** | **0.425** | **0.569** | **0.182** | **0.000** |

Forget detail (n_scored=11): changed forget 0.00 / remember 0.00; removed forget 0.667 / remember 0.00.

Stable vs base: **−0.012** (within ≤0.02 loss budget).

## Transfer closed-book (v0.21.0→v0.22.0)

| condition | exact | changed | removed | added | stable | forget_rate | remember_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| base CB (frozen) | 0.245 | 0.278 | 0.536 | 0.589 | 0.305 | — | — |
| **B claim-local CB** | **0.632** | **0.722** | **0.476** | **0.406** | **0.722** | **0.471** | **0.000** |

Transfer `changed` does **not** collapse (~0.72).

## Pass bar

Plan bar: CB `changed ≥ 0.50`, `removed ≥ 0.50`, forget ≤ 0.25, stable loss ≤ 0.02, transfer changed ≉ 0.

| gate | primary B | transfer B |
|---|---|---|
| changed ≥ 0.50 | **fail** (0.00) | pass (0.72) |
| removed ≥ 0.50 | pass (0.50) | fail (0.48) |
| forget ≤ 0.25 | pass (0.18) | **fail** (0.47) |
| stable loss ≤ 0.02 | pass (−0.012) | pass (gain) |
| transfer changed ≉ 0 | — | pass |

**Arm verdict:** local edits composed without canary abort, and primary forget fell to 0.18 (changed forget 0.0), but **remember stayed dead** (`changed=0`, `remember_rate=0`). Falsifier shape: surgical patches did not lift global `changed` on the sealed 22→23 bank. Transfer still carries `changed≈0.72`.

## What can die

- Per-edit `runs/.../edits/*/adapter` snapshots after curve is archived
- Local `/tmp/remember_forget_b_modal.log`

## What must survive

- Manifest + canary under `data/experiments/e1_vllm/claim_local_edits_v1/`
- Composed adapter on B2 under train run_id
- Interference curve + primary/transfer summaries with forget blocks
- This arm note (negative on remember / primary changed)

## Command

```bash
set -a; source .env; set +a

.venv/bin/python experiments/e1_vllm/eval_factory/build_claim_local_edits.py \
  --probes data/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1/probes_train.jsonl \
  --out-dir data/experiments/e1_vllm/claim_local_edits_v1 \
  --max-claims 10 --claim-copies 8 --stable-replay-n 12 --canary-n 24 \
  --seed 20260729 --canary-budget 0.02

.venv/bin/modal run infra/modal/app.py \
  --run-id e1-vllm-remember-forget-b-claim-local-qwen35-08b-modal \
  --config configs/experiments/e1_vllm/remember_forget_b_claim_local_qwen35_08b_modal.yaml \
  --task train_then_eval --gpu H100 \
  --eval-configs 'configs/experiments/e1_vllm/remember_forget_b_cb_eval_qwen35_08b_modal.yaml,configs/experiments/e1_vllm/remember_forget_b_t2122_cb_eval_qwen35_08b_modal.yaml'
```
