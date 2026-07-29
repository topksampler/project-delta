# Remember/Forget Experiment D — world-revision adapter attach vs merge

**Date:** 2026-07-29
**Arm:** D only (not ABCD compare)
**Model:** `Qwen/Qwen3.5-0.8B`
**FT:** reopened for this arc (`ft_reopened: true`); **no new train** — reuses A's DPO LoRA.

## Invariant

```text
Same lineage can mean shared backbone + swappable memory layer.
D1 = adapter attached (not merged) = operational world swap.
D2 = merge into base = can the memory become the blob?
Detach ⇒ old-world closed-book returns (rollback sanity).
Closed-book + forget is the scoreboard.
```

## Bet

A dedicated `world_v023` LoRA (here: A's remember/forget DPO adapter) can be attached as the active world; merging it should preserve closed-book behavior if the memory can live in a single durable blob.

## What goes where

| path | role |
|---|---|
| `configs/experiments/e1_vllm/remember_forget_d1_attach_*.yaml` | D1 attach CB primary + t2122 |
| `configs/experiments/e1_vllm/remember_forget_d2_merge_*.yaml` | D2 merge CB primary + t2122 (`merge_adapter: true`) |
| `runs/e1-vllm-remember-forget-a-dpo-qwen35-08b-modal/adapter/` | reused world adapter (no retrain) |
| `runs/e1-vllm-remember-forget-d{1,2}-*/summary.json` | forget-aware per-run summaries |
| `artifacts/reports/eval_factory_remember_forget_d*.json` | copied summaries + arm json |
| this file | arm verdict only |

## Run IDs

| stage | run_id |
|---|---|
| world adapter (reuse A) | `e1-vllm-remember-forget-a-dpo-qwen35-08b-modal` |
| D1 attach primary CB (22→23) | `e1-vllm-remember-forget-d1-attach-eval-qwen35-08b-modal` |
| D1 attach transfer CB (21→22) | `e1-vllm-remember-forget-d1-attach-t2122-eval-qwen35-08b-modal` |
| D2 merge primary CB (22→23) | `e1-vllm-remember-forget-d2-merge-eval-qwen35-08b-modal` |
| D2 merge transfer CB (21→22) | `e1-vllm-remember-forget-d2-merge-t2122-eval-qwen35-08b-modal` |
| modal batch wrapper | `e1-vllm-remember-forget-d-eval-batch-qwen35-08b-modal` |

## Configs

- `configs/experiments/e1_vllm/remember_forget_d1_attach_eval_qwen35_08b_modal.yaml`
- `configs/experiments/e1_vllm/remember_forget_d1_attach_t2122_eval_qwen35_08b_modal.yaml`
- `configs/experiments/e1_vllm/remember_forget_d2_merge_eval_qwen35_08b_modal.yaml`
- `configs/experiments/e1_vllm/remember_forget_d2_merge_t2122_eval_qwen35_08b_modal.yaml`

Harness: `model.merge_adapter` in `lab.eval_vllm_qa` (`false` → Peft attach; `true` → `merge_and_unload`). Platform: `eval_batch` in `infra/modal/app.py` (eval-only chain, one adapter pull).

## Primary closed-book (v0.22.0→v0.23.0)

Content means by drift + forget:

| condition | exact | changed | removed | added | stable | forget_rate | remember_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| base CB (frozen, detach) | 0.512 | 0.000 | 0.750 | 0.725 | 0.581 | — | — |
| **D1 attach** | **0.669** | **0.000** | **0.750** | **0.750** | **0.816** | **0.546** | **0.000** |
| **D2 merge** | **0.673** | **0.000** | **0.750** | **0.750** | **0.816** | **0.455** | **0.000** |
| A DPO CB (same adapter) | 0.669 | 0.000 | 0.750 | 0.750 | 0.816 | 0.546 | 0.000 |

D1 matches A bit-for-bit on this bank (same adapter, attach path). D2 merge ≈ D1 (exact +0.003; forget −0.091 on n=11).

Stable vs base: D1/D2 **gain** (~+0.23); no stable collapse.

## Transfer closed-book (v0.21.0→v0.22.0)

| condition | exact | changed | removed | added | stable | forget_rate | remember_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| base CB (frozen, detach) | 0.245 | 0.278 | 0.536 | 0.589 | 0.305 | — | — |
| **D1 attach** | **0.666** | **0.722** | **0.583** | **0.635** | **0.820** | **0.529** | **0.235** |
| **D2 merge** | **0.663** | **0.722** | **0.571** | **0.615** | **0.821** | **0.529** | **0.176** |

Transfer `changed` does **not** collapse (~0.72) for both D1 and D2.

## Detach sanity (rollback)

| bank | detach (= base, no adapter) | D1 attach | returns old world? |
|---|---|---|---|
| 22→23 | exact 0.512 / stable 0.581 | exact 0.669 / stable 0.816 | **yes** — detach drops to frozen base |
| 21→22 | exact 0.245 / stable 0.305 | exact 0.666 / stable 0.820 | **yes** |

Detach was not re-run; it is the frozen base CB scoreboard (no `adapter_run_id`). Attaching A's adapter moves stable/exact; removing it returns the base table. Merge has **no** detach path (weights baked).

## Pass bar (candidate interesting)

Plan bar: CB `changed ≥ 0.50`, `removed ≥ 0.50`, forget ≤ 0.25, stable loss ≤ 0.02, transfer changed ≉ 0.

| gate | D1 primary | D2 primary | D1/D2 transfer |
|---|---|---|---|
| changed ≥ 0.50 | **fail** (0.00) | **fail** (0.00) | pass (0.72) |
| removed ≥ 0.50 | pass | pass | pass |
| forget ≤ 0.25 | **fail** (0.55 / 0.45) | **fail** | **fail** (~0.53) |
| stable loss ≤ 0.02 | pass (gain) | pass (gain) | pass (gain) |
| transfer changed ≉ 0 | — | — | pass |

## Verdict

- **D1:** operational world swap works — attach changes CB vs base; detach restores base. Inherit A's primary failure (`changed=0`, forget high).
- **D2:** merge does **not** destroy attach behavior (primary/transfer ≈ D1). Plan falsifier “merge destroys stable/forget with no durable blob path” **did not hit** for this adapter — merge is a viable blob path **for the same (weak) remember/forget profile**.
- Arm does **not** clear the remember/forget pass bar; it answers the lineage/swap question, not the remember verb.

## What can die

- Local `/tmp/remember_forget_d_modal.log`
- Batch wrapper run_id (no artifacts of its own)

## What must survive

- Four D configs + `merge_adapter` harness flag
- Per-run `summary.json` with forget block (and copies under `artifacts/reports/`)
- Detach = base comparison table above
- Negative: attach/merge both fail primary `changed`/forget (adapter reuse from A)

## Command

```bash
set -a; source .env; set +a
.venv/bin/modal run infra/modal/app.py \
  --run-id e1-vllm-remember-forget-d-eval-batch-qwen35-08b-modal \
  --config configs/experiments/e1_vllm/remember_forget_d1_attach_eval_qwen35_08b_modal.yaml \
  --task eval_batch --gpu H100 \
  --eval-configs 'configs/experiments/e1_vllm/remember_forget_d1_attach_eval_qwen35_08b_modal.yaml,configs/experiments/e1_vllm/remember_forget_d1_attach_t2122_eval_qwen35_08b_modal.yaml,configs/experiments/e1_vllm/remember_forget_d2_merge_eval_qwen35_08b_modal.yaml,configs/experiments/e1_vllm/remember_forget_d2_merge_t2122_eval_qwen35_08b_modal.yaml'
```
