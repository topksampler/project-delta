# DECIDE demo — vLLM v0.22.0 → v0.26.0

**Date:** 2026-07-29
**Serves:** first end-to-end SENSE → DECIDE → VERIFY on a multi-release jump
**Bank:** factory seeds (unsealed) — teacher surfaces + human audit **not** done
**FT:** stopped (no new LoRA)

## Invariant

```text
Policy from eval_factory_decide_policy.md, applied without retuning.
Never promote on pooled exact.
Seed bank ≠ sealed bank — numbers are directional for DECIDE, not a freeze.
```

## Environment

| piece | value |
|---|---|
| before | `v0.22.0` (`doc_0`) |
| after | `v0.26.0` (`doc_11`, pinned `568afb3`) |
| claims | 1139 (85 added / 29 changed / 21 removed / 1004 stable) |
| eval probes | 756 seed (`probes_eval_seed` → `probes_eval`) |
| diff-hunk corpus | 14 612 chunks |

## SENSE (base closed-book)

Run: `e1-vllm-eval-factory-v022-v026-base-qwen35-08b-modal`

| exact | added | changed | removed | stable |
|---:|---:|---:|---:|---:|
| 0.118 | 0.588 | **0.069** | 0.667 | 0.209 |

**Failure mass:** `changed` ≈ 0. Ceiling check (oracle, debug only): `changed` **0.708** — so the hole is real and repairable in principle.

## DECIDE (policy walk)

```text
1. scoreboard split → changed mass is the fire
2. no adapter for this transition; FT paused
3. cheap deployable action for transition mass → diff-hunk BM25 on base
4. do not prescribe diet LoRA (paused) or oracle as product
```

Prescription: **base × diff-hunk**.

## VERIFY

| condition | exact | added | changed | removed | stable |
|---|---:|---:|---:|---:|---:|
| base closed-book | 0.118 | 0.588 | 0.069 | 0.667 | 0.209 |
| **base × diff-hunk** (prescribed) | 0.183 | 0.687 | **0.389** | 0.542 | 0.239 |
| base × evidence (ceiling) | 0.569 | 0.778 | 0.708 | 0.896 | 0.654 |

Path-hit (offline, before QA):

| | overall | added | changed | removed | stable |
|---|---:|---:|---:|---:|---:|
| diff_hunk BM25 | 0.348 | 0.788 | 0.722 | 0.615 | 0.271 |

## Verdict

1. On a **four-release** jump, SENSE correctly flags `changed` as broken (0.07).
2. DECIDE’s cheap action lifts `changed` **0.07 → 0.39** without FT — same specialty as adjacent-release banks.
3. Pooled exact stays low (0.18) — expected for a transition-only corpus; do not promote on it.
4. Gap to oracle `changed` (0.71) remains — ranking/router still open; diet reserved until FT reopen.
5. `removed` did not improve vs closed-book here (0.67→0.54) — policy should not claim universal win on every drift class.

## What can die

- Treating this seed bank as sealed / promotional
- Reopening factory LoRA just because the jump is bigger

## What must survive

- Pin `v0.26.0` + `doc_11`
- Claims + seed probes under `eval_factory/v0.22.0_to_v0.26.0/`
- This DECIDE trace (sense numbers → prescription → verify)
- Diff-hunk corpus `corpus_diff_hunk_022_026.jsonl`

## Command

```bash
# rebuild claims
.venv/bin/python -m experiments.e1_vllm.eval_factory.cli build --before doc_0 --after doc_11

# prescribed intervention
./scripts/lab run --target modal --gpu H100 \
  --config configs/experiments/e1_vllm/eval_factory_v022_v026_c1_diff_hunk_qwen35_08b_modal.yaml
```
