# DELTA v2 stability base baseline — Modal v1

Date: 2026-07-30

Run: `delta-v2-d4-stability-base-qwen35-08b-modal-v1`

Condition: `d4_stability_base`

Result: valid zero-update reference

## Receipt

- Model: `Qwen/Qwen3.5-0.8B`
- Revision: `2fc06364715b967f1860aea9cf38778875588b17`
- Adapter: none
- Device: NVIDIA A10G
- Evaluation items: 60
- Deterministic outputs: 120, two byte-identical repeats per item
- Model updates: zero
- Elapsed compute: 599.4 seconds
- Estimated dispatch cost: $0.1901
- Receipt SHA-256:
  `e91cd432da6d524bfcbb05626386335336a540e14f4d99f017cc11ade50fec97`
- Metrics SHA-256:
  `e6382bca092a87ff5587f7aaf5f1d113f46f00ee9f07f288efa3725be3858158`

## Cell results

| Probe | Correct | Parseable | Interpretation |
|---|---:|---:|---|
| Choice | 0/15 | 0/15 | Untouched model does not follow the exact choice surface |
| Boolean true | 1/15 | 15/15 | Strong false-default bias; true stable claims are rarely accepted |
| Boolean false | 14/15 | 15/15 | Existing false-rejection behavior is strong |
| Recall | 0/15 | 13/15 | Advisory only; no exact-contract recalls |

No pooled score is reported.

## Decision

The baseline is suitable for the preregistered d4 control. It establishes a
clear stability target: the candidate must lift choice parseability and
stable-true accuracy while preserving stable-false accuracy at or above 0.90.

The result does not authorize full evaluation or promotion. It authorizes only
the separately frozen 60-step stability-replay LoRA training run and its
pre-gate evaluations.
