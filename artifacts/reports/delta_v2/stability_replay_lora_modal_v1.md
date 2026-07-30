# DELTA v2 d4 stability-replay LoRA — Modal v1

Date: 2026-07-30

Candidate: `d4_stability_replay_sft_control`

Training run: `delta-v2-d4-stability-replay-lora-qwen35-08b-modal-v1`

Decision: reject before full evaluation; retain as informative partial result

## Training receipt

- Fresh base: `Qwen/Qwen3.5-0.8B` at
  `2fc06364715b967f1860aea9cf38778875588b17`.
- LoRA: rank 8, alpha 16, dropout 0.05.
- Optimizer steps: 60.
- Training units: 240.
- Acquisition units: 180.
- Source-disjoint replay units: 60.
- Schedule SHA-256:
  `7af9409e76c5c6242c6c1e949dce597c71668ccef75b98a7b4c5d0c1685a577a`.
- Trainable parameters: 5,411,328; vision modules trainable: false.
- Acquisition dev loss: 2.3945 to 0.2308.
- Adapter SHA-256:
  `0260638f253090c6ab89e38e10ff025fbdc57ef9b736f5457faa26a339fed4c7`.

## Acquisition margin pre-gate

All preregistered acquisition-margin gates passed:

- Verified true: 60/63, 95.2%.
- Verified false: 58/63, 92.1%.
- Truth-conditioned pairs: 57/63, 90.5%.
- Annotation-swap false: 19/21, 90.5%.
- Configuration-default-swap false: 8/9, 88.9%.
- Environment-getter-swap false: 12/12, 100%.
- Full-card-swap false: 19/21, 90.5%.

## Held-out stability pre-gate

| Probe | Untouched base | d4 | Required |
|---|---:|---:|---:|
| Choice parseable | 0/15 | 15/15 | at least 90% |
| Choice correct | 0/15 | 8/15 | diagnostic |
| Boolean true | 1/15 | 11/15 | at least 80% |
| Boolean false | 14/15 | 11/15 | at least 90% |
| Recall | 0/15 | 1/15 | advisory |

The adapter produced a large improvement in instruction following and
stable-true recognition, but both Boolean accuracy gates failed at 73.3%.
Most importantly, false rejection regressed from 93.3% to 73.3%.

## Decision

The current full evaluation and sealed `verify_v2` remain blocked. d4 is not
promotable.

The next controlled candidate changes only replay share from 25% to 50% while
holding the fresh base, LoRA topology, objective, learning rate, 60 optimizer
steps, gradient accumulation, and seed fixed. This tests whether more
source-disjoint stability rehearsal preserves false rejection without losing
the strong acquisition-margin behavior.
