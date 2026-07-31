# DELTA v2 stability LoRA d8-d19 stop decision

Date: 2026-07-31

Decision: stop the current small-base multisurface LoRA recipe lineage; do not
scale its steps, rank, model size, quantization, or update scope.

No candidate passed the frozen held-out stability pre-gate. The pinned base
model remains active. This is a negative result about the present data,
objective, and decoding contract; it is not evidence that all LoRA-based
knowledge adaptation is impossible.

## Comparable held-out results

Every row below uses the same 15-source, 60-probe stability structure. Counts
are out of 15. Recall is advisory; the Boolean and choice-format cells own the
gate.

| candidate | true correct | false correct | choice parseable | choice correct | recall correct | gate |
|---|---:|---:|---:|---:|---:|---|
| d8 claim-covered | 14 | 9 | 15 | 6 | 3 | fail |
| d9 multisurface | 13 | 12 | 15 | 6 | 3 | fail |
| d10 weighted ranking | 14 | 12 | 15 | 5 | 2 | fail |
| d11 hard negative | 14 | 13 | 0 | 0 | 3 | fail |
| d12 choice-preserved, rank 0.5 | 13 | 14 | 1 | 1 | 0 | fail |
| d13 choice-preserved, rank 0.25 | 13 | 12 | 15 | 9 | 1 | fail |
| d14 choice-preserved, rank 0.375 | 14 | 13 | 8 | 6 | 3 | fail |
| d15 doubled choice SFT | 13 | 15 | 0 | 0 | 0 | fail |
| d16 four-times choice SFT | 14 | 15 | 0 | 0 | 0 | fail |
| d17 sequence format ranking | 13 | 15 | 7 | 6 | 0 | fail |
| d18 four-times sequence format ranking | 13 | 15 | 1 | 1 | 0 | fail |
| d19 multi-negative sequence format ranking | 13 | 15 | 9 | 4 | 0 | fail |

The series does establish one real capability: d15-d19 preserve the paired
Boolean distinction well enough to pass the frozen true, false, and Boolean
parseability gates. They do not jointly preserve exact contract recall and
choice behavior.

## Parallel residual-format batch

The final batch changed only the format-ranking treatment from d17:

- d18 raised the single-negative sequence-ranking weight from 1 to 4;
- d19 kept weight 1 and covered three observed verbose-answer families.

D18 training:

- run:
  `delta-v2-d18-stability-choice-format-weight4-lora-qwen35-08b-modal-v1`
- adapter SHA-256:
  `7e9e614a83e1a20dfd8087617265094b73d326f97f85257bf9aaebb87659155c`
- receipt SHA-256:
  `4d50a701196ec15fa45d8f113a5b9feb08b0148ebdda0004c1eaa0518f9900b9`
- metrics SHA-256:
  `43bbff026c475a5bb80ab4f6b56afbf8ecae4bb4e286ce97a7a7e086f3dffe0c`

D18 evaluation:

- run: `delta-v2-d18-stability-choice-format-weight4-eval-modal-v1`
- receipt SHA-256:
  `e6d5b1d1340a28067dde362670073cfa2ffefe6d008b941c11382d4503486a9d`
- metrics SHA-256:
  `9ae2f6b7094e332ea405243b94a6ac9217b4609d7076a5ae83ea1411bee93db8`
- deterministic samples SHA-256:
  `e7649dca346686077b6cdb7a4d978fb67dd4359f3897f892303c0a817fab0cc5`
- compute: 687.7 seconds on NVIDIA A10; estimated dispatch cost
  USD 0.2173.

D19 training:

- run:
  `delta-v2-d19-stability-choice-format-multineg-lora-qwen35-08b-modal-v1`
- adapter SHA-256:
  `68f60d78dcf10c555de46b89ed25347327b989902e6256c4f396ed7c475ba951`
- receipt SHA-256:
  `4b26f0f55dc626914a7cab06b5316ee4b2a02dc1290c563ce8b3c5e1ba6da554`
- metrics SHA-256:
  `705837ea05a4ae3f841ed5d84e0bf2baa7347fd1d59c54e32b1a307bdcf6aadf`

D19 evaluation:

- run: `delta-v2-d19-stability-choice-format-multineg-eval-modal-v1`
- receipt SHA-256:
  `3ec84cff44e5e11f5114fd7a2b53421a927cb4f06b6d3f62a8ac707450250a63`
- metrics SHA-256:
  `4493a39a2d251eea369ec8836615841d3bf9a21053f68ca5e90b88b1dcce78e2`
- deterministic samples SHA-256:
  `1cfcf8a2800b4d4c4e22a320757cd07b2ae90567b5f0c8e4b4eea313098afb44`
- compute: 380.3 seconds on NVIDIA A10; estimated dispatch cost
  USD 0.1231.

Both evaluations made zero model updates and left promotion unauthorized.

## What failed

The failure is not “the adapter always says yes” or “the adapter always says
no.” The latest candidates retain the Boolean truth distinction. The failure
has two coupled parts:

1. exact contract generation falls to zero on d15-d19;
2. unconstrained choice generation does not reliably terminate after one
   letter.

D18 is especially diagnostic. Fourteen of its fifteen first-repeat choice
responses begin with a letter, but only one response contains the required
letter and nothing else. Most continue with analysis or reasoning. Raising the
whole-sequence ranking weight therefore did not teach the token-local stop
decision.

The implemented format loss compares the mean log probability of the complete
gold sequence with complete verbose negative sequences. Greedy decoding needs
a different local property: after the selected letter, the chat-termination
token must outrank newline and prose continuations. Whole-sequence preference
does not guarantee that property. D18's regression from 7/15 parseable at d17
to 1/15 at d18 is evidence against increasing this loss further.

D19's broader negative family improves parseability from 7/15 at d17 to 9/15,
but choice correctness falls from 6/15 to 4/15 and recall remains 0/15. That is
not a promotion-worthy improvement.

## Stop boundary

The following continuations are rejected as scientifically unjustified:

- more optimizer steps with the same objective;
- another interpolation of the same loss weights;
- higher LoRA rank or a larger base model used only to scale this recipe;
- QLoRA as a presumed quality fix;
- full-weight fine-tuning with this objective;
- verifier-grounded RL before its separate environment and safety contract.

QLoRA changes memory use, not the missing learning signal. Full-weight training
would spend more to optimize the same flawed target. RL remains outside this
decision because its reward, anti-hacking, reference-policy, retention, and
acceptance-separation contracts are not frozen.

## One justified diagnostic

Before designing another weight-changing candidate, freeze and run a
zero-update token-local diagnostic on representative checkpoints:

- d13: best exact choice result in this lineage;
- d18: strongest single-negative format pressure;
- d19: broadest residual-format coverage.

For each held-out choice prompt, measure separately:

1. the highest-probability first answer token among A, B, C, and D;
2. whether that forced first token is correct;
3. after the gold letter, whether the chat-termination token outranks every
   non-termination continuation;
4. the termination log-probability margin.

This diagnostic does not move a gate or authorize promotion. It decides the
next architecture:

- correct letter but negative termination margin: replace sequence ranking
  with a token-local termination objective and keep LoRA as a live mechanism;
- wrong letter and negative termination margin: redesign both knowledge and
  response objectives before any more training;
- weak results across d13, d18, and d19: close this 0.8B LoRA branch and move
  to a separately contracted capacity/mechanism study.
