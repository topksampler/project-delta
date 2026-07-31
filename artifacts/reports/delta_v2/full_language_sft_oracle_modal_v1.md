# DELTA v2 full-language SFT feasibility oracle — Modal v1

Date: 2026-07-31

Status: `implemented`; training completed, held-out stability VERIFY failed,
training-surface diagnostic failed, promotion unauthorized.

Loop position: `DECIDE`. The fresh-base full-language candidate remains an
isolated `CandidateState`; the pinned base remains active. Acquisition-margin
and full frozen evaluation are blocked by the failed stability pre-gate.

## Question

This experiment deliberately removed adapter capacity as the first
uncertainty. It asked whether plain supervised fine-tuning of every language
weight in the pinned 0.8B model could learn the acquisition and replay
surfaces while preserving exact false-claim rejection. It did not test QLoRA,
full-model vision updates, preference training, or reinforcement learning.

## Full-language training

- run: `delta-v2-f1-full-language-sft-qwen35-08b-modal-v1`
- successful launch commit: `154dd00`
- base: `Qwen/Qwen3.5-0.8B` at revision
  `2fc06364715b967f1860aea9cf38778875588b17`
- objective: plain assistant-token cross-entropy
- optimizer steps: 60
- schedule units: 240
- model forwards: 420
- acquisition/replay split: 120/120 units
- Boolean-pair/recall split: 180/60 units
- trainable parameters: 752,393,024 of 852,985,920 (88.21%)
- frozen parameters: 100,592,896; vision remained frozen
- initial/final recall-dev loss: 2.394504 / 0.292759
- compute: 217.61 seconds on NVIDIA H100
- estimated dispatch cost: USD 0.365
- checkpoint SHA-256:
  `0037e2e25837c0ea7e36766f68c487b80853b6051c0ae382b399a59701e2171a`
- metrics SHA-256:
  `f53ed0cfe52aae73411bbb2b33a9789bc936f413f1e872c4e07a84d9381cfeaa`
- receipt SHA-256:
  `d2de9cf7eeafac0ba54fad077dcaa8a8cb16422e4d98450d447bdeea17f263c1`

Training was numerically healthy and the full checkpoint was preserved in B2.
Loss reduction alone did not authorize promotion.

## Held-out stability VERIFY

- run: `delta-v2-f1-full-language-stability-eval-modal-v1`
- preregistration commit: `398781c`
- evaluation: 60 same-claim, new-wording probes; 120 deterministic outputs
- model updates: 0
- compute: 109.03 seconds on NVIDIA A10
- estimated dispatch cost: USD 0.048
- metrics SHA-256:
  `cfb54dbb3f7010c5ca6ab4be55142a27deef31e8df2324a6e7a64a0512f25d69`
- samples SHA-256:
  `212e5fefeeabba78398306447ec6b92639ee5c5c836212da275ce752a3370f13`

| cell | pinned base | full-language candidate | required | result |
|---|---:|---:|---:|---|
| choice correct | 0/15 | 11/15 | diagnostic | improved |
| choice parseable | 0/15 | 15/15 | at least 14/15 | pass |
| Boolean true correct | 1/15 | 13/15 | at least 12/15 | pass |
| Boolean false correct | 14/15 | 10/15 | at least 14/15 | **fail** |
| recall exact, advisory | 0/15 | 0/15 | advisory | no gain |

All required gates were conjunctive. The false-rejection failure blocks the
candidate even though choice and true-claim behavior improved substantially.
Five false prompts received a deterministic `yes`; raw-answer inspection
confirmed a model error rather than a parser or scorer defect.

## Exact training-surface diagnostic

The failed VERIFY left two live explanations: the model might not have fit its
own schedule, or it might have fit the schedule but failed to transfer across
wording. The frozen zero-update diagnostic replayed every distinct row
referenced by the exact training schedule.

- run: `delta-v2-f2-full-language-train-surface-diagnostic-modal-v1`
- preregistration commit: `e379c1d`
- evidence-input wiring commit: `415cfe8`
- rows: 282 across 36 source claims
- deterministic outputs: 564
- model updates: 0
- compute: 275.70 seconds on NVIDIA A10G
- estimated successful-run cost: USD 0.1015
- metrics SHA-256:
  `b3aa9ba397115d777436b5fa12a987e9bc1a21f9a578c4514a0a738e459e1424`
- samples SHA-256:
  `3a8d378ef4287926f105f43ab3b0db741cb64f5abced36bcf96415b4f9cdb4e7`
- receipt SHA-256:
  `f4fa6ef8bce18aa57d3becc980391f5e97c0cc55783cf837a229ff5cdbc5f6d7`

| exact training cell | observed | minimum | result |
|---|---:|---:|---|
| exact recall | 20/36, 55.56% | 80% | **fail** |
| verified true | 110/123, 89.43% | 95% | **fail** |
| verified false | 119/123, 96.75% | 95% | pass |

The first attempt stopped in preflight before model loading because six frozen
d20 evidence files were not declared as worker inputs. It performed zero model
invocations, cost about USD 0.013, and produced no diagnostic result. The retry
declared the exact hashes and completed successfully.

## Interpretation

Full fine-tuning is an update mechanism, not a correctness guarantee. This run
shows that the model has enough update capacity to acquire real choice and
truth signal: relative to the pinned base, held-out choice moved from 0/15 to
11/15 and held-out true claims from 1/15 to 13/15. The result therefore rejects
the claim that persistent weight adaptation is impossible at this scale.

It does not establish a complete solution for two independent reasons:

1. the candidate still misses its own exact recall and true-answer training
   surfaces, so the 60-step schedule did not fully fit the supervised task;
2. false rejection is 119/123 on exact training wording but only 10/15 on
   held-out wording, exposing a large transfer/calibration failure that more
   repetition of the same prompts cannot resolve by itself.

The correct conclusion is neither “LoRA is useless” nor “full fine-tuning
works.” Capacity was not the sole blocker. The present small, narrow prompt
bank and plain generative objective do not yield reliable transferable
knowledge behavior, even when 88% of model parameters can change.

## Decision and next controlled candidate

Reject this checkpoint and retain the pinned base. Do not run acquisition or
promotion gates for it. Do not merely extend the identical 60-step schedule.

The next justified feasibility candidate remains full-language SFT from the
same fresh base, but must change the supervised coverage before adding adapter
or RL complexity:

1. train every claim on multiple independently worded recall, choice, true,
   and false surfaces;
2. reserve disjoint wording templates and corruption families for VERIFY;
3. keep true and false examples exactly balanced per claim;
4. freeze an early-stop rule that first requires exact training-surface fit;
5. preserve the same held-out stability false-rejection gate;
6. continue to block QLoRA, LoRA, full-model vision updates, and RL.

This candidate must have a separate immutable data manifest, training
contract, and VERIFY gate before compute. Its purpose is to test supervised
coverage and transfer, not to rescue this rejected checkpoint.
