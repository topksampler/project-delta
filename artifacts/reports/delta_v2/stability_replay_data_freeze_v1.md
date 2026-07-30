# DELTA v2 stability-replay data freeze v1

Date: 2026-07-30

Dataset: `delta-v2-knowledge-stability-replay-v1`

Candidate: `d4_stability_replay_sft_control`

Result: data and stability-development exam frozen; no model invoked

## Outcome

The next authorized DELTA step is complete. The source-disjoint rehearsal data
and held-out stability-development exam were built deterministically from the
frozen partitions in the system design.

- Replay training: 15 sources, 105 rows.
- Replay recall: 15 rows.
- Replay Boolean pairs: 45 pairs, comprising 45 positive and 45 negative rows.
- Stability development: 15 different sources, 60 probes.
- Stability probe mix: 15 choice, 15 true Boolean, 15 false Boolean, and 15
  recall probes.

The exact data, builder, tests, inputs, and audit summary are bound by
`knowledge_stability_replay_data_freeze.yaml`.

## Shortcut audit

The first pre-freeze build exposed a prompt-length shortcut. True and false
contracts were value-balanced, but their character lengths differed often
enough that a length-only classifier achieved 93.3% across the Boolean rows.
That build was rejected and never frozen.

The repaired builder pads the shorter member of every positive/negative pair
at the contract boundary. Padding is model-visible but label-neutral: each pair
now has identical total prompt length. The contract also contains a hard
length-only ceiling so this property fails closed.

Final frozen audit:

- Prompt-length difference within all 45 pairs: zero.
- Empirical prompt-length-only accuracy: 0.50.
- Empirical value-only accuracy:
  - annotation swap: 0.50;
  - family-value swap: 0.50;
  - full-card swap: 0.50.
- Maximum single corruption-family share: 1/3.
- Donor values with verified true support: 100%.
- Replay/acquisition source overlap: zero.
- Replay/current-evaluation source overlap: zero.
- Replay/stability-development source overlap: zero.
- Replay/sealed-verify-v2 source overlap: zero.
- Exact replay/acquisition prompt overlap: zero.
- Exact replay/current-evaluation prompt overlap: zero.
- Exact replay/stability-development prompt overlap: zero.

The `verify_v2` source partition is frozen, but its prompt wording remains
sealed until a candidate receipt exists. Its exact prompt-overlap audit is
therefore explicitly deferred until that post-receipt materialization step.

## Frozen output hashes

- `replay_train.jsonl`:
  `3cce6b51aefb3ee876599485c7726aebe3568b3f86cec021c66eda49b1f4d98f`
- `replay_pairs.jsonl`:
  `4c99f8d95ec5e365f50b897659683d1b227d00ef6c5c582bff3e4761f5c1acb3`
- `stability_dev.jsonl`:
  `f9c4c150aaf5db7edd2caba10801acd05b8e34cced443afdfa678aca3ed71b3a`
- `summary.json`:
  `be065cfbbeba6d8ad1e5286538b9aaa428f5ac793ecca7d5418a0982f36ab048`

## Execution boundary

Dataset construction completed with zero model invocations and zero optimizer
steps. Baseline inference, LoRA training, Modal jobs, B2 writes, and promotion
remain unauthorized by this freeze. QLoRA, full-weight fine-tuning, and
reinforcement learning remain blocked by their separate-contract requirement.

## Next controlled action

Freeze a zero-update base-baseline protocol for the 60 stability-development
probes, then run that baseline. The baseline establishes how well the untouched
model performs before the stability-replay candidate is trained; it does not
authorize LoRA training.
