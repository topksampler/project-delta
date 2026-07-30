# DELTA v2 stability-replay system design v1

Date: 2026-07-30

Candidate: `d4_stability_replay_sft_control`

Loop position: `DECIDE` complete; next controlled action is data construction

Execution state: design frozen, training blocked

## Decision

The next LoRA candidate keeps the Stage 1 generative training objective and
adds source-disjoint stability rehearsal. It tests one narrow hypothesis:
whether interleaving old, unchanged knowledge prevents the false-statement
regression without sacrificing acquisition of the new revision.

This is not evidence that LoRA is generally useful or useless. Stage 1 showed
that this LoRA configuration could learn some changed facts but did not retain
the required rejection behavior. Stage 2 showed that the paired-ranking
objective was not a repair: its training-surface margins improved while its
full-evaluation behavior became worse. The justified next move is therefore a
controlled data intervention, not more unconstrained training.

## Plain-language system

Imagine teaching someone a software update. The first attempt studied only the
new material and became less reliable on some old facts. The next candidate
will use three of every four study slots for the update and one slot to rehearse
an unrelated fact that is still true. It takes the same 60 learning steps as
Stage 1, so the experiment changes the lesson mixture rather than simply giving
the model more training.

Three separate stability groups prevent the model from studying its exam:

1. `replay_train`: 15 stable facts used as rehearsal during training.
2. `stability_dev`: 15 different stable facts used for an early forgetting
   check.
3. `verify_v2`: 21 more stable facts reserved for the final promotion check.

All three groups exclude the 42 acquisition and retention sources in the
current full evaluation. The groups do not share sources with one another.
Within each group, configuration facts use different file paths.

## Frozen source selection

- Stable facts in the atomic-fact input: 1,031.
- Stable facts in the two supported families: 732.
- Eligible after excluding current evaluation sources: 711.
- Deterministic partition hash:
  `c87130408a7c0524bc4267df9ead6574f57c5deb1ff8a97bbeaa177a0b887608`.
- `replay_train`: 6 configuration facts and 9 environment-variable facts.
- `stability_dev`: 6 configuration facts and 9 environment-variable facts.
- `verify_v2`: 9 configuration facts and 12 environment-variable facts.

Selection is deterministic from the bound input bytes, a partition-specific
salt, and each source ID. Any source-byte, source-membership, or partition
change makes validation fail.

## Frozen training comparison

The candidate starts from the same fresh base and preserves the Stage 1
assistant-only generative SFT settings:

- LoRA rank 8, alpha 16, dropout 0.05.
- Learning rate 0.0001.
- 60 optimizer steps.
- Four training units per optimizer step.
- Seed 20260730.

The 240 total units are:

- 180 acquisition units.
- 60 replay units.
- 180 Boolean-pair units.
- 60 recall units.

For 45 optimizer steps, the four units are two acquisition Boolean pairs, one
acquisition recall, and one replay Boolean pair. For 15 optimizer steps, they
are three acquisition Boolean pairs and one replay recall. Step types and each
input stream are shuffled deterministically.

This deliberately keeps the total training budget fixed. The candidate tests
the effect of rehearsal, not the combined effect of rehearsal plus extra
optimizer steps.

## Required gates

Before training:

- audit the source partitions;
- audit false-example corruptions for shortcuts;
- prove zero exact prompt overlap with all evaluation groups;
- record a zero-update base-model baseline on `stability_dev`.

Before running the existing full evaluation:

- acquisition true, false, truth-conditioned, and each corruption-family
  margin accuracy must each be at least 0.80;
- held-out stability choice outputs must be at least 0.90 parseable;
- held-out stability Boolean outputs must be fully parseable;
- held-out stable-true accuracy must be at least 0.80;
- held-out stable-false accuracy must be at least 0.90.

Recall is reported only as an advisory diagnostic. Scores from distinct probe
types are never pooled into one headline number.

The existing DELTA v2 full evaluation and its gates remain unchanged. Even if
the candidate passes them, it cannot be promoted until it also passes the
separate `verify_v2` stability check. The wording for that check is materialized
only after the candidate receipt exists.

## Execution boundary

Only deterministic source partitioning is currently authorized. Replay-data
construction, baseline inference, LoRA training, Modal jobs, B2 writes, and
promotion remain blocked until their inputs are built, audited, and frozen.
QLoRA, full-weight fine-tuning, and reinforcement learning remain separately
blocked and require their own contracts and safety/verification gates.

## Next controlled action

Build the 105-row replay dataset from the 15 frozen `replay_train` sources:
one recall row and three verified true/false Boolean pairs per source. Then run
the contract validator and human-readable shortcut/overlap audit. No model job
is needed for that step.
