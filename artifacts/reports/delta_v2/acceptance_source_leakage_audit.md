# delta_v2 acceptance source-leakage audit

Status: corrected before any target-model run.

## What failed

The first acceptance EvalItem build was internally deterministic and
truth-grounded, but it checked source identities only inside the acceptance
transition. Comparing those items with the already assigned development source
identities found:

- 60 acceptance EvalItems;
- 35 source identities also assigned to development `train`;
- 38 source identities also present anywhere in development.

Different revision wording does not make the underlying semantic fact unseen.
The first 60-item bank is therefore not a valid acceptance environment.

## Corrective amendment

`acceptance_eval_item_contract_v2.yaml` excludes every source identity present
in `development_source_splits.jsonl` before choosing changed facts or stable
controls. Excluded records are retained in
`acceptance_eval_exclusions_v2.jsonl`; the extracted facts and their gold
statuses are not edited.

The amendment was implemented after acceptance was unsealed, but before any
target model, training, fine-tuning, Modal, Lambda, B2, or evaluation job ran.
This timing is a protocol limitation and is bound into the v2 contracts.

## Corrected result

- acceptance facts inspected: 1,061;
- excluded development identities: 995;
- unseen candidate facts: 66;
- unseen nonstable facts: 21, all `added`;
- unseen stable candidates: 45;
- selected acceptance EvalItems: 42;
- selected labels: 21 `added`, 21 `stable`;
- cross-transition source overlap: 0;
- exclusion audit SHA-256:
  `806c6f3adc6107cb641322412db5f3e4f4d51129eaec92364e06c3547f2f343a`;
- corrected EvalItems SHA-256:
  `38d3324a0cde106349394f9d6ebf55954aace3d3efc38ecd4b8f3379a73e2a08`;
- corrected audit packet SHA-256:
  `461cb4d11371d2c28333c47ad6ae1d57b7cac04ac1e88b9dfaba3927d329bcb8`.

A clean rebuild reproduced the EvalItems, exclusions, audit packet, grounded
judge evidence, judge verdicts, and both summaries byte-for-byte.

## Grounded LLM review

The LLM reviewed 32 exact audit rows. For each row, deterministic code first
proved that the proposed label matched the independently extracted
`AtomicFactDelta`, that evidence existed on the revisions required by the
label, and that the prompt regenerated from the same source record.

The judge then returned:

- truth: 32/32 pass;
- version status: 32/32 pass;
- answerability: 32/32 pass.

The evidence bundle SHA-256 is
`a7cfc5525d93efb1e141c686317f8a6c58c42b29c35e58a28e9e1a886d5819e7`.
The LLM was review-only and could not create or rewrite gold truth.

## Remaining limitations

This acceptance bank tests only unseen additions plus stable controls. All six
removed facts and all three changed facts shared semantic identities with
development and were correctly excluded. Acceptance feature verification is
also not generalized yet, so there are no acceptance `FeatureDelta` items.
The v0.26.0 endpoint appeared in earlier broad e1_vllm work, although this
exact adjacent v0.25.1-to-v0.26.0 transition was not previously used.
