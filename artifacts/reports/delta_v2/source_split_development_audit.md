# delta_v2 development source split audit

Status: **implemented and unfrozen**

This audit covers source-level assignment for verified development facts and
features before any model-facing wording exists. It does not access the sealed
acceptance transition or create EvalItems.

## Contract

- Contract ID: `delta-v2-development-source-split-v1`
- Contract SHA-256:
  `35625bbcb021c1162ce9d12ca19795189cefcf8ae3a00e45f0157d60fc6ad1a9`
- Development allocation: 80% train hash space, 20% dev hash space
- Development eval allocation: forbidden
- Future acceptance allocation: eval, after explicit unsealing
- Freeze state: unfrozen

The assignment input is the SHA-256 of a versioned salt, a NUL separator, and
the split unit's stable anchor ID. No source content, wording, model result, PR
description, or human preference influences the split.

## Leakage unit

A source is not always assigned alone. FeatureDelta-to-AtomicFactDelta
references form a graph, and each connected component is one indivisible split
unit. The unit anchor is its lexicographically smallest source ID.

This prevents one underlying contract from appearing in train as a fact and in
dev as a feature. All future wording variants inherit the already assigned
`source_id` split.

The current verified prefix-cache feature and its environment-variable fact
form one two-member unit:

- `feature:vllm-prefix-cache-retention`
- `fact:717e42584616457338dae6cae10686391a1ae66e37bbb96a467e93dbb97ed440`

The unit deterministically landed in train. It was not manually placed.

## Inputs

- 1,023 deterministic AtomicFactDelta records
- 1 promoted FeatureDelta record
- 1,024 eligible sources total
- 1,023 split units
- 1 multi-source unit

Input hashes:

- Atomic facts:
  `a5827d0a66289552bc707f75e20ce4ecd74a1e22dba6b3144a87ab8f27241017`
- FeatureDelta:
  `85852d298636f40ce9f947a87c2edf3c48440207890669d901b2373150711229`

Candidates, FileDeltas, PR records, probe records, and rejected fact
observations are not scored split sources.

## Assignment result

| Split | Atomic facts | Features | Total |
|---|---:|---:|---:|
| train | 827 | 1 | 828 |
| dev | 196 | 0 | 196 |
| eval | 0 | 0 | 0 |

Fact-family development holdouts:

- CLI options: 50
- config fields: 90
- environment variables: 50
- literal domains: 6

Status development holdouts:

- stable: 185
- added: 5
- changed: 4
- removed: 2

The generated assignment has SHA-256
`6961545f88805c962390b79ba44ebeee3613472113682940e83b42aae0839f53`.
An independent rebuild produced byte-identical assignment and summary files.

## Invariant audit

- duplicate source IDs: none
- missing eligible sources: none
- extra sources: none
- invalid splits: none
- cross-split units: none
- eval sources in development: zero
- acceptance accessed: false
- deterministic ordering: pass
- complete `experiments/delta_v2` suite: 77 tests passed

## Interpretation

The observed dev share is 19.1%, which is normal for a content-blind 80/20 hash
threshold. The contract deliberately does not force exact quotas because quota
ranking would reshuffle existing assignments as new verified sources arrive.

Only one FeatureDelta exists, so it is impossible to place verified features
in both train and dev without duplicating or manually overriding that source.
Future verified features will be assigned by the same policy.

The split is not frozen. Adding verified sources can merge overlap components
and may move a source when a newly discovered feature connects facts that were
previously separate. This must be re-audited before the EvalEnvironment freeze.

## Next boundary

The next controlled BUILD slice is an EvalItem eligibility and wording
contract. It must consume these source assignments, preserve the source split
for every wording variant, and generate gold answers only from verified source
records.
