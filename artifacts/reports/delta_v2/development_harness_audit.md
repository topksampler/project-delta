# delta_v2 development harness audit

Status: **implemented for the current development slice**

This report audits the development-only path from pinned SourceSnapshots
through one verified FeatureDelta. It does not access the sealed acceptance
transition, freeze an EvalEnvironment, generate EvalItems, or run a model.
Canonical program delivery status remains in `docs/delta-roadmap.md`.

## Verified feature

- Feature ID: `feature:vllm-prefix-cache-retention`
- Status: `added`
- Source transition: vLLM `v0.22.0` to `v0.23.0`
- Source candidate:
  `feature-candidate:vllm-prefix-cache-retention`
- Merged-PR provenance:
  `pull-request:vllm-project/vllm#43447`

The FeatureDelta promotion audit passed every candidate link, evidence link,
probe ID, claim scope, result status, source role, and exact result hash.

## End-to-end rebuild

Every deterministic development artifact was rebuilt into a fresh temporary
directory and compared with the current output.

| Stage | Result | Deterministic evidence |
|---|---:|---|
| SourceSnapshot verification | pass | exact commits and Git tree hashes |
| FileDelta inventory | pass | 5,515 rows; SHA-256 `129831f4b6fa4b0604fe2c1f31854947df66df3dd4bf9101700e2386cb95c6b9` |
| AtomicFactDelta extraction | pass | 1,023 rows; SHA-256 `a5827d0a66289552bc707f75e20ce4ecd74a1e22dba6b3144a87ab8f27241017` |
| FeatureCandidate and PR trace | pass | one fact, seven files, zero unresolved links |
| Environment-interface probe | pass | SHA-256 `2838adddfc48ef334a948c9d9e2d55a34abf37a07be59cdf1cce270a1fcea884` |
| Cache-mechanics probe | pass | SHA-256 `0180c2e7a688a4a87c1a418c4ced09289f1d51c447337f56835f3d3d08d4587b` |
| FeatureDelta promotion | pass | both probe hashes and development roles match |

The rebuilt FileDelta JSONL, all five atomic-fact JSONL files, the atomic-fact
summary, and both BehaviorProbe results were byte-identical. The FileDelta
summary also matched after removing its intentionally location-dependent
`inventory_path` field.

## Behavior verification

The external old/new mechanics contract passed all five cases:

1. unset remains dense in both revisions;
2. interval `64` is ignored by the old revision and produces sparse retained
   checkpoints in the new revision;
3. interval `0` is ignored by the old revision and retains only the latest
   replay boundary in the new revision;
4. negative values are ignored by the old revision and rejected by the new
   revision;
5. misaligned values are ignored by the old revision and rejected by the new
   revision.

The v0.23.0 source also passed eight focused upstream CPU tests covering hybrid
alignment, invalid intervals, block recycling, latest-only reuse, pure
sliding-window sparse retention, and dense-default preservation.

The development runtime was:

- macOS arm64;
- Python 3.11.9;
- PyTorch 2.13.0;
- Transformers 4.57.6;
- dependency-source SHA-256
  `35d459d03e4f446eac159b8308b0a5a95ff9ba4ff8506512374dd9bac6275b6d`.

This runtime is explicitly `development-only-unfrozen`. It is not the future
EvalEnvironment.

## Harness tests

The complete `experiments/delta_v2` test suite passed:

- 64 tests;
- snapshot and tree verification;
- acceptance-transition sealing;
- FileDelta classification and reconstruction;
- atomic-fact family preregistration, extraction, rejection, joining, and
  reconstruction;
- FeatureCandidate and PR provenance validation;
- interface and mechanics probe validation;
- negative-path expectation and hash checks;
- FeatureDelta promotion validation.

## Defects found by the audit

Two harness defects were found before promotion:

1. The first mechanics run categorized a combined vLLM validation message by
   its first phrase. The classifier now uses the configured numeric constraint
   while still requiring the expected source error text.
2. A whitespace cleanup removed YAML continuation indentation after focused
   tests had passed. The subsequent full-suite run caught the malformed probe
   contract, and a focused correction was committed.

Neither defect was a vLLM feature failure. Both demonstrate why post-staging
full-suite execution belongs in the harness workflow.

## Boundary after this audit

The development SENSE-to-BUILD slice is now proven for one complete feature
trace:

`SourceSnapshot → FileDelta → AtomicFactDelta → FeatureCandidate → FeatureDelta`

BUILD remains partial at the experiment level. No source-level split,
model-facing EvalItem, audit/freeze, or EvalEnvironment exists yet. DECIDE,
ADAPT, VERIFY, and MEMORY have not started for `delta_v2`.

The next controlled slice is to define a source-level split contract for
verified facts and features before generating any question wording.
