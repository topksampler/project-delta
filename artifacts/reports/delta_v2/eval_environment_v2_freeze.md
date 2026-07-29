# delta_v2 EvalEnvironment v2 freeze

Status: frozen locally and reproducible; not published.

The manifest at `experiments/delta_v2/eval_environment_v2.yaml` passed its
freeze gate twice with byte-identical results.

## Frozen identity

- environment: `delta-v2-vllm-source-build-v2`;
- validator code commit:
  `431a712d821d519abd642fe48dfe10f7f365eb7e`;
- manifest SHA-256:
  `8e12ebdb685eceb8f9b590c67c05c449d71cc4b8a95f7d2bf3c8f3b4064991c4`;
- bound files: 56;
- harness tests: 147/147 pass;
- publication: local reproducible cache only;
- target-model, training, and fine-tuning runs: zero.

The committed manifest is the durable identity. Generated data remains in the
ignored local cache and is content-addressed by the manifest. No B2 write or
cloud job ran.

## Source-level SENSE

The selected repository is `vllm-project/vllm`.

- development transition: v0.22.0 to v0.23.0;
- acceptance transition: v0.25.1 to v0.26.0;
- all four revisions are pinned by commit SHA and Git-tree hash;
- complete FileDelta inventories contain 5,515 development rows and 6,088
  acceptance rows;
- the preregistered AST fact families produce 1,023 development and 1,061
  acceptance AtomicFactDelta records;
- merged-pull-request evidence is exact provenance for feature grouping, not
  gold truth.

Source-change SENSE is implemented for this vertical slice. Active-target-model
drift sensing is deferred because no target model has run.

## BUILD

Development contains 1,024 source identities, 87 EvalItems (65 train and 22
dev), and one verified prefix-cache-retention feature.

Acceptance contains 1,062 source identities and 43 eval-only EvalItems:

- 21 unseen added facts;
- 21 unseen stable controls;
- one endpoint-plugin feature matrix;
- 995 development source identities excluded before selection;
- zero cross-transition source overlap.

The acceptance feature is rooted in merged PR 47454, 12 exact FileDelta
records, a promoted FeatureDelta, and a passing old/new executable probe. The
probe verifies opt-in loading, task gating, failure isolation, route
attachment, and asynchronous state initialization. It executes exact pinned
function definitions in an isolated standard-library runtime with dependency
stubs; it does not boot a full vLLM server.

The v3 acceptance item build preserves all 42 previously audited fact items
byte-for-byte and adds exactly one structured feature item.

The evidence-grounded LLM audit reviewed a mandatory 32-item sample, including
the feature:

- truth: 32/32 pass;
- version status: 32/32 pass;
- answerability: 32/32 pass.

The LLM acted only as reviewer. Deterministic validators regenerated prompts
and gold from the bound facts and behavior probe before judgment. The LLM could
fail or abstain but could not rewrite truth.

## What remains in the first two stages

For the deliberately scoped source-level vertical slice, the SENSE-to-BUILD
path is complete and frozen.

The broader Project DELTA stages still have explicit open boundaries:

1. SENSE has not measured the active target model's knowledge or drift.
2. BUILD covers four preregistered static fact families and two manually traced
   features; it does not claim repository-wide semantic or behavioral
   coverage.
3. The endpoint-plugin probe is intentionally narrower than a full server
   integration test.
4. The acceptance feature was selected after unsealing and before any target
   model run. This timing is recorded rather than hidden.

These are declared coverage limits, not unfinished requirements for the frozen
v2 slice. The next loop action is to preregister a target-model baseline
protocol and enter DECIDE. It requires separate authorization before any local
or remote model-evaluation job runs.
