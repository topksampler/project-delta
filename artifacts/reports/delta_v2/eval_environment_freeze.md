# delta_v2 EvalEnvironment freeze

Status: frozen locally and reproducible; not published.

The manifest at `experiments/delta_v2/eval_environment.yaml` passed its freeze
gate twice with identical results.

## Frozen identity

- environment: `delta-v2-vllm-source-build-v1`;
- validator code commit:
  `399ee886566ade2cb40d4673cac2e0bf4e220c09`;
- manifest SHA-256:
  `d31ea1014da00a6d4ac7cd20c2a82bd3113e9e2dc47acf3512f6bb6fdc29b514`;
- bound files: 47;
- harness tests: 124/124 pass;
- publication: local reproducible cache only;
- target-model, training, and fine-tuning runs: zero.

The committed manifest is durable. Its generated data files remain in the
ignored local cache and can be regenerated from the pinned source snapshots;
no B2 publication has run.

## Development material

Transition: vLLM v0.22.0 to v0.23.0.

- complete FileDelta inventory: 5,515 rows;
- AtomicFactDelta records: 1,023;
- nonstable facts: 43;
- verified development FeatureDelta records: 1;
- development source identities: 1,024;
- EvalItems: 65 train, 22 dev, 0 eval;
- grounded LLM audit: 32/32 pass for truth, version status, and
  answerability.

The verified feature is prefix-cache retention. Its evidence includes exact
files, extracted facts, merged pull-request provenance, and executable old/new
behavior probes.

## Acceptance material

Transition: vLLM v0.25.1 to v0.26.0.

- complete FileDelta inventory: 6,088 rows;
- AtomicFactDelta records: 1,061;
- raw nonstable facts: 30;
- development identities excluded before selection: 995;
- leak-free EvalItems: 42;
- acceptance labels: 21 `added`, 21 `stable`;
- cross-transition source overlap: 0;
- grounded LLM audit: 32/32 pass for truth, version status, and
  answerability.

The first proposed acceptance transition, v0.25.0 to v0.25.1, remains recorded
as a negative result because it produced no nonstable facts in the
preregistered families. The first item selection for attempt 2 also remains
recorded as invalid because it exposed development source identities. Both
failures happened before any target model ran.

## LLM-as-judge boundary

The LLM receives an evidence bundle only after deterministic validators prove
that each proposed answer matches an independently extracted fact or executable
feature observation. It reviews truth, version status, and answerability. It
cannot create a scored source, rewrite gold, repair a conflict, or substitute a
teacher proposal for evidence. Any disagreement fails BUILD and returns the
item for investigation.

## Loop status

| DELTA stage | delta_v2 status | Exact boundary |
|---|---|---|
| SENSE | partial | Source snapshots, complete file changes, and source facts are implemented; active-target-model drift sensing has not run. |
| BUILD | partial, fact slice frozen | Development train/dev and acceptance fact eval are frozen; one development feature is verified. Acceptance feature evaluation is not generalized. |
| DECIDE | deferred | No intervention matrix or plan exists. |
| ADAPT | deferred | No context, RAG, LoRA, or training intervention ran. |
| VERIFY | deferred | No candidate model exists to compare. |
| MEMORY | deferred | No promotion decision or active-state update exists. |

The next scientifically clean step is not training. It is either:

1. accept this fact-level environment and preregister a small target-model
   baseline protocol; or
2. extend BUILD with an independently selected acceptance feature, executable
   old/new verification, and a new environment revision.

The current environment is honest about its narrower coverage: acceptance is
facts only, and its unseen nonstable facts are additions rather than removals
or changes. The v0.26.0 endpoint also appeared in broader historical e1_vllm
work, although this exact adjacent transition was not previously used.
