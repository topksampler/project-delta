# delta_v2 repaired-data SFT control — Modal v1 audit

Status: `implemented`; repaired-pair first gate passed, full frozen evaluation
failed the retention gate, promotion unauthorized.

Loop position: ADAPT produced one fresh-base supervised-LoRA `CandidateState`.
Its conditional training-surface gate passed and it entered the unchanged
169-item VERIFY environment. It failed one preregistered retention gate and
rolled back to the pinned base; DELTA is at `DECIDE` before the separately
contracted Stage 2 objective comparison. The active state has not changed.

## causal question

The earlier c5, c7, and c6 candidates used annotation-only false contracts and
learned a global `yes` or `no` intercept. This control asks whether repairing
the negative bank is sufficient while retaining the original assistant-only
generative SFT objective.

The paired-data view contains 21 recall rows and 63 balanced true/false pairs.
Every false value comes from another source's verified true contract. The four
corruption families are annotation swap, config-default swap,
environment-getter swap, and full-card swap. The strongest empirical
value-only classifier reaches 55.56%; frozen and original evaluation prompt
overlap are both zero.

Dataset prefix:
`datasets/experiments/delta_v2/knowledge_paired_v2/`.

## Stage 1 training

- run:
  `delta-v2-d2-knowledge-paired-data-sft-control-qwen35-08b-modal-v1`
- protocol:
  `delta-v2-knowledge-paired-sft-control-v1`
- successful launch commit: `fd3f8f3`
- objective: `assistant-only-generative-sft-v1`
- adapter SHA-256:
  `9ef3caaf1f1d83a26df485ee0abcd365d1f932e0752cb9a99d9c5ac3510565a5`
- config SHA-256:
  `14e96035b8b57ae84aab7c27fd5e760440f462143a572d84761922dc846f7f1d`
- protocol SHA-256:
  `7c97f2581636510403f7af8792e044fd7543c8be66500f5d923f73ef0feb72cd`
- metrics SHA-256:
  `4cc7035c23ff0b949810e7b7e2ad599243cbb574bb3d111a4a234d9ccfea27b2`
- steps: 60
- training units: 180 boolean pairs and 60 recall units
- model forwards: 420
- initial/final recall-dev loss: 2.394504 / 0.163693
- trainable parameters: 5,411,328
- matched language modules: 186
- vision modules trainable: false
- model work: 264.11 seconds on NVIDIA H100
- estimated successful-run cost: USD 0.3222

The adapter, metrics, config, protocol, train, pairs, and dev hashes were
independently recomputed after B2 pull and matched the signed receipt.

## repaired-pair margin gate

- run:
  `delta-v2-d2-knowledge-paired-data-sft-control-margin-modal-v1`
- protocol: `delta-v2-knowledge-paired-margin-v1`
- successful launch commit: `ccb8609`
- metrics SHA-256:
  `a954b4501fe9a60a2a8c842a69ad050fa278e9c28baa446300002aed5c02ce5b`
- samples SHA-256:
  `7ec5c4df42a965ff0ccecde833c363c8e9a32c4bfa5b461fee4720ced5e5b00a`
- model updates: 0
- model work: 67.45 seconds on NVIDIA A10
- estimated successful-run cost: USD 0.0275

| gate cell | observed | minimum | result |
|---|---:|---:|---|
| verified true margin | 60/63, 95.24% | 80% | pass |
| verified false margin | 58/63, 92.06% | 80% | pass |
| truth-conditioned pairs | 57/63, 90.48% | 80% | pass |
| annotation-swap false | 19/21, 90.48% | 80% | pass |
| config-default-swap false | 8/9, 88.89% | 80% | pass |
| environment-getter-swap false | 12/12, 100% | 80% | pass |
| full-card-swap false | 19/21, 90.48% | 80% | pass |

The mean true `yes-no` sequence margin is +5.9287; the mean false margin is
-5.5992. Mean within-pair separation is +11.5279. Unlike the earlier adapters,
this candidate learned the truth-conditioned sign on most pairs rather than
moving both surfaces together.

## unchanged full evaluation

- run:
  `delta-v2-d2-knowledge-paired-data-sft-control-eval-modal-v1`
- extension protocol:
  `delta-v2-knowledge-paired-control-eval-v1`
- launch commit:
  `f22349de8e0296a2396f342a3a73967e7238695c`
- parent evaluation protocol SHA-256:
  `0ba449b993cd9afb356d35952283b582d81519960ae362137456173c3978c226`
- extension protocol SHA-256:
  `5dd0a46e6c3a35bea15c2bba58514395c8bc02118716926a6487bee1a53bbd03`
- config SHA-256:
  `a9e33b40ba3c1cc0fb0deb68d8b1c95092d22bc197236f5ded1bf8b4effc406f`
- evaluation-bank SHA-256:
  `5cc70b14353d874f7d51117dbec341a5814f4a55606b611824dd5b2f0d86dd9c`
- request SHA-256:
  `ceb8875ff2803b202d9d9d05298d3bbaa346bd656323f61e3df65d18e223d5cf`
- metrics SHA-256:
  `c75b475e4af524948c3f7976b18f3b5a53a126d282bf47e7c47603706488e705`
- samples SHA-256:
  `caed7b014f0e7143a8a14fcd2d23a0c34db16d5a011eead7f6ee574e789762ab`
- run-receipt SHA-256:
  `90b9b64b1cce0fa11e8f7af146f49700b54f8317daf6972f4e9de6a5f95c1f19`
- model outputs: 338 deterministic continuations for 169 items
- model updates: 0
- model work: 342.44 seconds on NVIDIA A10
- estimated cost: USD 0.1115

The adapter and pinned base used the same frozen evaluation bank, parent
protocol, prompting, generation, scoring, and reporting. Recall remains
advisory, the feature cell remains separate, and no pooled score is defined.

| cell | pinned base | repaired-data LoRA | delta |
|---|---:|---:|---:|
| acquisition choice | 0/21 | 12/21 | +12 |
| acquisition boolean true | 0/21 | 20/21 | +20 |
| acquisition boolean false | 21/21 | 18/21 | -3 |
| acquisition recall, advisory | 0/21 | 9/21 | +9 |
| retention choice | 0/21 | 8/21 | +8 |
| retention boolean true | 1/21 | 18/21 | +17 |
| retention boolean false | 21/21 | 16/21 | -5 |
| retention recall, advisory | 0/21 | 1/21 | +1 |
| feature retention | 0/1 | 0/1 | 0 |

| preregistered gate | minimum | observed | result |
|---|---:|---:|---|
| acquisition choice | 50% | 57.14% | pass |
| acquisition boolean true | 80% | 95.24% | pass |
| acquisition boolean false | 80% | 85.71% | pass |
| retention boolean false | 90% | 76.19% | **fail** |

All candidate hashes above were independently recomputed after pulling the
uploaded B2 artifacts. The pinned-base metrics SHA-256 also matched its frozen
receipt:
`6b617bf893a2f9c49e16ba5bd17f16abe195b7b69a35c7e8e94681d81de6a0fb`.

## interpretation and decision

LoRA was not the failed component. The original annotation-only negative bank
was. With source-grounded, marginally realistic corruptions, the unchanged
generative SFT objective crosses every conditional truth-separation gate.

The full evaluation now narrows the remaining failure: the candidate acquires
the changed-source claims on all three preregistered acquisition cells, but it
does not preserve the frozen false-accept boundary on source-disjoint stable
facts. More steps under the same generative objective are not justified.

The frozen paired-data causal design makes one objective-only Stage 2 LoRA
comparison eligible: restart from the same pinned base and change only the
boolean-pair loss from generative cross-entropy to truth-conditioned logistic
ranking. This comparison tests whether the original objective caused an
unnecessarily broad answer-policy shift. It is not authorized until its
separate protocol and safety/verification gates are frozen, and it is not
expected to pass merely because Stage 1 failed. Rank, learning rate, steps,
data, QLoRA, full-weight training, and verifier-grounded RL remain frozen.

## platform preflight stops

Two attempts stopped before model loading and performed zero optimizer work:

1. the training worker initially lacked the frozen paired-data summary in its
   image;
2. the margin worker initially pulled the adapter but not the bound training
   receipt and metrics.

The first was fixed by mounting the already-frozen summary. The second was
fixed generically: Modal configs can declare exact run-evidence files, which
the worker pulls from B2 and verifies by SHA-256. Before each retry, the run
prefix contained no adapter or diagnostic completion artifacts. Estimated
preflight cost was USD 0.0286. Total Stage 1 plus gate and preflights was an
estimated USD 0.3783. Including the successful full evaluation, the audited
total is an estimated USD 0.4898.
