# delta_v2 repaired-data SFT control — Modal v1 audit

Status: `implemented`; repaired-pair first gate passed, full frozen evaluation
eligible, promotion unauthorized.

Loop position: ADAPT produced one fresh-base supervised-LoRA `CandidateState`.
Its conditional training-surface gate passed, so the candidate may enter the
unchanged 169-item VERIFY environment. The active state has not changed.

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

## interpretation and decision

LoRA was not the failed component. The original annotation-only negative bank
was. With source-grounded, marginally realistic corruptions, the unchanged
generative SFT objective crosses every conditional truth-separation gate.

DELTA's least-cost rule therefore blocks an automatic paired-objective training
run at this point. The simpler candidate must first run the unchanged frozen
169-item evaluation. A paired objective remains a separately preregistered
research comparison, not the next required intervention. Steps, learning rate,
rank, QLoRA, full-weight training, and verifier-grounded RL remain frozen.

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
estimated USD 0.3783.
