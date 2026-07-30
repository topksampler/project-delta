# delta_v2 paired-objective LoRA — Modal v1 audit

Status: `implemented`; adapter training and repaired-pair first gate passed,
unchanged full evaluation eligible, promotion unauthorized.

Loop position: ADAPT produced one fresh-base Stage 2 supervised-LoRA
`CandidateState` whose only causal change from the repaired-data SFT control is
the Boolean-pair learning objective. Its conditional training-surface gate
passed, so it may enter the unchanged 169-item VERIFY environment. The active
state has not changed.

## causal boundary

Stage 1 showed that source-grounded paired data repairs acquisition but that
ordinary generative SFT damages the source-disjoint retention false boundary.
Stage 2 keeps the pinned base revision, 147 training rows, unit order, LoRA
topology, optimizer, 60-step schedule, development bank, and evaluation bank
fixed. It replaces only the Boolean-pair cross-entropy loss with:

- truth-conditioned logistic classification for true and false prompts; and
- a logistic ranking term over true-minus-false pair separation.

QLoRA, full-weight training, reinforcement learning, additional training, and
promotion are outside this contract.

## training

- run:
  `delta-v2-d3-knowledge-paired-objective-qwen35-08b-modal-v1`
- protocol:
  `delta-v2-knowledge-paired-objective-v1`
- launch commit:
  `2086374e2a9e048f50e0fa9c87f935a569f778b1`
- objective:
  `paired-boolean-logistic-ranking-v1`
- config SHA-256:
  `147e5cbd1f5d586d518406bd89f0836105b47d5b3fa7f03b784870b5f358e811`
- protocol SHA-256:
  `610cf70ccc0678f2bb1d21e43f6142ea21a4ec0fc2da6a4a252af0fae8daaf67`
- metrics SHA-256:
  `4ae603a3d0d7219b183e366d28a92716dac21fca14ecca2c885fb77810461bd0`
- adapter SHA-256:
  `3de6466516530533e0a060224df99e05966e1639f30fc71cb52aed2fb3ef21ae`
- run-receipt SHA-256:
  `9b78887be36662b2087a988eb49cb34606f57acf7eae38cda40cdf84f653b171`
- steps: 60
- training units: 180 Boolean pairs and 60 recall units
- model forwards: 780
- initial/final recall-development loss: 2.394504 / 0.345351
- trainable parameters: 5,411,328
- matched language modules: 186
- vision modules trainable: false
- model work: 468.68 seconds on NVIDIA H100
- estimated dispatch cost: USD 0.5983

The config, protocol, train, pairs, development data, adapter, metrics, and
receipt hashes were independently recomputed after B2 pull and matched.

## repaired-pair margin gate

- run:
  `delta-v2-d3-knowledge-paired-objective-margin-modal-v1`
- protocol:
  `delta-v2-knowledge-paired-objective-margin-v1`
- launch commit:
  `421684e1e559b4f37ccf53bafef5949db9eef9bf`
- config SHA-256:
  `0f6afb8751e822fe45d214785bb7b5517a7e428af6c60dc29b117c248791dd7c`
- protocol SHA-256:
  `0bc60ddef307be99d380a5917cd12ebde5bb77068ca20eb2f8e9ef70a7744495`
- metrics SHA-256:
  `959a15dbc94efb27f1f4e6d22b788508371c4716a44a22f2a064b721cf63a09d`
- samples SHA-256:
  `4429f34ac1fb92b3b549a7db84b47368d0487d888e6965fce541a1a1f0b13252`
- run-receipt SHA-256:
  `1b815304bfa52bf5f665bf698d108ec6af58fd75b85ac6eddb1b1c2b2d34bfc7`
- model updates: 0
- model work: 67.34 seconds on NVIDIA A10
- estimated dispatch cost: USD 0.0277

| gate cell | observed | minimum | result |
|---|---:|---:|---|
| verified true margin | 60/63, 95.24% | 80% | pass |
| verified false margin | 60/63, 95.24% | 80% | pass |
| truth-conditioned pairs | 59/63, 93.65% | 80% | pass |
| annotation-swap false | 20/21, 95.24% | 80% | pass |
| config-default-swap false | 8/9, 88.89% | 80% | pass |
| environment-getter-swap false | 12/12, 100% | 80% | pass |
| full-card-swap false | 20/21, 95.24% | 80% | pass |

The mean true `yes-no` sequence margin is +9.3635; the mean false margin is
-9.8294. Mean within-pair separation is +19.1929.

Relative to the Stage 1 generative control on identical rows, verified-false
accuracy rises from 58/63 to 60/63, truth-conditioned pairs rise from 57/63 to
59/63, and mean pair separation rises from +11.5279 to +19.1929. This is
evidence about the frozen training surface only; it does not establish
source-disjoint retention.

All gate evidence hashes were independently recomputed after B2 pull. Training
plus gate cost an estimated USD 0.6260.

## decision

Every separate first-gate cell passed. The only authorized next action is the
unchanged 169-item evaluation with the original acquisition and retention
gates. No promotion or further training is authorized by this result.
