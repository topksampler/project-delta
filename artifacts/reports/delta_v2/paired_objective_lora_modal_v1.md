# delta_v2 paired-objective LoRA — Modal v1 audit

Status: `implemented`; repaired-pair first gate passed, unchanged full
evaluation failed acquisition-choice and retention-false gates, promotion
unauthorized.

Loop position: ADAPT produced one fresh-base Stage 2 supervised-LoRA
`CandidateState` whose only causal change from the repaired-data SFT control is
the Boolean-pair learning objective. Its conditional training-surface gate
passed, but it failed two gates in the unchanged 169-item VERIFY environment
and rolled back to the pinned base. DELTA is at `DECIDE` before any new
`InterventionPlan`; the active state has not changed.

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

## unchanged full evaluation

- run:
  `delta-v2-d3-knowledge-paired-objective-eval-modal-v1`
- extension protocol:
  `delta-v2-knowledge-paired-objective-eval-v1`
- successful launch commit:
  `5ed421788cfeb0a43cbddf075ff5ba1aff966499`
- parent evaluation protocol SHA-256:
  `0ba449b993cd9afb356d35952283b582d81519960ae362137456173c3978c226`
- extension protocol SHA-256:
  `0fefe6f2b1bbf687e75ce484183b4f231857eb42109c5cb1a826850987d89857`
- config SHA-256:
  `9a58babbc0b749d2f8b731bda6b4e404abaa3ac7d4d28b4a00f713cc2d163dee`
- evaluation-bank SHA-256:
  `5cc70b14353d874f7d51117dbec341a5814f4a55606b611824dd5b2f0d86dd9c`
- request SHA-256:
  `5d6d686cba3a20bef8ff7b0d00f456b7dd54e1141cd3218b2cb8057cb6371b8c`
- metrics SHA-256:
  `cfbffc48b24347e4f4701d29fb27823cfa0d3f61a5fdf35d6235892c8d97a740`
- samples SHA-256:
  `edffd8990058184a5aa28eca45e7df8e3fa1acaeeaeb0d38bd19d99bd1f50063`
- run-receipt SHA-256:
  `5d9cf9be0eb4dd31c3d40587fe3305934256805bd5bb4117294ad767ecbe6f57`
- model outputs: 338 deterministic continuations for 169 items
- model updates: 0
- model work: 1,615.72 seconds on NVIDIA A10
- estimated successful-run cost: USD 0.5026

The adapter, pinned base, and Stage 1 control used the same frozen evaluation
bank, parent protocol, prompting, generation, scoring, and reporting. Recall is
advisory, the feature cell is separate, and no pooled score is defined.

| cell | pinned base | Stage 1 generative | Stage 2 ranking |
|---|---:|---:|---:|
| acquisition choice | 0/21 | 12/21 | 0/21 |
| acquisition boolean true | 0/21 | 20/21 | 20/21 |
| acquisition boolean false | 21/21 | 18/21 | 19/21 |
| acquisition recall, advisory | 0/21 | 9/21 | 3/21 |
| retention choice | 0/21 | 8/21 | 0/21 |
| retention boolean true | 1/21 | 18/21 | 18/21 |
| retention boolean false | 21/21 | 16/21 | 15/21 |
| retention recall, advisory | 0/21 | 1/21 | 1/21 |
| feature retention | 0/1 | 0/1 | 0/1 |

| preregistered gate | minimum | observed | result |
|---|---:|---:|---|
| acquisition choice | 50% | 0% | **fail** |
| acquisition boolean true | 80% | 95.24% | pass |
| acquisition boolean false | 80% | 90.48% | pass |
| retention boolean false | 90% | 71.43% | **fail** |

All 42 choice outputs were unparseable under the exact-choice scorer. They were
not random: the model usually embedded a selected letter in a long explanation
instead of returning the required single letter. Ten of 169 unique
continuations reached the unchanged 256-token limit, all in the choice cell.
The scorer correctly applied no repair. This response-format regression also
explains why the run took materially longer than Stage 1.

All full-evaluation hashes were independently recomputed after B2 pull and
matched the signed receipt.

## interpretation and decision

The paired ranking objective improves the frozen Boolean-pair surface, but
that gain does not transfer to the complete task. Relative to Stage 1, it gains
one acquisition false item and loses one retention false item, twelve
acquisition choice items, and six advisory acquisition recall items.

The causal result is therefore narrow and negative:

- repairing the data is necessary and makes LoRA acquire changed-source facts;
- generative SFT transfers the required short-answer behavior but fails the
  source-disjoint retention boundary;
- Boolean-only ranking sharpens truth separation but removes that
  short-answer transfer and does not repair retention.

Neither more steps under Stage 1 nor more steps under Stage 2 are justified.
Any next supervised-LoRA candidate needs a new frozen contract for a
retention-preserving, multi-surface objective. It must preserve choice-format
supervision without copying evaluation wording and must use stability controls
that remain source-disjoint from the 21 frozen retention facts. That is a new
intervention, not a retry.

The first full-evaluation attempt stopped before model load because the bound
training receipt was not declared as a worker input. It produced no samples,
metrics, or run receipt and cost an estimated USD 0.0062. The packaging fix
mounted that already-frozen receipt; no scientific field changed. Including
training, margin gate, successful full evaluation, and this preflight stop,
Stage 2 cost an estimated USD 1.1348.

QLoRA, full-weight training, reinforcement learning, additional training, and
promotion remain unauthorized.
