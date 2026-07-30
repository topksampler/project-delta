# delta_v2 c5 knowledge LoRA — Modal v1 evaluation audit

Status: `implemented`; preregistered gates failed.

This report compares the rank-8 LoRA candidate against the base model on the
identical frozen `delta-v2-vllm-knowledge-adaptation-v1` bank. Every one of the
169 probes produced two byte-identical greedy continuations. There is no pooled
overall score.

## run identity

- evaluation run:
  `delta-v2-c5-knowledge-lora-eval-qwen35-08b-modal-v1`
- adapter-producing run:
  `delta-v2-c5-knowledge-lora-qwen35-08b-modal-v2`
- model: `Qwen/Qwen3.5-0.8B`
- model revision: `2fc06364715b967f1860aea9cf38778875588b17`
- launch commit: `f4e9a6106f540a21019b008a36fdc7984629117b`
- config SHA-256:
  `b88ac9501aeffc64a2708b066fc9ceb74c682b728d7629c2400a0580a9d1d44d`
- adapter SHA-256:
  `83fb04ae52502432df6b17d2aaa12cfeb12f457df6a77c83f28fed433d9966fa`
- protocol SHA-256:
  `0ba449b993cd9afb356d35952283b582d81519960ae362137456173c3978c226`
- eval-bank SHA-256:
  `5cc70b14353d874f7d51117dbec341a5814f4a55606b611824dd5b2f0d86dd9c`
- request SHA-256:
  `172a96225a22dab5378b062192f34a25320716f5a0a81119729a4313943b7e5c`
- samples SHA-256:
  `a6549894df5a8c1d9f9065e494e2fbefded5972460b02f296518996bbc2cafa6`
- metrics SHA-256:
  `592915e674e874e4e5874ed835ed4169127260c2faf88e7e8b92490322670a8b`
- comparison SHA-256:
  `8fb215dd38539c05cd60300b200c4b50e58e9f47af2629f11f7efe5c1e21ec8b`
- B2 prefix:
  `runs/delta-v2-c5-knowledge-lora-eval-qwen35-08b-modal-v1/`

## readout

| scoreboard | probe | base | LoRA | delta |
|---|---|---:|---:|---:|
| acquisition, added facts | choice | 0/21 | 10/21 | +10 |
| acquisition, added facts | boolean true | 0/21 | 0/21 | 0 |
| acquisition, added facts | boolean false | 21/21 | 21/21 | 0 |
| acquisition, added facts | recall, advisory | 0/21 | 10/21 | +10 |
| retention, stable facts | choice | 0/21 | 7/21 | +7 |
| retention, stable facts | boolean true | 1/21 | 1/21 | 0 |
| retention, stable facts | boolean false | 21/21 | 21/21 | 0 |
| retention, stable facts | recall, advisory | 0/21 | 3/21 | +3 |
| feature retention | verified behavior JSON | 0/1 | 0/1 | 0 |

All LoRA choice responses were parseable, compared with none of the base
responses. All 21 acquisition recalls became parseable. This is a real response
format improvement, but parseability does not substitute for correctness.

## preregistered gates

| gate | minimum | observed | result |
|---|---:|---:|---|
| acquisition choice | 50% | 47.62% | fail |
| acquisition boolean true | 80% | 0% | fail |
| acquisition boolean false | 80% | 100% | pass |
| retention boolean false | 90% | 100% | pass |

The candidate fails the evaluation gates and is not eligible for promotion.
Full-weight fine-tuning remains forbidden.

## truth-rooted LLM review

The deterministic scorer owns the verdict. A review of raw outputs against the
same frozen source truth explains the shape of the failure:

- the adapter learned some exact source contracts: 10 of 21 added-fact recall
  objects exactly match gold;
- it learned constrained output surfaces, but acquired a strong option-position
  shortcut: 19 of 21 added-fact choice outputs were `A`;
- it did not transfer the balanced yes/no training surface to the evaluation
  paraphrase: every added-fact true assertion was answered `no`;
- the unchanged 21/21 false-assertion scores therefore still reflect a
  false-reject policy, not calibrated honesty;
- the one feature-control output remained structurally parseable but mapped
  behavior states to string booleans, so it was deterministically incorrect.

This is same-claim, new-surface acquisition—not held-out-fact generalization.
The most informative next action is a training-surface fit diagnostic. It must
test the already frozen 63 training prompts without adding evaluation wording
to training. If true rows fit on their original surface, the failure is
paraphrase transfer; if they do not, the intervention did not overcome the
base model's negative-answer prior.

The evaluation used one NVIDIA A10 and took 333.04 seconds of model work.
Estimated run cost: USD 0.1071.
