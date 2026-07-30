# delta_v2 c4 knowledge base — Modal v1 audit

Status: `implemented`

This report records the closed-book base-model readout on the frozen
`delta-v2-vllm-knowledge-adaptation-v1` bank. It is the required pre-training
baseline for the first `delta_v2` LoRA candidate.

## run identity

- run: `delta-v2-c4-knowledge-base-qwen35-08b-modal-v1`
- condition: `c4_knowledge_base`
- model: `Qwen/Qwen3.5-0.8B`
- model revision: `2fc06364715b967f1860aea9cf38778875588b17`
- harness commit: `eb26727`
- config SHA-256:
  `87f029a255a25cdfc0272667b5d75d94639aefbea892d9271029c165c8e3bdf0`
- protocol SHA-256:
  `0ba449b993cd9afb356d35952283b582d81519960ae362137456173c3978c226`
- request SHA-256:
  `f835b4e57099961bac3740a781528ede80ac88fe932a4dbf6f6508810557196c`
- samples SHA-256:
  `50ab84e9800285bf8a3a257338ebd32485cfdec307ddabc4057a12f189dc8315`
- metrics SHA-256:
  `6b617bf893a2f9c49e16ba5bd17f16abe195b7b69a35c7e8e94681d81de6a0fb`
- B2 prefix:
  `runs/delta-v2-c4-knowledge-base-qwen35-08b-modal-v1/`

The exact runtime lock passed on one NVIDIA A10. All 169 probes had two
byte-identical greedy continuations.

## readout

| scoreboard | probe | correct | parseable |
|---|---|---:|---:|
| acquisition, added facts | choice | 0/21 | 0/21 |
| acquisition, added facts | boolean true | 0/21 | 21/21 |
| acquisition, added facts | boolean false | 21/21 | 21/21 |
| acquisition, added facts | recall, advisory | 0/21 | 18/21 |
| retention, stable facts | choice | 0/21 | 0/21 |
| retention, stable facts | boolean true | 1/21 | 21/21 |
| retention, stable facts | boolean false | 21/21 | 21/21 |
| retention, stable facts | recall, advisory | 0/21 | 17/21 |
| feature retention | verified behavior JSON | 0/1 | 1/1 |

There is no pooled overall accuracy.

## interpretation

The boolean bank exposes an extreme false-reject bias. The base model answered
`no` to every added-fact true assertion and to 20 of 21 stable-fact true
assertions. Its perfect false-assertion score therefore does not establish
honesty or source knowledge.

Choice responses were long explanations rather than the required single
letter. Some explanations contained a proposed option, but the preregistered
exact scorer correctly rejected every response without repair. Recall outputs
were often valid JSON but used invented schemas or unsupported meanings, so
parseability did not substitute for truth.

The feature output copied internal observation-key names into the values
instead of producing the requested user-facing behavior states.

## ADAPT decision

Run the already frozen, truth-balanced LoRA training set unchanged:

- 21 exact v0.26 source-card recalls;
- 21 verified true assertions with `yes`;
- 21 deterministic false assertions with `no`.

The training data predates this result and contains no original evaluation
prompt. Keeping it unchanged avoids adapting the dataset to individual c4
outputs. The LoRA candidate must be compared on the identical 169 probes.

Only same-claim acquisition may be claimed. Stable and feature rows remain
source-disjoint regression controls. Full-weight fine-tuning and promotion
remain deferred.

Estimated run cost: USD 0.3827.
