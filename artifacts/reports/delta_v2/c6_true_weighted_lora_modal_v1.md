# delta_v2 c6 true-weighted LoRA — Modal v1 audit

Status: `implemented`; training-surface gate failed, full evaluation blocked.

c6 tested one change from c5: the immutable verified-true rows received four
virtual sampler entries while false and recall rows received one. Training
started again from the pinned base model. No evaluation wording was added to
training.

## training

- run:
  `delta-v2-c6-knowledge-lora-true-weighted-qwen35-08b-modal-v1`
- launch commit: `40b64aa`
- adapter SHA-256:
  `8dae0e5cf7692542efc34f4fc836b1c96279bae680747edc368f646fdf938e36`
- protocol SHA-256:
  `0ac25278f228a7afd4661c67e41ffe9e6e787572fcc3af4f9cef5afb19fa2ddb`
- metrics SHA-256:
  `97083b6cc25e1839348505cf0af03d86d44e1964d1e00e69cf23f93bb4911f48`
- realized exposures: 163 true, 37 false, 40 recall
- initial/final recall-dev loss: 2.394504 / 0.263486
- model work: 159.30 seconds on NVIDIA H100
- estimated cost: USD 0.2117

## conditional training-surface gate

- run:
  `delta-v2-c6-knowledge-lora-true-weighted-train-surface-modal-v1`
- launch commit: `efc5799`
- samples SHA-256:
  `daeb972b64e1a87c617afbd3f3907b00e4d56e78c0894792333ae351bc310923`
- metrics SHA-256:
  `50e199846eaf500b32a4f11bba4ebb7d2997298b9cf0a51fbe1e766b6d8f93da`

| immutable training surface | c5, 1:1 | c6, 4:1 |
|---|---:|---:|
| verified true | 0/21 | 21/21 |
| deterministic false | 21/21 | 0/21 |
| source-card recall | 9/21 | 8/21 |

c6 changed the answer prior but did not learn the truth-conditioned
distinction. It moved from always `no` to always `yes`. Because the
preregistered gate required at least 17/21 on both boolean surfaces, the full
169-item frozen evaluation was not run.

This result brackets the decision boundary between the 1:1 and 4:1 sampling
ratios. The next minimal intervention is a fresh-base 2:1 true:false ratio with
all other parameters and gates unchanged. More steps, sequential training, or
evaluation-derived wording would confound that comparison.

The gate used 151.94 seconds of model work on NVIDIA A10 and cost an estimated
USD 0.0527. Total c6 estimate: USD 0.2644. Neither adapter promotion nor
full-weight fine-tuning is authorized.
