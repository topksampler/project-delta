# delta_v2 boolean-margin diagnostic — Modal v1 audit

Status: `implemented`; diagnostic complete, candidate promotion unauthorized.

Loop position: the c5, c7, and c6 supervised-LoRA `CandidateState`s have failed
their conditional training-surface gate and rolled back to the pinned base.
DELTA is at `DECIDE`, before a new `InterventionPlan`. This diagnostic measures
why the rejected ADAPT candidates failed; it performs zero optimizer steps and
does not create a candidate.

The frozen diagnostic measures the two boolean training surfaces without
generation parsing:

- teacher-forced gold-answer negative log likelihood;
- `yes` minus `no` sequence log-probability margin;
- truth-signed margin and accuracy;
- within-source true-versus-false pair separation.

No pooled overall score is defined. The two boolean surfaces remain separate.

## execution and integrity

All four runs used the pinned Qwen/Qwen3.5-0.8B revision, the same 42
verified-true and deterministic-false rows, deterministic eager bfloat16
execution, and NVIDIA A10 compute. Each manifest records launch commit
`6a96ec4c4db93bfec78449abcd9bcc2d79eaf1ee`; each run performed zero model
updates.

- protocol SHA-256:
  `57b79d276d869e85b9323c6bc1730a540bb89503e9628115d85fd1c685c8da2e`
- pair-audit SHA-256:
  `d804c9ea2d03cb2791bffd31edb5fe221032facf5ae1d759c6fdc4119661ff42`

| condition | config SHA-256 | metrics SHA-256 | samples SHA-256 |
|---|---|---|---|
| pinned base | `5009fd9136cbaa6a385ed48c602aba39c240b2e83b024bf163b4a320e1344b69` | `f11459b75bb60771e7ea2da2ffed51c9414a74a34395e8e7d3808f29c1c60598` | `88a5ef09a540705c695f82a577a4516faa2b8acfbf3e5ae628d6f2b0a402b5eb` |
| c5 LoRA, 1:1 | `41331d42f4224027afe61ba22325480b55f13acfe6680888a1564950eeb331a8` | `2bca25ac48ec9c11f47a175d4099d6c90ad4ef356793144c5b908d873eb4390d` | `cc994eaa7b0240576879f2aff3f0f8e3508fa834255567d743758590d4dc0b88` |
| c7 LoRA, 2:1 | `c2d3deb2120d749045820e6cb066b0a5da8fa2053493d1c608825cf96527f908` | `c2bea95b5ad82adfc09f960fd7a68ecd15068aceabbc9b1eddda5c773c5acc43` | `9ca3c002c81887d15be4fd21a633e9fdcb704716ad1209350bb56ad73a30bd29` |
| c6 LoRA, 4:1 | `6890dfdb3c8bec05ab0356f094695aa3dd696cc354dee80cd695abf6062195de` | `c6c61fa5b6a7567c6e509d0defd9161ed8fdeb115d8a9e56f420cedb2ea9ac3e` | `082ae242965a581263c3d6436b65075d42a7f9bd2e670b98a59db32b8698e1fb` |

The config, metrics, samples, and pair-audit hashes were independently
recomputed after pulling the uploaded artifacts and matched every run receipt.

## results

Positive margin means the model prefers `yes`; negative margin means it prefers
`no`.

| condition | true correct | true mean margin | false correct | false mean margin | mean pair separation | truth-conditioned pairs |
|---|---:|---:|---:|---:|---:|---:|
| pinned base | 9/21 | -0.2203 | 16/21 | -0.2500 | +0.0298 | 4/21 |
| c5 LoRA, 1:1 | 0/21 | -0.3094 | 21/21 | -0.4821 | +0.1726 | 0/21 |
| c7 LoRA, 2:1 | 21/21 | +0.7739 | 0/21 | +0.5061 | +0.2679 | 0/21 |
| c6 LoRA, 4:1 | 21/21 | +1.4883 | 0/21 | +1.4049 | +0.0833 | 0/21 |

The base weakly prefers `no` on both surfaces. c5 pushes both surfaces farther
toward `no`; c7 pushes both toward `yes`; c6 pushes both strongly toward `yes`.
No trained adapter places the true item above zero and its paired false item
below zero for even one of the 21 sources.

Teacher-forced losses tell the same story. From c5 to c7 to c6, true-answer NLL
falls from 0.8635 to 0.3860 to 0.2068 while false-answer NLL rises from 0.4857
to 0.9926 to 1.6274. The sampler ratio controls a shared label intercept rather
than producing a truth-conditioned discriminator. More steps under the same
objective are therefore not justified.

## pair-construction audit

The true and false prompts have an identical outer template, identical contract
key order, and no explicit label leak. Every false card, however, changes only
the `annotation` field. The 21 false annotations contain only 15 `str` and 6
`int` values; the true annotations span ten value families. The audit status is
`pass-with-narrow-negative-warning`.

That warning blocks an immediate paired-objective training run on this frozen
bank. A contrastive loss could improve the measured pair margins by detecting
the annotation-value distribution rather than comparing the factual contract.
The next BUILD step must freeze a source-grounded paired view with multiple
corruption families and an explicit shortcut audit before DECIDE may authorize
one objective-change LoRA candidate.

Steps, learning rate, and rank remain frozen. They become eligible one factor at
a time only after a clean paired candidate shows truth-conditioned separation.
QLoRA, full-weight fine-tuning, and verifier-grounded RL remain ineligible until
their separate contracts and safety or verification gates are frozen.

## cost

The four diagnostic runs cost an estimated USD 0.0650 in total. They performed
zero optimizer steps and are not training candidates.
