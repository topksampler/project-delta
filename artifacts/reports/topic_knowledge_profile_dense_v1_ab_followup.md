# A+B follow-up — paraphrase holdout + eval_v3 transfer

## A — memorization check

Same 85 verified claims, **0 exact question overlap** with `train_v0.22.0_v5`.
New templates + reshuffled recognition options.
Run: `e1-vllm-profile-dense-v1-para-ft-v5-modal`

| form | base | FT train wording | FT paraphrase |
|------|-----:|-----------------:|--------------:|
| recognition | 34.5% | 97.6% | **98.4%** |
| accept true | 95.6% | 97.6% | 97.2% |
| reject false | 23.7% | 96.8% | 96.4% |
| honesty fail | 95.3% | 11.8% | 12.9% |

Train→paraphrase deltas ≈ 0. Gains survive new wording. Not surface memorization.

## B — eval_v3 transfer

| condition | accuracy | notes |
|-----------|---------:|-------|
| c0 base | 23.9% | baseline |
| c3 mill v4 | 32.6% | prior mill mix |
| c3 profile wheel v5 | **21.7%** | below base; +4 confident_hallucination |

Profile-targeted v5 did **not** transfer to the hand `eval_v3` intervention set. It specialized the closed-book meaning/honesty manifold and regressed the broader QA eval.

## implication

```text
Profile loop works for its own manifold (and generalizes across paraphrases).
It is not yet a substitute for mill mixes aimed at eval_v3.
Next bet should either (1) blend v5 + mill rows, or (2) add eval_v3-aligned
holes into the profile wheel — not assume transfer.
```
