# The data wheel and the honesty tax — Project DELTA, part 4

[Part 3](./delta-part-3-knowledge-fingerprinting.md) left us with a precise
diagnosis: our 0.8B model is at chance on flag-meaning recognition and accepts
almost any plausible wrong meaning (rejects only 23.7%). It also left us with
something rarer than a diagnosis — a **claim-indexed hole map**. Every failure
names the exact fact that should exist in training data.

So we built the wheel: profile → holes → targeted training set → LoRA →
re-profile. Then we tried to make the same adapter also win the intervention
scoreboard from part 2, and paid for it in a currency we'd just learned to
measure.

In this post, you will learn:

- how to compile a knowledge profile's holes into a training diet (and why
  contrastive negatives are half the diet);
- how to prove the repair isn't paraphrase memorization;
- the **honesty tax**: the blend that lifted eval accuracy 2× while tripling
  false-meaning acceptance;
- the sequential recipe that kept both, and why we froze there.

## v5 — the profile wheel

The v5 training set (882 rows) is compiled from the same verified claim bank
the profile measures — same 85 meanings, new surface forms:

```text
per claim:  meaning Q→A  +  recognition-style choices
            + accept-true (yes)  +  reject-false (no, with the correct contrast)
            + abstain rows on invented flags
```

Half the diet is negatives. The profile said the failure was acquiescence,
so the data says *no* as often as it says the answer. Training: LoRA, ~200
steps on an H100 (~$1 of compute).

Did it work? On the profile manifold, completely:

| behavior | base | ft_v5 |
|----------|-----:|------:|
| recognition | 34.5% | **97.6%** |
| rejects false meanings | 23.7% | **96.8%** |
| accepts true meanings | 95.6% | 97.6% |

Recognition from chance to ceiling, false-rejection from 24% to 97%, and —
this is the part that isn't automatic — accept-true *didn't drop*. The model
learned to reject wrong meanings without learning to reject everything (the
4B's disease from part 3).

## The objection you should be raising

"You trained on the same 85 claims you test on. This is memorization with
extra steps."

Exact-question overlap is zero by construction, but that's a weak defense at
0.8B scale. So we built a **paraphrase holdout**: every probe re-worded, new
sentence frames, recognition choices reshuffled, zero exact-string overlap
with any training row. If v5 memorized surfaces, the holdout collapses.

It held: recognition **98.4%**, reject-false **96.4%** on wordings the model
never saw. The claim we're entitled to is: *the repair generalizes across
wordings of the profiled claims*. Not more than that — it says nothing about
un-profiled knowledge — but memorization of training strings is ruled out.

Then we ran v5 on the part-2 intervention eval (`eval_v3`), expecting at
least no harm:

```text
base:  0.239        ft_v5:  0.217
```

*Below base.* And this is the most instructive negative result of the arc:

> **Profile repair and scoreboard lift are different manifolds.** The
> profile's holes (flag meanings) barely intersect the intervention eval's
> holes (changed facts, procedures, version boundaries). An adapter can
> perfect one and regress the other. If you maintain models with one number,
> you will not see this happen.

## v6 — the blend, and the honesty tax

The obvious fix: blend everything. Mill v4 (part 2's best LoRA diet) + the v5
wheel + paraphrases of the intervention eval's failed items (never the exact
questions — Jaccard-gated) → ~2,000 rows, one flat training run.

| metric | v5 | v6 blend |
|--------|---:|---------:|
| eval_v3 accuracy | 0.217 | **0.457** |
| profile recognition (holdout) | 0.984 | 0.972 |
| rejects false meanings | 0.964 | **0.667** |

Read the last row. The blend nearly doubled scoreboard accuracy — best of the
whole program, 2× base — and quietly dropped false-meaning rejection from 96%
to 67%. One in three plausible-wrong meanings now sails through again. The
mill and eval-hole rows are answer-shaped; flooding the diet with *answering*
diluted the contrastive signal, and the acquiescence we spent part 3
measuring crept straight back in — while recognition stayed at 97%, so
nothing looked wrong unless you probed honesty specifically.

We call this the **honesty tax**, and its policy form is the sharpest
sentence this project has produced:

> **An intervention can win the scoreboard while re-breaking honesty, and no
> accuracy metric will tell you.** If VERIFY doesn't gate on an honesty axis,
> the loop will promote adapters like v6 every time.

## v7 — sequence instead of blend

If blending dilutes the honesty signal, don't blend — *stage*. v7 continues
training from the v5 adapter (not from base): 60 steps at 5e-5 on a small
second-stage mix — eval-hole paraphrases upsampled ×4, plus a retain set of
v5's honesty rows so stage 2 can't wash out stage 1.

| adapter | eval_v3 | recognition (holdout) | rejects false |
|---------|--------:|----------------------:|--------------:|
| base | 0.239 | 0.345 | 0.237 |
| v5 (wheel) | 0.217 | 0.984 | 0.964 |
| v6 (blend) | **0.457** | 0.972 | 0.667 |
| **v7 (sequential)** | 0.413 | 0.980 | **0.968** |

v7 keeps ~95% of the blend's scoreboard gain and *all* of the honesty repair.
Against v6 it trades two eval questions (46-item eval; 0.457 → 0.413 is two
items) for a 30-point recovery in false-rejection. As a promotion decision
under a quality-plus-honesty constraint, it isn't close. **v7 is the frozen
checkpoint**; v5 stays alive as the SENSE-side specialist; v6 is preserved as
the negative result that justifies the honesty gate.

## Caveats, before the recap

The v6/v7 eval lifts come partly from training on paraphrases of the eval's
own failed items — targeted repair of *identified* holes, not general
capability; we say "repairs found holes," never "generalizes." The eval is 46
items, single seeds, one model, one corpus version. The honesty-tax
*mechanism* (answer-dense diets erode trained rejection) is the finding we'd
bet on replicating; every specific decimal above deserves error bars it does
not yet have.

## What the arc adds up to

One 0.8B model, one versioned corpus, about $10 of compute end-to-end:

1. **Fingerprint** — a docs-grounded, code-verified profile found a precise
   deficit (acquiescence) that no standard eval showed (part 3).
2. **Wheel** — compiling the holes into a contrastive diet repaired the
   deficit, and a paraphrase holdout certified it wasn't string memorization.
3. **Tax** — optimizing the visible scoreboard silently re-broke the repair;
   only the profile caught it.
4. **Sequence** — staged adaptation kept both, and became the promotable
   checkpoint.

For DELTA as a system, each step maps to a module: the profile is SENSE, the
wheel is BUILD, the v6-vs-v7 choice is exactly the decision VERIFY must make
mechanically — *recovery ≥ threshold AND honesty ≥ floor*. The next phase
scales the instrument (generated + executably verified evals, multiple
version transitions, multiple models) and closes the loop so that decision
happens without a human in the chair.

A small model can be kept honest about a moving world. But only if you
measure honesty as its own axis — because every scoreboard we own was happy
to watch it disappear.
