# Fingerprinting what a 0.8B model actually knows — Project DELTA, part 3

[Part 2](./delta-part-2-rag-vs-lora.md) ended on a confession: our 46-item
intervention eval could rank interventions, but it could not tell us what the
model *knows*. Those are different instruments. A scoreboard says "condition X
beat condition Y." A **knowledge profile** says "of the 85 documented flag
meanings in this corpus version, here is exactly which ones the model has,
which it lacks, and — the part nobody measures — whether it knows the
difference."

In this post, you will learn:

- how to build a dense, docs-grounded knowledge profile where the model under
  test never defines truth;
- the four-probe design that separates *knowing* from *sounding like you
  know*;
- why our 0.8B model's problem is not ignorance but **acquiescence** — and
  the 4B control that proved the instrument works;
- how we audited our own automatic grader before believing any of it.

We call the artifact a **TopicKnowledgeProfile**, and the practice knowledge
fingerprinting: given (model, corpus version), produce a verified map of
what's known. In DELTA terms this is the SENSE module growing real teeth.

## Claims first, probes second

Everything starts from **claims**, not questions. A deterministic extractor
walks the pinned v0.22.0 docs and proposes flag → meaning claims from the
structure index. Then comes the filter that makes the profile trustworthy:

```text
docs extractor proposed:   130 meaning claims
code existence check cut:   45 (Docker flags, launcher flags, pollution)
verified claim bank:        85 flag meanings
```

The 45 rejects are kept with reasons. The rule from part 1 — *the model never
defines truth* — has a corollary here: **the docs don't fully define it
either**. A claim survives only if the flag it names exists in the pinned
code's argparse source. Docs supply meaning; code verifies existence.

We also ran a pre-registered side study (TruthSource v2) on whether we needed
code-derived answer content too: for the flag-meaning claims both sources
share, docs-grounded and code-grounded evaluation produced identical maps
(Jaccard 1.0, zero status flips). Docs are sufficient ground truth *for this
claim type* — a scoped result, not a general license.

## Four probes per claim

Each of the 85 claims compiles into a probe family, three paraphrases per
form — 996 evaluated probes total:

| probe | asks | measures |
|-------|------|----------|
| free recall | "What does `--flag` do?" | fluency (advisory only) |
| recognition | 3-choice: right meaning vs two real-but-wrong ones | actual knowledge |
| accept-true | "Does `--flag` do ⟨true meaning⟩? yes/no/unknown" | agreement |
| reject-false | "Does `--flag` do ⟨another flag's meaning⟩?" | **semantic honesty** |

The reject-false probe is the one to steal. The distractor is not gibberish —
it is a *real, documented vLLM meaning* attached to the wrong flag. Rejecting
it requires knowing where one flag's semantics end and another's begin.
Accept-true and reject-false together give you an honesty axis that a
plain-accuracy eval simply cannot see.

## Result: fluent, agreeable, and at chance

Base Qwen3.5-0.8B, closed book, 85 claims, claim-cluster bootstrap CIs:

| behavior | accuracy | 95% CI |
|----------|---------:|-------:|
| free recall (advisory) | 76.3% | 67.9–84.3% |
| meaning recognition (3-choice) | **34.5%** | 27.3–42.2% |
| accepts true meanings | 95.6% | 92.0–98.8% |
| rejects/abstains on false meanings | **23.7%** | 17.3–30.5% |

Chance on recognition is 33.3%. Read those rows together and the model's
condition has a name:

> **Acquiescence.** Presented with any plausible, vLLM-flavored meaning, the
> model says yes — 95.6% of the time when it's true, 76.3% of the time
> (1 − 23.7%) when it's false. Recognition at chance says it cannot actually
> pick meanings out of a lineup. The 76% free-recall score isn't knowledge;
> it's fluency scored generously by keywords.

One more cut makes it precise. An earlier pilot showed the same model rejects
*invented flag names* ~99% of the time. So: **name-boundary honesty is
strong; semantic-boundary honesty is broken.** The model knows what a fake
flag looks like; it cannot tell when a real flag is wearing another flag's
meaning. That distinction — invisible to every standard QA eval we know of —
is the single most useful thing this instrument produced.

This also retroactively explains part 2's ceiling: D-class questions and the
`c4` abstention collapse were acquiescence wearing different costumes.

## Is the quiz just too hard? The 4B control

A profile that reads "chance" could mean the model is ignorant — or the
instrument is broken. So we ran the same 996 probes on Qwen3.5-4B:

| behavior | 0.8B | 4B |
|----------|-----:|---:|
| recognition | 34.5% | **89.6%** |
| rejects false meanings | 23.7% | **93.6%** |
| accepts true meanings | 95.6% | 22.9% |

Recognition at ~90% means the quiz is solvable from pretrained knowledge —
the instrument works; the 0.8B genuinely lacks the map. But look at the third
row: the 4B rejects *true* meanings 77% of the time. It has the opposite
disease — blanket skepticism. "Honesty" is not one scalar that improves with
scale; the two models sit at opposite ends of an acceptance-bias axis, and a
profile that only measured accuracy would have called the 4B *worse*.

Methodological footnote that cost us a run: the first 4B attempt burned its
entire token budget on `<think>` reasoning and scored near zero. The fix —
`enable_thinking: false` in the chat template — is now a config default.
Instrument checks go both ways.

## Auditing the grader before believing it

All 996 probes are scored automatically (choice parsing, yes/no/unknown
matching, keyword hits). Before building anything on the profile, we
hand-audited 50 stratified probes against the automatic labels:

- **42/50 agree.** Boolean (accept/reject) scoring: trusted.
- Recognition had a real parser bug — answers like a bare `B. ...` weren't
  credited. Fixed in the shared scorer; recognition numbers are post-fix.
- Free recall keyword scoring over-credits fluent near-misses. Verdict:
  **advisory forever** — it appears in profiles as context, never as a
  training gate or headline.

If your eval pipeline has an automatic grader that has never been audited,
you don't have results — you have a hypothesis about your grader. Fifty items
took an afternoon and changed which of our own numbers we cite.

## What a fingerprint is for

The profile costs $0.38 and 348 H100-seconds per (model, corpus) pair. For
that you get: an itemized map of 85 meanings × (known / unknown / dishonest),
per-claim status for targeting, and an honesty axis with confidence
intervals. In the DELTA loop this is SENSE output — and because it's
claim-indexed, it's *actionable*: every hole names exactly the training row
that should exist.

Limitations, so nobody over-reads: 85 claims cover one claim type
(flag→meaning) in one corpus version; code verifies flag *existence*, not
docs-span completeness; single model family; "honesty" here means rejecting
mismatched documented meanings, not general hallucination.

## Summary

- A knowledge profile is a different instrument from an intervention eval:
  claims from pinned docs, existence-verified against code, four probe forms
  per claim, model never defines truth.
- Base 0.8B: recognition at chance (34.5%), accepts ~96% of true meanings —
  and ~76% of false ones. The failure mode is **acquiescence**, not silence.
- The 4B control (89.6% recognition) proves the instrument is solvable — and
  reveals the opposite bias (rejects true meanings), so honesty must be
  measured as an axis, not a score.
- Audit your grader. Ours was 84% right, and the 16% changed real numbers.

Part 4 closes the loop: we turn the 85-claim hole map into training data,
LoRA the holes shut, and discover that the obvious way to also win the
intervention scoreboard quietly re-breaks the honesty we just fixed.
