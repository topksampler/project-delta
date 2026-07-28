# TruthSource v2 — plain English

**Question:** On the same shared facts, does labeling truth from docs vs careful code vs lots of code change what the model looks like?

**Answer:** No. Same questions → same answers. Per the rule we locked before running: **docs-only for now.**

## Shared map (75 facts)

| status | count |
|--------|------:|
| known | 23 |
| partial | 17 |
| unknown | 31 |
| unstable | 4 |

Fake “does this exist?”: **99% honest** on every arm.

## Why careful vs thorough matched
Thorough only adds extra private symbols. It does not change the shared quiz.

## Why the first run lied
Multiple-choice wrong options differed by arm. Fixed: one frozen shared quiz.

## For your data wheel
```text
profile → find holes → make train data for holes → LoRA → profile again
```

Fair takeaway: fill **meanings** (and keep honesty). Putting code into the profile claim bank did not move closed-book scores on shared facts. Code still matters later as a **checker** of real behavior — different experiment.

Artifact: `artifacts/reports/truthsource/compare_v2.json`
