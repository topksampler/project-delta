# TruthSource v2 — results (plain English)

**Invariant:** Same model, same questions on the shared “bridge” facts. Only the *labeling authority* (docs vs code) differs in metadata; the probe text is frozen identical across arms.

**Command that produced this:** three parallel Modal evals on H100, then `compare_arms.py`.

## Decision (pre-registered rule)

**Docs-only for now.** Adding code (careful or thorough) did **not** change the shared knowledge map.

| Pair | Gap overlap (Jaccard) | Status flip rate |
|------|----------------------:|-----------------:|
| Docs vs careful code | 1.00 | 0.00 |
| Careful code vs big code dump | 1.00 | 0.00 |
| Docs vs big code dump | 1.00 | 0.00 |

Honesty on “does this fake thing exist?” was **99%** on every arm — same.

## Why “careful vs thorough” looked the same

On the **shared** questions (flags that appear in both docs and code), the three arms asked the **exact same** wording after the fix. Same model + same questions + same gold ⇒ same statuses. Thorough code only adds *extra private* symbols; it does not rewrite the shared quiz.

## Why the first (v1) run lied

The “which flag is real?” questions used different multiple-choice foils per arm. That made docs vs code look different even when the underlying fact was the same. **v2 freezes one shared quiz for the bridge.**

## What the model actually looks like on the shared map

On the 75 shared bridge claims (same questions):

| status | count |
|--------|------:|
| known | 23 |
| partial | 17 |
| unknown | 31 |
| unstable | 4 |

So: strong on “does this flag exist?”, weak on “what does it mean?” — same hole you cared about.

## Artifact

`artifacts/reports/truthsource/compare_v2.json` (and `compare.json` after this run).
