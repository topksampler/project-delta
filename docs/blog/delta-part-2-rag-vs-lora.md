# RAG vs. LoRA on a moving target — Project DELTA, part 2

In [part 1](./delta-part-1-cicd-for-model-knowledge.md) we built the
environment: pinned vLLM docs at v0.22.0 (`doc_0`) and v0.23.0 (`doc_8`), a
deterministic data mill, and a frozen 46-item eval with stable (A), changed
(B), procedural (C), and version-binary (D) classes.

Now the question every applied team argues about in Slack: **when the world
moves, do you refresh the index or retrain the model?**

In this post, you will learn:

- how we structured a six-condition intervention matrix so the answer would be
  attributable, not vibes;
- why a *placebo* retrieval condition is the control most RAG evals are
  missing;
- the results — including the two failure modes that surprised us more than
  the headline;
- what any of this implies for an automated intervention policy.

## The matrix

Six conditions, one frozen eval, one model (Qwen3.5-0.8B), greedy decoding:

| id | recipe |
|----|--------|
| `c0_base` | base model, closed book |
| `c1_rag_fresh` | base + BM25 top-k from **v0.23.0** docs |
| `c2_rag_stale` | base + BM25 top-k from **v0.22.0** docs |
| `c3_mill_v3` / `c3_mill_v4` | LoRA trained on v0.22.0 mill data, closed book |
| `c4_ft_rag` | the v0.22.0 LoRA **plus** fresh v0.23.0 retrieval |
| `c6_shuffle` | base + *random* v0.23.0 chunks — the retrieval placebo |

Design notes worth stealing:

- **The placebo (`c6`).** If retrieval helps, is it the *relevant text* or just
  "any grounding context calms the model down"? Random chunks from the same
  fresh corpus answer that directly.
- **Stale retrieval (`c2`) is a real production state**, not a strawman — it's
  what every RAG deployment becomes the day upstream releases and nobody
  rebuilds the index.
- **`c4` tests the folk wisdom** that fine-tuning plus RAG is strictly better
  than either. It is not.
- BM25, not embeddings — deliberately. Lexical retrieval is reproducible from
  a JSON index with zero model dependencies. Establish the frontier with the
  simple thing first.

## Results

Accuracy on `eval_v3` (46 items), with per-class breakdown:

| condition | acc | A (stable) | B (changed) | C (how-to) | D (version) |
|-----------|----:|----:|----:|----:|----:|
| `c6_shuffle` | 0.17 | 0.20 | 0.00 | 0.38 | 0.50 |
| `c0_base` | 0.24 | 0.20 | 0.10 | 0.44 | 0.50 |
| `c3_mill_v3` | 0.30 | 0.42 | 0.10 | 0.50 | 0.38 |
| `c4_ft_rag` | 0.30 | 0.40 | 0.10 | 0.50 | 0.12 |
| `c3_mill_v4` | 0.33 | 0.45 | 0.10 | 0.56 | 0.25 |
| `c2_rag_stale` | 0.39 | 0.40 | 0.20 | 0.62 | 0.38 |
| **`c1_rag_fresh`** | **0.46** | 0.45 | **0.30** | 0.62 | 0.50 |

Six findings, in the order we came to trust them:

**1. Fresh retrieval wins, and wins where it matters.** `c1` leads overall and
is the *only* condition that moves B-class (changed facts): 0.30 vs 0.10 for
everything else. On drifted knowledge, index freshness beat every LoRA we
trained on the old corpus. This should not surprise anyone, and yet the size
of the gap — fine-tuning never touched B at all — is the number to remember.

**2. The placebo flunks, which validates the win.** Shuffled fresh chunks
(`c6`, 0.17) score *below* closed-book base (0.24). Irrelevant context
actively hurts a 0.8B model. So `c1`'s lift is retrieval *relevance*, not the
mere presence of documentation-shaped text. If your RAG eval doesn't have this
control, you don't know which one you're measuring.

**3. Stale RAG is better than nothing and worse than you think.** `c2` beats
base overall (0.39) but drags D-class below base (0.38 vs 0.50) — wrong-era
chunks talk the model *out of* correct version boundaries. A stale index
doesn't just miss new facts; it actively contaminates version reasoning.

**4. LoRA on old docs helps what it saw, not what changed.** `c3` lifts A and
C (facts and procedures present in its training corpus) but leaves B at 0.10.
Fine-tuning on v0.22.0 cannot conjure v0.23.0. Obvious in retrospect;
quantified now.

**5. LoRA + fresh RAG _interferes_.** The folk-wisdom stack (`c4`) matched
plain `c3_mill_v3` overall (0.30) but collapsed D-class to 0.12 and spiked
`abstain_wrong` to 11 items (vs 0–2 elsewhere). Our reading: the LoRA learned
v0.22-era answer habits, retrieval injected v0.23 text, and the model split
the difference into confused abstention. Composing interventions is not free;
they can fight.

**6. Even the winner has a ceiling.** `c1`'s D-class is flat vs base. Many
D-items require answering *no* ("that flag doesn't exist in v0.22"), and
handing the model fresh relevant text biases it toward *yes*. Retrieval
cannot fix a model that won't reject — a thread that becomes the whole story
in part 3.

## Cost, since DECIDE will need it

Eval-only conditions (`c0/c1/c2/c6`) cost ~100–140 A10G-seconds each. The LoRA
conditions each require a prior H100 training run (~200 steps, ~460 s for mill
v4). On this slice, the cheapest effective intervention for drift was also the
best one: rebuild a BM25 index — no GPU required at all. That ordering (no-op
→ context injection → index refresh → LoRA → post-training) is exactly the
cost ladder DELTA's DECIDE module will walk with thresholds instead of a
human.

## Caveats, stated plainly

Forty-six items, one version transition, one model, single seeds, substring
scoring. Class-level numbers (n = 8–20 per class) move by whole percentage
points per question. We treat this matrix as *directionally* locked — fresh
retrieval dominates stale-corpus fine-tuning for drifted facts; composition
interferes; irrelevant context hurts — and none of the deltas as
publication-precise. Scaling the eval is the program's next phase.

## Summary

- On a moving target, **fresh BM25 beat every stale-corpus LoRA**, and was the
  only intervention that recovered changed facts (B: 0.30 vs 0.10).
- **Random-context placebo scored below closed-book** — always run this
  control before crediting your retriever.
- **Stale RAG contaminates version reasoning** and **LoRA+RAG interfere** —
  intervention composition is a real design decision, not a free upgrade.
- The winner's ceiling is the model's inability to say *no* — which is not a
  retrieval problem, and not visible on a 46-item eval at all.

That last limitation is what part 3 is about: building an instrument that
measures what the model actually knows — nearly a thousand probes instead of
46 — and discovering our 0.8B model has a much stranger problem than
ignorance.
