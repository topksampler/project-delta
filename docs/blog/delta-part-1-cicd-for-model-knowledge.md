# Your model's knowledge is rotting — Project DELTA, part 1

You fine-tuned a small model on your product docs. It ships. Three weeks later
your project cuts a new release: two flags renamed, one config file moved, a
deprecation. Your model doesn't know. Worse — it doesn't know that it doesn't
know. It will answer questions about the old flag with total confidence,
forever, until someone notices.

Software teams solved a version of this problem decades ago. Code changes are
not "hope someone re-tests everything" events; they flow through CI/CD —
detect the change, build the artifacts, run the tests, promote or roll back.
Model knowledge gets none of that. Project DELTA is our attempt to build it.

In this post, you will learn:

- the DELTA loop and why "CI/CD for model knowledge" is more than a metaphor;
- how we built a fully pinned, provenance-bearing change environment from a
  real fast-moving project (vLLM, v0.22.0 → v0.23.0);
- how a deterministic "data mill" turns a docs snapshot into typed training
  data with contamination gates;
- what a frozen drift eval looks like when you refuse to let the eval move.

This is part 1 of a 4-part series. Parts 2–4 cover the intervention shootout,
knowledge fingerprinting, and the data wheel.

## The problem, stated precisely

Take a small expert model — small enough to run cheaply, ideally sub-1B — that
must stay correct about a source of truth that keeps changing. The research
question is not "can we make the model better." It is:

> When the source of truth moves from version X to version Y, how do we
> **detect** what the model now gets wrong, **build** evidence for exactly that
> knowledge, **choose** the cheapest intervention that repairs it, and
> **verify** the repair didn't break anything else — automatically?

That decomposition gives DELTA its module contract:

```text
world changes → SENSE → BUILD → DECIDE → VERIFY → promote or rollback
```

Each module has a typed, durable output: SENSE emits a `DriftEvent`, BUILD
emits an `EvalEnvironment` plus training candidates, DECIDE emits an
`InterventionPlan`, VERIFY emits a `PromotionDecision`. Nothing is promoted
without measured recovery *and* regression checks. Workers can die; source
snapshots, datasets, decisions, and promoted state must survive.

One invariant does most of the work in this program, so it's worth stating
early:

> **The model under test never defines truth.** Truth comes from a pinned,
> versioned source — documentation spans, and code where behavior is runnable.

Every result in this series traces back to that rule.

## Why vLLM docs as the first vertical slice

We needed a subject that (a) changes fast enough for drift to be real, (b) is
technical enough that answers are checkable, and (c) has ground truth in two
forms — prose docs *and* executable code. vLLM fits all three: its CLI surface
(flags, environment variables, config paths) evolves every release, and every
documented flag either exists in the argparse source or it doesn't.

We pinned two snapshots:

| snapshot | meaning |
|----------|---------|
| `doc_0` | vLLM docs @ **v0.22.0** — the "what the model was trained/adapted on" era |
| `doc_8` | vLLM docs @ **v0.23.0** — the "world moved" era; truth for new facts |

Pinning is literal: clone at tag, hash the content (`content_sha`), chunk into
a provenance-bearing corpus (`chunk_id` carries the tag), and store the whole
thing on B2. If you cannot reproduce the corpus byte-for-byte, nothing
downstream is trustworthy.

The probe model for the whole series is **Qwen3.5-0.8B** — deliberately small.
Part of DELTA's thesis is that CPU-viable expert models are worth maintaining;
the other part, as you'll see in part 3, is that small models fail in
*interesting* ways.

## The mill: deterministic training data from a snapshot

Hand-writing training data does not scale past one experiment, and
teacher-model-generated data has a provenance problem: if a big LLM invents
your training rows, your ground truth is whatever the big LLM believed. Our
compromise is a **data mill** — a deterministic extractor + generator pipeline
where teacher models may (later) propose surface forms, but every row must
survive mechanical gates.

The mill has three stages:

```text
pin      clone at tag → snapshot_meta.json + content hash
extract  docs → corpus.jsonl + structure_index.json (flags, env vars, paths)
generate T1–T7 typed candidates → gate → emit train_v{N}.jsonl + manifest
```

The seven generator types matter more than the implementation:

| type | skill it teaches |
|------|------------------|
| T1 | declarative flag → answer |
| T2 | procedural how-to |
| T3 | version-conditioned ("In v0.22, …") |
| T4 | abstain on invented flags |
| T5 | contrast ("not this flag") |
| T6 | answer + cite source path |
| T7 | multi-hop across two chunks |

T4 and T5 are the ones most teams skip: rows that teach the model to say *no*
and *unknown*. Hold that thought — the entire second half of this series is
about what happens when honesty is and isn't in the training diet.

Every candidate then passes through a gate: grounding checks (the answer span
must exist in the pinned corpus), deduplication, and an **eval denylist** — a
token-Jaccard filter (≥ 0.85) against the frozen eval questions, so training
data cannot simply contain the test. Rejected rows are kept in
`rejected.jsonl`; a gate you can't audit is a gate you can't trust.

The emitted artifact is a train JSONL plus a manifest binding every row to the
source revision and generator version. `train_v0.22.0_v4.jsonl` (~1,200 rows,
T1–T7) is the mill's main output for this arc.

An honest caveat, because this series will keep being honest: the denylist is
currently *optional* — a flag you pass, not a gate the emitter enforces. That
is on the hardening list, and we'd rather tell you than have you find it.

## The eval that refuses to move

Drift measurement dies the moment your eval drifts too. So `eval_v3` is small,
hand-authored, versioned, and **frozen**: 46 questions in four classes.

| class | n | tests |
|-------|--:|-------|
| A | 20 | stable facts (true in both versions) |
| B | 10 | what changed / only-in-v0.23 |
| C | 8 | procedural how-to |
| D | 8 | version-binary ("does X exist in v0.22?") |

The class structure is the point. A-class answers should *never* change — they
are the regression alarm. B-class answers *must* change with the world — they
are the drift signal. D-class questions test whether the model can say "no,
that doesn't exist in this version," which turns out to be the hardest thing
a small model does.

Scoring is deliberately mechanical (substring/choice/boolean matching, no
LLM-as-judge in the loop for the headline numbers), and every response gets a
failure-mode label: `correct`, `wrong`, `stale_version`, `abstain_wrong`, and
— added after we watched the base model calmly answer questions about
nonexistent features — `confident_hallucination`.

Forty-six questions is small, and we'll say so every time we show a number
from it. It was built to compare interventions on one version transition, not
to be a benchmark. Scaling this eval up (generated, executably verified,
hundreds of items per transition) is the next phase of the program.

## The platform underneath

None of this matters if runs aren't reproducible, so the boring parts are
load-bearing: a dispatcher (`./scripts/lab run`) that ships configs to Modal
GPUs, pushes every run's `metrics.json` + `samples.jsonl` to B2 under a stable
`run_id`, and a ledger that records cost. A dense profile run from part 3
costs $0.38 — 348 H100-seconds. A LoRA training run is ~460 seconds. The whole
research arc in this series cost less than a tank of gas, which is itself a
finding about doing drift research at 0.8B scale.

## Summary

- Model knowledge rots, and unlike code, nothing alerts you when it does.
  DELTA treats knowledge maintenance as CI/CD: SENSE → BUILD → DECIDE →
  VERIFY → promote or rollback, with typed durable artifacts at each stage.
- The first vertical slice pins vLLM docs at v0.22.0 and v0.23.0, mills
  deterministic typed training data with grounding and contamination gates,
  and freezes a 46-item classed eval (`eval_v3`) so the yardstick can't move.
- The model under test never defines truth. Pinned docs do — and where
  behavior is runnable, code does.

In part 2, we run the shootout everyone asks about: on a moving target, does
retrieval beat fine-tuning? The answer embarrassed our LoRAs.
