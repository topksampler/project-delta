# e1_vllm — thesis

This is the first empirical slice of [Project DELTA](../project-delta.md). DELTA
owns the closed-loop system claim; this document owns one research question and
its measurements.

**Research question (not “RAG vs LoRA”):**

> How do small language models handle **versioned technical knowledge**, and which adaptation strategy best preserves **version fidelity** when documentation moves?

vLLM v0.22.0 → v0.23.0 is the **vehicle**. The finding generalizes to any versioned API, changelog, or doc set.

---

## invariant

```text
Ground truth is knowable  →  pinned doc snapshots + corpus-validated eval gold
Methods are instruments   →  not the headline
Failure modes are data    →  primary output alongside accuracy
```

## what this is NOT

| saturated framing | why we skip it |
|-------------------|----------------|
| “RAG for facts, LoRA for style” | literature converged on “it depends” — no new information |
| “which method wins overall?” | one number hides when each method lies |
| politeness / format tuning | not version knowledge |

## contribution under test

```
1. Version delta as surgical knowledge update
   - Same product, two doc snapshots, measurable diff
   - Train on doc_0, eval on doc_8 → what breaks vs survives?

2. Failure mode taxonomy (publishable unit)
   - correct | wrong | abstain | stale_version | retrieval_miss | confident_hallucination

3. Version attribution
   - Does the model answer as v0.22 or v0.23 when both could apply?

4. Combination attribution
   - c4 (LoRA + RAG): what did each layer contribute?

5. Verifier reuse
   - If the failure labels become trustworthy, the same environment can later
     verify interventions and supply executable reward.
```

These are hypotheses, not established novelty claims. Related-work comparison and
experimental evidence must exist before publication language calls them novel.

---

## experiment object

```text
Corpus  : vLLM git docs @ v0.22.0 (doc_0) and v0.23.0 (doc_8)
Model   : sub-1B probe frozen in experiments/e1_vllm/snapshots.yaml
Task    : version-aware short-form Q&A (see eval task below)
Methods : c0 base | c1 RAG fresh | c2 RAG stale | c3 LoRA@doc_0 | c4 LoRA+RAG | c6 shuffle
Goal    : failure modes + version fidelity, not leaderboard accuracy
```

---

## eval task (decision)

**Use: structured short-answer Q&A (free-form generation, rubric scoring).**

| format | verdict |
|--------|---------|
| **Short free-form Q&A** | **yes — primary** |
| Multiple choice | no — hides calibration and attribution |
| Full code generation | later — Type C subset only |
| Extractive span | optional auxiliary metric, not primary |

**Prompt shape:**

```text
You are answering questions about vLLM documentation.
Answer in 1–3 sentences. If the docs for the requested version do not contain the answer, say "unknown".
Question: {question}
```

Optional later: `Requested doc version: 0.23.0` in prompt for attribution experiments.

**Per-item output stored:**

```json
{
  "id": "api_02",
  "answer": "...",
  "scores": {"content": 1, "attribution": 1, "abstain_ok": 1},
  "failure_mode": "correct"
}
```

---

## question taxonomy (eval_class)

Build eval around these — not `concept|api|config` alone (keep `type` as topic tag).

| class | name | example | current v2 coverage |
|-------|------|---------|---------------------|
| **A** | stable | “What flag sets tensor parallelism?” (true in both versions) | ~31 items (`requires_doc: 0.22.0`, gold in both corpora) |
| **B** | delta | “What changed in LLM Compressor doc layout 0.22→0.23?” | **weak** — add 8–10 |
| **C** | procedural | “How do I serve with LoRA enabled?” | **weak** — add 6–8 |
| **D** | version-sensitive binary | “Is moriio_connector_usage.md in v0.22.0 docs?” → no | **weak** — add 6–8 |
| **E** | adversarial / stale | same question as A but eval under train@doc_0 + no RAG | **condition**, not question — run c3 on class A |

**Type E is a run configuration:** train LoRA on doc_0, ask doc_8-only questions, measure stale_version rate.

---

## metrics (beyond accuracy)

| metric | definition |
|--------|------------|
| **content_accuracy** | `must_contain` rubric (current) |
| **version_attribution** | answer cites correct version OR `unknown` when only other version has truth |
| **abstain_quality** | said unknown when gold missing from model’s effective knowledge |
| **calibration** | confidence (if elicited) vs correctness — planned |
| **failure_mode** | label per sample (see taxonomy below) |

### failure_mode labels

```text
correct              — rubric pass
wrong                — rubric fail, fluent wrong answer
stale_version        — cites doc_0 fact on doc_8-only item (or vice versa)
retrieval_miss       — RAG run but gold strings only in non-retrieved chunks
abstain_correct      — unknown and gold expects abstain
abstain_wrong        — unknown but answer was knowable
confident_hallucination — specific false claim with no retrieval support
```

## interpretation boundaries

- The current doc_0 training builder teaches chunk summarization, while the eval
  asks short-answer Q&A. c3 results are provisional until train and eval task
  shapes align.
- The current scorer does not implement the complete failure taxonomy above.
- Retrieval conditions cannot be interpreted until an index, retriever, and
  retrieval trace exist.
- Documentation changes are a controlled proxy for world change. Generalization
  to other repositories or executable API behavior must be demonstrated.
- This experiment measures DELTA's BUILD/SENSE foundation. It does not establish
  an intervention policy or automatic promotion.

---

## what goes where

- this document: research question, eval classes, and metrics;
- `e1_vllm.md`: conditions and operational state;
- `e1_vllm_data.md`: corpus facts and validation rules;
- `experiments/e1_vllm/snapshots.yaml`: authoritative model and source pins.

---

## what can die

- “RAG vs LoRA” blog framing
- eval questions not in corpus (release-note trivia)
- accuracy-only tables without failure breakdown

## what must survive

- pinned doc_0 / doc_8 snapshots
- corpus-validated gold
- per-sample `failure_mode` in `samples.jsonl`

## command

```bash
python experiments/e1_vllm/inspect_data.py validate-eval eval_v2.jsonl
```
