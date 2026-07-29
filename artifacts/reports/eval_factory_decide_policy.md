# DECIDE policy table (post diff-hunk)

**Date:** 2026-07-29
**Serves:** DELTA DECIDE — first measured action table from factory arc
**FT:** stopped unless explicit reopen

## Invariant

```text
Never promote on pooled exact alone.
Route by drift failure mass; interventions compose only when measured.
Oracle RAG is debug/ceiling, not deploy.
```

## Feature → action

Inputs are *observed* failure mass on the sealed factory board (0.8B, greedy).
Costs are relative (GPU train ≫ index rebuild ≫ closed-book eval).

| if you observe… | first action | do **not** | promote when |
|---|---|---|---|
| `changed`≈0 and `stable` high (existence LoRA trap) | **delta-diet** SFT (δ×40, stable_cap=200) | more balanced existence SFT; trust pooled exact | `changed` ≥ closed-book oracle floor **and** `stable` non-regression vs prior |
| `changed`/`removed` low; no adapter yet; can spend context | **diff-hunk BM25** on base | after-era-only BM25/docs RAG for removals; `diff_bm25` allowlist | transition lifts without collapsing the slice you care about (accept pooled drop if product is transition QA) |
| need upper bound / debugger | **oracle evidence** | ship oracle as product | n/a (not a promote path) |
| deployable name routing, mostly stable/existence | **symbol** (or code BM25) | claim it fixes `removed` | existence/`stable` lift; do not credit transitions |
| delta-diet already strong closed-book | keep closed-book **or** add retrieval only with a **router** | blind diet+diff_hunk stack (regressed `changed` 0.38→0.13 on v022→v023) | stack only if held-out `changed` ≥ closed-book diet |
| nothing moves `removed` with after-era corpus | switch corpus to **diff hunks** | expand-same-file / allowlist hacks | `removed` path-hit > 0 and QA moves |

## Cost ladder (DECIDE walk order)

```text
1. scoreboard split (stable / added / changed / removed) — free
2. fresh index / corpus choice (code vs diff-hunk) — cheap
3. delta-diet LoRA — GPU, paused by default after freeze
4. oracle evidence — debug only
5. post-training beyond diet — require explicit reopen
```

## Scoreboard glue

| board | gates |
|---|---|
| **factory** (this table) | transition interventions; diet; deployable retrieval |
| **eval_v3** | hand intervention compare; do not mix into factory promote |
| **TopicKnowledgeProfile** | meanings/honesty; do not substitute for `changed` |

## Measured anchors (v0.22→v0.23)

| action | exact | changed | note |
|---|---:|---:|---|
| base closed-book | 0.51 | 0.00 | |
| base × evidence | 0.72 | 0.38 | ceiling |
| base × diff_hunk | 0.10 | **0.50** | transition specialist |
| delta-diet closed-book | 0.82 | 0.38 | best balanced closed-book |
| delta-diet × diff_hunk | 0.65 | 0.13 | **interference** |
| delta-diet × symbol | 0.84 | 0.25 | weak transition add-on |

## Held-out transfer (v0.21→v0.22)

| action | exact | changed | note |
|---|---:|---:|---|
| base closed-book | 0.245 | 0.278 | |
| base × evidence | 0.642 | 0.667 | ceiling |
| base × diff_hunk | 0.134 | **0.500** | same `changed` as v023; no retune |
| delta-diet closed-book | 0.721 | **0.778** | best balanced |
| delta-diet × diff_hunk | 0.240 | 0.667 | pooled collapse again |

## Multi-release DECIDE demo (v0.22→v0.26, seed bank)

| stage | exact | changed | note |
|---|---:|---:|---|
| SENSE base CB | 0.118 | **0.069** | fire = changed |
| VERIFY base × diff_hunk | 0.172 | 0.347 | first prescription (fixed corpus) |
| VERIFY base × **hybrid** | **0.335** | **0.347** | upgraded prescription — pooled recovers too |
| oracle ceiling | 0.569 | 0.708 | debug only |

Full trace: `artifacts/reports/eval_factory_decide_v022_v026.md`.
Router: `artifacts/reports/eval_factory_hybrid_router.md`.

**Policy amendment (2026-07-29):** the "transition mass, no adapter" row now
prescribes **hybrid_diff** (entity-touched routing) instead of raw diff-hunk
BM25. Caveat: if closed-book pooled is already strong (≥ ~0.5), retrieval
taxes stable — compare against closed-book before promoting.

Full table: `artifacts/reports/eval_factory_diff_hunk_transfer.md`.

## What can die

- One-retriever-fits-all defaults
- Promote-on-pooled-exact
- Factory LoRA reopen without a new DECIDE question

## What must survive

- This table + stop-FT seal
- Diff-hunk = searchable corpus finding
- Diet recipe

## Command

```bash
# re-read measured inputs
cat artifacts/reports/eval_factory_diff_hunk.md
cat artifacts/reports/eval_factory_ft_stop_freeze.md
```
