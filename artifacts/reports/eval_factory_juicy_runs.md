# Juicy runs — symbol recovery + factory LoRA

## invariant

```text
Report by drift_type. Pooled accuracy can lie.
Oracle evidence ≠ non-oracle symbol routing ≠ LoRA on train surfaces.
eval_v3 remains a separate scoreboard.
```

## A — symbol/path retriever (v0.22→v0.23)

| condition | exact | changed |
|-----------|------:|--------:|
| closed-book | 0.512 | 0.000 |
| code BM25 | 0.096 | 0.125 |
| **symbol fresh** | **0.424** | 0.000 |
| symbol stale | 0.441 | 0.000 |
| evidence oracle | 0.724 | 0.375 |

Symbol ≫ naive code-BM25, but **below** closed-book on this bank and still << oracle. Not era-sensitive.

## B — same ladder on frozen v0.21→v0.22

| condition | exact | changed |
|-----------|------:|--------:|
| closed-book | 0.245 | 0.278 |
| **symbol** | **0.355** | 0.333 |
| evidence | 0.642 | 0.667 |

Here symbol **helps** vs closed-book (+0.11 exact). Method is bank-dependent.

## C — LoRA on factory train (pause lifted for this run)

Train: `e1-vllm-c3-ft-factory-v1-qwen35-08b-modal` · H100 · ~$0.54 · 300 steps · loss→0.028
Eval: `…-eval-…` closed-book on sealed factory eval

| | exact | stable | added | removed | **changed** |
|--|------:|-------:|------:|--------:|------------:|
| base | 0.512 | 0.581 | 0.725 | 0.750 | **0.000** |
| **factory LoRA** | **0.884** | **0.952** | 0.350 | 0.333 | **0.000** |

Juicy result: LoRA **memorizes stable/existence** (existence form 0.995) and **does not open the changed hole**. Added/removed regress vs base. Same moral as mill@doc_0 on eval_v3 B/D — diet without deltas cannot teach deltas.

## paths

- `artifacts/reports/eval_factory_symbol_recovery.json`
- `artifacts/reports/eval_factory_v1_lora_summary.json`
- adapter: `runs/e1-vllm-c3-ft-factory-v1-qwen35-08b-modal/adapter/`

## sequel (diet × evidence × RAG-SFT)

See `artifacts/reports/eval_factory_diet_lever_freeze.md`.

| condition | exact | changed |
|--|------:|--------:|
| LoRA delta-diet CB | 0.815 | 0.375 |
| LoRA delta-diet + evidence | 0.815 | **0.625** |
| RAG-SFT + evidence | **0.925** | **0.750** |
| RAG-SFT CB (no context) | 0.461 | 0.000 |

## what must survive

- symbol vs BM25 vs evidence ordering
- LoRA 0.884 with changed=0 (negative on version-delta)
- diet unlocks combo; RAG-SFT is context-conditioned
- explicit reopen of FT for factory-v1 only (v7 pause otherwise intact)
