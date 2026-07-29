# Symbol→evidence bridge (no FT)

**Date:** 2026-07-28
**Bank:** sealed `v0.22.0→v0.23.0` / `e1_eval_factory_v1`

## Invariant

```text
Deployable retrieval must not use probe-bound oracle spans.
Path hit ≠ span hit ≠ QA accuracy.
Expanding siblings in the same file cannot fix a wrong-file miss.
```

## What goes where

| artifact | role |
|---|---|
| `measure_symbol_evidence_bridge.py` | path-overlap measurement |
| `mode=symbol_expand` in `eval_vllm_qa.py` | seed symbol files → sibling chunks |
| this report | bridge verdict |

## Path overlap vs gold evidence files

| mode | path hit | changed path hit | removed path hit |
|---|---:|---:|---:|
| symbol | 0.684 | 0.375 | 0.000 |
| symbol_expand | 0.684 | 0.375 | 0.000 |

Expand does **not** find new files — only more chunks in files symbol already picked.

## QA (0.8B)

| condition | exact | changed |
|---|---:|---:|
| base × symbol | 0.424 | 0.000 |
| base × symbol_expand | 0.392 | 0.125 |
| base × evidence (oracle) | 0.724 | **0.375** |
| delta-diet × symbol | 0.841 | 0.250 |
| delta-diet × symbol_expand | 0.848 | 0.250 |
| delta-diet × evidence | 0.815 | **0.625** |

## Verdict

File-expand is **not** the bridge. On the diet backbone it is ~flat; on base it trades pooled accuracy for a tiny `changed` bump still far below oracle. The remaining gap is **wrong file on transitions** (esp. removed/changed) plus **span selection inside the file** — not “give me more of the same file.”

## What can die

- symbol_expand as a default intervention

## What must survive

- path-hit table (symbol ≈ expand)
- QA flat/negative vs evidence
- next lever: better **file routing** for deltas, then span ranking — still no FT required to test

## Command

```bash
PYTHONPATH=src .venv/bin/python experiments/e1_vllm/eval_factory/measure_symbol_evidence_bridge.py \
  --probes data/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1/probes_eval.jsonl \
  --evidence-map data/experiments/e1_vllm/indexes/factory_v1_evidence_map_v022_v023.json \
  --corpus data/experiments/e1_vllm/corpus_code_8.jsonl \
  --index data/experiments/e1_vllm/indexes/code_8_bm25.json \
  --symbol-index data/experiments/e1_vllm/indexes/symbol_code_8_v022_v023_train.json \
  --out artifacts/reports/eval_factory_symbol_evidence_bridge.json
```
