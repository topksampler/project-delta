# e1_vllm — Phase B drift baseline

## invariant

```text
DriftEvent compares the same eval_v3 probes under two conditions.
Stable knowledge = class A. Changed knowledge = classes B and D.
A file diff alone is not drift; this v1 artifact is behavioral (+ deferred corpus).
```

## what goes where

- DriftEvent JSON: `artifacts/reports/drift_events/`
- this report: `artifacts/reports/e1_vllm_drift_baseline.md`
- generator: `experiments/e1_vllm/sense_drift.py`

## what can die

- local metric/sample caches used to rebuild events

## what must survive

- frozen eval_v3 + cited run_ids
- DriftEvent JSON + this baseline report

## events

### `c0_base` → `c3_ft`

- baseline run: `e1-vllm-c0-base-eval-v3-qwen35-08b-modal`
- probe run: `e1-vllm-c3-ft-mill-v3-eval-v3-qwen35-08b-modal`
- accuracy_delta: **0.0869**
- class deltas A/B/C/D: {'A': 0.225, 'B': 0.0, 'C': 0.0625, 'D': 0.0}
- stable (A) delta: 0.225 (regression if negative)
- changed B/D deltas: 0.0 / 0.0
- probe_failures listed: 30

### `c0_base` → `c1_rag_fresh`

- baseline run: `e1-vllm-c0-base-eval-v3-qwen35-08b-modal`
- probe run: `e1-vllm-c1-rag-fresh-eval-v3-qwen35-08b-modal`
- accuracy_delta: **0.1956**
- class deltas A/B/C/D: {'A': 0.25, 'B': 0.2, 'C': 0.1875, 'D': -0.125}
- stable (A) delta: 0.25 (regression if negative)
- changed B/D deltas: 0.2 / -0.125
- probe_failures listed: 25

## reading

```text
c0→c3: LoRA@doc_0. Expect A/C up or flat; B/D not recovered (stale diet).
c0→c1: fresh RAG. Expect B up; D mixed (negative-existence contamination).
```

## command

```bash
python experiments/e1_vllm/sense_drift.py \
  --metrics-dir /tmp/e1_compare \
  --out-dir artifacts/reports/drift_events \
  --report artifacts/reports/e1_vllm_drift_baseline.md
```
