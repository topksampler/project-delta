# Version-diff file prior (no FT)

**Date:** 2026-07-29
**Bank:** sealed `v0.22.0→v0.23.0`

## Invariant

```text
Diff allowlist is computable from tags alone.
Covering gold files ≠ retrieving the right span ≠ QA win.
Restricting BM25 to diff paths can raise transition path-hit and wreck stable.
```

## What goes where

| artifact | role |
|---|---|
| `build_version_diff_paths.py` | hash `.py` between snapshots → `diff_paths` |
| `indexes/version_diff_paths_v022_v023.json` | 647 paths (90 added / 37 removed / 520 changed) |
| `mode=symbol_diff` | symbol candidates, prefer diff paths |
| `mode=diff_bm25` | BM25 hits filtered to diff paths |

## Coverage (offline)

Gold evidence paths ∈ diff set: **100%** of added/changed/removed probes; **78%** of stable.

Path hit of retrieved top-k vs gold files:

| mode | overall | changed | removed | added | stable |
|---|---:|---:|---:|---:|---:|
| symbol | 0.684 | 0.375 | 0.000 | 1.000 | 0.688 |
| symbol_diff | 0.686 | 0.375 | 0.000 | 1.000 | 0.689 |
| diff_bm25 | 0.338 | **0.500** | **0.333** | 0.625 | 0.332 |

## QA (0.8B)

| condition | exact | changed |
|---|---:|---:|
| base × symbol | 0.424 | 0.000 |
| base × symbol_diff | 0.441 | 0.000 |
| base × diff_bm25 | 0.218 | 0.000 |
| delta-diet × symbol | 0.841 | 0.250 |
| delta-diet × symbol_diff | 0.844 | 0.250 |
| evidence oracle | 0.724 | **0.375** |

## Verdict

Diff prior is a **true** statement about where transition gold lives, but the naive uses fail as interventions: `symbol_diff` ≈ symbol; `diff_bm25` improves some transition path-hits while collapsing pooled QA. Need better **ranking inside the allowlist** (or entity→diff-file join), not raw restriction.

## What can die

- `diff_bm25` as a default recipe

## What must survive

- 100% transition gold-path ∈ diff set
- QA flat/negative table above

## Command

```bash
.venv/bin/python experiments/e1_vllm/eval_factory/build_version_diff_paths.py \
  --before data/experiments/e1_vllm/snapshots/v0.22.0 \
  --after data/experiments/e1_vllm/snapshots/v0.23.0 \
  --out data/experiments/e1_vllm/indexes/version_diff_paths_v022_v023.json
```
