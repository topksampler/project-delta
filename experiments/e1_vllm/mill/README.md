# mill

`(repo, tag) → typed train rows + manifest`

DELTA BUILD dataset factory for `e1_vllm`. Generators are plugins; the pipeline
is version-agnostic. Start here; promote shared pieces to `src/lab/mill/` only
when a second experiment needs them.

## invariant

```text
One tag per mill run. Cross-tag delta is a different product.
Every row is corpus-anchored or explicitly synthetic-from-corpus.
Eval fixtures are a denylist input — never generator seed.
Teacher (if any) proposes surface form; corpus defines ground truth.
```

## stages

```text
0 pin       repo + tag → snapshot_meta.json
1 extract   snapshot → corpus.jsonl + structure_index.json
2 generate  structure + corpus + recipe → candidates (T1–T7)
3 gate      validate + dedupe + eval denylist + balance
4 emit      train_*.jsonl + manifest + rejected.jsonl
```

## status

```text
implemented  pin, extract, generate (T1–T7), gate, emit, build
```

T7 is default-capped at 200 candidates so it does not drown the mix.

## train classes

| id | skill | c3@doc_0 |
|----|-------|----------|
| T1 | declarative Q→A | yes |
| T2 | procedural | yes |
| T3 | version-conditioned | yes |
| T4 | abstain | yes |
| T5 | negatives / contrast | yes |
| T6 | cite-augmented | yes |
| T7 | multi-hop | yes (capped) |
| T8 | delta (two tags) | never — separate entrypoint |

## what goes where

| path | role |
|------|------|
| `mill/` | pipeline code (this package) |
| `mill/generators/` | one module per train class |
| `configs/experiments/e1_vllm/mill_recipe_*.yaml` | mix %, gates, paths |
| `data/experiments/e1_vllm/snapshots/{tag}/` | local pin cache (die) |
| `data/experiments/e1_vllm/train_{tag}_vN.jsonl` | milled train set |
| `data/experiments/e1_vllm/manifest_{tag}_vN.json` | counts, SHAs, recipe hash |

## what can die

- local `snapshots/` clones
- `candidates/` shards and `rejected_*.jsonl`
- teacher drafts that fail gates

## what must survive

- recipe yaml + generator version pins
- `snapshot_meta` (`tag`, `commit_sha`, `content_sha`)
- `manifest_*.json` next to every frozen train set
- gate rules (especially eval denylist)

## command

```bash
python experiments/e1_vllm/mill/cli.py build \
  --repo https://github.com/vllm-project/vllm.git \
  --tag v0.22.0 \
  --eval-denylist experiments/e1_vllm/fixtures/eval_v3.jsonl
```

Or step by step: `pin` → `extract` → `generate` → `gate` → `emit`.

Train artifact: `data/experiments/e1_vllm/train_v0.22.0_v4.jsonl` + matching manifest.
