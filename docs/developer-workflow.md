# developer workflow

Read this before adding experiment code. The repo has two layers that must not be confused.

This document owns platform/experiment mechanics. The research system contract is
[Project DELTA](./project-delta.md); its delivery status is the
[DELTA roadmap](./delta-roadmap.md).

## invariant

```text
Platform = shared dispatch + storage contract (many experiments, one cockpit)
Experiment = hypothesis + harness + configs + run matrix (you own this)
```

A worker can die. GitHub must still explain how to reproduce a run. B2 must still hold the artifacts.

## mental model (one picture)

```text
                         ┌─────────────────────────────────────┐
                         │  Mac cockpit: ./scripts/lab           │
                         │  src/lab/dispatch/*                 │
                         └──────────────┬──────────────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
              ▼                         ▼                         ▼
        configs/**/*.yaml         experiments/eN/          .env (secrets)
        (one run per file)        (your harness code)      (never commit)
              │                         │
              │    manifest + config ───┼──► B2 runs/{run_id}/
              │                         │
              ▼                         ▼
        ┌──────────┐              ┌──────────┐
        │  Modal   │              │  Lambda  │
        │ infra/   │              │ infra/   │
        │ modal/   │              │ lambda/  │
        └────┬─────┘              └────┬─────┘
             │                         │
             └────────────┬────────────┘
                          ▼
              src/lab/train_*.py  eval_*.py  (shared entrypoints today)
              experiments/eN/*.py (your entrypoints tomorrow)
                          │
                          ▼
              runs/{run_id}/  locally (cache)  +  B2 (source of truth for outputs)
```

**You write:** `experiments/`, experiment configs, eval JSONL, indexes, comparison reports.

**Platform already provides:** dispatch, manifest upload, remote bootstrap, B2 cp, timings, cost estimate.

Do not fork `dispatch/` per experiment. Do not put experiment logic inside `modal_driver.py`.

---

## what goes where

| path | layer | commits to git? | purpose |
|------|-------|-----------------|--------|
| `src/lab/dispatch/` | platform | yes | cockpit CLI, B2, Modal, Lambda, manifest, timing, costs |
| `src/lab/train_sft_lora.py` | shared entrypoint | yes | generic LoRA SFT (`--config`) — usable by any experiment until you replace it |
| `src/lab/eval_*.py` | shared entrypoint | yes | generic eval runners — same |
| `scripts/lab` | platform | yes | cockpit shell wrapper (`PYTHONPATH`, `set -a` `.env`) |
| `infra/lambda/` | platform | yes | Lambda worker bootstrap + `run_job.sh` |
| `infra/modal/` | platform | yes | Modal image + `modal run` target |
| `configs/runtime/` | platform | yes | Lambda/Modal/pricing defaults — not experiment-specific |
| `configs/evals/`, `configs/sft/` | legacy runs | yes | early smoke configs; new work uses `configs/experiments/` |
| `configs/experiments/{eN}/` | experiment | yes | one YAML = one dispatched run |
| `experiments/{eN}/` | experiment | yes | harness code, dataset builders, index builders, compare scripts |
| `data/` | experiment data | **no** (large files) | local copy of datasets; recipes live in git |
| `data/experiments/{eN}/` | experiment | recipes yes, blobs no | eval JSONL, retrieval indexes (build instructions in git) |
| `runs/` | artifact cache | **no** | local mirror of `B2 runs/{run_id}/` |
| `tmp/` | scratch | **no** | throwaway |
| `docs/experiments/` | experiment | yes | charter per experiment (hypothesis, matrix, metrics) |
| `artifacts/reports/` | experiment | yes (small) | cross-run tables; large dumps go to B2 `artifacts/reports/` |
| `.env` | secrets | **never** | B2, Lambda, HF tokens |

### B2 layout (canonical for outputs)

Current code uses a **flat** run namespace:

```text
s3://{S3_BUCKET}/
  runs/{run_id}/manifest.yaml
  runs/{run_id}/config.yaml
  runs/{run_id}/timings.json
  runs/{run_id}/cost.json
  runs/{run_id}/metrics.json
  runs/{run_id}/samples.jsonl
  runs/{run_id}/adapter/          # if train uploaded weights
  datasets/...                    # workers pull inputs
  indexes/{experiment_id}/...     # you add — RAG indexes per git SHA
  artifacts/reports/{experiment_id}/...
```

`docs/b2-tree.md` shows an aspirational tree (`runs/sft/`, `runs/evals/`). **Dispatch today writes `runs/{run_id}/` only.** New experiments follow the flat `run_id` key; use naming — not extra folders — to group.

---

## what can die

| thing | OK to lose? |
|-------|------------|
| Lambda instance | yes — terminate after job |
| Modal container | yes |
| `runs/` on Mac | yes — re-pull from B2 |
| `tmp/` | yes |
| in-flight GPU time | yes (money spent, lesson kept) |
| uncommitted experiment code | **no** |
| only copy of adapter on worker disk | **no** — must be on B2 before terminate |
| only copy of eval JSONL on worker | **no** |
| `.env` | don't commit; keep backup offline |

---

## what must survive

| thing | where |
|-------|--------|
| experiment harness source | GitHub `experiments/{eN}/` |
| run config (exact YAML used) | GitHub `configs/experiments/{eN}/` + B2 `runs/{run_id}/config.yaml` |
| git SHA at dispatch time | B2 `runs/{run_id}/manifest.yaml` (`git_commit` field) |
| metrics, samples, adapter | B2 `runs/{run_id}/` |
| frozen eval set | B2 `datasets/experiments/{eN}/` + recipe in git |
| RAG index for SHA | B2 `indexes/{eN}/{sha}/` |
| cross-run report | git `artifacts/reports/` and/or B2 |

---

## naming (multi-experiment, multi-model, multi-condition)

Three IDs. Do not overload them.

| ID | scope | example | rule |
|----|-------|---------|------|
| `experiment_id` | whole study | `e1_vllm` | stable; maps to `experiments/e1_vllm/`, `docs/experiments/e1_vllm.md` |
| `condition_id` | factorial arm | `c0_base`, `c3_ft` | defined in experiment charter |
| `run_id` | one dispatched job | `e1-vllm-c0-base-eval-qwen35-08b-modal` | **unique globally**; appears in YAML `run_id:` |

Suggested `run_id` pattern:

```text
{eN}-{short}-{condition}-{model_slug}-{target}[-{suffix}]
```

Examples:

```text
e1-vllm-c0-base-eval-qwen35-08b-modal
e1-vllm-c3-ft-qwen35-08b-modal
e2-example-c0-base-qwen15b-lambda
```

**Parallel models:** same `condition_id`, different `model_slug` → different `run_id` → dispatch in parallel (separate terminals, or your own sweep script). Platform does not batch runs; you orchestrate the matrix.

**Parallel conditions:** independent `run_id`s; compare after via your `experiments/e1/compare.py` (you write).

---

## one run lifecycle (platform — already implemented)

```text
1. Mac: ./scripts/lab run --target {modal|lambda} --config <yaml>
2. cockpit writes runs/{run_id}/manifest.yaml locally
3. cockpit uploads manifest.yaml + config.yaml to B2
4. worker runs train or eval (inferred from config: training/lora → train, else eval)
5. worker uploads run outputs to B2 runs/{run_id}/
6. cockpit writes timings.json + cost.json locally, uploads to B2
```

Task inference (`manifest.infer_task`): config contains `training` or `lora` → `train`; else → `eval`.

Remote entry (`infra/lambda/run_job.sh`): pull config from B2 → pull dataset paths referenced in config → `python -m lab.<module> --config`.

**To add a new entrypoint:** you change `run_job.sh` or add a field in config that remote script respects — that is experiment integration work, not dispatch rewrite.

---

## developer workflow (what you do for a new experiment)

### step 1 — charter (docs only, no GPU)

1. Create `docs/experiments/{experiment_id}.md` — hypothesis, conditions, metrics, frozen SHA plan.
2. Create `experiments/{experiment_id}/README.md` — pointer to charter + file map you intend to create.
3. Do **not** touch `src/lab/dispatch/` unless fixing platform bugs.

### step 2 — harness (you code)

Typical files you add (names yours):

```text
experiments/eN_example/
  README.md
  build_eval.py          # source snapshots → eval JSONL
  build_index.py         # source snapshot → B2 index
  compare.py             # read runs/*/metrics.json → report
  lib/                   # optional helpers

configs/experiments/eN_example/
  c0_base_modal.yaml
  c1_rag_fresh_modal.yaml
  ...

data/experiments/eN_example/
  .gitkeep               # actual JSONL/indexes gitignored or in B2 only
```

### step 3 — data on B2

```bash
set -a && source .env && set +a
# upload eval set + indexes (your commands)
s5cmd --endpoint-url "${S3_ENDPOINT_URL%%[[:space:]]}" cp ...
```

Workers pull via paths in config YAML (`eval.path`, custom `retrieval.index_prefix`, etc.). Config schema is **yours** — document fields in experiment charter.

### step 4 — run matrix

```bash
./scripts/lab run --target modal --config configs/experiments/eN_example/c0_base_modal.yaml
./scripts/lab run --target modal --config configs/experiments/eN_example/c1_rag_fresh_modal.yaml
# parallel: open second terminal for another model/condition
```

### step 5 — compare (you code)

Read from `runs/{run_id}/` locally or pull with `./scripts/lab status` / s5cmd.

Write `artifacts/reports/eN_example.md` with the experiment's declared metrics.

### step 6 — credits

```bash
./scripts/lab credits runs
./scripts/lab credits summary
```

---

## config YAML contract (minimum today)

Every run config must have:

```yaml
run_id: <unique>
```

Train configs also need what `train_sft_lora.py` expects (`model`, `data`, `output`, `lora`, `training`).

Eval configs need what your eval module expects (`model`, `eval`, …).

**Experiment-specific fields** (RAG index path, condition tag, git SHA for index): add freely; your harness must read them. Document them in the experiment charter. Platform ignores unknown keys.

Recommended metadata:

```yaml
experiment_id: e1_vllm
condition_id: c0_base
model_id: <provider/model>
index_sha: abc1234
```

---

## platform vs experiment — boundary rules

| do | don't |
|----|-------|
| add `experiments/eN/*.py` | put e1 logic in `dispatch/cli.py` |
| add `configs/experiments/eN/*.yaml` | hardcode e1 paths in `lambda_driver.py` |
| extend `run_job.sh` to call your module | duplicate B2 upload logic in experiment code |
| reuse `b2.upload_file` from `src/lab/dispatch/b2.py` if importing in harness on Mac | commit datasets or adapters |
| document new config fields in charter | create `e2_*` forks of entire dispatch stack |

---

## commands (cheat sheet)

```bash
# storage gate
bash scripts/check_storage.sh

# dispatch one run
./scripts/lab run --target modal --config configs/experiments/<exp>/<run>.yaml
./scripts/lab run --target lambda --launch --terminate --config ...

# inspect artifacts
./scripts/lab status --run-id <run_id>
./scripts/lab credits summary

# lambda ops
./scripts/lab lambda list
./scripts/lab env build --target lambda --instance-ip <ip>
```

---

## relation to DELTA

| platform capability | status | DELTA use |
|---|---|---|
| storage and manifests | implemented | preserve datasets, evidence, and lineage |
| train → B2 → reload | implemented for LoRA | execute adaptation candidates |
| eval dispatch | partial for experiment-specific harnesses | execute BUILD environments |
| state promotion/rollback | absent | required by VERIFY/MEMORY |

The module roadmap and phase completion gates live only in
[delta-roadmap.md](./delta-roadmap.md).

---

## assistants (Cursor / agents)

Agents help with: docs, reviews, debugging platform bugs, explaining errors.

Agents do **not** implement whole experiment harnesses unless you explicitly ask for a small slice.

You own `experiments/` so you learn the harness.

### AI doing the work vs you in the loop

| mode | who | examples |
|------|-----|----------|
| **you in the loop** | default | `build_corpus.py`, eval design, reading `samples.jsonl`, thesis, `compare.py` |
| **AI executes** | platform + explicit ask | `./scripts/lab run`, Modal/Lambda dispatch, B2 sync, re-run after infra bugs |
| **AI built while you were out** | flag it | e1 eval harness + c0/c3 Modal runs (Jul 2026) — results are valid; ownership of *why* the harness looks like this is catch-up work |

**The exceptional part:** one config → job on Modal or Lambda → artifacts on B2. That can run unattended.

**The learning part:** experiment code and interpretation. If AI did that slice without you, note it and revisit before trusting the narrative.

---

## further reading

- [the split](./the-split.md) — Mac / GitHub / B2 / workers
- [b2 tree](./b2-tree.md) — storage intent (flat `runs/{run_id}` is current reality)
- [Project DELTA](./project-delta.md) — system contract
- [DELTA roadmap](./delta-roadmap.md) — active delivery order
- [experiments index](./experiments/README.md) — experiment registry
- [e1_vllm](./experiments/e1_vllm.md) — active vertical slice
