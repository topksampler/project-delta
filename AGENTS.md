# lalith-ai-lab — agent notes

## workflow

- **Project DELTA** is the research program: CI/CD for model knowledge. Its
  invariant and module contracts live only in `docs/project-delta.md`; delivery
  status lives only in `docs/delta-roadmap.md`.
- **Platform** (`src/lab/dispatch/`, `scripts/lab`, `infra/modal/`, `infra/lambda/`) is shared across experiments. Do not embed experiment-specific logic there.
- **Experiments** live in `experiments/{id}/`, configs in `configs/experiments/{id}/`, charters in `docs/experiments/{id}.md`.
- `e1_vllm` is DELTA's first vertical slice, not an alternate name for the
  program. Keep its machine ID, run IDs, and B2 paths stable.
- User writes experiment harness code to learn. Assist with docs, review, small fixes, platform bugs — do not implement whole experiment pipelines unless explicitly asked.
- See `docs/developer-workflow.md`.

### documentation ownership

- Do not restate the DELTA loop, current phase, model ID, or storage contract in
  multiple documents.
- Experiment charters own hypotheses and condition matrices. Data guides own
  corpus/eval facts. Harness READMEs own commands.
- Use `implemented`, `partial`, `planned`, or `deferred`. Do not describe a
  planned module as if a closed loop exists.

### AI doing the work vs user in the loop

**Default:** user in the loop on `experiments/` — writes harness, interprets failures, owns hypotheses.

**Exception (explicit ask or platform):** dispatch, Modal/Lambda wiring, B2 pulls, re-running jobs, fixing platform bugs — fine for agents to execute while user is away.

**Current drift (e1_vllm, Jul 2026):** user stepped out; agent built `eval_vllm_qa.py`, eval configs, dispatched c0/c3 on Modal, fixed eval JSONL + ledger bugs. User was not in the loop for that slice. **Note this** when interpreting results — the harness is real but the learning debt is on experiment code, not on `./scripts/lab run`.

**Invariant:** launching jobs (Modal/Lambda) is platform — exceptional and safe to automate. *What* to run and *what it means* stays with the user unless they explicitly delegate.

## e1_vllm knowledge-profile guardrails (frozen Jul 2026)

**Invariant:** two scoreboards, never one. `TopicKnowledgeProfile` (dense bank:
85 claims / 996 probes) measures what the model knows. `eval_v3` (46 hand items)
compares interventions. A number from one never substitutes for the other.

**Adapter roles (do not swap):**

- `v5` = SENSE/profile wheel — repairs meanings+honesty; **regresses** eval_v3.
- `v6` = negative result — lifts eval_v3, wrecks false-reject (~67%). Never
  promote as the honesty adapter; its hand bank mixes v0.23 facts into a
  v0.22-tagged train set on purpose (Phase C intervention, not a knowledge pin).
- `v7` = frozen checkpoint — sequential from v5; ~41% eval_v3, ~97% false-reject.
- FT (LoRA and full-weight) is **paused** — profile wheel past v7 **and**
  factory-v1 LoRA (stopped 2026-07-28 after symbol-SFT closeout;
  `artifacts/reports/eval_factory_ft_stop_freeze.md`). Do not restart without
  an explicit go.

**Traps a fresh agent will fall into:**

- The mill eval denylist is **optional** (`--eval-denylist`) and Jaccard-only;
  `build_wheel.py` (v5) has no eval_v3 check at all. Any new train set must pass
  the denylist explicitly — policy, not enforced by code.
- `dense.py`, `build_wheel.py`, `aggregate*.py` hardcode `v0.22.0` / `doc_0`
  strings. Moving to v0.23.0 requires parameterizing revisions, not editing
  outputs.
- Free-recall keyword scoring is **advisory** (one OR-group = full credit). Do
  not headline free-recall numbers; trust choice/boolean scorers (audited 42/50).
- c0/c3 accuracy differs between `e1_vllm_failure_modes.md` (pre-rescore) and
  the DriftEvent JSONs (rescored). Cite the source you actually used.
- Profile-wheel train surfaces ≈ dense probe surfaces by design; paraphrase
  holdout is the anti-memorization control. Do not claim generalization beyond
  new-wording-on-same-claims.

**What can die:** rejected adapters after B2 push, local run caches.
**What must survive:** manifests (`manifest_v0.22.0_v{4..7}.json`), frozen
`eval_v3.jsonl`, probe banks, the role split above, negative results (v6).

**Command (readouts, safe to re-run):**

```bash
./scripts/lab run --target modal --gpu H100 \
  --config configs/experiments/e1_vllm/profile_dense_v1_para_ft_v7_modal.yaml
./scripts/lab run --target modal --gpu H100 \
  --config configs/experiments/e1_vllm/c3_ft_seq_v7_eval_v3_qwen35_08b_modal.yaml
```

## lessons

### B2 + s5cmd: export creds before every child process

**Symptom:** `s5cmd cp` or `s5cmd ls` hangs for minutes (or forever). No clear error unless you `timeout` the command (`fetching region failed: RequestCanceled`).

**Cause:** `source .env` without `set -a` sets shell variables but does **not export** them. `s5cmd` is a child process and never receives `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`. It stalls on credential/region discovery instead of failing fast.

**Fix (shell):**
```bash
set -a
source .env
set +a
s5cmd --endpoint-url "${S3_ENDPOINT_URL%%[[:space:]]}" cp ...
```

**Fix (Python dispatch):** set `os.environ` before `subprocess.run`, or pass env explicitly. `./scripts/lab` already uses `set -a`.

**Also:** trim trailing whitespace on `S3_ENDPOINT_URL` in `.env`.

**Invariant:** every storage child process must receive B2 creds — shell (`set -a`), Python (`os.environ`), Modal (`lalith-lab` secret), Lambda (SSH env injection).

### Lambda: private GitHub repo cannot `git clone` over HTTPS

**Symptom:** remote job exits 128 — `fatal: could not read Username for 'https://github.com'`.

**Cause:** worker tries to clone a private repo without credentials.

**Fix:** cockpit pushes code via `rsync` before the job (`LAB_SKIP_GIT=1`). GitHub clone is fallback for public repos or when a `GITHUB_TOKEN` is configured later.

### Lambda CUDA: never install unqualified `torch` on GPU workers

**Symptom:** `nvidia-smi` works, but PyTorch prints `CUDA initialization: The NVIDIA driver on your system is too old` and `torch.cuda.is_available()` is false. Jobs silently fall back to CPU unless guarded.

**Cause:** `pip install torch` can install a PyTorch wheel whose bundled CUDA runtime is newer than the Lambda image's NVIDIA driver supports.

**Fix:** probe `nvidia-smi`, select an explicit PyTorch wheel profile (`cu128`, `cu126`, `cu124`, `cu121`, or `cpu`), install it first, then install the rest of the runtime stack. Verify CUDA before running GPU jobs.

**Invariant:** GPU workers must write `hardware.json` and `env.json`, and GPU runs must fail fast when `torch.cuda.is_available()` is false unless `--allow-cpu` is explicit.
