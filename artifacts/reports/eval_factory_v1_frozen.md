# Eval factory v1 — frozen EvalEnvironment

Transition: vLLM v0.22.0 → v0.23.0
Protocol: `e1_eval_factory_v1`
Status: **frozen** (`freeze_ready=true`, sealed)

## invariant

```text
The target model never defines truth.
Executable AST verification and leakage gates precede every model score.
This artifact is the primary version-delta instrument for spine C.
eval_v3 remains a historical 46-item comparison, not this scoreboard.
```

## frozen counts

| measure | value |
|---------|------:|
| verified claims | 1,070 |
| deltas (added/changed/removed) | 42 |
| train / dev / eval claims | 639 / 212 / 219 |
| final eval probes | 623 |
| train/eval max Jaccard | 0.55 |
| audit agreement | 50/50 (independent agent pair; provenance in `audit_50.jsonl`) |

Family deltas: CLI 10, config 20, env 12, public exports 0.

## what goes where

- local sealed dir:
  `data/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1/`
- B2:
  `datasets/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1/`
- protocol/code: `experiments/e1_vllm/eval_factory/`
- first model instrument check:
  `configs/experiments/e1_vllm/eval_factory_v1_base_qwen35_08b_modal.yaml`

## what can die

- pre-seal draft surfaces and rejected teacher candidates after B2 durability;
- local Modal caches.

## what must survive

- sealed `manifest.json` and all hashed artifacts;
- `audit_50.jsonl` with auditor provenance;
- `validation.json`;
- this freeze report.

## command

```bash
python -m experiments.e1_vllm.eval_factory.cli validate \
  --artifact-dir \
  data/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1

./scripts/lab run --target modal --gpu A10G \
  --config configs/experiments/e1_vllm/eval_factory_v1_base_qwen35_08b_modal.yaml
```
