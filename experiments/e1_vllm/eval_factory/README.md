# e1_vllm eval factory

Builds a version-bound, claim-first evaluation artifact for one vLLM release
transition. The protocol and freeze gates are in [PROTOCOL.md](./PROTOCOL.md).

## invariant

```text
No model defines gold.
Claims split before questions are generated.
Executable verification and leakage checks run before model evaluation.
An artifact with freeze_ready=false is not an eval.
```

## current pilot

```text
transition       v0.22.0 → v0.23.0
protocol         e1_eval_factory_v1
claim families   CLI flags, typed config fields, env vars, public exports
claims           1,070 (1,028 stable; 42 added/changed/removed)
status           frozen (`freeze_ready=true`, sealed on B2)
```

The initial CLI-only census contained 10 deltas. Amendment A broadened the
preregistered executable population and reached 42 without lowering the gate.
All automatic gates pass; the 50-claim audit remains blocking.

The deterministic seed surfaces also produce high train/eval template
Jaccard. They are seeds only. A separate 4B teacher rewrites eval wording
without receiving or changing gold; the validator then measures overlap again.
The first teacher pass was rejected for presupposing change. Multi-family v3
plus deterministic cleanup and neutral eval-only fallbacks yields 623/657
surfaces and maximum train/eval Jaccard 0.55. See
`artifacts/reports/eval_factory_v1_prefreeze.md`.

## what goes where

- `cli_flags.py`: AST verifier for literal argparse contracts;
- `factory.py`: version diff, claim-level split, deterministic seed probes;
- `generate_surfaces.py`: teacher paraphrase job; wording only, never gold;
- `validate.py`: mandatory freeze gates;
- `cli.py`: build/import/validate entrypoint;
- local artifacts:
  `data/experiments/e1_vllm/eval_factory/{transition}/{protocol}/`;
- frozen artifacts: matching B2 path under `datasets/`.

## what can die

- deterministic eval seed wording after teacher surfaces pass;
- rejected teacher generations;
- local snapshot checkouts and pre-freeze artifacts.

## what must survive

- protocol, source hashes, claim IDs and splits;
- verified and rejected claims;
- deterministic seeds, teacher prompts/model IDs, and raw rejects;
- leakage report, human audit, and frozen manifest.

## command

```bash
python -m experiments.e1_vllm.eval_factory.cli build \
  --before doc_0 --after doc_8

./scripts/lab run --target modal --gpu A10G \
  --config \
  configs/experiments/e1_vllm/eval_factory_v1_surface_gen_v3_multifamily_qwen35_4b_modal.yaml

python -m experiments.e1_vllm.eval_factory.cli finalize-surfaces \
  --artifact-dir \
  data/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1 \
  --generated \
  runs/e1-vllm-eval-factory-v1-surface-gen-v3-multifamily-qwen35-4b-modal/probes_eval_generated.jsonl \
  --rejected \
  runs/e1-vllm-eval-factory-v1-surface-gen-v3-multifamily-qwen35-4b-modal/rejected.jsonl \
  --metrics \
  runs/e1-vllm-eval-factory-v1-surface-gen-v3-multifamily-qwen35-4b-modal/metrics.json

python -m experiments.e1_vllm.eval_factory.cli build-audit \
  --artifact-dir \
  data/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1

python -m experiments.e1_vllm.eval_factory.cli validate \
  --artifact-dir \
  data/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1

# Only after a human completes audit_50.jsonl and validate passes:
python -m experiments.e1_vllm.eval_factory.cli seal \
  --artifact-dir \
  data/experiments/e1_vllm/eval_factory/v0.22.0_to_v0.23.0/e1_eval_factory_v1
```
