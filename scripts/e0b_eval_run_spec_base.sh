#!/usr/bin/env bash
set -euo pipefail

python -m src.lab.eval_run_spec \
  --config configs/evals/e0b_run_spec_base.yaml
