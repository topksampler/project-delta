#!/usr/bin/env bash
set -euo pipefail

python -m src.lab.eval_base \
  --config configs/evals/e0_qwen_0_5b_base.yaml
