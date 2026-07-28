# Project DELTA — blog series

Narrative writeups of the `e1_vllm` research arc. These posts restate numbers
for readability; the reports under `artifacts/reports/` and the run artifacts
on B2 are the source of truth. If a number here disagrees with a report, the
report wins.

| post | subject |
|------|---------|
| [Part 1 — Your model's knowledge is rotting](./delta-part-1-cicd-for-model-knowledge.md) | The thesis: CI/CD for model knowledge, and building a grounded change environment |
| [Part 2 — RAG vs. LoRA on a moving target](./delta-part-2-rag-vs-lora.md) | The intervention matrix: six conditions, one frozen eval, one upset |
| [Part 3 — Fingerprinting what a 0.8B model actually knows](./delta-part-3-knowledge-fingerprinting.md) | Dense knowledge profiles, acquiescence, and auditing your own grader |
| [Part 4 — The data wheel and the honesty tax](./delta-part-4-data-wheel-honesty-tax.md) | Turning profile holes into training data without teaching the model to lie |

Written July 2026, at the freeze of the seq v7 checkpoint.
