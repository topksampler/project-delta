# A+B+C result — blend v6

## C — role split (frozen)

v5 remains the SENSE/profile adapter. Do not expect eval_v3 lift from v5 alone.
See `adapter_roles_profile_vs_eval_v3.md`.

## A+B — blend train

`train_v0.22.0_v6`: mill v4 (1192) + profile v5 (872) + eval_v3 hole paraphrases (29, denylisted).
Adapter: `e1-vllm-c3-ft-mill-v6-qwen35-08b-modal`

## eval_v3

| condition | accuracy |
|-----------|---------:|
| c0 base | 0.239 |
| mill v4 | 0.326 |
| profile v5 | 0.217 |
| **blend v6** | **0.457** |

by_class: {'A': 0.55, 'B': 0.1, 'C': 0.4375, 'D': 0.5}
failures: {'correct': 21, 'wrong': 22, 'stale_version': 2, 'confident_hallucination': 1}

## profile paraphrase (should not destroy SENSE)

| form | v5 | v6 |
|------|---:|---:|
| recognition | 0.984 | 0.972 |
| true | 0.972 | 0.956 |
| false | 0.964 | 0.667 |
| honesty fail | 0.129 | 0.671 |
