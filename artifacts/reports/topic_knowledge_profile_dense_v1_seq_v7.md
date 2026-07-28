# Sequential FT v7 (fork C)

## method

```text
start = profile v5 adapter
stage-2 = 60 steps @ 5e-5 on eval_v3 hole paraphrases (x4) + honesty retain
```

Run: `e1-vllm-c3-ft-seq-v7-qwen35-08b-modal`

## eval_v3

| adapter | accuracy |
|---------|---------:|
| base | 0.239 |
| mill v4 | 0.326 |
| profile v5 | 0.217 |
| blend v6 | 0.457 |
| **seq v7** | **0.413** |

## profile paraphrase honesty

| adapter | recognition | reject false | honesty fail |
|---------|------------:|-------------:|-------------:|
| v5 | 0.984 | 0.964 | 0.129 |
| v6 blend | 0.972 | 0.667 | 0.671 |
| **v7 seq** | 0.980 | 0.968 | 0.141 |
