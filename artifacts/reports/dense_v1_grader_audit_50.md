# Dense profile grader audit (50 probes)

Serves: trust the automatic scorer before the data wheel uses it.

For each item, fill in `audit.jsonl`:
- `human_agree_with_auto`: true/false
- `human_correct_label`: correct | wrong | abstain_ok | unclear
- `human_notes`: optional

Source run: `e1-vllm-profile-dense-v1-qwen35-08b-modal`
Claims: `/Users/lalithnarayanc/lalith-ai-lab/data/experiments/e1_vllm/profile/dense_v1/claims_verified.jsonl`

## A01 — `--fig-by` · free_recall

Auto: **wrong** (score=0.0)

Question:
```
What behavior does the vLLM v0.22.0 option `--fig-by` affect?
```

Model output:
```
The `--fig-by` option in vLLM v0.22.0 restricts the figure generation to only the first `fig_by` number of figures, effectively limiting the output to a subset of the provided figures rather than generating all of them.
```

Gold: `{"must_contain_any": [["variables", "plot", "via", "var-x", "and"]], "advisory": true}`

Docs evidence:
```
Control the variables to plot via `--var-x` and `--var-y`, optionally applying `--filter-by` and `--bin-by` to the values. The plot is organized according to `--fig-by`, `--row-by`, `--col-by`, and `--curve-by`.
```

Path: ['docs/benchmarking/sweeps.md']

Your call: agree with auto? ____  label: ____

## A02 — `--mm-encoder-attn-backend` · meaning_false

Auto: **confident_hallucination** (score=0.0)

Question:
```
Does the documentation support this description for vLLM v0.22.0?
`--mm-encoder-attn-backend`: You can adjust the number of runs by setting `--num-runs`.
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
Yes, the documentation supports the description for vLLM v0.22.0 regarding the `--mm-encoder-attn-backend` option, which allows users to adjust the number of runs.
```

Gold: `{"boolean": "no_or_unknown", "abstain_if_unknown": true}`

Docs evidence:
```
Enable FP8 ViT attention by passing `--mm-encoder-attn-dtype fp8` together with `--mm-encoder-attn-backend FLASHINFER`:.
```

Path: ['docs/design/cuda_graphs_multimodal.md']

Your call: agree with auto? ____  label: ____

## A03 — `--trust-remote-code` · meaning_false

Auto: **correct** (score=1.0)

Question:
```
Is the following statement true for vLLM v0.22.0?
`--trust-remote-code`: Controls traffic variability using a Gamma distribution (range: > 0).
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
No, the `--trust-remote-code` flag in vLLM v0.22.0 does not control traffic variability using a Gamma distribution; it is a flag for verifying remote code execution and is not related to traffic variability.
```

Gold: `{"boolean": "no_or_unknown", "abstain_if_unknown": true}`

Docs evidence:
```
For `Alibaba-NLP/gte-Qwen2-*`, you need to enable `--trust-remote-code` for the correct tokenizer to be loaded.
```

Path: ['docs/benchmarking/cli.md']

Your call: agree with auto? ____  label: ____

## A04 — `--var-y` · meaning_recognition

Auto: **wrong** (score=0.0)

Question:
```
In vLLM 0.22.0, what is the documented purpose of `--var-y`?
A. You can use `--dry-run` to preview the commands to be run.
B. y-axis: tokens/s/GPU = `output_throughput` ÷ GPU count (`--gpu-count-var` if set; else gpu_count is TP×PP*DP).
C. Control the variables to plot via `--var-x` and `--var-y`, optionally applying `--filter-by` and `--bin-by` to the values.
Reply exactly as `ANSWER: <letter>`.
```

Model output:
```
B. y-axis: tokens/s/GPU = `output_throughput` ÷ GPU count (`--gpu-count-var` if set; else gpu_count is TP×PP*DP).
```

Gold: `{"choice": "C"}`

Docs evidence:
```
Control the variables to plot via `--var-x` and `--var-y`, optionally applying `--filter-by` and `--bin-by` to the values.
```

Path: ['docs/benchmarking/sweeps.md']

Your call: agree with auto? ____  label: ____

## A05 — `--all2all-backend` · meaning_false

Auto: **correct** (score=1.0)

Question:
```
Does the documentation support this description for vLLM v0.22.0?
`--all2all-backend`: NixlConnector currently does not distinguish `kv_role`; the actual prefiller/decoder roles are determined by the upper-level proxy (e.g., `toy_proxy_server.py` using `--prefiller-hosts` and `--decoder-hosts`).
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
No, the documentation does not support this description because it incorrectly states that NixlConnector currently does not distinguish `kv_role`, whereas the actual behavior is determined by the upper-level proxy.
```

Gold: `{"boolean": "no_or_unknown", "abstain_if_unknown": true}`

Docs evidence:
```
Currently, DBO is only supported with DeepEP, so DeepEP must be installed and the `--all2all-backend` argument must be set to `deepep_low_latency` if your workload is primarily decode requests, or `deepep_high_throughput` if your workload is primarily prefill requests.
```

Path: ['docs/design/dbo.md']

Your call: agree with auto? ____  label: ____

## A06 — `--show-hidden-metrics-for-version` · free_recall

Auto: **wrong** (score=0.0)

Question:
```
What behavior does the vLLM v0.22.0 option `--show-hidden-metrics-for-version` affect?
```

Model output:
```
This option controls whether hidden metrics are displayed in the output of the vLLM v0.22.0 command-line interface, which is useful for debugging and understanding internal state.
```

Gold: `{"must_contain_any": [["but", "can", "re-enabled", "show-hidden-metrics-for-version", "escape"]], "advisory": true}`

Docs evidence:
```
but can be re-enabled using the `--show-hidden-metrics-for-version=X.Y` escape hatch.
```

Path: ['docs/usage/metrics.md']

Your call: agree with auto? ____  label: ____

## A07 — `--dbo-decode-token-threshold` · meaning_true

Auto: **correct** (score=1.0)

Question:
```
Is the following statement true for vLLM v0.22.0?
`--dbo-decode-token-threshold`: * `--dbo-decode-token-threshold` the minimum number of tokens in a decode-only batch required to enable DBO for that batch.
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
Yes, the `--dbo-decode-token-threshold` flag in vLLM v0.22.0 specifies the minimum number of tokens required to enable Decode-Only (DBO) decoding for a specific batch.
```

Gold: `{"boolean": "yes"}`

Docs evidence:
```
* `--dbo-decode-token-threshold` the minimum number of tokens in a decode-only batch required to enable DBO for that batch.
```

Path: ['docs/design/dbo.md']

Your call: agree with auto? ____  label: ____

## A08 — `--filter-by` · meaning_recognition

Auto: **correct** (score=1.0)

Question:
```
Which description best matches the vLLM v0.22.0 flag `--filter-by`?
A. GPU-to-NUMA mapping and uses `--cpunodebind=<node> --membind=<node>` for each.
B. Control the variables to plot via `--var-x` and `--var-y`, optionally applying `--filter-by` and `--bin-by` to the values.
C. x-axis: tokens/s/user = `output_throughput` ÷ concurrency (`--user-count-var`, default `max_concurrency`, fallback `max_concurrent_requests`).
Reply exactly as `ANSWER: <letter>`.
```

Model output:
```
ANSWER: B
```

Gold: `{"choice": "B"}`

Docs evidence:
```
Control the variables to plot via `--var-x` and `--var-y`, optionally applying `--filter-by` and `--bin-by` to the values.
```

Path: ['docs/benchmarking/sweeps.md']

Your call: agree with auto? ____  label: ____

## A09 — `--mm-encoder-fp8-scale-save-margin` · meaning_true

Auto: **correct** (score=1.0)

Question:
```
Is the following statement true for vLLM v0.22.0?
`--mm-encoder-fp8-scale-save-margin`: Saved scales are multiplied by `--mm-encoder-fp8-scale-save-margin` (default `1.5`) to leave headroom against activation outliers not present in the calibration set. The default has been validated to generalize across datasets (e.g. VisionArena-Chat calibration maintains BF16 accuracy on ChartQA).
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
Yes, the statement is true. The `--mm-encoder-fp8-scale-save-margin` flag in vLLM v0.22.0 is designed to apply a scaling factor (default 1.5) to saved FP8 model scales to account for activation outliers in the calibration set, ensuring the model generalizes across datasets like VisionArena-Chat.
```

Gold: `{"boolean": "yes"}`

Docs evidence:
```
Saved scales are multiplied by `--mm-encoder-fp8-scale-save-margin` (default `1.5`) to leave headroom against activation outliers not present in the calibration set. The default has been validated to generalize across datasets (e.g. VisionArena-Chat calibration maintains BF16 accuracy on ChartQA).
```

Path: ['docs/features/quantization/fp8_vit_attn.md']

Your call: agree with auto? ____  label: ____

## A10 — `--n` · meaning_false

Auto: **confident_hallucination** (score=0.0)

Question:
```
Does the documentation support this description for vLLM v0.22.0?
`--n`: To customize the ITL color thresholds, one can use `--timeline-itl-thresholds` flag (default: 25ms, 50ms).
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
Yes, the documentation supports the description for vLLM v0.22.0 regarding the `--timeline-itl-thresholds` flag.
```

Gold: `{"boolean": "no_or_unknown", "abstain_if_unknown": true}`

Docs evidence:
```
with an embedding model, also set `--no_reranker`.
```

Path: ['docs/benchmarking/cli.md']

Your call: agree with auto? ____  label: ____

## A11 — `--host` · meaning_true

Auto: **wrong** (score=0.0)

Question:
```
Is this description accurate for vLLM v0.22.0?
`--host`: !!! warning Please note that `VLLM_PORT` and `VLLM_HOST_IP` set the port and ip for vLLM's **internal usage**. It is not the port and ip for the API server. If you use `--host $VLLM_HOST_IP` and `--port $VLLM_PORT` to start the API server, it will not work.
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
No, this description is inaccurate. The warning indicates that `VLLM_PORT` and `VLLM_HOST_IP` are internal configuration parameters for the vLLM server itself, not the API server. The API server is typically started via the `--port` flag, and the internal port is used for communication between the vLLM server and the API server.
```

Gold: `{"boolean": "yes"}`

Docs evidence:
```
!!! warning Please note that `VLLM_PORT` and `VLLM_HOST_IP` set the port and ip for vLLM's **internal usage**. It is not the port and ip for the API server. If you use `--host $VLLM_HOST_IP` and `--port $VLLM_PORT` to start the API server, it will not work.
```

Path: ['docs/cli/README.md']

Your call: agree with auto? ____  label: ____

## A12 — `--ignore-eos` · meaning_false

Auto: **correct** (score=1.0)

Question:
```
Is the following statement true for vLLM v0.22.0?
`--ignore-eos`: you should set `--backend openai-embeddings` and `--endpoint /v1/embeddings` to use the Embeddings API.
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
No, the `--ignore-eos` flag is not supported in vLLM v0.22.0; this feature was introduced in v0.21.0.
```

Gold: `{"boolean": "no_or_unknown", "abstain_if_unknown": true}`

Docs evidence:
```
Benchmark. It is recommended to use the flag `--ignore-eos` to simulate real responses. You can set the size of the output via the arg `random-output-len`.
```

Path: ['docs/benchmarking/cli.md']

Your call: agree with auto? ____  label: ____

## A13 — `--max-num-seqs` · meaning_true

Auto: **wrong** (score=0.0)

Question:
```
Is the following statement true for vLLM v0.22.0?
`--max-num-seqs`: If the `--max-num-seqs` parameter for P instances is set to a large value, due to the large batch size, P instances will generate a large amount of KVCache simultaneously.
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
No, the statement is false. The `--max-num-seqs` parameter controls the maximum number of sequences to generate in a single batch, not the number of P instances simultaneously. A large batch size causes the model to generate many sequences in parallel, but the `--max-num-seqs` limit restricts how many sequences are generated at once, not how many instances run.
```

Gold: `{"boolean": "yes"}`

Docs evidence:
```
If the `--max-num-seqs` parameter for P instances is set to a large value, due to the large batch size, P instances will generate a large amount of KVCache simultaneously.
```

Path: ['docs/benchmarking/sweeps.md']

Your call: agree with auto? ____  label: ____

## A14 — `--var-y` · meaning_false

Auto: **confident_hallucination** (score=0.0)

Question:
```
Is this description accurate for vLLM v0.22.0?
`--var-y`: y-axis: tokens/s/GPU = `output_throughput` ÷ GPU count (`--gpu-count-var` if set; else gpu_count is TP×PP*DP).
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
Yes, the description is accurate for vLLM v0.22.0, as it correctly defines the output throughput as the ratio of tokens per second to the GPU count, with a fallback to the total compute capability if the `--gpu-count-var` flag is not provided.
```

Gold: `{"boolean": "no_or_unknown", "abstain_if_unknown": true}`

Docs evidence:
```
Control the variables to plot via `--var-x` and `--var-y`, optionally applying `--filter-by` and `--bin-by` to the values.
```

Path: ['docs/benchmarking/sweeps.md']

Your call: agree with auto? ____  label: ____

## A15 — `--kv-cache-metrics-sample` · free_recall

Auto: **correct** (score=1.0)

Question:
```
Explain the documented purpose of `--kv-cache-metrics-sample` in vLLM version 0.22.0.
```

Model output:
```
The `--kv-cache-metrics-sample` flag in vLLM 0.22.0 is used to retrieve a single sample of the KV cache metrics from the underlying backend, allowing users to inspect the current state of their cache without triggering a full re-evaluation of the entire dataset.
```

Gold: `{"must_contain_any": [["also", "emit", "set", "histograms", "describe"]], "advisory": true}`

Docs evidence:
```
We also emit a set of histograms that describe how long sampled KV cache blocks stay resident and how often they are reused. Sampling (`--kv-cache-metrics-sample`) keeps the overhead tiny; when a block is chosen we record:.
```

Path: ['docs/design/metrics.md']

Your call: agree with auto? ____  label: ____

## A16 — `--moe-backend` · meaning_recognition

Auto: **correct** (score=1.0)

Question:
```
Which description best matches the vLLM v0.22.0 flag `--moe-backend`?
A. Combine with `--moe-backend` to pin a specific kernel family.
B. You can adjust the number of runs by setting `--num-runs`.
C. Enable FP8 ViT attention by passing `--mm-encoder-attn-dtype fp8` together.
Reply exactly as `ANSWER: <letter>`.
```

Model output:
```
ANSWER: A
```

Gold: `{"choice": "A"}`

Docs evidence:
```
Combine with `--moe-backend` to pin a specific kernel family.
```

Path: ['docs/features/quantization/online.md']

Your call: agree with auto? ____  label: ____

## A17 — `--workload-iters` · free_recall

Auto: **correct** (score=1.0)

Question:
```
What does `--workload-iters` control in vLLM v0.22.0?
```

Model output:
```
`--workload-iters` controls the number of iterations for the model's inference, which determines how many times the model processes the input to generate the final output.
```

Gold: `{"must_contain_any": [["you", "can", "override", "number", "iterations"]], "advisory": true}`

Docs evidence:
```
You can override the number of iterations in the algorithm by setting `--workload-iters`.
```

Path: ['docs/benchmarking/sweeps.md']

Your call: agree with auto? ____  label: ____

## A18 — `--var-y` · meaning_true

Auto: **correct** (score=1.0)

Question:
```
Is the following statement true for vLLM v0.22.0?
`--var-y`: Control the variables to plot via `--var-x` and `--var-y`, optionally applying `--filter-by` and `--bin-by` to the values.
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
Yes, the `--var-y` flag in vLLM v0.22.0 allows users to control variables for plotting by specifying `--var-x` and `--var-y`, optionally applying `--filter-by` and `--bin-by` to the values.
```

Gold: `{"boolean": "yes"}`

Docs evidence:
```
Control the variables to plot via `--var-x` and `--var-y`, optionally applying `--filter-by` and `--bin-by` to the values.
```

Path: ['docs/benchmarking/sweeps.md']

Your call: agree with auto? ____  label: ____

## A19 — `--attention-backend` · free_recall

Auto: **wrong** (score=0.0)

Question:
```
Explain the documented purpose of `--attention-backend` in vLLM version 0.22.0.
```

Model output:
```
The `--attention-backend` flag specifies the underlying attention mechanism used by vLLM, such as `torch_attention` or `torch_attention_impl`, which determines how the model processes attention weights during inference.
```

Gold: `{"must_contain_any": [["when", "you", "explicitly", "set", "via"]], "advisory": true}`

Docs evidence:
```
When you explicitly set a backend via `--attention-backend` or `AttentionConfig`:.
```

Path: ['docs/design/attention_backends.md']

Your call: agree with auto? ____  label: ____

## A20 — `--enable-cumem-allocator` · meaning_false

Auto: **correct** (score=1.0)

Question:
```
Is the following statement true for vLLM v0.22.0?
`--enable-cumem-allocator`: NixlConnector currently does not distinguish `kv_role`; the actual prefiller/decoder roles are determined by the upper-level proxy (e.g., `toy_proxy_server.py` using `--prefiller-hosts` and `--decoder-hosts`).
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
No, the statement is false. The `--enable-cumem-allocator` flag in vLLM v0.22.0 does not explicitly disable the distinction between KV roles; rather, it enables the Cumem allocator which handles role-based memory management internally, and the actual role assignment is still determined by the proxy server configuration (like `--prefiller-hosts` and `--decoder-hosts`).
```

Gold: `{"boolean": "no_or_unknown", "abstain_if_unknown": true}`

Docs evidence:
```
To enable this feature, you need to set `--enable-cumem-allocator` or `--enable-sleep-mode` flags, and set `UCX_CUDA_IPC_ENABLE_MNNVL: 'y'` env.
```

Path: ['docs/features/nixl_connector_usage.md']

Your call: agree with auto? ____  label: ____

## A21 — `--trust-remote-code` · meaning_false

Auto: **confident_hallucination** (score=0.0)

Question:
```
Is this description accurate for vLLM v0.22.0?
`--trust-remote-code`: Controls traffic variability using a Gamma distribution (range: > 0).
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
Yes, the description is accurate for vLLM v0.22.0, as the `--trust-remote-code` flag is indeed used to control traffic variability by applying a Gamma distribution with a specified rate parameter.
```

Gold: `{"boolean": "no_or_unknown", "abstain_if_unknown": true}`

Docs evidence:
```
For `Alibaba-NLP/gte-Qwen2-*`, you need to enable `--trust-remote-code` for the correct tokenizer to be loaded.
```

Path: ['docs/benchmarking/cli.md']

Your call: agree with auto? ____  label: ____

## A22 — `--language-model-only` · meaning_recognition

Auto: **correct** (score=1.0)

Question:
```
Which description best matches the vLLM v0.22.0 flag `--language-model-only`?
A. For hybrid-only models such as Llama-4, Step3, Mistral-3 and Qwen-3.5, a text-only mode can be enabled by setting all supported multimodal modalities to 0 (`--language-model-only`) so that their multimodal modules will not be loaded to free up more GPU memory for KV cache.
B. Both are enabled by the `--enable-prompt-embeds` flag in `vllm serve`.
C. | `--plot` / `--no-plot` | `bool` | `True` | Controls whether performance plots are generated.
Reply exactly as `ANSWER: <letter>`.
```

Model output:
```
ANSWER: A
```

Gold: `{"choice": "A"}`

Docs evidence:
```
For hybrid-only models such as Llama-4, Step3, Mistral-3 and Qwen-3.5, a text-only mode can be enabled by setting all supported multimodal modalities to 0 (`--language-model-only`) so that their multimodal modules will not be loaded to free up more GPU memory for KV cache.
```

Path: ['docs/models/supported_models.md']

Your call: agree with auto? ____  label: ____

## A23 — `--port` · meaning_recognition

Auto: **correct** (score=1.0)

Question:
```
Choose the accurate meaning of `--port` for vLLM version 0.22.0.
A. !!! warning Please note that `VLLM_PORT` and `VLLM_HOST_IP` set the port and ip for vLLM's **internal usage**. It is not the port and ip for the API server. If you use `--host $VLLM_HOST_IP` and `--port $VLLM_PORT` to start the API server, it will not work.
B. A full set of options is available in the `vllm serve --help`.
C. At inference time, use the option `--dataset-name custom`.
Reply exactly as `ANSWER: <letter>`.
```

Model output:
```
ANSWER: A
```

Gold: `{"choice": "A"}`

Docs evidence:
```
!!! warning Please note that `VLLM_PORT` and `VLLM_HOST_IP` set the port and ip for vLLM's **internal usage**. It is not the port and ip for the API server. If you use `--host $VLLM_HOST_IP` and `--port $VLLM_PORT` to start the API server, it will not work.
```

Path: ['docs/benchmarking/cli.md']

Your call: agree with auto? ____  label: ____

## A24 — `--dbo-decode-token-threshold` · meaning_false

Auto: **correct** (score=1.0)

Question:
```
Is this description accurate for vLLM v0.22.0?
`--dbo-decode-token-threshold`: For `Alibaba-NLP/gte-Qwen2-*`, you need to enable `--trust-remote-code` for the correct tokenizer to be loaded.
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
No, this description is inaccurate. The `--trust-remote-code` flag is required for the `Alibaba-NLP/gte-Qwen2-*` tokenizer to be loaded, but it is not a general flag for all vLLM versions or all tokenizers; it is specific to the Qwen2 tokenizer and its associated inference engine.
```

Gold: `{"boolean": "no_or_unknown", "abstain_if_unknown": true}`

Docs evidence:
```
* `--dbo-decode-token-threshold` the minimum number of tokens in a decode-only batch required to enable DBO for that batch.
```

Path: ['docs/design/dbo.md']

Your call: agree with auto? ____  label: ____

## A25 — `--row-by` · meaning_false

Auto: **confident_hallucination** (score=0.0)

Question:
```
Is this description accurate for vLLM v0.22.0?
`--row-by`: | `--plot` / `--no-plot` | `bool` | `True` | Controls whether performance plots are generated.
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
Yes, the description is accurate for vLLM v0.22.0, as the `--row-by` flag controls whether performance plots are generated, and the `--plot` / `--no-plot` options are indeed boolean flags that control this behavior.
```

Gold: `{"boolean": "no_or_unknown", "abstain_if_unknown": true}`

Docs evidence:
```
Control the variables to plot via `--var-x` and `--var-y`, optionally applying `--filter-by` and `--bin-by` to the values. The plot is organized according to `--fig-by`, `--row-by`, `--col-by`, and `--curve-by`.
```

Path: ['docs/benchmarking/sweeps.md']

Your call: agree with auto? ____  label: ____

## A26 — `--gpu-count-var` · meaning_recognition

Auto: **correct** (score=1.0)

Question:
```
Which description best matches the vLLM v0.22.0 flag `--gpu-count-var`?
A. y-axis: tokens/s/GPU = `output_throughput` ÷ GPU count (`--gpu-count-var` if set; else gpu_count is TP×PP*DP).
B. In case you are using a custom `--serve-cmd`, you can override the commands used for resetting the state by setting `--after-bench-cmd`.
C. | `--debug` | `bool` | `False` | Enables debug mode.
Reply exactly as `ANSWER: <letter>`.
```

Model output:
```
ANSWER: A
```

Gold: `{"choice": "A"}`

Docs evidence:
```
y-axis: tokens/s/GPU = `output_throughput` ÷ GPU count (`--gpu-count-var` if set; else gpu_count is TP×PP*DP).
```

Path: ['docs/benchmarking/sweeps.md']

Your call: agree with auto? ____  label: ____

## A27 — `--label-by` · meaning_true

Auto: **correct** (score=1.0)

Question:
```
Is the following statement true for vLLM v0.22.0?
`--label-by`: Show the configuration used in each data point `--label-by` (default: `max_concurrency,gpu_count`).
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
Yes, the `--label-by` flag in vLLM v0.22.0 is available and defaults to `max_concurrency,gpu_count`.
```

Gold: `{"boolean": "yes"}`

Docs evidence:
```
Show the configuration used in each data point `--label-by` (default: `max_concurrency,gpu_count`).
```

Path: ['docs/benchmarking/sweeps.md']

Your call: agree with auto? ____  label: ____

## A28 — `--allowed-media-domains` · free_recall

Auto: **wrong** (score=0.0)

Question:
```
What does `--allowed-media-domains` control in vLLM v0.22.0?
```

Model output:
```
`--allowed-media-domains` restricts which specific media formats (e.g., MP4, WebM, AVI) are permitted for streaming, ensuring that only supported codecs are used to prevent compatibility issues.
```

Gold: `{"must_contain_any": [["when", "serving", "multi-modal", "models", "consider"]], "advisory": true}`

Docs evidence:
```
When serving multi-modal models, consider setting `--allowed-media-domains` to restrict domain that vLLM can access to prevent it from accessing arbitrary endpoints that can potentially be vulnerable to Server-Side Request Forgery (SSRF) attacks.
```

Path: ['docs/features/multimodal_inputs.md']

Your call: agree with auto? ____  label: ____

## A29 — `--enable-cumem-allocator` · meaning_true

Auto: **correct** (score=1.0)

Question:
```
Does the documentation support this description for vLLM v0.22.0?
`--enable-cumem-allocator`: To enable this feature, you need to set `--enable-cumem-allocator` or `--enable-sleep-mode` flags, and set `UCX_CUDA_IPC_ENABLE_MNNVL: 'y'` env.
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
Yes, the documentation for vLLM v0.22.0 explicitly supports the `--enable-cumem-allocator` flag and the `UCX_CUDA_IPC_ENABLE_MNNVL` environment variable.
```

Gold: `{"boolean": "yes"}`

Docs evidence:
```
To enable this feature, you need to set `--enable-cumem-allocator` or `--enable-sleep-mode` flags, and set `UCX_CUDA_IPC_ENABLE_MNNVL: 'y'` env.
```

Path: ['docs/features/nixl_connector_usage.md']

Your call: agree with auto? ____  label: ____

## A30 — `--ir-op-priority` · meaning_recognition

Auto: **wrong** (score=0.0)

Question:
```
In vLLM 0.22.0, what is the documented purpose of `--ir-op-priority`?
A. NixlConnector currently does not distinguish `kv_role`; the actual prefiller/decoder roles are determined by the upper-level proxy (e.g., `toy_proxy_server.py` using `--prefiller-hosts` and `--decoder-hosts`).
B. Configured with `--oltp-traces-endpoint` and `--collect-detailed-traces`.
C. Use `--ir-op-priority.<op_name>=<provider1>,<provider2>,...`:.
Reply exactly as `ANSWER: <letter>`.
```

Model output:
```
ANSWER: A
```

Gold: `{"choice": "C"}`

Docs evidence:
```
Use `--ir-op-priority.<op_name>=<provider1>,<provider2>,...`:.
```

Path: ['docs/design/vllm_ir.md']

Your call: agree with auto? ____  label: ____

## A31 — `--plot-dataset-stats` · free_recall

Auto: **correct** (score=1.0)

Question:
```
Explain the documented purpose of `--plot-dataset-stats` in vLLM version 0.22.0.
```

Model output:
```
The `--plot-dataset-stats` flag in vLLM 0.22.0 is used to visualize the performance metrics of the dataset being processed, such as token generation speed, memory usage, and latency, to help users understand the efficiency of their model configuration.
```

Gold: `{"must_contain_any": [["plot-timeline", "and", "plot-dataset-stats", "can", "generate"]], "advisory": true}`

Docs evidence:
```
The `--plot-timeline` and `--plot-dataset-stats` can be used to generate respectively the requests completion timeline and dataset prompt and output tokens statistics, which can be useful for debugging purpose or for deeper analysis.
```

Path: ['docs/benchmarking/cli.md']

Your call: agree with auto? ____  label: ____

## A32 — `--resume` · meaning_false

Auto: **confident_hallucination** (score=0.0)

Question:
```
Is this description accurate for vLLM v0.22.0?
`--resume`: Control the variables to plot via `--var-x` and `--var-y`, optionally applying `--filter-by` and `--bin-by` to the values.
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
Yes, the description is accurate for vLLM v0.22.0, as the `--resume` flag is used to control the variables to plot via `--var-x` and `--var-y` with optional filtering and binning.
```

Gold: `{"boolean": "no_or_unknown", "abstain_if_unknown": true}`

Docs evidence:
```
You can use the `--resume` option to continue the parameter sweep if an unexpected error occurs, e.g., timeout when connecting to HF Hub.
```

Path: ['docs/benchmarking/sweeps.md']

Your call: agree with auto? ____  label: ____

## A33 — `--numa-bind` · free_recall

Auto: **correct** (score=1.0)

Question:
```
What does `--numa-bind` control in vLLM v0.22.0?
```

Model output:
```
`--numa-bind` in vLLM v0.22.0 controls the specific memory layout and memory management strategy for the NUMA (Non-Uniform Memory Access) architecture, allowing you to optimize memory access patterns and reduce latency by assigning specific memory regions to different CPU cores.
```

Gold: `{"must_contain_any": [["use", "numa-bind", "feature"]], "advisory": true}`

Docs evidence:
```
Use `--numa-bind` to enable the feature.
```

Path: ['docs/configuration/optimization.md']

Your call: agree with auto? ____  label: ____

## A34 — `--all2all-backend` · meaning_true

Auto: **wrong** (score=0.0)

Question:
```
Is the following statement true for vLLM v0.22.0?
`--all2all-backend`: Currently, DBO is only supported with DeepEP, so DeepEP must be installed and the `--all2all-backend` argument must be set to `deepep_low_latency` if your workload is primarily decode requests, or `deepep_high_throughput` if your workload is primarily prefill requests.
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
No, the statement is false. The `--all2all-backend` argument is not currently restricted to DeepEP; it is supported by both DeepEP and DBO, and the recommended value depends on the workload type (prefill vs. decode) rather than the backend.
```

Gold: `{"boolean": "yes"}`

Docs evidence:
```
Currently, DBO is only supported with DeepEP, so DeepEP must be installed and the `--all2all-backend` argument must be set to `deepep_low_latency` if your workload is primarily decode requests, or `deepep_high_throughput` if your workload is primarily prefill requests.
```

Path: ['docs/design/dbo.md']

Your call: agree with auto? ____  label: ____

## A35 — `--convert` · free_recall

Auto: **wrong** (score=0.0)

Question:
```
Explain the documented purpose of `--convert` in vLLM version 0.22.0.
```

Model output:
```
The `--convert` flag in vLLM 0.22.0 is used to convert the model's output format to a specific target format, such as JSON or a custom protocol, which is essential for integrating the model with downstream systems that require structured data.
```

Gold: `{"must_contain_any": [["you", "can", "explicitly", "set", "type"]], "advisory": true}`

Docs evidence:
```
You can explicitly set `--convert <type>` to specify how to convert the model.
```

Path: ['docs/models/pooling_models/README.md']

Your call: agree with auto? ____  label: ____

## A36 — `--request-rate` · free_recall

Auto: **correct** (score=1.0)

Question:
```
What behavior does the vLLM v0.22.0 option `--request-rate` affect?
```

Model output:
```
The `--request-rate` option in vLLM v0.22.0 controls the maximum throughput of the underlying HTTP requests, effectively setting the maximum number of concurrent requests per second.
```

Gold: `{"must_contain_any": [["target", "generation", "requests", "per", "second"]], "advisory": true}`

Docs evidence:
```
Controls the target request generation rate (requests per second).
```

Path: ['docs/benchmarking/cli.md']

Your call: agree with auto? ____  label: ____

## A37 — `--trust-remote-code` · meaning_true

Auto: **wrong** (score=0.0)

Question:
```
Is this description accurate for vLLM v0.22.0?
`--trust-remote-code`: For `Alibaba-NLP/gte-Qwen2-*`, you need to enable `--trust-remote-code` for the correct tokenizer to be loaded.
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
No, this description is inaccurate. The `--trust-remote-code` flag is a configuration option for the `--trust-remote-code` command-line argument, which is used to enable remote code execution in the context of the `vLLM` library. It does not directly relate to the `Alibaba-NLP/gte-Qwen2-*` tokenizer loading requirement.
```

Gold: `{"boolean": "yes"}`

Docs evidence:
```
For `Alibaba-NLP/gte-Qwen2-*`, you need to enable `--trust-remote-code` for the correct tokenizer to be loaded.
```

Path: ['docs/benchmarking/cli.md']

Your call: agree with auto? ____  label: ____

## A38 — `--plot-dataset-stats` · free_recall

Auto: **correct** (score=1.0)

Question:
```
What behavior does the vLLM v0.22.0 option `--plot-dataset-stats` affect?
```

Model output:
```
The `--plot-dataset-stats` option in vLLM v0.22.0 generates a detailed visualization of the dataset's statistics, including metrics like token count, sequence length, and token utilization, to help users understand the performance and efficiency of their model during inference.
```

Gold: `{"must_contain_any": [["plot-timeline", "and", "plot-dataset-stats", "can", "generate"]], "advisory": true}`

Docs evidence:
```
The `--plot-timeline` and `--plot-dataset-stats` can be used to generate respectively the requests completion timeline and dataset prompt and output tokens statistics, which can be useful for debugging purpose or for deeper analysis.
```

Path: ['docs/benchmarking/cli.md']

Your call: agree with auto? ____  label: ____

## A39 — `--dataset-name` · meaning_true

Auto: **wrong** (score=0.0)

Question:
```
Is the following statement true for vLLM v0.22.0?
`--dataset-name`: At inference time, use the option `--dataset-name custom`.
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
No, the `--dataset-name` option in vLLM v0.22.0 does not support the `custom` suffix for inference time; it only accepts `--dataset-name` or `--dataset-name=custom` (with a space) for specifying the dataset name.
```

Gold: `{"boolean": "yes"}`

Docs evidence:
```
At inference time, use the option `--dataset-name custom`.
```

Path: ['docs/benchmarking/cli.md']

Your call: agree with auto? ____  label: ____

## A40 — `--random-mm-num-mm-items-range-ratio` · meaning_recognition

Auto: **wrong** (score=0.0)

Question:
```
Which description best matches the vLLM v0.22.0 flag `--random-mm-num-mm-items-range-ratio`?
A. Enable this mode with `--data-parallel-hybrid-lb` while still launching every node with the global data-parallel size.
B. Determine per-request item count k by sampling uniformly from the integer range defined by `--random-mm-base-items-per-request` and `--random-mm-num-mm-items-range-ratio`, then clamp k to at most the sum of per-modality limits.
C. The ramp-up strategy to use (`linear` or `exponential`).
Reply exactly as `ANSWER: <letter>`.
```

Model output:
```
B. Determine per-request item count k by sampling uniformly from the integer range defined by `--random-mm-base-items-per-request` and `--random-mm-num-mm-items-range-ratio`, then clamp k to at most the sum of per-modality limits.
```

Gold: `{"choice": "B"}`

Docs evidence:
```
Determine per-request item count k by sampling uniformly from the integer range defined by `--random-mm-base-items-per-request` and `--random-mm-num-mm-items-range-ratio`, then clamp k to at most the sum of per-modality limits.
```

Path: ['docs/benchmarking/cli.md']

Your call: agree with auto? ____  label: ____

## A41 — `--enable-sleep-mode` · meaning_recognition

Auto: **wrong** (score=0.0)

Question:
```
In vLLM 0.22.0, what is the documented purpose of `--enable-sleep-mode`?
A. | `--plot` / `--no-plot` | `bool` | `True` | Controls whether performance plots are generated.
B. We also emit a set of histograms that describe how long sampled KV cache blocks stay resident and how often they are reused. Sampling (`--kv-cache-metrics-sample`) keeps the overhead tiny; when a block is chosen we record:.
C. To enable this feature, you need to set `--enable-cumem-allocator` or `--enable-sleep-mode` flags, and set `UCX_CUDA_IPC_ENABLE_MNNVL: 'y'` env.
Reply exactly as `ANSWER: <letter>`.
```

Model output:
```
B. We also emit a set of histograms that describe how long sampled KV cache blocks stay resident and how often they are reused. Sampling (`--kv-cache-metrics-sample`) keeps the overhead tiny; when a block is chosen we record:.
```

Gold: `{"choice": "C"}`

Docs evidence:
```
To enable this feature, you need to set `--enable-cumem-allocator` or `--enable-sleep-mode` flags, and set `UCX_CUDA_IPC_ENABLE_MNNVL: 'y'` env.
```

Path: ['docs/features/nixl_connector_usage.md']

Your call: agree with auto? ____  label: ____

## A42 — `--after-bench-cmd` · meaning_recognition

Auto: **wrong** (score=0.0)

Question:
```
In vLLM 0.22.0, what is the documented purpose of `--after-bench-cmd`?
A. For `vllm serve`, you can control the hashing algorithm via .
B. In case you are using a custom `--serve-cmd`, you can override the commands used for resetting the state by setting `--after-bench-cmd`.
C. The `--generation-config` argument specifies where the generation config will be loaded from when calling `LLM.get_default_sampling_params()`.
Reply exactly as `ANSWER: <letter>`.
```

Model output:
```
B. In case you are using a custom `--serve-cmd`, you can override the commands used for resetting the state by setting `--after-bench-cmd`.
```

Gold: `{"choice": "B"}`

Docs evidence:
```
In case you are using a custom `--serve-cmd`, you can override the commands used for resetting the state by setting `--after-bench-cmd`.
```

Path: ['docs/benchmarking/sweeps.md']

Your call: agree with auto? ____  label: ____

## A43 — `--col-by` · free_recall

Auto: **wrong** (score=0.0)

Question:
```
What behavior does the vLLM v0.22.0 option `--col-by` affect?
```

Model output:
```
The `--col-by` option in vLLM v0.22.0 allows you to specify a custom column name to use as the primary key for grouping data, which is useful for filtering or aggregating datasets based on a specific column rather than the default `id` or `row_id`.
```

Gold: `{"must_contain_any": [["variables", "plot", "via", "var-x", "and"]], "advisory": true}`

Docs evidence:
```
Control the variables to plot via `--var-x` and `--var-y`, optionally applying `--filter-by` and `--bin-by` to the values. The plot is organized according to `--fig-by`, `--row-by`, `--col-by`, and `--curve-by`.
```

Path: ['docs/benchmarking/sweeps.md']

Your call: agree with auto? ____  label: ____

## A44 — `--max-num-batched-tokens` · meaning_recognition

Auto: **wrong** (score=0.0)

Question:
```
In vLLM 0.22.0, what is the documented purpose of `--max-num-batched-tokens`?
A. `--max-num-batched-tokens`, defines the limit of token numbers in a single batch, has more impacts on the first token performance.
B. Another delightful way is to use `uv run` with `--with [dependency]` option, which allows you to run commands such as `vllm serve` without creating any permanent environment:.
C. The `--rope-scaling` parameter used in older versions of vLLM is no longer supported.
Reply exactly as `ANSWER: <letter>`.
```

Model output:
```
A. `--max-num-batched-tokens` defines the limit of token numbers in a single batch, has more impacts on the first token performance.
```

Gold: `{"choice": "A"}`

Docs evidence:
```
`--max-num-batched-tokens`, defines the limit of token numbers in a single batch, has more impacts on the first token performance.
```

Path: ['docs/benchmarking/sweeps.md']

Your call: agree with auto? ____  label: ____

## A45 — `--mm-encoder-attn-dtype` · meaning_recognition

Auto: **wrong** (score=0.0)

Question:
```
In vLLM 0.22.0, what is the documented purpose of `--mm-encoder-attn-dtype`?
A. Note that this might result in errors due to engine config `--limit-mm-per-prompt`.
B. Enable FP8 ViT attention by passing `--mm-encoder-attn-dtype fp8` together.
C. Combine with `--moe-backend` to pin a specific kernel family.
Reply exactly as `ANSWER: <letter>`.
```

Model output:
```
B. Enable FP8 ViT attention by passing `--mm-encoder-attn-dtype fp8` together.
```

Gold: `{"choice": "B"}`

Docs evidence:
```
Enable FP8 ViT attention by passing `--mm-encoder-attn-dtype fp8` together.
```

Path: ['docs/features/quantization/fp8_vit_attn.md']

Your call: agree with auto? ____  label: ____

## A46 — `--label-by` · meaning_recognition

Auto: **correct** (score=1.0)

Question:
```
Choose the accurate meaning of `--label-by` for vLLM version 0.22.0.
A. Show the configuration used in each data point `--label-by` (default: `max_concurrency,gpu_count`).
B. You can pass in the argument `--api-key` or environment variable `VLLM_API_KEY` to enable the server to check for API key in the header.
C. Control the variables to plot via `--var-x` and `--var-y`, optionally applying `--filter-by` and `--bin-by` to the values.
Reply exactly as `ANSWER: <letter>`.
```

Model output:
```
ANSWER: A
```

Gold: `{"choice": "A"}`

Docs evidence:
```
Show the configuration used in each data point `--label-by` (default: `max_concurrency,gpu_count`).
```

Path: ['docs/benchmarking/sweeps.md']

Your call: agree with auto? ____  label: ____

## A47 — `--limit-mm-per-prompt` · meaning_false

Auto: **correct** (score=1.0)

Question:
```
Is the following statement true for vLLM v0.22.0?
`--limit-mm-per-prompt`: Only `X-Request-Id` HTTP request header is supported for now. It can be enabled with `--enable-request-id-headers`.
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
No, the `--limit-mm-per-prompt` flag is not supported for vLLM v0.22.0; instead, the `--enable-request-id-headers` flag is used to enable the `X-Request-Id` HTTP request header.
```

Gold: `{"boolean": "no_or_unknown", "abstain_if_unknown": true}`

Docs evidence:
```
Note that this might result in errors due to engine config `--limit-mm-per-prompt`.
```

Path: ['docs/benchmarking/cli.md']

Your call: agree with auto? ____  label: ____

## A48 — `--enable-sleep-mode` · meaning_true

Auto: **correct** (score=1.0)

Question:
```
Is this description accurate for vLLM v0.22.0?
`--enable-sleep-mode`: To enable this feature, you need to set `--enable-cumem-allocator` or `--enable-sleep-mode` flags, and set `UCX_CUDA_IPC_ENABLE_MNNVL: 'y'` env.
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
Yes, the description is accurate for vLLM v0.22.0, as it correctly identifies the required flags (`--enable-cumem-allocator` or `--enable-sleep-mode`) and the necessary environment variable (`UCX_CUDA_IPC_ENABLE_MNNVL`) to enable the sleep mode feature.
```

Gold: `{"boolean": "yes"}`

Docs evidence:
```
To enable this feature, you need to set `--enable-cumem-allocator` or `--enable-sleep-mode` flags, and set `UCX_CUDA_IPC_ENABLE_MNNVL: 'y'` env.
```

Path: ['docs/features/nixl_connector_usage.md']

Your call: agree with auto? ____  label: ____

## A49 — `--allowed-local-media-path` · meaning_recognition

Auto: **correct** (score=1.0)

Question:
```
Which description best matches the vLLM v0.22.0 flag `--allowed-local-media-path`?
A. Launch the API server/engine with the --allowed-local-media-path argument.
B. Enable this mode with `--data-parallel-hybrid-lb` while still launching every node with the global data-parallel size.
C. !!! warning Please note that `VLLM_PORT` and `VLLM_HOST_IP` set the port and ip for vLLM's **internal usage**. It is not the port and ip for the API server. If you use `--host $VLLM_HOST_IP` and `--port $VLLM_PORT` to start the API server, it will not work.
Reply exactly as `ANSWER: <letter>`.
```

Model output:
```
ANSWER: A
```

Gold: `{"choice": "A"}`

Docs evidence:
```
Launch the API server/engine with the --allowed-local-media-path argument.
```

Path: ['docs/benchmarking/cli.md']

Your call: agree with auto? ____  label: ____

## A50 — `--all2all-backend` · meaning_true

Auto: **wrong** (score=0.0)

Question:
```
Does the documentation support this description for vLLM v0.22.0?
`--all2all-backend`: Currently, DBO is only supported with DeepEP, so DeepEP must be installed and the `--all2all-backend` argument must be set to `deepep_low_latency` if your workload is primarily decode requests, or `deepep_high_throughput` if your workload is primarily prefill requests.
Answer yes or no. If you cannot verify it, answer unknown.
```

Model output:
```
No, the documentation does not support this description for vLLM v0.22.0.
```

Gold: `{"boolean": "yes"}`

Docs evidence:
```
Currently, DBO is only supported with DeepEP, so DeepEP must be installed and the `--all2all-backend` argument must be set to `deepep_low_latency` if your workload is primarily decode requests, or `deepep_high_throughput` if your workload is primarily prefill requests.
```

Path: ['docs/design/dbo.md']

Your call: agree with auto? ____  label: ____

