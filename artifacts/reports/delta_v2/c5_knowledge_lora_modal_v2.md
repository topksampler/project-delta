# delta_v2 c5 knowledge LoRA — Modal v2 training audit

Status: `implemented`; candidate trained, not promoted.

This run trained one rank-8 LoRA candidate on the frozen
`delta-v2-vllm-knowledge-adaptation-v1` source cards. It did not alter the base
weights. A lower training or dev loss establishes that the adapter fit the
training objective; it does not establish knowledge acquisition or retention.
Those claims require the separately frozen base-versus-LoRA evaluation.

## run identity

- run: `delta-v2-c5-knowledge-lora-qwen35-08b-modal-v2`
- condition: `c5_knowledge_lora`
- base model: `Qwen/Qwen3.5-0.8B`
- base revision: `2fc06364715b967f1860aea9cf38778875588b17`
- launch commit: `81ffe6a24407f5cd24e80ed9593817b4a32373c0`
- config SHA-256:
  `78b8dc1d53a7d54f25cbf535e46fcd6ea68bf4fa058057d8ab3d700ab8917ca7`
- protocol SHA-256:
  `73f7dde372ebdbeafd56cd25808b9dcc7135bbeafd829f91ff4d970abdd44331`
- training data SHA-256:
  `16ad0d2c12a823722501800227c739e59131fd7c7e0f774e2f1faab2e4bd4b95`
- dev data SHA-256:
  `dc8dc048de89c6306d1950acbda2a5e67ace7a712a174eb68d2d22ed3e68cff9`
- adapter SHA-256:
  `83fb04ae52502432df6b17d2aaa12cfeb12f457df6a77c83f28fed433d9966fa`
- metrics SHA-256:
  `b2bcf6ba69fed86ac378f19cd9171b28a4a3d2f75f2bb099bbde2c6f9c276cf0`
- B2 prefix:
  `runs/delta-v2-c5-knowledge-lora-qwen35-08b-modal-v2/`

## training readout

| field | observed |
|---|---:|
| optimizer steps | 60 |
| examples seen with accumulation | 240 |
| initial dev loss | 2.394504 |
| final dev loss | 0.141030 |
| dev-loss reduction | 2.253475 |
| trainable LoRA parameters | 5,411,328 |
| total parameters in wrapped model | 858,397,248 |
| trainable fraction | 0.6304% |
| matched language modules | 186 |
| trainable vision modules | 0 |
| longest training row | 129 tokens |
| longest dev row | 122 tokens |

The exact runtime lock passed on one NVIDIA H100 80GB HBM3. Deterministic
algorithms were enabled. Model work took 196.37 seconds; the dispatch estimate
was USD 0.2359.

The adapter remains a candidate. Promotion is forbidden until the identical
169-item frozen bank reports acquisition and retention separately and the
preregistered gates are applied.
