# delta_v2 c5 knowledge LoRA — v1 launch failure

Status: failed before model load; no optimizer step or adapter was created.

The immutable run
`delta-v2-c5-knowledge-lora-qwen35-08b-modal-v1` reached the Modal H100
worker, wrote its environment and hardware receipts, and then stopped while
resolving the configured training module. The experiment-owned Modal app called
`lab.dispatch.modal_worker.train_module`, but the shared worker did not yet
provide that generic routing helper.

This is an infrastructure failure, not a training result. It says nothing about
the dataset, loss, optimizer, LoRA topology, knowledge acquisition, or
retention. The model was not loaded, optimizer steps were zero, and no adapter
exists under the run prefix.

Evidence:

- run prefix:
  `runs/delta-v2-c5-knowledge-lora-qwen35-08b-modal-v1/`
- launch commit:
  `ec7987b1067b7e1ce0eccdc398bb9e73185abc64`
- recorded Modal dispatch time: 28.94 seconds
- estimated cost: USD 0.0318
- durable objects: config, manifest, environment, hardware, timings, and cost
  receipts only

The generic route was added and tested in commit `bfa54b6`. The amended
protocol is `delta-v2-knowledge-lora-v2`; it preserves all scientific
parameters and uses a fresh run ID.
