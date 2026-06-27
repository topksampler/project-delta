Personal AI lab.

Fine-tuning, evals, post-training, distributed training.

```text
Mac = cockpit | GitHub = code | B2 = data | Lambda/Modal = workers
```

Workers are disposable. Code and data are not.

Dispatch runs from the Mac cockpit (no SSH session required):

```bash
./scripts/lab run --target modal --config configs/evals/e0b_run_spec_base.yaml
./scripts/lab run --target lambda --instance-ip 1.2.3.4 --config configs/sft/e1_qwen_0_5b_run_spec_lora_r8.yaml
```

See `configs/runtime/` for Lambda/Modal defaults. Phase 1 — clone, `.env`, B2, run job, push artifacts, kill machine.

```bash
bash scripts/check_storage.sh && bash scripts/test_storage_roundtrip.sh
```

Fix storage before Lambda.
