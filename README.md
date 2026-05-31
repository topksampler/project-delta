# lalith-ai-lab

Personal AI systems lab for fine-tuning, post-training, evals, and distributed training.

## Mental model

- GitHub stores code, scripts, configs, and documentation.
- Backblaze B2 stores durable data, checkpoints, logs, and artifacts.
- Lambda and Modal are disposable compute workers.
- Secrets live in local `.env` files or platform secret managers, never in Git.

## Local setup

```bash
cp .env.example .env
# fill in .env
bash scripts/check_storage.sh
bash scripts/test_storage_roundtrip.sh
