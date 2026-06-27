# lalith-ai-lab — agent notes

## lessons

### B2 + s5cmd: export creds before every child process

**Symptom:** `s5cmd cp` or `s5cmd ls` hangs for minutes (or forever). No clear error unless you `timeout` the command (`fetching region failed: RequestCanceled`).

**Cause:** `source .env` without `set -a` sets shell variables but does **not export** them. `s5cmd` is a child process and never receives `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`. It stalls on credential/region discovery instead of failing fast.

**Fix (shell):**
```bash
set -a
source .env
set +a
s5cmd --endpoint-url "${S3_ENDPOINT_URL%%[[:space:]]}" cp ...
```

**Fix (Python dispatch):** set `os.environ` before `subprocess.run`, or pass env explicitly. `./scripts/lab` already uses `set -a`.

**Also:** trim trailing whitespace on `S3_ENDPOINT_URL` in `.env`.

**Invariant:** every storage child process must receive B2 creds — shell (`set -a`), Python (`os.environ`), Modal (`lalith-lab` secret), Lambda (SSH env injection).

### Lambda: private GitHub repo cannot `git clone` over HTTPS

**Symptom:** remote job exits 128 — `fatal: could not read Username for 'https://github.com'`.

**Cause:** worker tries to clone a private repo without credentials.

**Fix:** cockpit pushes code via `rsync` before the job (`LAB_SKIP_GIT=1`). GitHub clone is fallback for public repos or when a `GITHUB_TOKEN` is configured later.

### Lambda CUDA: never install unqualified `torch` on GPU workers

**Symptom:** `nvidia-smi` works, but PyTorch prints `CUDA initialization: The NVIDIA driver on your system is too old` and `torch.cuda.is_available()` is false. Jobs silently fall back to CPU unless guarded.

**Cause:** `pip install torch` can install a PyTorch wheel whose bundled CUDA runtime is newer than the Lambda image's NVIDIA driver supports.

**Fix:** probe `nvidia-smi`, select an explicit PyTorch wheel profile (`cu128`, `cu126`, `cu124`, `cu121`, or `cpu`), install it first, then install the rest of the runtime stack. Verify CUDA before running GPU jobs.

**Invariant:** GPU workers must write `hardware.json` and `env.json`, and GPU runs must fail fast when `torch.cuda.is_available()` is false unless `--allow-cpu` is explicit.
