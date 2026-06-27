#!/usr/bin/env bash
# Runs on Lambda via SSH one-shot from the Mac cockpit.
# Expects LAB_RUN_ID, LAB_REPO_URL, LAB_REPO_BRANCH, and B2 env vars in the environment.

REPO_DIR="${LAB_REPO_DIR:-$HOME/lalith-ai-lab}"
PYTHON="${LAB_PYTHON:-python3}"
LAB_STAGE_T0=$SECONDS

lab_timing() {
  local now=$SECONDS
  local delta=$((now - LAB_STAGE_T0))
  echo "[timing] $1: ${delta}s"
  LAB_STAGE_T0=$now
}

echo "=== lab remote job ==="
echo "run_id=${LAB_RUN_ID}"
echo "repo_dir=${REPO_DIR}"
export S3_ENDPOINT_URL="${S3_ENDPOINT_URL%%[[:space:]]}"

if ! command -v s5cmd >/dev/null 2>&1; then
  echo "installing s5cmd..."
  tmp="$(mktemp -d)"
  wget -q -O "${tmp}/s5cmd.tgz" \
    https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz
  tar -xzf "${tmp}/s5cmd.tgz" -C "${tmp}"
  sudo mv "${tmp}/s5cmd" /usr/local/bin/s5cmd
  rm -rf "${tmp}"
fi
lab_timing "s5cmd_install"

if [ "${LAB_SKIP_GIT:-0}" = "1" ]; then
  echo "using cockpit-synced code (skipping git)"
elif [ ! -d "$REPO_DIR/.git" ]; then
  echo "cloning repo..."
  git clone --branch "${LAB_REPO_BRANCH}" --depth 1 "${LAB_REPO_URL}" "$REPO_DIR"
fi

cd "$REPO_DIR"
if [ "${LAB_SKIP_GIT:-0}" != "1" ]; then
  git fetch origin "${LAB_REPO_BRANCH}"
  git checkout "${LAB_REPO_BRANCH}"
  git pull --ff-only origin "${LAB_REPO_BRANCH}" || true
fi
lab_timing "git_sync"

scripts/remote/bootstrap_env.sh
lab_timing "env_bootstrap"

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

MANIFEST_LOCAL="/tmp/lab-${LAB_RUN_ID}-manifest.yaml"
CONFIG_LOCAL="/tmp/lab-${LAB_RUN_ID}-config.yaml"

s5cmd --endpoint-url "$S3_ENDPOINT_URL" cp "s3://${S3_BUCKET}/runs/${LAB_RUN_ID}/manifest.yaml" "$MANIFEST_LOCAL"
s5cmd --endpoint-url "$S3_ENDPOINT_URL" cp "s3://${S3_BUCKET}/runs/${LAB_RUN_ID}/config.yaml" "$CONFIG_LOCAL"

TASK="$(python - <<'PY'
import yaml, os
with open(f"/tmp/lab-{os.environ['LAB_RUN_ID']}-manifest.yaml") as f:
    print(yaml.safe_load(f)["task"])
PY
)"

echo "task=${TASK}"

# Pull datasets referenced by the config.
python - <<'PY'
import os
import yaml
from pathlib import Path

run_id = os.environ["LAB_RUN_ID"]
with open(f"/tmp/lab-{run_id}-config.yaml") as f:
    cfg = yaml.safe_load(f)

paths = []
for key in ("data", "eval"):
    block = cfg.get(key, {})
    for field in ("train_path", "eval_path", "path"):
        p = block.get(field)
        if p:
            paths.append(p)

for rel in paths:
    local = Path(rel)
    if local.exists():
        continue
    local.parent.mkdir(parents=True, exist_ok=True)
    remote = f"s3://{os.environ['S3_BUCKET']}/datasets/{local.as_posix()}"
    os.system(
        f"s5cmd --endpoint-url {os.environ['S3_ENDPOINT_URL']} cp {remote} {local} "
        f"|| s5cmd --endpoint-url {os.environ['S3_ENDPOINT_URL']} cp "
        f"s3://{os.environ['S3_BUCKET']}/{local.as_posix()} {local}"
    )
PY
lab_timing "b2_pull_inputs"

if [ "$TASK" = "train" ]; then
  .venv/bin/python -m lab.train_sft_lora --config "$CONFIG_LOCAL"
else
  .venv/bin/python -m lab.eval_run_spec --config "$CONFIG_LOCAL"
fi
lab_timing "compute"

RUN_DIR="$(python - <<'PY'
import os, yaml
from pathlib import Path
with open(f"/tmp/lab-{os.environ['LAB_RUN_ID']}-config.yaml") as f:
    cfg = yaml.safe_load(f)
if "output" in cfg:
    print(cfg["output"]["dir"])
else:
    print(cfg["eval"]["output_dir"])
PY
)"

echo "uploading outputs from ${RUN_DIR}..."
s5cmd --endpoint-url "$S3_ENDPOINT_URL" cp "${RUN_DIR}/*" "s3://${S3_BUCKET}/runs/${LAB_RUN_ID}/"
lab_timing "b2_push_outputs"

echo "=== remote job complete (${SECONDS}s total) ==="
