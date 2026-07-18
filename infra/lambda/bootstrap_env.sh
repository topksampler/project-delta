#!/usr/bin/env bash
# Build and verify the worker Python environment.
#
# Durable outputs:
#   runs/<run-id>/hardware.json
#   runs/<run-id>/env.json

set -euo pipefail

REPO_DIR="${LAB_REPO_DIR:-$HOME/lalith-ai-lab}"
RUN_ID="${LAB_RUN_ID:-manual-env-build}"
PYTHON_VERSION="${LAB_PYTHON_VERSION:-3.11}"
ENV_PROFILE="${LAB_ENV_PROFILE:-auto}"
REQUIRE_GPU="${LAB_REQUIRE_GPU:-1}"

cd "$REPO_DIR"
mkdir -p "runs/${RUN_ID}"

export S3_ENDPOINT_URL="${S3_ENDPOINT_URL%%[[:space:]]}"
export PATH="$HOME/.local/bin:$PATH"

upload_receipt() {
  local path="$1"
  if [ -n "${S3_BUCKET:-}" ] && [ -n "${S3_ENDPOINT_URL:-}" ] && command -v s5cmd >/dev/null 2>&1; then
    s5cmd --endpoint-url "$S3_ENDPOINT_URL" cp "$path" "s3://${S3_BUCKET}/runs/${RUN_ID}/$(basename "$path")" || true
  fi
}

echo "[env] probing hardware"
python - <<'PY'
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

run_id = os.environ.get("LAB_RUN_ID", "manual-env-build")
out = Path("runs") / run_id / "hardware.json"

def run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except Exception as exc:
        return f"ERROR: {exc}"

nvidia_smi = run(["nvidia-smi"])
cuda_version = None
match = re.search(r"CUDA Version:\s*([0-9.]+)", nvidia_smi)
if match:
    cuda_version = match.group(1)

query = run([
    "nvidia-smi",
    "--query-gpu=name,driver_version,memory.total",
    "--format=csv,noheader",
])

out.write_text(json.dumps({
    "created_at": datetime.now(timezone.utc).isoformat(),
    "hostname": platform.node(),
    "platform": platform.platform(),
    "nvidia_smi": nvidia_smi,
    "gpu_query": query,
    "cuda_version": cuda_version,
}, indent=2) + "\n", encoding="utf-8")
print(out)
PY
upload_receipt "runs/${RUN_ID}/hardware.json"

if ! command -v uv >/dev/null 2>&1; then
  echo "[env] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

PROFILE="$(
python - <<'PY'
import json
import os
from pathlib import Path

requested = os.environ.get("LAB_ENV_PROFILE", "auto")
if requested != "auto":
    print(requested)
    raise SystemExit

run_id = os.environ.get("LAB_RUN_ID", "manual-env-build")
hardware = json.loads((Path("runs") / run_id / "hardware.json").read_text())
cuda = hardware.get("cuda_version")
if not cuda:
    print("cpu")
    raise SystemExit

major, minor = [int(part) for part in cuda.split(".")[:2]]
if (major, minor) >= (12, 8):
    print("cu128")
elif (major, minor) >= (12, 6):
    print("cu126")
elif (major, minor) >= (12, 4):
    print("cu124")
elif (major, minor) >= (12, 1):
    print("cu121")
else:
    print("cpu")
PY
)"

echo "[env] profile=${PROFILE} python=${PYTHON_VERSION}"
export LAB_RESOLVED_ENV_PROFILE="$PROFILE"

if [ ! -d .venv ]; then
  uv python install "$PYTHON_VERSION"
  uv venv --python "$PYTHON_VERSION" .venv
fi

uv pip install --python .venv/bin/python -q -r "requirements/torch-${PROFILE}.txt"
uv pip install --python .venv/bin/python -q -r requirements/runtime.txt

.venv/bin/python - <<'PY'
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

run_id = os.environ.get("LAB_RUN_ID", "manual-env-build")
profile = os.environ.get("PROFILE", "")
require_gpu = os.environ.get("LAB_REQUIRE_GPU", "1") == "1"
cuda_available = torch.cuda.is_available()
device_count = torch.cuda.device_count() if cuda_available else 0

receipt = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(),
    "python_executable": sys.executable,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": cuda_available,
    "device_count": device_count,
    "gpu_names": [torch.cuda.get_device_name(i) for i in range(device_count)] if cuda_available else [],
    "profile": os.environ.get("LAB_ENV_PROFILE", "auto"),
    "resolved_profile": os.environ.get("LAB_RESOLVED_ENV_PROFILE"),
}

out = Path("runs") / run_id / "env.json"
out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2))

if require_gpu and not cuda_available:
    raise SystemExit("GPU required but torch.cuda.is_available() is false")
PY
upload_receipt "runs/${RUN_ID}/env.json"

echo "[env] ready"
