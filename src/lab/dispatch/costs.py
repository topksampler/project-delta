import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from lab.dispatch import lambda_driver


def load_pricing(repo_root: Path) -> dict:
    path = repo_root / "configs/runtime/pricing.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def lambda_rate_usd_per_hour(instance_type: str, pricing: dict) -> float:
    try:
        data = lambda_driver._request("GET", "/instance-types")
        entry = data.get("data", {}).get(instance_type, {})
        cents = entry.get("instance_type", {}).get("price_cents_per_hour")
        if cents is not None:
            return float(cents) / 100.0
    except Exception:
        pass
    fallback = pricing.get("lambda", {}).get("instance_usd_per_hour", {})
    return float(fallback.get(instance_type, fallback.get("gpu_1x_a10", 1.29)))


def modal_rate_usd_per_hour(gpu: str, pricing: dict) -> float:
    rates = pricing.get("modal", {}).get("gpu_usd_per_hour", {})
    return float(rates.get(gpu, rates.get("T4", 0.59)))


def billable_seconds(target: str, stages: dict[str, float], *, launched: bool) -> float:
    if target == "modal":
        return float(stages.get("modal_dispatch", stages.get("modal_run", 0.0)))

    if launched:
        keys = ("lambda_launch", "lambda_boot", "rsync_code", "remote_job", "lambda_terminate")
        return sum(float(stages.get(k, 0.0)) for k in keys)

    return sum(float(stages.get(k, 0.0)) for k in ("rsync_code", "remote_job"))


def estimate_run_cost(
    *,
    repo_root: Path,
    run_id: str,
    target: str,
    stages: dict[str, float],
    total_sec: float,
    launched: bool = False,
    modal_gpu: str | None = None,
    lambda_instance_type: str | None = None,
) -> dict:
    pricing = load_pricing(repo_root)
    billable = billable_seconds(target, stages, launched=launched)

    if target == "modal":
        gpu = modal_gpu or pricing.get("modal", {}).get("default_gpu", "T4")
        rate = modal_rate_usd_per_hour(gpu, pricing)
        resource = gpu
    else:
        instance_type = (
            lambda_instance_type
            or pricing.get("lambda", {}).get("default_instance_type", "gpu_1x_a10")
        )
        rate = lambda_rate_usd_per_hour(instance_type, pricing)
        resource = instance_type

    estimated = round((billable / 3600.0) * rate, 4)

    return {
        "run_id": run_id,
        "target": target,
        "resource": resource,
        "rate_usd_per_hour": rate,
        "billable_seconds": round(billable, 2),
        "wall_seconds": round(total_sec, 2),
        "estimated_cost_usd": estimated,
        "method": "timings_x_list_rate",
        "launched_instance": launched,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "notes": "Estimate only. Lambda credits deduct weekly; Modal may include free credits.",
    }


def estimate_from_timings(
    *,
    repo_root: Path,
    timings_path: Path,
    launched: bool = False,
    modal_gpu: str | None = None,
    lambda_instance_type: str | None = None,
) -> dict | None:
    if not timings_path.exists():
        return None
    data = json.loads(timings_path.read_text(encoding="utf-8"))
    stages = data.get("stages_sec", {})
    run_id = data.get("run_id", timings_path.parent.name)
    target = data.get("target", "unknown")
    return estimate_run_cost(
        repo_root=repo_root,
        run_id=run_id,
        target=target,
        stages=stages,
        total_sec=float(data.get("total_sec", 0)),
        launched=launched,
        modal_gpu=modal_gpu,
        lambda_instance_type=lambda_instance_type,
    )


def write_cost(path: Path, cost: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cost, indent=2) + "\n", encoding="utf-8")


def credit_baselines() -> dict[str, float | None]:
    pricing_path = Path(__file__).resolve().parents[3] / "configs/runtime/pricing.yaml"
    with open(pricing_path, "r", encoding="utf-8") as f:
        pricing = yaml.safe_load(f)
    credits = pricing.get("credits", {})

    def _val(env_key: str, yaml_key: str) -> float | None:
        if os.environ.get(env_key):
            return float(os.environ[env_key])
        val = credits.get(yaml_key)
        return float(val) if val is not None else None

    return {
        "lambda_usd": _val("LAMBDA_CREDITS_USD", "lambda_usd"),
        "modal_usd": _val("MODAL_CREDITS_USD", "modal_usd"),
    }
