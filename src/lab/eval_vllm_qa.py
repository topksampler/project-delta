"""e1_vllm short-answer eval with optional LoRA adapter and failure-mode labels."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from rich import print
from transformers import AutoModelForCausalLM, AutoTokenizer

from lab.eval_base import append_jsonl, choose_device, read_jsonl, write_json


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def score_must_contain(output: str, needles: list[str]) -> tuple[float, list[str], list[str]]:
    out = output.lower()
    hits = [n for n in needles if n.lower() in out]
    misses = [n for n in needles if n.lower() not in out]
    score = len(hits) / max(1, len(needles))
    return score, hits, misses


def classify_failure_mode(
    *,
    score: float,
    output: str,
    requires_doc: str,
    misses: list[str],
) -> str:
    lower = output.lower()
    if score >= 1.0:
        return "correct"
    if any(p in lower for p in ("unknown", "not sure", "don't know", "do not know")):
        return "abstain_wrong"
    if requires_doc == "0.23.0" and any(
        x in lower for x in ("llm_compressor.md", "0.22", "v0.22")
    ):
        return "stale_version"
    return "wrong"


def resolve_adapter_path(cfg: dict, repo_root: Path) -> Path | None:
    model_cfg = cfg.get("model", {})
    if model_cfg.get("adapter_path"):
        path = Path(model_cfg["adapter_path"])
        return path if path.is_absolute() else repo_root / path
    run_id = model_cfg.get("adapter_run_id")
    if run_id:
        return repo_root / "runs" / run_id / "adapter"
    return None


def load_model(cfg: dict, device: str, repo_root: Path):
    model_name = cfg["model"]["name"]
    trust = cfg["model"].get("trust_remote_code", False)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=trust,
    )
    adapter_path = resolve_adapter_path(cfg, repo_root)
    if adapter_path and adapter_path.exists():
        print(f"loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(base, str(adapter_path))
    else:
        if adapter_path:
            print(f"[yellow]adapter missing at {adapter_path}, using base[/yellow]")
        model = base
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    *,
    question: str,
    device: str,
    eval_cfg: dict,
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions about vLLM documentation. "
                "Reply in 1–3 sentences. If unsure, say unknown."
            ),
        },
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    temperature = float(eval_cfg.get("temperature", 0.0))
    gen_kwargs = {
        "max_new_tokens": int(eval_cfg.get("max_new_tokens", 256)),
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature <= 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = float(eval_cfg.get("top_p", 0.9))

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def aggregate(rows: list[dict]) -> dict:
    by_type: dict[str, list[float]] = defaultdict(list)
    by_requires: dict[str, list[float]] = defaultdict(list)
    by_failure: dict[str, int] = defaultdict(int)
    for row in rows:
        by_type[row["type"]].append(row["content_score"])
        by_requires[row["requires_doc"]].append(row["content_score"])
        by_failure[row["failure_mode"]] += 1
    return {
        "by_type": {k: float(sum(v) / len(v)) for k, v in by_type.items()},
        "by_requires_doc": {k: float(sum(v) / len(v)) for k, v in by_requires.items()},
        "by_failure_mode": {k: int(v) for k, v in by_failure.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    repo_root = Path(__file__).resolve().parents[2]
    run_id = cfg["run_id"]
    eval_cfg = cfg["eval"]
    eval_path = Path(eval_cfg["path"])
    if not eval_path.is_absolute():
        eval_path = repo_root / eval_path
    out_dir = Path(eval_cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    samples_path = out_dir / "samples.jsonl"
    metrics_path = out_dir / "metrics.json"
    ledger_path = out_dir / "ledger.yaml"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path.unlink(missing_ok=True)

    device = choose_device()
    print(f"[bold]e1_vllm eval[/bold] run_id={run_id} device={device}")
    model, tokenizer = load_model(cfg, device, repo_root)

    rows: list[dict] = []
    total_score = 0.0
    n = 0

    for ex in read_jsonl(str(eval_path)):
        n += 1
        question = ex["question"]
        needles = ex.get("gold", {}).get("must_contain", [])
        requires_doc = ex.get("requires_doc", "?")

        output = generate_answer(model, tokenizer, question=question, device=device, eval_cfg=eval_cfg)
        score, hits, misses = score_must_contain(output, needles)
        failure_mode = classify_failure_mode(
            score=score,
            output=output,
            requires_doc=requires_doc,
            misses=misses,
        )
        total_score += score

        row = {
            "id": ex["id"],
            "type": ex.get("type", "unknown"),
            "requires_doc": requires_doc,
            "question": question,
            "gold_must_contain": needles,
            "output": output,
            "content_score": score,
            "hits": hits,
            "misses": misses,
            "failure_mode": failure_mode,
        }
        append_jsonl(samples_path, row)
        rows.append(row)
        print(f"{ex['id']}: score={score:.2f} mode={failure_mode}")

    avg = total_score / max(1, n)
    metrics = {
        "run_id": run_id,
        "experiment_id": cfg.get("experiment_id"),
        "condition_id": cfg.get("condition_id"),
        "model": cfg["model"]["name"],
        "adapter_run_id": cfg.get("model", {}).get("adapter_run_id"),
        "num_examples": n,
        "avg_content_score": round(avg, 4),
        "accuracy": round(sum(1 for r in rows if r["failure_mode"] == "correct") / max(1, n), 4),
        "breakdown": aggregate(rows),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    write_json(metrics_path, metrics)
    ledger = {
        "run_id": run_id,
        "type": "e1_vllm_qa",
        "model": cfg["model"]["name"],
        "adapter_run_id": cfg.get("model", {}).get("adapter_run_id"),
        "device": device,
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "metrics": metrics,
    }
    with open(ledger_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(ledger, f, sort_keys=False)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"\n[bold green]eval complete[/bold green] avg={avg:.3f} accuracy={metrics['accuracy']:.3f}")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
