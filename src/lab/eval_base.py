import argparse
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import yaml
from rich import print
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def choose_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def score_contains(output: str, expected):
    out = output.lower()
    hits = []
    misses = []
    for item in expected:
        if item.lower() in out:
            hits.append(item)
        else:
            misses.append(item)
    score = len(hits) / max(1, len(expected))
    return score, hits, misses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_id = cfg["run_id"]
    model_name = cfg["model"]["name"]
    eval_path = cfg["eval"]["path"]
    out_dir = Path(cfg["eval"]["output_dir"])
    samples_path = out_dir / "samples.jsonl"
    metrics_path = out_dir / "metrics.json"
    ledger_path = out_dir / "ledger.yaml"
    config_copy_path = out_dir / "config.yaml"

    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path.unlink(missing_ok=True)

    device = choose_device()
    print(f"[bold]E0 baseline eval[/bold]")
    print(f"run_id: {run_id}")
    print(f"model:  {model_name}")
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
    )

    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
        torch_dtype=dtype,
    )

    model.to(device)
    model.eval()

    total = 0
    total_score = 0.0
    rows = []

    for ex in read_jsonl(eval_path):
        total += 1

        messages = [
            {
                "role": "system",
                "content": "You are a precise technical assistant. Answer directly.",
            },
            {"role": "user", "content": ex["prompt"]},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg["eval"].get("max_new_tokens", 128),
                do_sample=True,
                temperature=cfg["eval"].get("temperature", 0.2),
                top_p=cfg["eval"].get("top_p", 0.9),
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        score, hits, misses = score_contains(output, ex.get("expected_contains", []))
        total_score += score

        row = {
            "id": ex["id"],
            "prompt": ex["prompt"],
            "expected_contains": ex.get("expected_contains", []),
            "output": output,
            "contains_score": score,
            "hits": hits,
            "misses": misses,
        }

        append_jsonl(samples_path, row)
        rows.append(row)

        print(f"\n[bold]{ex['id']}[/bold]")
        print(f"prompt: {ex['prompt']}")
        print(f"output: {output}")
        print(f"score: {score:.2f}")

    avg_score = total_score / max(1, total)

    metrics = {
        "run_id": run_id,
        "model": model_name,
        "eval_path": eval_path,
        "num_examples": total,
        "avg_contains_score": avg_score,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    ledger = {
        "run_id": run_id,
        "type": "baseline_no_training",
        "model": model_name,
        "method": "base_generation",
        "quantized": False,
        "trained": False,
        "device": device,
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "output_dir": str(out_dir),
        "metrics": metrics,
    }

    write_json(metrics_path, metrics)

    with open(ledger_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(ledger, f, sort_keys=False)

    with open(config_copy_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print("\n[bold green]E0 complete[/bold green]")
    print(f"samples: {samples_path}")
    print(f"metrics: {metrics_path}")
    print(f"avg_contains_score: {avg_score:.3f}")


if __name__ == "__main__":
    main()
