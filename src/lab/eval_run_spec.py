import argparse
import json
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import yaml
from rich import print
from transformers import AutoModelForCausalLM, AutoTokenizer


REQUIRED_KEYS = [
    "task",
    "model",
    "method",
    "dataset",
    "max_steps",
    "lora_r",
    "quantized",
    "output_target",
]

ALLOWED = {
    "task": {"sft", "eval", "generate"},
    "model": {
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    },
    "method": {"none", "lora", "qlora"},
    "dataset": {"lab-instructions-v0", "run-spec-v0", "json-schema-v0"},
    "output_target": {"local", "b2"},
}


def read_jsonl(path):
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


def extract_json(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text.strip()).strip()

    try:
        return json.loads(text), text, None
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None, text, "no_json_object_found"

    candidate = match.group(0)
    try:
        return json.loads(candidate), candidate, None
    except Exception as e:
        return None, candidate, f"json_parse_error: {e}"


def schema_errors(obj):
    errors = []

    if not isinstance(obj, dict):
        return ["not_object"]

    for key in REQUIRED_KEYS:
        if key not in obj:
            errors.append(f"missing_key:{key}")

    for key in obj.keys():
        if key not in REQUIRED_KEYS:
            errors.append(f"extra_key:{key}")

    for key, allowed in ALLOWED.items():
        if key not in obj:
            continue
        value = obj[key]
        if not isinstance(value, str):
            errors.append(f"bad_type:{key}")
            continue
        if value not in allowed:
            errors.append(f"bad_value:{key}:{value}")

    if "max_steps" in obj and not isinstance(obj["max_steps"], int):
        errors.append("bad_type:max_steps")
    if "lora_r" in obj and not isinstance(obj["lora_r"], int):
        errors.append("bad_type:lora_r")
    if "quantized" in obj and not isinstance(obj["quantized"], bool):
        errors.append("bad_type:quantized")

    if obj.get("method") == "qlora" and obj.get("quantized") is not True:
        errors.append("invariant:qlora_requires_quantized_true")
    if obj.get("method") == "none":
        if obj.get("max_steps") != 0:
            errors.append("invariant:none_requires_zero_steps")
        if obj.get("lora_r") != 0:
            errors.append("invariant:none_requires_zero_rank")

    return errors


def field_accuracy(pred, expected):
    if not isinstance(pred, dict):
        return 0.0, {}

    hits = {}
    for key in REQUIRED_KEYS:
        hits[key] = pred.get(key) == expected.get(key)

    return sum(hits.values()) / len(REQUIRED_KEYS), hits


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
    print(f"[bold]Structured E0 baseline[/bold]")
    print(f"run_id: {run_id}")
    print(f"model:  {model_name}")
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
    )

    dtype = torch.float16 if device == "cuda" else torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    rows = list(read_jsonl(eval_path))

    valid_json = 0
    schema_valid = 0
    exact_match = 0
    total_field_acc = 0.0

    for i, ex in enumerate(rows):
        expected = ex["spec"]

        messages = [
            {
                "role": "system",
                "content": (
                    "Convert the user's experiment request into strict JSON. "
                    "Return JSON only. No prose. No markdown."
                ),
            },
            {"role": "user", "content": ex["prompt"]},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(device)

        gen_kwargs = {
            "max_new_tokens": cfg["eval"].get("max_new_tokens", 256),
            "pad_token_id": tokenizer.eos_token_id,
        }

        temperature = cfg["eval"].get("temperature", 0.0)
        if temperature and temperature > 0:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": cfg["eval"].get("top_p", 1.0),
                }
            )
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        parsed, extracted, parse_error = extract_json(output)
        is_valid_json = parsed is not None
        errors = schema_errors(parsed) if is_valid_json else ["invalid_json"]
        is_schema_valid = is_valid_json and not errors
        is_exact = parsed == expected if is_valid_json else False
        acc, field_hits = field_accuracy(parsed, expected)

        valid_json += int(is_valid_json)
        schema_valid += int(is_schema_valid)
        exact_match += int(is_exact)
        total_field_acc += acc

        row = {
            "id": i,
            "prompt": ex["prompt"],
            "expected": expected,
            "raw_output": output,
            "extracted_json_text": extracted,
            "parsed": parsed,
            "parse_error": parse_error,
            "schema_errors": errors,
            "valid_json": is_valid_json,
            "schema_valid": is_schema_valid,
            "exact_match": is_exact,
            "field_accuracy": acc,
            "field_hits": field_hits,
        }

        append_jsonl(samples_path, row)

        print(f"\n[bold]example {i}[/bold]")
        print(f"prompt: {ex['prompt']}")
        print(f"output: {output}")
        print(f"valid_json={is_valid_json} schema_valid={is_schema_valid} exact={is_exact} field_acc={acc:.2f}")

    n = len(rows)

    metrics = {
        "run_id": run_id,
        "model": model_name,
        "eval_path": eval_path,
        "num_examples": n,
        "valid_json_rate": valid_json / n,
        "schema_valid_rate": schema_valid / n,
        "exact_match_rate": exact_match / n,
        "avg_field_accuracy": total_field_acc / n,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    ledger = {
        "run_id": run_id,
        "type": "structured_output_baseline_no_training",
        "model": model_name,
        "method": "base_generation",
        "trained": False,
        "quantized": False,
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

    print("\n[bold green]E0b complete[/bold green]")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
