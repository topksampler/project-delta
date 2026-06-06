import json
import random
from pathlib import Path

RNG = random.Random(1337)

MODELS = {
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "7B": "Qwen/Qwen2.5-7B-Instruct",
}

DATASETS = [
    "lab-instructions-v0",
    "run-spec-v0",
    "json-schema-v0",
]

TASKS = ["sft", "eval", "generate"]
METHODS = ["none", "lora", "qlora"]
STEPS = [0, 25, 50, 100, 200]
RANKS = [0, 4, 8, 16, 32]
OUTPUTS = ["local", "b2"]


def choose_valid_spec():
    task = RNG.choice(TASKS)

    if task in {"eval", "generate"}:
        method = "none"
        max_steps = 0
        lora_r = 0
        quantized = False
    else:
        method = RNG.choice(["lora", "qlora"])
        max_steps = RNG.choice([25, 50, 100, 200])
        lora_r = RNG.choice([4, 8, 16, 32])
        quantized = method == "qlora"

    size = RNG.choice(list(MODELS.keys()))

    return {
        "task": task,
        "model": MODELS[size],
        "method": method,
        "dataset": RNG.choice(DATASETS),
        "max_steps": max_steps,
        "lora_r": lora_r,
        "quantized": quantized,
        "output_target": RNG.choice(OUTPUTS),
    }


def natural_request(spec):
    model_label = {
        "Qwen/Qwen2.5-0.5B-Instruct": "Qwen 0.5B",
        "Qwen/Qwen2.5-1.5B-Instruct": "Qwen 1.5B",
        "Qwen/Qwen2.5-7B-Instruct": "Qwen 7B",
    }[spec["model"]]

    out_phrase = {
        "local": RNG.choice(["keep outputs local", "do not upload artifacts", "save locally"]),
        "b2": RNG.choice(["upload artifacts to B2", "save the run to B2", "push outputs to object storage"]),
    }[spec["output_target"]]

    if spec["task"] == "sft":
        method_phrase = "QLoRA" if spec["method"] == "qlora" else "LoRA"
        templates = [
            f"Run a {method_phrase} SFT job on {model_label} for {spec['max_steps']} steps using {spec['dataset']}. Use rank {spec['lora_r']} and {out_phrase}.",
            f"Fine-tune {model_label} with {method_phrase}. Dataset is {spec['dataset']}. Train for {spec['max_steps']} steps, rank {spec['lora_r']}. {out_phrase}.",
            f"Do an SFT smoke run: {model_label}, {method_phrase}, {spec['dataset']}, {spec['max_steps']} steps, LoRA rank {spec['lora_r']}. {out_phrase}.",
        ]
    elif spec["task"] == "eval":
        templates = [
            f"Evaluate {model_label} on {spec['dataset']} with no training. {out_phrase}.",
            f"Run eval only for {model_label} using the {spec['dataset']} dataset. {out_phrase}.",
            f"No fine-tuning. Just evaluate {model_label} on {spec['dataset']} and {out_phrase}.",
        ]
    else:
        templates = [
            f"Generate samples from {model_label} using {spec['dataset']}. {out_phrase}.",
            f"Run generation only on {model_label}. Use {spec['dataset']} prompts and {out_phrase}.",
            f"No training, just generate outputs with {model_label} for {spec['dataset']}. {out_phrase}.",
        ]

    return RNG.choice(templates)


def to_chat_example(spec):
    prompt = natural_request(spec)
    answer = json.dumps(spec, indent=2)
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Convert the user's experiment request into strict JSON. "
                    "Return JSON only. No prose. No markdown."
                ),
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "spec": spec,
        "prompt": prompt,
    }


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    out_dir = Path("data/run_spec_v0")
    rows = [to_chat_example(choose_valid_spec()) for _ in range(260)]

    RNG.shuffle(rows)
    train = rows[:220]
    eval_rows = rows[220:]

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "eval.jsonl", eval_rows)

    schema = {
        "task": ["sft", "eval", "generate"],
        "model": list(MODELS.values()),
        "method": ["none", "lora", "qlora"],
        "dataset": DATASETS,
        "max_steps": "integer",
        "lora_r": "integer",
        "quantized": "boolean",
        "output_target": ["local", "b2"],
    }

    with open(out_dir / "schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print(f"Wrote {len(train)} train rows")
    print(f"Wrote {len(eval_rows)} eval rows")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
