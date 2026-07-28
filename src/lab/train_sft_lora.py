import argparse
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from lab.eval_base import choose_device
from lab.eval_run_spec import read_jsonl


def resolve_continue_adapter(cfg: dict, repo_root: Path) -> Path | None:
    """Continue training an existing LoRA when model.adapter_run_id is set."""
    model_cfg = cfg.get("model", {})
    if model_cfg.get("adapter_path"):
        path = Path(model_cfg["adapter_path"])
        return path if path.is_absolute() else repo_root / path
    run_id = model_cfg.get("adapter_run_id")
    if run_id:
        return repo_root / "runs" / run_id / "adapter"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_id = cfg["run_id"]
    model_name = cfg["model"]["name"]
    train_path = cfg["data"]["train_path"]
    eval_path = cfg["data"]["eval_path"]
    out_dir = Path(cfg["output"]["dir"])
    repo_root = Path(__file__).resolve().parents[2]

    out_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    print(f"Using device: {device}")

    train_rows = list(read_jsonl(train_path))
    eval_rows = list(read_jsonl(eval_path))

    print(f"Loaded {len(train_rows)} train rows")
    print(f"Loaded {len(eval_rows)} eval rows")

    train_dataset = Dataset.from_list(train_rows).select_columns(["messages"])
    eval_dataset = Dataset.from_list(eval_rows).select_columns(["messages"])

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )
    print(f"Loaded tokenizer: {tokenizer}")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )

    continue_adapter = resolve_continue_adapter(cfg, repo_root)
    peft_config = None
    if continue_adapter and continue_adapter.exists():
        print(f"continuing LoRA from {continue_adapter}")
        model = PeftModel.from_pretrained(
            base, str(continue_adapter), is_trainable=True
        )
    else:
        if continue_adapter:
            raise FileNotFoundError(
                f"continue adapter missing at {continue_adapter}"
            )
        model = base
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            target_modules=cfg["lora"]["target_modules"],
        )
    model.to(device)
    model.train()
    print(f"Loaded model: {model}")

    training_config = cfg["training"]

    sft_args = SFTConfig(
        output_dir=str(out_dir),
        max_steps=training_config["max_steps"],
        learning_rate=training_config["learning_rate"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        logging_steps=training_config["logging_steps"],
        save_strategy=training_config["save_strategy"],
        bf16=training_config["bf16"],
        fp16=training_config["fp16"],
        save_steps=training_config["max_steps"],
        assistant_only_loss=True,
    )

    trainer_kwargs = {
        "model": model,
        "args": sft_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")

    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(out_dir / "ledger.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Training complete: run_id={run_id} output={out_dir}")


if __name__ == "__main__":
    main()
