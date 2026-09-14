"""LoRA SFT on explicit response tokens. Run in the Colab environment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from .core import SYSTEM
from .evaluate import read_rows


def encode_row(tokenizer, row, max_length):
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["user"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    # Confidence=1 is an oracle target on trivial fixtures, not calibrated confidence.
    # Use reviewed assistant_text fields in a real corpus for richer answers.
    answer = row.get("assistant_text", json.dumps({"answer": row["expected"], "confidence": 1.0}, ensure_ascii=False))
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    suffix = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(prefix) + len(suffix) > max_length:
        raise ValueError(f"Row {row['id']} exceeds max_length; edit the dataset, do not silently truncate")
    return {"input_ids": prefix + suffix, "attention_mask": [1]*(len(prefix)+len(suffix)),
            "labels": [-100]*len(prefix) + suffix}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", default="main")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--resume", default=None)
    p.add_argument("--max-length", type=int, default=2048)
    a = p.parse_args()
    if a.steps < 1 or not 128 <= a.max_length <= 2048:
        p.error("steps >=1 and max-length between 128 and 2048 required")
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; use the Colab notebook")
    gpu = torch.cuda.get_device_properties(0)
    if "A100" not in gpu.name or gpu.total_memory < 70 * 1024**3:
        raise RuntimeError(f"Expected A100 80 GB, got {gpu.name}: {gpu.total_memory / 1024**3:.1f} GiB")
    out = Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.resume:
        raise ValueError("Use a new output directory or --resume checkpoint path")
    out.mkdir(parents=True, exist_ok=True)
    train_path, val_path = Path(a.data)/"train.jsonl", Path(a.data)/"validation.jsonl"
    train_rows, val_rows = read_rows(train_path), read_rows(val_path)
    if {r["user"] for r in train_rows} & {r["user"] for r in val_rows}:
        raise ValueError("Training/validation prompt leakage")
    manifest_path = out / "run.json"
    if a.resume:
        previous = json.loads(manifest_path.read_text())
        if previous["model"] != a.model or previous["arguments"]["max_length"] != a.max_length:
            raise ValueError("Resume model/context mismatch")
        revision = previous["revision"]
    else:
        revision = HfApi().model_info(a.model, revision=a.revision).sha
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (train_path, val_path)}
    if a.resume and hashes != previous["dataset_sha256"]:
        raise ValueError("Resume dataset mismatch")
    set_seed(17)
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_data = Dataset.from_list([encode_row(tokenizer, r, a.max_length) for r in train_rows])
    val_data = Dataset.from_list([encode_row(tokenizer, r, a.max_length) for r in val_rows])
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=revision,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM", bias="none"))
    model.enable_input_require_grads()
    args = TrainingArguments(output_dir=str(out), max_steps=a.steps,
        per_device_train_batch_size=2, per_device_eval_batch_size=2,
        gradient_accumulation_steps=8, learning_rate=5e-5, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=20, save_steps=20, save_total_limit=2,
        logging_steps=5, report_to="none", seed=17, data_seed=17, warmup_ratio=0.03)
    manifest = {"model": a.model, "revision": revision, "dataset_sha256": hashes,
        "gpu": gpu.name, "vram_bytes": gpu.total_memory, "arguments": vars(a),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "claim": "format adaptation only; subjective consciousness not established"}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (out/"pip-freeze.txt").write_text(subprocess.check_output(["python", "-m", "pip", "freeze"], text=True))
    trainer = Trainer(model=model, args=args, train_dataset=train_data, eval_dataset=val_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
        processing_class=tokenizer)
    torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=a.resume)
    trainer.save_model(str(out/"adapter"))
    tokenizer.save_pretrained(out/"adapter")
    metrics = trainer.evaluate()
    metrics["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    (out/"metrics.json").write_text(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
