"""Actual local HF inference; do not confuse these outputs with rule baselines."""
import argparse
import json
from pathlib import Path
from .core import SYSTEM
from .evaluate import read_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", default="main")
    p.add_argument("--adapter")
    p.add_argument("--ablation", choices=["none", "no-system", "no-state"], default="none")
    a = p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from huggingface_hub import HfApi
    set_seed(17)
    revision = a.revision
    if a.adapter:
        run = json.loads((Path(a.adapter).parent/"run.json").read_text())
        if run["model"] != a.model:
            raise ValueError("Adapter/base mismatch")
        revision = run["revision"]
    else:
        revision = HfApi().model_info(a.model, revision=revision).sha
    tok = AutoTokenizer.from_pretrained(a.model, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=revision,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    rows = read_rows(a.data)
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f, torch.inference_mode():
        for row in rows:
            user = row["user"]
            if a.ablation == "no-state":
                user = user.replace(json.dumps(row["state"], ensure_ascii=False), "{}", 1)
            messages = [] if a.ablation == "no-system" else [{"role": "system", "content": SYSTEM}]
            messages.append({"role": "user", "content": user})
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            if ids.input_ids.shape[-1] > 1856:
                raise ValueError("Prompt exceeds mobile test budget")
            output = model.generate(**ids, max_new_tokens=192, do_sample=False, pad_token_id=tok.eos_token_id)
            text = tok.decode(output[0, ids.input_ids.shape[-1]:], skip_special_tokens=True)
            f.write(json.dumps({"id": row["id"], "output": text}, ensure_ascii=False)+"\n")
            f.flush()
    path.with_suffix(".manifest.json").write_text(json.dumps({"model": a.model, "revision": revision,
        "adapter": a.adapter, "ablation": a.ablation, "n": len(rows)}, indent=2))

if __name__ == "__main__":
    main()
