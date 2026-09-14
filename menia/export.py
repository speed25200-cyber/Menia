"""Merge the trained LoRA with its exact original base; no adapter-only export."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    run, out = Path(a.run), Path(a.out)
    meta = json.loads((run/"run.json").read_text())
    if out.exists() and any(out.iterdir()):
        raise ValueError("Export destination must be empty")
    model = AutoModelForCausalLM.from_pretrained(meta["model"], revision=meta["revision"], torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, run/"adapter").merge_and_unload(safe_merge=True)
    model.config.use_cache = True
    model.save_pretrained(out, safe_serialization=True, max_shard_size="2GB")
    tok = AutoTokenizer.from_pretrained(run/"adapter")
    tok.save_pretrained(out)
    checksums = {}
    for f in out.iterdir():
        if f.is_file():
            with f.open("rb") as stream:
                checksums[f.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    (out/"menia-export.json").write_text(json.dumps({"base": meta, "sha256": checksums}, indent=2))
    print(f"Merged HF model exported to {out}. Quantize separately on Mac.")

if __name__ == "__main__":
    main()
