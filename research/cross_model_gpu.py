"""Actual GPU backend for the fixed cross-model pilot. No CPU or model fallback."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

from research.cross_model_prediction import MODELS, SETTINGS, SEED, collect, analyze, digest


def generate_text(model, tokenizer, messages, seed, *, settings=SETTINGS):
    """Fresh generation per request; also exercised with a tiny random Qwen on CPU."""
    import torch
    from transformers import GenerationConfig
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=settings["enable_thinking"])
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False).to(model.device)
    length = inputs.input_ids.shape[-1]
    if length > settings["max_input_tokens"]:
        raise ValueError(f"Input has {length} tokens; limit is {settings['max_input_tokens']}. No truncation performed.")
    torch.manual_seed(seed)
    if model.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
    # Construct the full sampling config, avoiding extra defaults in downloaded generation configs.
    config = GenerationConfig(
        max_new_tokens=settings["max_new_tokens"], do_sample=settings["do_sample"],
        temperature=settings["temperature"], top_p=settings["top_p"], top_k=settings["top_k"],
        min_p=settings["min_p"], repetition_penalty=settings["repetition_penalty"],
        renormalize_logits=settings["renormalize_logits"],
        use_cache=settings["use_cache"], num_beams=1, num_return_sequences=1,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    with torch.inference_mode():
        output = model.generate(**inputs, generation_config=config)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    generated = output[0, length:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), dict(
        inputTokens=length, outputTokens=len(generated),
        reachedTokenLimit=len(generated) == settings["max_new_tokens"], dtype=str(model.dtype),
        device=str(model.device), effectiveGeneration=config.to_dict())


def environment():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA absent. Choisir un GPU A100 dans Colab.")
    gpu = torch.cuda.get_device_properties(0)
    # Both models are resident (~24.4 GB of weights). This profile deliberately leaves ample workspace.
    if "A100" not in gpu.name or gpu.total_memory < 35 * 1024**3:
        raise RuntimeError("Ce profil exige un A100 de 40 ou 80 Go. Aucun repli automatique.")
    required = {"torch": "2.8.0", "transformers": "4.56.2", "numpy": "2.2.6",
                "accelerate": "1.10.1", "safetensors": "0.6.2", "huggingface-hub": "0.36.2",
                "tokenizers": "0.22.2"}
    versions = {name: importlib.metadata.version(name) for name in required}
    if any(versions[name].split("+")[0] != version for name, version in required.items()):
        raise RuntimeError(f"Dependency versions differ from protocol: {versions}")
    packages = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
    source = {name: Path(__file__).with_name(name).read_text(encoding="utf-8")
              for name in ("cross_model_prediction.py", "cross_model_gpu.py",
                           "iphone_capability_learning_report.py", "iphone_coupling_report.py")}
    repo = Path(__file__).resolve().parent.parent
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    return dict(gpu=gpu.name, gpuBytes=gpu.total_memory, cuda=torch.version.cuda, python=platform.python_version(),
                packages=packages, requiredVersions=versions, sourceHash=digest(source), codeCommit=commit,
                deterministicAlgorithms=True, cublasWorkspace=":4096:8", tf32=False,
                models=MODELS, settings=SETTINGS)


class GPUBackend:
    origin = "transformers_gpu"

    def __init__(self):
        # Set before importing torch or initializing CUDA; the notebook uses a fresh subprocess.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        self.metadata = environment()
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.loaded = {}

    def generate(self, request):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        key = request["call"]["model"]
        if key not in self.loaded:
            spec = MODELS[key]
            print(f"Loading {spec['id']} @ {spec['revision']}", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"], trust_remote_code=False)
            model = AutoModelForCausalLM.from_pretrained(spec["id"], revision=spec["revision"],
                        torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
                        attn_implementation="sdpa", use_safetensors=True, trust_remote_code=False)
            model.eval()
            if model.config._commit_hash != spec["revision"]:
                raise RuntimeError("Loaded model revision mismatch")
            self.loaded[key] = model, tokenizer
        return generate_text(*self.loaded[key], request["messages"], request["call"]["seed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Technical partial run; never presented as a complete pilot")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    backend = GPUBackend()
    try:
        collect(args.journal, backend, resume=args.resume, limit=args.limit, seed=args.seed)
    finally:
        if args.journal.exists():
            output = args.journal.with_suffix(".summary.json")
            output.write_text(json.dumps(analyze(args.journal), ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                              encoding="utf-8")
            print(f"Summary: {output}", flush=True)


if __name__ == "__main__":
    main()
