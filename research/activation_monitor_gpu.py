"""Read-only hooks on the actual first prefill; no repeated or altered answer pass."""
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np

from research.activation_monitor import DIM, PROJECTIONS, collect, messages
from research.cross_model_prediction import MODELS, SETTINGS, digest
from research.cross_model_gpu import environment, generate_text


@lru_cache(maxsize=12)
def projection(width, dimension, seed):
    return np.random.default_rng(seed).standard_normal((width, dimension)) / np.sqrt(width)


def project(vector, name, dimension=DIM):
    v = vector.detach().float().cpu().numpy().astype(np.float64)
    return (v @ projection(len(v), dimension, PROJECTIONS[name])).tolist()


def sample_with_states(model, tokenizer, task, capture, *, settings=SETTINGS):
    """Capture and persist states before lm_head/sampling. Hooks return None."""
    cached = {}
    def embedding_hook(module, inputs, output):
        if "input" not in cached:
            # Four positional means preserve coarse order in the input-only control.
            import torch
            require_length = output.shape[1]
            if require_length < 4:
                raise ValueError("Prompt too short for four positional input bins")
            cached["input"] = sum((project(chunk.mean(0), "input", DIM//4)
                                   for chunk in torch.tensor_split(output[0], 4, dim=0)), [])
    def middle_hook(module, inputs, output):
        if "middle" not in cached:
            value = output[0] if isinstance(output, tuple) else output
            cached["middle"] = project(value[0, -1], "middle")
    def final_hook(module, inputs, output):
        if "final" not in cached:
            cached["final"] = project(output[0, -1], "final")
            # This hook executes before the LM head and before multinomial sampling.
            capture(dict(cached))
    handles = []
    try:
        handles.append(model.get_input_embeddings().register_forward_hook(embedding_hook))
        middle_index = len(model.model.layers)//2 - 1
        if middle_index < 0:
            raise ValueError("At least two transformer layers required")
        handles.append(model.model.layers[middle_index].register_forward_hook(middle_hook))
        handles.append(model.model.norm.register_forward_hook(final_hook))
        return generate_text(model, tokenizer, messages(task), task["seed"], settings=settings)
    finally:
        for handle in handles:
            handle.remove()


class ActivationBackend:
    origin = "transformers_gpu"

    def __init__(self):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.metadata = environment()
        self.metadata["models"] = {"A": MODELS["A"]}
        self.metadata["monitorSourceHash"] = digest({name: Path(__file__).with_name(name).read_text(encoding="utf-8")
            for name in ("activation_monitor.py", "activation_monitor_gpu.py")})
        spec = MODELS["A"]
        self.tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"], trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(spec["id"], revision=spec["revision"],
            torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa",
            use_safetensors=True, trust_remote_code=False).eval()
        if self.model.config._commit_hash != spec["revision"]:
            raise RuntimeError("Model revision mismatch")
        self.metadata["readoutSites"] = dict(middleLayerIndex=len(self.model.model.layers)//2-1,
            final="model.norm output", position="last prompt token", projectionDimension=DIM)

    def generate(self, task, capture):
        return sample_with_states(self.model, self.tokenizer, task, capture)


if __name__ == "__main__":
    import argparse
    from research.activation_monitor import analyze
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    backend = ActivationBackend()
    collect(args.journal, backend, resume=args.resume)
    result = analyze(args.journal)
    args.journal.with_suffix(".summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
