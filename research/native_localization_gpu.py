"""Train small Q/V LoRA updates and evaluate Qwen's own output head.

No external classifier, remote API, confidence JSON in the prompt, or model fallback.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from research.native_localization import ARMS, CONFIG, MODEL, SEED, analyze, digest, plan, prompt, read_journal


class LowRank(nn.Module):
    def __init__(self, base, rank=8, scale=1.0):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale, self.enabled = scale, True

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + (nn.functional.linear(nn.functional.linear(x.float(), self.a), self.b) * self.scale).to(y.dtype)
        return y


def install_adapters(model, rank=8):
    model.requires_grad_(False)
    modules = {}
    for name, module in list(model.named_modules()):
        if name.endswith((".q_proj", ".v_proj")) and isinstance(module, nn.Linear):
            parent, field = name.rsplit(".", 1)
            replacement = LowRank(module, rank, CONFIG["scale"])
            setattr(model.get_submodule(parent), field, replacement)
            modules[name] = replacement
    if not modules:
        raise ValueError("No Q/V attention projections found")
    return modules


def adapter_state(modules):
    return {name+"."+part: getattr(m, part).detach().cpu().clone().contiguous() for name, m in modules.items() for part in ("a", "b")}


def load_adapter(modules, state):
    if set(state) != set(adapter_state(modules)):
        raise ValueError("Adapter key mismatch")
    with torch.no_grad():
        for name, m in modules.items():
            for part in ("a", "b"):
                target = getattr(m, part); source = state[name+"."+part]
                if target.shape != source.shape or not torch.isfinite(source).all():
                    raise ValueError("Adapter shape/value mismatch")
                target.copy_(source)


@contextmanager
def intervention(model, layer, spans, position, strength, seed):
    """One full forward; prefix spans only. Cloning is an exact sham control."""
    stats = dict(applications=0, normRelativeError=0.0, changed=False)
    if position == 0:
        yield stats
        return
    def hook(module, inputs, output):
        v = output[0] if isinstance(output, tuple) else output
        if v.ndim != 3 or v.shape[0] != 1:
            raise ValueError("Only a single unpadded full prefix is supported")
        result = v.clone()
        stats["applications"] += 1
        if 1 <= position <= 5:
            ids = spans[position-1]
            if not ids or max(ids) >= v.shape[1]:
                raise ValueError("Intervention outside prefix")
            z = v[0, ids].float()
            norms = torch.linalg.vector_norm(z, dim=-1, keepdim=True)
            if torch.any(norms == 0):
                raise ValueError("Zero activation norm")
            noise = torch.as_tensor(np.random.default_rng(seed).standard_normal(z.shape[-1]), dtype=torch.float32, device=z.device)
            unit = z / norms
            orthogonal = noise - (unit * noise).sum(-1, keepdim=True) * unit
            onorm = torch.linalg.vector_norm(orthogonal, dim=-1, keepdim=True)
            if torch.any(onorm < 1e-8):
                raise ValueError("Degenerate direction")
            rotated = ((z + strength * norms * orthogonal / onorm) / math.sqrt(1 + strength**2)).to(v.dtype)
            error = ((torch.linalg.vector_norm(rotated.float(), dim=-1, keepdim=True)-norms).abs()/norms).max()
            stats["normRelativeError"] = float(error.detach())
            stats["changed"] = not torch.equal(rotated.detach(), v[0, ids].detach())
            if stats["normRelativeError"] > 0.01 or not stats["changed"]:
                raise ValueError("Ineffective or invalid rotation")
            result[0, ids] = rotated
        elif position != 6:
            raise ValueError("Invalid position")
        return (result, *output[1:]) if isinstance(output, tuple) else result
    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield stats
    finally:
        handle.remove()


def encode(tokenizer, block, task, device):
    body, char_spans = prompt(block, task)
    text = tokenizer.apply_chat_template([dict(role="user", content=body)], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    if text.count(body) != 1:
        raise ValueError("Cannot locate user text in chat template")
    offset = text.index(body)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False,
                        return_offsets_mapping=True, return_token_type_ids=False)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    if len(offsets) > CONFIG["maxInputTokens"]:
        raise ValueError("Prefix exceeds fixed token budget; no truncation")
    spans = [[i for i, (a, b) in enumerate(offsets) if b > lo+offset and a < hi+offset and b > a]
             for lo, hi in char_spans]
    if any(not s for s in spans) or len(set(sum(spans, []))) != sum(map(len, spans)):
        raise ValueError("Empty or overlapping sentence token spans")
    return encoded.to(device), spans, digest(body)


def forward(model, inputs, spans, block, position):
    with intervention(model, block["layer"], spans, position, block["strength"], block["noiseSeed"]) as stats:
        # Decode nothing; score the native vocabulary at the same final prefix token.
        out = model(**inputs, use_cache=False, logits_to_keep=1)
    if stats["applications"] != int(position != 0):
        raise ValueError("Intervention applied an unexpected number of times")
    return out.logits[0, -1].float(), stats


def summarize_logits(logits, choices):
    probabilities = torch.softmax(logits, dim=-1)
    raw = int(logits.argmax())
    return dict(choiceLogits=logits[choices].detach().cpu().tolist(),
                choiceMass=float(probabilities[choices].sum()), rawTokenId=raw,
                rawChoice=choices.index(raw) if raw in choices else -1)


def append(path, event, **values):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(event=event, **values), ensure_ascii=False, allow_nan=False)+"\n")
        f.flush(); os.fsync(f.fileno())


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(path, resume=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file, save_file
    from research.cross_model_gpu import environment
    path = Path(path)
    if path.exists() and not resume:
        raise ValueError("Existing journal requires --resume")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metadata = environment()
    metadata["origin"] = "transformers_gpu"
    metadata["models"] = {"A": MODEL}
    metadata["settings"] = dict(dtype="bfloat16", useCache=False, thinking=False, sampling=False,
                                 firstTokenOnly=True, attention="sdpa", **CONFIG)
    metadata["nativeSourceHash"] = digest({n: Path(__file__).with_name(n).read_text(encoding="utf-8")
                                         for n in ("native_localization.py", "native_localization_gpu.py")})
    tokenizer = AutoTokenizer.from_pretrained(MODEL["id"], revision=MODEL["revision"], trust_remote_code=False)
    choices = []
    for i in range(6):
        ids = tokenizer.encode(str(i), add_special_tokens=False)
        if len(ids) != 1 or ids[0] in choices:
            raise ValueError("Choice digits must be six distinct single tokens")
        choices.append(ids[0])
    metadata["choiceTokenIds"] = choices
    fixed = plan()
    if path.exists():
        header, rows, trained, pending, events = read_journal(path)
        if header["metadata"] != metadata:
            raise ValueError("Environment or code differs from the recorded run")
        if len(rows) == len(fixed["evaluation"])*len(ARMS) and pending is None:
            for arm, sha in trained.items():
                if file_hash(path.with_suffix("."+arm+".safetensors")) != sha:
                    raise ValueError("Completed adapter changed")
            analyze(path)
            print("Tentative déjà complète ; aucune nouvelle inférence.", flush=True)
            return
        if pending is not None:
            append(path, "interrupted_request", arm=pending[0], id=pending[1],
                   reason="Interrupted deterministic forward; retry recorded separately")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        append(path, "header", plan=fixed, planHash=digest(fixed), metadata=metadata)
        rows, trained, events = {}, {}, []
    model = AutoModelForCausalLM.from_pretrained(MODEL["id"], revision=MODEL["revision"], torch_dtype=torch.bfloat16,
                 device_map={"": "cuda:0"}, attn_implementation="sdpa", use_safetensors=True, trust_remote_code=False).eval()
    if model.config._commit_hash != MODEL["revision"] or len(model.model.layers) != 36:
        raise ValueError("Unexpected Qwen revision/architecture")
    torch.manual_seed(SEED)
    modules = install_adapters(model, CONFIG["rank"])
    initial = adapter_state(modules)
    blocks = {b["id"]: b for b in fixed["blocks"]}
    cache = {}
    def inputs_for(b, task):
        key = (b["id"], task)
        if key not in cache:
            cache[key] = encode(tokenizer, b, task, model.device)
        return cache[key]
    params = [p for p in model.parameters() if p.requires_grad]
    for arm in ARMS[1:]:
        checkpoint = path.with_suffix("."+arm+".safetensors")
        if arm in trained:
            if not checkpoint.exists() or file_hash(checkpoint) != trained[arm]:
                raise ValueError("Completed adapter absent or modified; refusing to retrain after evaluation")
            continue
        if any(e["event"] == "training_start" and e["arm"] == arm for e in events):
            append(path, "training_restart", arm=arm, reason="Incomplete training restarted from identical initialization; old logs retained")
        append(path, "training_start", arm=arm, examples=len(fixed["training"]),
               trainableParameters=sum(p.numel() for p in params), initializationSeed=SEED)
        load_adapter(modules, initial)
        optimizer = torch.optim.AdamW(params, lr=CONFIG["learningRate"], weight_decay=0.0)
        for step, (block_id, position) in enumerate(fixed["training"], 1):
            b = blocks[block_id]
            inputs, spans, _ = inputs_for(b, "localize")
            optimizer.zero_grad(set_to_none=True)
            logits, stats = forward(model, inputs, spans, b, position)
            target = position if arm == "aligned" else b["shuffledTargets"][position]
            loss = nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([choices[target]], device=logits.device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, CONFIG["clipNorm"], error_if_nonfinite=True)
            optimizer.step()
            append(path, "training_step", arm=arm, step=step, block=block_id, position=position,
                   loss=float(loss.detach()), gradientNorm=float(norm), intervention=stats)
            if step % 24 == 0:
                print(f"{arm} : entraînement {step}/{len(fixed['training'])}, perte {float(loss.detach()):.3f}", flush=True)
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        state = adapter_state(modules)
        if not any(not torch.equal(state[k], initial[k]) for k in state):
            raise ValueError("Training changed no adapter weights")
        save_file(state, str(checkpoint))
        trained[arm] = file_hash(checkpoint)
        append(path, "training_complete", arm=arm, steps=len(fixed["training"]), sha256=trained[arm], checkpoint=checkpoint.name)
    for arm in ARMS:
        for m in modules.values(): m.enabled = arm != "base"
        if arm != "base":
            checkpoint = path.with_suffix("."+arm+".safetensors")
            if file_hash(checkpoint) != trained[arm]: raise ValueError("Checkpoint hash changed")
            load_adapter(modules, load_file(str(checkpoint)))
        for index, request in enumerate(fixed["evaluation"], 1):
            if (arm, request["id"]) in rows: continue
            b = blocks[request["block"]]
            inputs, spans, prompt_hash = inputs_for(b, request["task"])
            append(path, "request", arm=arm, id=request["id"], promptHash=prompt_hash,
                   adapterHash=trained.get(arm), inputTokens=int(inputs.input_ids.shape[-1]))
            torch.cuda.synchronize(); start = time.perf_counter()
            with torch.inference_mode():
                logits, stats = forward(model, inputs, spans, b, request["position"])
                result = summarize_logits(logits, choices)
            torch.cuda.synchronize()
            append(path, "result", arm=arm, id=request["id"], promptHash=prompt_hash,
                   seconds=time.perf_counter()-start, intervention=stats, **result)
            if index % 28 == 0:
                print(f"{arm} : évaluation {index}/{len(fixed['evaluation'])}", flush=True)
    report = analyze(path)
    path.with_suffix(".summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("Étude terminée ; les scores ne sont pas un verdict de conscience.", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("journal", type=Path); p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    try:
        run(a.journal, a.resume)
    except BaseException as exc:
        if a.journal.exists(): append(a.journal, "error", errorType=type(exc).__name__, message=str(exc))
        raise
