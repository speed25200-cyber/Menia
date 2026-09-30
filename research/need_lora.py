"""mlx-lm LoRA on lives of the need world, the loss on the kept choices only (docs/LLM_NEED_PROTOCOL.md).

Each document is one life (or one life ending on a report question) with one weight per target: each "Choix : X"
and each "Réponse : d" in order. The target token weighs its weight, every other token weighs nothing; the loss is
the weighted mean. The wrapper replaces mlx-lm's text dataset, batches and loss, then runs mlx_lm.lora, as
research/llm_weighted_lora.py does for the adjusted body.

A document with "workspace" (docs/LLM_NEED_WORKSPACE_PROTOCOL.md) ends on a question whose tokens see only the
header, the three tokens of the pending "Choix :" and the question itself: its batch carries an attention mask, which
the Qwen3 blocks use in place of the causal mask. A document with "reader" too (docs/LLM_NEED_READER_PROTOCOL.md)
carries the positions of the question: the adapter adds its update there only, so that it reads "Choix :" without
changing how it is computed.
"""
import functools
import re
import sys
import numpy as np

from .need_world import HEADER

TARGET = re.compile(r"Choix : ([RM])|Réponse : ([01])")
WORKSPACE = 3  # " Cho" "ix" " :"


def target_tokens(text, weights, encode):
    """Tokens of the text and the weight of each token as a target; the text is cut before and after each target
    and each piece encoded alone, which the caller checks against encoding the whole text. The target of a choice
    is " R" or " M" (one token with its space); that of an answer is the digit alone (the space before a digit is
    its own token)."""
    matches = list(TARGET.finditer(text))
    if len(matches) != len(weights):
        raise ValueError(f"{len(matches)} targets in the text, {len(weights)} weights")
    tokens, out, start = [], [], 0
    for match, w in zip(matches, weights):
        group = 1 if match.group(1) is not None else 2
        cut = match.start(group) - (1 if group == 1 else 0)
        before = list(encode(text[start:cut]))
        target = list(encode(text[cut:match.end(group)]))
        tokens += before + target
        out += [0.0] * len(before) + [float(w)] * len(target)
        start = match.end(group)
    rest = list(encode(text[start:]))
    return tokens + rest, out + [0.0] * len(rest)


def workspace_mask(size, span=None):
    """Which positions each position attends to (size x size, True = seen): causal; with span = (header, end), the
    positions from `end` on (the question) see only the header, the WORKSPACE tokens before `end` (the pending
    "Choix :") and the question up to themselves."""
    rows, cols = np.arange(size)[:, None], np.arange(size)[None, :]
    mask = cols <= rows
    if span is not None:
        header, end = span
        seen = (cols < header) | ((cols >= end - WORKSPACE) & (cols < end)) | (cols >= end)
        mask &= (rows < end) | seen
    return mask


def pad_batch(items, max_seq_length):
    """Token and target-weight arrays of a batch: weights[:, j] is the weight of predicting token j + 1. If an item
    has a workspace span, a third array: the attention mask of the inputs (batch x 1 x inputs x inputs)."""
    length = min(max(len(item[0]) for item in items), max_seq_length)
    width = min(1 + 32 * ((length + 31) // 32), max_seq_length)
    tokens = np.zeros((len(items), width), np.int32)
    weights = np.zeros((len(items), width), np.float32)
    for j, (t, w, *_) in enumerate(items):
        n = min(len(t), width)
        tokens[j, :n] = t[:n]
        weights[j, :n] = w[:n]
    if all(len(item) == 2 for item in items):
        return tokens, weights[:, 1:]
    masks = np.stack([workspace_mask(width - 1, item[2] if len(item) >= 3 else None) for item in items])
    if all(len(item) < 4 for item in items):
        return tokens, weights[:, 1:], masks[:, None]
    positions = np.zeros((len(items), width - 1, 1), np.float32)
    for j, item in enumerate(items):
        if len(item) == 4:
            positions[j, item[2][1]:] = 1.0  # the question, where the reader acts
    return tokens, weights[:, 1:], masks[:, None], positions


class NeedText:
    """Replaces mlx-lm's TextDataset: each document gives its tokens and target weights."""

    def __init__(self, data, tokenizer, text_key="text"):
        self._data, self.tokenizer, self.text_key = data, tokenizer, text_key

    def process(self, d):
        encode = lambda s: self.tokenizer.encode(s, add_special_tokens=False)
        tokens, weights = target_tokens(d[self.text_key], d["weights"], encode)
        if tokens != list(encode(d[self.text_key])):
            raise RuntimeError("cutting at the targets changed the tokens")
        if sum(weights) <= 0:
            raise RuntimeError("a document without a weighted target")
        if "workspace" not in d:
            return tokens, weights
        header, before = list(encode(HEADER)), list(encode(d[self.text_key][:d["workspace"]]))
        end = len(before)
        if tokens[:len(header)] != header or tokens[:end] != before:
            raise RuntimeError("the header or the text up to the pending choice is cut differently")
        if tokens[end - WORKSPACE:end] != list(encode(" Choix :")):
            raise RuntimeError("the workspace is not the tokens of \" Choix :\"")
        if d.get("reader"):
            return tokens, weights, (len(header), end), True
        return tokens, weights, (len(header), end)

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


def need_batches(dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None):
    import mlx.core as mx
    order = list(range(len(dataset)))
    batches = [order[i:i + batch_size] for i in range(0, len(order) - batch_size + 1, batch_size)]
    masked = any(len(dataset[j]) >= 3 for j in order)  # then every batch carries a mask
    if any(len(dataset[j]) == 4 for j in order) and not all(len(dataset[j]) == 4 for j in order):
        raise ValueError("a reader learns from questions only")
    if seed:
        np.random.seed(seed)
    while True:
        for i in np.random.permutation(len(batches)):
            arrays = pad_batch([dataset[j] for j in batches[i]], max_seq_length)
            if masked and len(arrays) == 2:
                arrays += (workspace_mask(arrays[0].shape[1] - 1)[None, None].repeat(len(batches[i]), 0),)
            yield tuple(mx.array(x) for x in arrays)
        if not loop:
            break


MASK = []  # the attention mask of the batch being run, read by the patched Qwen3 blocks
READER = []  # where the adapter acts (batch x inputs x 1), read by the patched LoRA layers


def masked_attention(original):
    def create_attention_mask(h, cache=None, *args, **kwargs):
        return MASK[-1] if MASK else original(h, cache, *args, **kwargs)
    return create_attention_mask


def reader_lora(lora_class):
    """LoRALinear whose update is multiplied by READER[-1] when set (the linear part is unchanged)."""
    def __call__(self, x):
        y = self.linear(x)
        z = self.scale * ((self.dropout(x) @ self.lora_a) @ self.lora_b)
        if READER:
            z = z * READER[-1]
        return y + z.astype(x.dtype)
    lora_class.__call__ = __call__


def need_loss(model, batch, weights, mask=None, positions=None):
    import mlx.core as mx
    import mlx.nn as nn
    if mask is not None:
        MASK.append(mask)
    if positions is not None:
        READER.append(positions)
    try:
        logits = model(batch[:, :-1])
    finally:
        if mask is not None:
            MASK.pop()
        if positions is not None:
            READER.pop()
    ce = nn.losses.cross_entropy(logits, batch[:, 1:]) * weights
    return ce.astype(mx.float32).sum() / weights.sum(), (weights > 0).sum()


def main(argv=None):
    from mlx_lm import lora
    from mlx_lm.models import qwen3
    from mlx_lm.tuner import datasets, trainer
    from mlx_lm.tuner.lora import LoRALinear
    qwen3.create_attention_mask = masked_attention(qwen3.create_attention_mask)
    reader_lora(LoRALinear)
    datasets.TextDataset = NeedText
    lora.train = functools.partial(trainer.train, loss=need_loss, iterate_batches=need_batches)
    sys.argv = ["mlx_lm.lora"] + list(sys.argv[1:] if argv is None else argv)
    lora.main()


if __name__ == "__main__":
    main()
