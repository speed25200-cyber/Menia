"""mlx-lm LoRA on lives of the need world, the loss on the kept choices only (docs/LLM_NEED_PROTOCOL.md).

Each document is one life (or one life ending on a report question) with one weight per target: each "Choix : X"
and each "Réponse : d" in order. The target token weighs its weight, every other token weighs nothing; the loss is
the weighted mean. The wrapper replaces mlx-lm's text dataset, batches and loss, then runs mlx_lm.lora, as
research/llm_weighted_lora.py does for the adjusted body.
"""
import functools
import re
import sys
import numpy as np

TARGET = re.compile(r"Choix : ([RM])|Réponse : ([01])")


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


def pad_batch(items, max_seq_length):
    """Token and target-weight arrays of a batch: weights[:, j] is the weight of predicting token j + 1."""
    length = min(max(len(t) for t, _ in items), max_seq_length)
    width = min(1 + 32 * ((length + 31) // 32), max_seq_length)
    tokens = np.zeros((len(items), width), np.int32)
    weights = np.zeros((len(items), width), np.float32)
    for j, (t, w) in enumerate(items):
        n = min(len(t), width)
        tokens[j, :n] = t[:n]
        weights[j, :n] = w[:n]
    return tokens, weights[:, 1:]


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
        return tokens, weights

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


def need_batches(dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None):
    import mlx.core as mx
    order = list(range(len(dataset)))
    batches = [order[i:i + batch_size] for i in range(0, len(order) - batch_size + 1, batch_size)]
    if seed:
        np.random.seed(seed)
    while True:
        for i in np.random.permutation(len(batches)):
            tokens, weights = pad_batch([dataset[j] for j in batches[i]], max_seq_length)
            yield mx.array(tokens), mx.array(weights)
        if not loop:
            break


def need_loss(model, batch, weights):
    import mlx.core as mx
    import mlx.nn as nn
    logits = model(batch[:, :-1])
    ce = nn.losses.cross_entropy(logits, batch[:, 1:]) * weights
    return ce.astype(mx.float32).sum() / weights.sum(), (weights > 0).sum()


def main(argv=None):
    from mlx_lm import lora
    from mlx_lm.tuner import datasets, trainer
    datasets.TextDataset = NeedText
    lora.train = functools.partial(trainer.train, loss=need_loss, iterate_batches=need_batches)
    sys.argv = ["mlx_lm.lora"] + list(sys.argv[1:] if argv is None else argv)
    lora.main()


if __name__ == "__main__":
    main()
