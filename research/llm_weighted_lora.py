"""mlx-lm LoRA on whole lives, the landing squares weighing more (docs/LLM_MOTOR_OBJECTIVE_PROTOCOL.md, amendment 1).

Each life is one text document, as in the adjusted body. Every token is a target, as with mlx-lm's text loss, but
the digit of each landing square ("... de la case p à la case q") weighs WEIGHT times more; padding weighs
nothing; the loss is the weighted mean. The wrapper replaces mlx-lm's text dataset, batches and loss, then runs
mlx_lm.lora with the given arguments.
"""
import functools
import re
import sys
import numpy as np

WEIGHT = 20.0
LANDING = re.compile(r"à la case (\d)")


def landing_tokens(text, encode):
    """Tokens of the text and the weight of each token as a target; the text is cut before and after each landing
    digit and each piece encoded alone, which the caller checks against encoding the whole text."""
    tokens, weights, start = [], [], 0
    for match in LANDING.finditer(text):
        before = list(encode(text[start:match.start(1)]))
        digit = list(encode(match.group(1)))
        tokens += before + digit
        weights += [1.0] * len(before) + [WEIGHT] * len(digit)
        start = match.end(1)
    rest = list(encode(text[start:]))
    return tokens + rest, weights + [1.0] * len(rest)


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


class WeightedText:
    """Replaces mlx-lm's TextDataset: each document gives its tokens and target weights."""

    def __init__(self, data, tokenizer, text_key="text"):
        self._data, self.tokenizer, self.text_key = data, tokenizer, text_key

    def process(self, d):
        encode = lambda s: self.tokenizer.encode(s, add_special_tokens=False)
        tokens, weights = landing_tokens(d[self.text_key], encode)
        if tokens != list(encode(d[self.text_key])):
            raise RuntimeError("cutting at the landing digits changed the tokens")
        return tokens + [self.tokenizer.eos_token_id], weights + [1.0]

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


def summary(rows, tokenizer):
    """Processes every document (raising if a cut changes the tokens) and gives the share of the loss weight on the
    landing digits."""
    dataset = WeightedText(rows, tokenizer)
    weights = [w for d in rows for w in dataset.process(d)[1][1:]]
    heavy = sum(w for w in weights if w == WEIGHT)
    return {"documents": len(rows), "targets": len(weights), "landing_targets": int(heavy / WEIGHT),
            "landing_share_of_loss": heavy / sum(weights)}


def weighted_batches(dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None):
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


def weighted_loss(model, batch, weights):
    import mlx.core as mx
    import mlx.nn as nn
    logits = model(batch[:, :-1])
    ce = nn.losses.cross_entropy(logits, batch[:, 1:]) * weights
    return ce.astype(mx.float32).sum() / weights.sum(), (weights > 0).sum()


def main(argv=None):
    from mlx_lm import lora
    from mlx_lm.tuner import datasets, trainer
    datasets.TextDataset = WeightedText
    lora.train = functools.partial(trainer.train, loss=weighted_loss, iterate_batches=weighted_batches)
    sys.argv = ["mlx_lm.lora"] + list(sys.argv[1:] if argv is None else argv)
    lora.main()


if __name__ == "__main__":
    main()
