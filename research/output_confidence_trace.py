"""Observe raw output uncertainty without changing the frozen answer generator.

This is instrumentation, not a calibrated probability of correctness. Prefill
statistics precede sampling; sequence likelihoods require the completed answer.
Only the single-sequence generator in cross_model_gpu is supported.
"""
import math

from research.cross_model_gpu import generate_text
from research.cross_model_prediction import SETTINGS


def distribution_summary(logits):
    """Float64 CPU arithmetic on the complete, unfiltered vocabulary."""
    import torch
    values = logits.detach().to(device="cpu", dtype=torch.float64)
    if values.ndim != 1 or len(values) < 2 or not torch.isfinite(values).all():
        raise ValueError("Expected a finite full-vocabulary logit vector")
    logp = values.log_softmax(dim=0)
    p = logp.exp()
    top = p.topk(2)
    entropy = float(-(p * logp).sum())
    return dict(vocabularySize=len(values), topTokenId=int(top.indices[0]),
                maxProbability=float(top.values[0]),
                topTwoMargin=float(top.values[0] - top.values[1]),
                entropyNats=entropy, normalizedEntropy=entropy / math.log(len(values)))


class _ObservedTokenizer:
    """Pass through unchanged; retain actual IDs before decoding drops EOS."""
    def __init__(self, tokenizer):
        self.original = tokenizer
        self.generated_ids = None

    def __getattr__(self, key):
        return getattr(self.original, key)

    def __call__(self, *args, **kwargs):
        return self.original(*args, **kwargs)

    def decode(self, ids, *args, **kwargs):
        if self.generated_ids is not None:
            raise ValueError("Expected exactly one final decoding call")
        if ids.ndim != 1:
            raise ValueError("Expected one generated sequence")
        self.generated_ids = ids.detach().cpu().tolist()
        return self.original.decode(ids, *args, **kwargs)


def trace_generation(model, tokenizer, messages, seed, *, settings=SETTINGS,
                     on_prefill=None):
    """Return (text, original_metrics, trace) from one unmodified generation.

    The optional on_prefill callback runs once after the first LM head and before
    sampling; use only a read-only recorder that neither samples nor edits state.
    All hooks are removed even if recording fails. Raw logits are copied to CPU
    temporarily and are never returned or journalled in full. This adds transfer
    and CPU cost; original generation timings are not a valid cost comparison.
    """
    import torch
    if model.training:
        raise ValueError("Confidence tracing requires an evaluation-mode model")
    if not hasattr(model, "lm_head"):
        raise ValueError("Model has no supported LM head")
    observed = _ObservedTokenizer(tokenizer)
    raw = []
    prefill = None

    def capture(module, inputs, output):
        nonlocal prefill
        if output.ndim != 3 or output.shape[0] != 1:
            raise ValueError("Only one unbranched sequence is supported")
        # CPU copy, not a view that a downstream processor could overwrite.
        raw.append(output[0, -1].detach().to(device="cpu", dtype=torch.float64, copy=True))
        if len(raw) == 1:
            prefill = distribution_summary(raw[0])
            if on_prefill is not None:
                on_prefill(dict(prefill))
        # No replacement output, RNG call, or model mutation.

    handle = model.lm_head.register_forward_hook(capture)
    try:
        text, metrics = generate_text(model, observed, messages, seed, settings=settings)
    finally:
        handle.remove()
    ids = observed.generated_ids
    if not raw or ids is None or len(raw) != len(ids) or len(ids) != metrics["outputTokens"]:
        raise ValueError("Generation steps and actual token IDs do not align")
    if any(type(i) is not int or not 0 <= i < len(row) for i, row in zip(ids, raw)):
        raise ValueError("Generated token outside the observed vocabulary")
    log_probabilities = [float(row.log_softmax(0)[i]) for i, row in zip(ids, raw)]
    summaries = [distribution_summary(row) for row in raw]
    trace = dict(
        schema="menia-output-confidence-trace-v1",
        logitSource="lm_head output, before generation processors and sampling",
        distribution="raw full vocabulary; temperature 1; no top-k/top-p filtering",
        preAnswer=prefill,
        completion=dict(tokenIds=ids, tokenLogProbabilities=log_probabilities,
                        entropyNats=[s["entropyNats"] for s in summaries],
                        includesStopTokens=True,
                        sumLogProbability=math.fsum(log_probabilities),
                        meanLogProbability=math.fsum(log_probabilities) / len(ids)),
    )
    return text, metrics, trace
