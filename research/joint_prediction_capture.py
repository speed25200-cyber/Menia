"""Capture input, intermediate state and output confidence before one sampled answer.

This records features for an external predictor. It neither implements a native
self-report nor establishes introspection. Frozen collectors remain unchanged.
"""
import copy

from research.activation_monitor_gpu import project
from research.activation_monitor import DIM, validate_state
from research.cross_model_prediction import SETTINGS
from research.output_confidence_trace import trace_generation


def sample_with_joint_capture(model, tokenizer, messages, seed, capture, *, settings=SETTINGS):
    """Persist all prospective inputs once, after the head and before sampling.

    `capture` receives only input/middle/final projected states and the first
    full-vocabulary distribution summary. Completion scores and actual sampled
    token IDs are available only in the returned trace, after generation.
    """
    import torch
    cached = {}
    emitted = False
    def embedding_hook(module, inputs, output):
        if 'input' not in cached:
            if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] < 4:
                raise ValueError('One sequence with at least four prompt tokens is required')
            cached['input'] = sum((project(chunk.mean(0), 'input', DIM//4)
                                   for chunk in torch.tensor_split(output[0], 4, dim=0)), [])
    def middle_hook(module, inputs, output):
        if 'middle' not in cached:
            value = output[0] if isinstance(output, tuple) else output
            cached['middle'] = project(value[0, -1], 'middle')
    def final_hook(module, inputs, output):
        if 'final' not in cached:
            cached['final'] = project(output[0, -1], 'final')
    def prefill(confidence):
        nonlocal emitted
        if emitted:
            raise ValueError('Duplicate pre-answer capture')
        validate_state(cached)
        emitted = True
        # Isolated lists prevent a recorder from modifying the retained capture.
        capture(dict(state=copy.deepcopy(cached), preAnswer=dict(confidence)))

    handles = []
    try:
        index = len(model.model.layers)//2 - 1
        if index < 0:
            raise ValueError('At least two transformer layers required')
        handles.append(model.get_input_embeddings().register_forward_hook(embedding_hook))
        handles.append(model.model.layers[index].register_forward_hook(middle_hook))
        handles.append(model.model.norm.register_forward_hook(final_hook))
        result = trace_generation(model, tokenizer, messages, seed, settings=settings, on_prefill=prefill)
        if not emitted:
            raise ValueError('Generation completed without pre-answer capture')
        return result
    finally:
        for handle in handles:
            handle.remove()
