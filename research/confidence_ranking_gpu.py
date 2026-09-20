"""Differentiable paired confidence loss on the native 0/1 vocabulary logits.

Two separate full forwards per pair; neither example can attend to its partner.
Backpropagate a pair at a time to avoid retaining eight activation graphs.
"""
import torch
from torch import nn

from research.confidence_ranking import ARMS, CONFIG
from research.iphone_coupling_report import require


def example_terms(model, encoded, eos, codes):
    inputs, target = encoded
    require(not model.training, 'Evaluation mode required; gradients remain enabled')
    require(len(codes) == 2 and codes[0] != codes[1] and target in codes,
            'Distinct binary code ids required')
    require(inputs['input_ids'].ndim == 2 and inputs['input_ids'].shape[0] == 1 and
            inputs['input_ids'].shape[1] >= 2 and int(inputs['input_ids'][0, -1]) == target,
            'One unpadded example ending in its supervised code required')
    require(bool(torch.all(inputs['attention_mask'] == 1)), 'Unpadded input required')
    logits = model(**inputs, use_cache=False, logits_to_keep=2).logits[0].float()
    require(logits.shape[0] == 2, 'Code and EOS causal positions required')
    ce = nn.functional.cross_entropy(logits, torch.tensor([target, eos], device=logits.device))
    # -2 predicts the appended code, hence cannot attend to that code. At -1
    # the code is visible, but that position is used only for EOS supervision.
    score = logits[0, codes[1]] - logits[0, codes[0]]
    require(bool(torch.isfinite(ce)) and bool(torch.isfinite(score)), 'Nonfinite loss/score')
    return ce, score


def auxiliary_loss(left, right, left_target, right_target, arm):
    require(arm in ARMS and left_target in (0, 1) and right_target in (0, 1),
            'Unknown objective or target')
    delta = left-right
    if left_target == right_target or arm == 'ce':
        return delta*0.
    if arm == 'rank':
        return nn.functional.softplus(-(left_target-right_target)*delta)
    # Expected loss of an independent fair binary target, evaluated exactly.
    # This is an ACTIVE regularizer that shrinks pair differences, not a sham
    # that is expected to leave the model unchanged. CE is the no-auxiliary arm.
    return .5*(nn.functional.softplus(delta)+nn.functional.softplus(-delta))


def train_paired_step(model, optimizer, parameters, pairs, eos, codes, arm):
    require(arm in ARMS and len(pairs) == CONFIG['pairsPerBatch'], 'Paired batch')
    require(len({id(p) for p in parameters}) == len(parameters) and
            {id(p) for p in parameters} == {id(p) for p in model.parameters() if p.requires_grad},
            'Exactly the trainable parameters required')
    optimizer.zero_grad(set_to_none=True)
    losses = []; pair_losses = []; differences = []; lengths = []; eligible = []
    for left, right in pairs:
        ce_left, s_left = example_terms(model, left, eos, codes)
        ce_right, s_right = example_terms(model, right, eos, codes)
        y_left, y_right = (int(item[1] == codes[1]) for item in (left, right))
        extra = auxiliary_loss(s_left, s_right, y_left, y_right, arm)
        loss = (ce_left+ce_right)/(2*len(pairs)) + CONFIG['pairWeight']*extra/len(pairs)
        require(bool(torch.isfinite(loss)), 'Nonfinite paired loss')
        loss.backward()
        losses.extend((float(ce_left.detach()), float(ce_right.detach())))
        pair_losses.append(float(extra.detach()))
        differences.append(float((s_left-s_right).detach()))
        lengths.extend(int(item[0]['input_ids'].shape[1]) for item in (left, right))
        eligible.append(y_left != y_right)
    norm = nn.utils.clip_grad_norm_(parameters, CONFIG['clipNorm'], error_if_nonfinite=True)
    optimizer.step()
    return dict(losses=losses, pairLosses=pair_losses, scoreDifferences=differences,
                eligible=eligible, inputTokens=lengths, gradientNorm=float(norm),
                objective=sum(losses)/len(losses)+CONFIG['pairWeight']*sum(pair_losses)/len(pairs))
