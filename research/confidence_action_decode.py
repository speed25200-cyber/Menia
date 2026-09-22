"""Greedy native decoding from a common prefix, optionally patched each pass.

An action is accepted only when the model itself emits one code followed by
EOS. Candidate-normalized probability is descriptive and never selects it.
"""
import torch

from research.answer_confidence_data import confidence_from_logits
from research.confidence_prefix_interventions import forward_at_shared_prefix, tensor_hash
from research.iphone_coupling_report import require


def parse_native_tokens(token_ids, candidates, meanings, eos):
    eos_ids = [eos] if type(eos) is int else eos
    require(type(eos_ids) in (list, tuple) and len(eos_ids) > 0 and
            all(type(i) is int for i in eos_ids) and len(set(eos_ids)) == len(eos_ids), 'Distinct EOS ids required')
    require(len(candidates) == len(meanings) == 2 and candidates[0] != candidates[1] and
            len(set(meanings)) == 2 and not set(eos_ids).intersection(candidates),
            'Two candidate meanings distinct from EOS required')
    valid = len(token_ids) == 2 and token_ids[0] in candidates and token_ids[1] in eos_ids
    return dict(validNativeResponse=valid,
                decision=meanings[candidates.index(token_ids[0])] if valid else None,
                firstTokenMeaning=meanings[candidates.index(token_ids[0])] if token_ids and token_ids[0] in candidates else None)


def decode_branch(model, tokenizer, branch, prefix_ids, layer, *, donor=None, basis=None,
                  max_new_tokens=2, max_input_tokens=1792):
    require(type(max_new_tokens) is int and max_new_tokens == 2, 'Fixed code/EOS budget required')
    require(type(max_input_tokens) is int and max_input_tokens > 0, 'Positive input budget')
    eos = model.generation_config.eos_token_id
    eos = [eos] if type(eos) is int else eos
    require(type(eos) is list and tokenizer.eos_token_id in eos, 'Configured EOS must include tokenizer EOS')
    candidates = branch['candidateTokenIds']; meanings = [branch['negative'], branch['positive']]
    require(all(type(c) is int and 0 <= c < model.config.vocab_size for c in candidates), 'Candidate vocabulary IDs')
    parse_native_tokens([], candidates, meanings, eos)
    original = list(branch['inputIds'])
    require(original[:len(prefix_ids)] == list(prefix_ids) and len(original) > len(prefix_ids), 'Pre-query shared prefix required')
    require(len(original)+max_new_tokens-1 <= max_input_tokens, 'Decoding would exceed budget; no truncation')
    decoded = []; captures = []; first_scores = None; first_logits = None
    reference_state = None; max_state_drift = 0.
    for step in range(max_new_tokens):
        ids = torch.tensor([original+decoded], dtype=torch.long, device=model.device)
        logits, state = forward_at_shared_prefix(model, dict(input_ids=ids, attention_mask=torch.ones_like(ids)),
                                                 prefix_ids, layer, donor=donor, basis=basis)
        require(bool(torch.isfinite(logits).all()), 'Nonfinite vocabulary logits')
        if step == 0:
            values = logits.double().cpu().tolist()
            first_scores = confidence_from_logits(values, candidates[0], candidates[1])
            first_logits = [float(values[c]) for c in candidates]
            reference_state = state['before'].detach().clone()
        else:
            max_state_drift = max(max_state_drift, float((state['before'].double()-reference_state.double()).abs().max()))
        token = int(logits.argmax())  # Full vocabulary argmax, never candidate restriction.
        decoded.append(token)
        captures.append(dict(step=step, inputTokens=ids.shape[1], layer=layer, position=state['position'],
            prefixHash=state['prefixHash'], rank=state['rank'], displacementNorm=state['displacementNorm'],
            beforeHash=tensor_hash(state['before']), afterHash=tensor_hash(state['after']),
            untouchedTokensEqual=state.get('untouchedTokensEqual', True)))
        if token in eos: break
    parsed = parse_native_tokens(decoded, candidates, meanings, eos)
    return dict(schema='menia-native-branch-greedy-decode-v1', tokenIds=decoded,
        text=tokenizer.decode(decoded, skip_special_tokens=True), stoppedAtEOS=decoded[-1] in eos,
        eosTokenIds=list(eos), reachedTokenLimit=len(decoded) == max_new_tokens and decoded[-1] not in eos,
        **parsed, candidateTokenIds=list(candidates), candidateMeanings=meanings,
        conditionalPositive=first_scores['conditionalCorrect'], candidateMass=first_scores['candidateMass'],
        candidateLogits=first_logits, firstTopIsCode=decoded[0] in candidates,
        passes=captures, maxPrefixStateDrift=max_state_drift,
        settings=dict(do_sample=False, use_cache=False, candidateRestriction=False, max_new_tokens=max_new_tokens),
        scope='Native full-vocabulary greedy code plus EOS; conditional candidate probability does not '
              'select the action. Prefix is recomputed and patched on every pass. No measured task utility.')
