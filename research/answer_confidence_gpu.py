"""Native code likelihood after an observed answer; no calibration or routing."""
import torch

from research.answer_confidence_data import input_messages, confidence_from_logits
from research.cross_model_prediction import digest
from research.iphone_coupling_report import require


def encode_confidence_training(tokenizer, example, device, *, max_input_tokens=1792):
    """Context is unsupervised; final two causal positions predict code and EOS."""
    require(example['target'] in ('0','1'), 'Binary confidence code required')
    messages=example['messages']
    require(messages==input_messages(messages[1]['content'],messages[2]['content']), 'Unexpected confidence input')
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prefix=tokenizer.encode(text,add_special_tokens=False)
    label=tokenizer.encode(example['target'],add_special_tokens=False)
    codes=[tokenizer.encode(s,add_special_tokens=False) for s in ('0','1')]
    require(all(len(c)==1 for c in codes) and codes[0]!=codes[1], 'Distinct single-token codes required')
    require(0<len(prefix) and len(prefix)+1<=max_input_tokens, 'Training overflow; no truncation')
    inputs=torch.tensor([prefix+label],device=device)
    return dict(input_ids=inputs,attention_mask=torch.ones_like(inputs)),label[0]


def assess_answer(model, tokenizer, question, answer, *, max_input_tokens=1792):
    require(not model.training, 'Evaluation mode required')
    messages=input_messages(question,answer)
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prefix=tokenizer.encode(text,add_special_tokens=False)
    codes=[tokenizer.encode(s,add_special_tokens=False) for s in ('0','1')]
    require(all(len(c)==1 for c in codes) and codes[0]!=codes[1], 'Distinct single-token codes required')
    require(0<len(prefix)<=max_input_tokens, 'Input overflow or empty input; no truncation')
    inputs=torch.tensor([prefix],device=model.device)
    with torch.inference_mode():
        logits=model(input_ids=inputs,attention_mask=torch.ones_like(inputs),use_cache=False,logits_to_keep=1).logits
        require(logits.ndim==3 and logits.shape[:2]==(1,1), 'Single final-position vocabulary required')
        values=logits[0,0].detach().double().cpu().tolist()
    result=confidence_from_logits(values,codes[0][0],codes[1][0])
    top=max(range(len(values)),key=values.__getitem__)
    result.update(inputHash=digest(messages),inputTokens=len(prefix),codeTokenIds=[c[0] for c in codes],
                  topTokenId=top,topIsCode=top in (codes[0][0],codes[1][0]),
                  sampled=False,source='Recomputed full text prefix, raw lm_head logits at temperature 1')
    return result
