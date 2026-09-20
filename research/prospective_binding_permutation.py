"""Locate only observed table-bit tokens for a future, more selective KV test.

This prepares a reversible permutation; it does not run the model or establish
that the edited hidden values encode only those bits. All other positions stay
fixed, including role markers, instructions and the generated acknowledgement.
"""
import re

from research.cross_model_prediction import digest
from research.iphone_coupling_report import require


def binding_swap(tokenizer,case,prompt_ids,prefix_length):
    require(type(prefix_length) is int and prefix_length>len(prompt_ids)>0,'Completed prefix length')
    rendered=tokenizer.apply_chat_template(case['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    encoded=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
    ids=encoded['input_ids']; offsets=encoded['offset_mapping']
    require(ids==list(prompt_ids) and len(ids)==len(offsets),'Raw observed prompt mismatch')
    matches=list(re.finditer(r'(?m)^([A-Z]{4}) = ([01])$',rendered))
    require(len(matches)==case['bindings'] and len({m[1] for m in matches})==len(matches),'Distinct complete binding table')
    require([int(m[2]) for m in matches]==case['values'],'Observed binding values changed')
    positions=[]; rows=[]
    for match in matches:
        candidates=[i for i,span in enumerate(offsets) if tuple(span)==match.span(2)]
        require(len(candidates)==1,'One token exactly covering each bit')
        position=candidates[0]; code=tokenizer.encode(match[2],add_special_tokens=False)
        require(len(code)==1 and ids[position]==code[0],'Single bit token identity')
        positions.append(position); rows.append(dict(key=match[1],bit=int(match[2]),position=position))
    zero=[r['position'] for r in rows if r['bit']==0]; one=[r['position'] for r in rows if r['bit']==1]
    require(len(zero)==len(one)>0 and len(set(positions))==len(positions),'Balanced distinct bit positions')
    permutation=list(range(prefix_length))
    for a,b in zip(zero,one): permutation[a],permutation[b]=b,a
    require(sorted(permutation)==list(range(prefix_length)) and
        all(permutation[permutation[i]]==i for i in range(prefix_length)),'Involutive permutation')
    require([i for i in range(prefix_length) if permutation[i]!=i]==sorted(positions),'Only bit positions may move')
    return dict(schema='menia-binding-position-permutation-v1',case=case['id'],promptHash=digest(prompt_ids),
        prefixLength=prefix_length,rows=rows,permutation=permutation,permutationHash=digest(permutation),
        movedPositions=sorted(positions),unchangedPositions=prefix_length-len(positions),
        scope='Position mask from observed public bindings. No native inference, task effect or self-monitoring result.')
