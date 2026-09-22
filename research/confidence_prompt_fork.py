"""Compile plain Qwen chat branches with an explicit pre-query shared prefix.

Rendering a history ending at an assistant turn can differ from rendering
that same history followed by a new user turn. Do not infer the intervention
boundary from a standalone rendering or the longest common token prefix.
"""
from research.cross_model_prediction import digest
from research.iphone_coupling_report import require


def compile_fork(tokenizer, shared_messages, queries):
    require(type(shared_messages) is list and len(shared_messages)==3, 'System, question, answer required')
    require([m.get('role') for m in shared_messages]==['system','user','assistant'], 'Unexpected shared roles')
    require(all(set(m)=={'role','content'} and type(m['content']) is str for m in shared_messages), 'Plain text only')
    require(type(queries) is dict and len(queries)>=2 and all(type(k) is str and k for k in queries), 'Named branches required')
    require(all(type(q) is str and q for q in queries.values()) and len(set(queries.values()))==len(queries), 'Distinct nonempty branch queries required')
    texts=[m['content'] for m in shared_messages]+list(queries.values())
    require(not any(marker in s for s in texts for marker in ('<|im_start|>','<|im_end|>','<tool_response>','</tool_response>')), 'Reserved role/tool markers unsupported')
    def render(messages, generation):
        return tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=generation,enable_thinking=False)
    # A following empty USER fixes the role-sensitive rendering of the shared
    # assistant. Remove only its checked, complete Qwen role framing.
    empty_user='<|im_start|>user\n<|im_end|>\n'
    future_render=render(shared_messages+[dict(role='user',content='')],False)
    require(future_render.endswith(empty_user), 'Unsupported chat framing')
    prefix_text=future_render[:-len(empty_user)]
    require(prefix_text.endswith('<|im_end|>\n'), 'Unclosed shared assistant turn')
    prefix=tokenizer.encode(prefix_text,add_special_tokens=False)
    require(bool(prefix), 'Empty shared token prefix')
    standalone=render(shared_messages,False)
    standalone_ids=tokenizer.encode(standalone,add_special_tokens=False)
    branches={}
    for name,query in queries.items():
        messages=shared_messages+[dict(role='user',content=query)]
        text=render(messages,True)
        require(text.startswith(prefix_text+'<|im_start|>user\n'+query+'<|im_end|>\n'), 'Branch changed shared rendering')
        ids=tokenizer.encode(text,add_special_tokens=False)
        require(ids[:len(prefix)]==prefix and len(ids)>len(prefix), 'Tokenization changed shared boundary')
        branches[name]=dict(messages=messages,text=text,inputIds=ids,
                            inputHash=digest(messages),standaloneIsPrefix=ids[:len(standalone_ids)]==standalone_ids)
    return dict(schema='menia-confidence-prompt-fork-v1',sharedMessagesHash=digest(shared_messages),
                prefixText=prefix_text,prefixIds=prefix,prefixTokenHash=digest(prefix),
                position=len(prefix)-1,branches=branches,
                standaloneText=standalone,standaloneDiffers=standalone!=prefix_text,
                scope='Closed shared assistant turn before new user role. Full recomputation; no privileged generation cache or learned confidence direction.')
