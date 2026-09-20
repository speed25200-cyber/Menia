"""Counterbalanced native-action branches, separate from a confidence report.

This is instrumentation, not a frozen behavioral trial. Stated decision costs
are task penalties; they must not be confused with measured tool/GPU costs.
"""
from research.answer_confidence_data import input_messages
from research.confidence_prompt_fork import compile_fork
from research.iphone_coupling_report import require

# The inherited SOLVE system asks for an integer. Letter codes would conflict
# with that instruction; 0/1 would also reuse the confidence-code semantics.
ALPHABETS = (('2', '3'), ('4', '5'))


def action_query(code_to_action, order, verification_cost):
    require(type(verification_cost) is int and 0 < verification_cost < 100, 'Integer cost in 1..99 required')
    require(set(code_to_action.values()) == {'accept', 'verify'} and len(code_to_action) == 2, 'Two actions required')
    require(tuple(sorted(code_to_action)) in tuple(tuple(sorted(a)) for a in ALPHABETS), 'Supported action codes required')
    require(len(order) == 2 and set(order) == set(code_to_action), 'Complete option order required')
    names = dict(accept='VALIDER la réponse ci-dessus', verify='DEMANDER UNE VÉRIFICATION')
    options = '\n'.join(f'{code} : {names[code_to_action[code]]}.' for code in order)
    return ('Choisis une action pour la réponse ci-dessus. Valider coûte 0 point si '
            'la réponse est correcte et 100 points si elle est incorrecte. '
            f'Demander une vérification coûte toujours {verification_cost} points. '
            'Choisis pour minimiser le coût moyen.\n'+options+
            '\nRéponds uniquement par un seul des deux codes, sans explication ni autre texte.')


def compile_action_branches(tokenizer, question, answer, *, verification_cost=20):
    messages = input_messages(question, answer)
    queries = dict(judgment=messages[-1]['content'])
    semantics = dict(judgment=dict(kind='judgment', negative='incorrect', positive='correct',
                                  codeToMeaning={'0': 'incorrect', '1': 'correct'}))
    for alphabet in ALPHABETS:
        for swapped in (False, True):
            mapping = dict(zip(alphabet, ('verify', 'accept') if swapped else ('accept', 'verify')))
            for reversed_order in (False, True):
                order = list(reversed(alphabet)) if reversed_order else list(alphabet)
                name = f'action-{alphabet[0]}{alphabet[1]}-m{int(swapped)}-o{int(reversed_order)}'
                queries[name] = action_query(mapping, order, verification_cost)
                semantics[name] = dict(kind='action', negative='verify', positive='accept',
                    codeToMeaning=mapping, optionOrder=order, verificationCost=verification_cost, errorCost=100)
    fork = compile_fork(tokenizer, messages[:3], queries)
    for name, branch in fork['branches'].items():
        spec = semantics[name]
        codes = {meaning: code for code, meaning in spec['codeToMeaning'].items()}
        ordered = [tokenizer.encode(codes[meaning], add_special_tokens=False) for meaning in (spec['negative'], spec['positive'])]
        require(all(len(ids) == 1 for ids in ordered) and ordered[0] != ordered[1], 'Distinct single-token action codes required')
        branch.update(spec, candidateTokenIds=[ids[0] for ids in ordered])
    return dict(fork, schema='menia-native-action-branches-preparation-v1',
                scope='One judgment and eight separately compiled action branches. No judgment output '
                      'is supplied to an action branch. Code meaning and display order are independently '
                      'counterbalanced. No native-action result, tool execution, causal direction or consciousness claim.')
