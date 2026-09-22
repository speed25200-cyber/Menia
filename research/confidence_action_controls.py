"""Explicit-information controls in the same shared-prefix action context.

The supplied probability is stipulated task information, never a claim about
the real answer. These controls separate expected-loss use from self-estimation.
They do not modify the frozen Colab 19 or 24 experiments.
"""
from research.confidence_action_branches import compile_action_branches
from research.confidence_prompt_fork import compile_fork
from research.iphone_coupling_report import require
from research.risk_presentation import PERCENTAGES, COST_CENTS


def expected_costs(p_percent, verification_cost):
    require(type(p_percent) is int and 0 <= p_percent <= 100, 'Integer probability in 0..100 required')
    require(type(verification_cost) is int and 0 < verification_cost < 100, 'Integer cost in 1..99 required')
    return dict(accept=100-p_percent, verify=verification_cost)


def control_grid():
    """Reuse Colab 19's finite grid, converting its cents into task points."""
    return [dict(pPercent=p, verificationCost=c) for p in PERCENTAGES for c in COST_CENTS]


def compile_explicit_controls(tokenizer, question, answer, *, p_percent, verification_cost):
    losses = expected_costs(p_percent, verification_cost)
    native = compile_action_branches(tokenizer, question, answer, verification_cost=verification_cost)
    queries = {}; specifications = {}
    common = ('Pour ce choix hypothétique, utilise les valeurs stipulées ci-dessous, '
              'sans résoudre à nouveau la question ni estimer toi-même la justesse de la réponse. ')
    for name, branch in native['branches'].items():
        if branch['kind'] != 'action':
            continue
        for information in ('probability', 'expectedLoss'):
            if information == 'probability':
                cue = (f'La probabilité que la réponse ci-dessus soit correcte est fixée '
                       f'exactement à {p_percent} %.')
            else:
                labels = dict(accept='VALIDER', verify='DEMANDER UNE VÉRIFICATION')
                ordered = [branch['codeToMeaning'][code] for code in branch['optionOrder']]
                cue = ('Les coûts moyens sont déjà calculés exactement : '+
                       ' ; '.join(f'{labels[action]} = {losses[action]} points' for action in ordered)+'.')
            key = information+'/'+name
            queries[key] = common+cue+'\n'+branch['messages'][-1]['content']
            spec = {k: branch[k] for k in ('negative', 'positive', 'codeToMeaning', 'optionOrder',
                    'verificationCost', 'errorCost', 'candidateTokenIds')}
            specifications[key] = dict(spec, kind='explicitActionControl', information=information,
                pPercent=p_percent, stipulatedExpectedCosts=dict(losses), nativeBranch=name,
                informationProvenance='stipulated_control_not_a_model_estimate')
    fork = compile_fork(tokenizer, native['branches']['judgment']['messages'][:3], queries)
    require(fork['prefixIds'] == native['prefixIds'], 'Control changed native pre-query prefix')
    for name, branch in fork['branches'].items():
        branch.update(specifications[name])
    return dict(fork, schema='menia-explicit-action-controls-preparation-v1',
        scope='Sixteen action branches with stipulated probability or precomputed expected loss. '
              'Same pre-query prefix as native action and judgment. No model self-estimation, '
              'actual tool cost, behavioral result or consciousness claim.')


def grade_decision(decision, *, p_percent, verification_cost):
    """Invalid output fails optimal-choice accuracy; its cost is undefined.

    Keep invalid counts in any accuracy denominator. Report regret only with
    its valid-response denominator, never impute an arbitrary invalid cost.
    """
    require(decision is None or decision in ('accept', 'verify'), 'Use parsed native decision or None')
    losses = expected_costs(p_percent, verification_cost)
    best = min(losses.values())
    optimal = [action for action, cost in losses.items() if cost == best]
    valid = decision is not None
    return dict(validNativeDecision=valid, decision=decision, optimalActions=optimal,
                optimalChoice=valid and decision in optimal, expectedCosts=losses,
                selectedExpectedCost=losses[decision] if valid else None,
                expectedRegret=losses[decision]-best if valid else None,
                regretDefined=valid, costsAreStipulated=True)
