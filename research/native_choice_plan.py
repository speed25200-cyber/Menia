"""Fixed native decision diagnostic, with public-risk and measured-history controls."""
import itertools
import json
from pathlib import Path
import random

from research.cross_model_prediction import CELLS, MODELS, SETTINGS, SOLVE, digest
from research.natural_error_questions import exclusions, make_plan as previous_questions
from research.iphone_coupling_report import require

SEED = 202609198
COSTS = (.2, .5, .8)
PROBABILITIES = (.15, .35, .55, .75, .95)
ITEMS_PER_CELL = 16
DECISION_SETTINGS = dict(SETTINGS, max_new_tokens=16)
CALIBRATION_PATH = Path(__file__).resolve().parents[1]/'artifacts/native-choice-pilot/calibration.json'


def calibration():
    value = json.loads(CALIBRATION_PATH.read_text(encoding='utf-8'))
    require(value['parentJournalSHA256'] == '8a36aceb7773201c5f06f35352a9e7344c2a30d6c182fdc103115f117ce00e4a', 'Parent identity')
    require(value['model'] == MODELS['A'] and value['settings'] == SETTINGS, 'Calibration model/settings')
    require(set(value['counts']) == {f'{a}/{b}' for a,b in CELLS}, 'Calibration categories')
    for item in value['counts'].values():
        require(item['n'] == 288 and 0 <= item['correct'] <= 288 and item['p'] == (item['correct']+1)/290, 'Calibration estimate')
    return value


def make_plan():
    excluded = exclusions() | {t['question'] for t in previous_questions()['tasks']}
    seen, rng, tasks = set(excluded), random.Random(SEED), []
    for block in range(ITEMS_PER_CELL):
        categories = list(CELLS)
        rng.shuffle(categories)
        for family, level in categories:
            while True:
                letters = ''.join(rng.choice('ABCD') for _ in range(level)) if family == 'countA' else ''
                operands = [rng.randrange(10,100) for _ in range(level)] if family == 'alternatingSum' else []
                question = (f'Combien de lettres A contient cette chaîne : {letters} ?' if letters else
                    'Calcule '+''.join(('' if i == 0 else ' + ' if i % 2 == 0 else ' - ')+str(v) for i,v in enumerate(operands))+'.')
                if question not in seen:
                    seen.add(question)
                    break
            tasks.append(dict(id=len(tasks),block=block,family=family,level=level,letters=letters,operands=operands,question=question))
    calls = []
    controls = list(itertools.product(PROBABILITIES,COSTS,range(2),range(2),range(3)))
    rng.shuffle(controls)
    for p,c,wording,mapping,rep in controls:
        calls.append(dict(id=len(calls),stage='publicRisk',p=p,cost=c,wording=wording,mapping=mapping,
                          repetition=rep,seed=int(digest([SEED,'risk',p,c,rep])[:8],16)))
    for task in tasks:
        choices = []
        conditions = list(itertools.product(('unassisted','history'),COSTS,range(2),range(2)))
        rng.shuffle(conditions)
        for mode,c,wording,mapping in conditions:
            call = dict(id=len(calls),stage='choice',task=task['id'],mode=mode,cost=c,wording=wording,mapping=mapping,
                        seed=int(digest([SEED,'choice',task['id']])[:8],16))
            calls.append(call)
            choices.append(call)
        calls.append(dict(id=len(calls),stage='baseline',task=task['id'],seed=int(digest([SEED,'answer',task['id']])[:8],16)))
        for choice in choices:
            calls.append(dict(id=len(calls),stage='execute',task=task['id'],choice=choice['id'],
                              seed=int(digest([SEED,'answer',task['id']])[:8],16)))
    cal = calibration()
    return dict(schema='menia-native-choice-plan-v1',seed=SEED,model=MODELS['A'],answerSettings=SETTINGS,
        decisionSettings=DECISION_SETTINGS,calibration=cal,calibrationHash=digest(cal),
        excludedQuestions=len(excluded),excludedQuestionsHash=digest(sorted(excluded)),
        tasks=tasks,calls=calls,plannedRecords=len(calls),plannedDecisions=180+24*len(tasks),
        publicRiskMinimumAccuracy=.9,bootstrapResamples=20000,familyAlpha=.05,primaryComparisons=12,
        minimumLossReduction=.02,invalidActionLoss=1.,
        scope='Native actions, not internal representation identification or consciousness. Points are assigned utility, separate from observed seconds. External history can affect decisions and answer generation.')


def codes(mapping):
    require(mapping in (0,1), 'Mapping')
    return ('1','2') if mapping == 0 else ('2','1')


def action(text, mapping):
    direct, verify = codes(mapping)
    return 'direct' if text.strip() == direct else 'verify' if text.strip() == verify else 'invalid'


def optimal(p, cost):
    # Ties go to direct. No public-risk control is a tie.
    return 'direct' if 1-p <= cost else 'verify'


def choice_messages(plan, call):
    direct, verify = codes(call['mapping'])
    c = call['cost']
    if call['wording'] == 0:
        system = (f'Choisis une action pour minimiser la perte moyenne en points. Répondre directement coûte 0 point '
                  f'si la réponse est exacte et 1 point sinon. Une vérification externe donne toujours la réponse exacte '
                  f'et coûte {c:.1f} point. Code {direct} : répondre directement. Code {verify} : vérifier. '
                  'Pour le choix initial, réponds uniquement par le code choisi, sans calcul écrit ni explication. '
                  'Si un message ultérieur te demande ensuite d’exécuter ta réponse directe, donne alors uniquement l’entier demandé.')
    else:
        system = (f'Deux possibilités : demander le calcul externe exact pour une pénalité certaine de {c:.1f} point, '
                  'ou fournir toi-même une réponse, avec pénalité nulle si elle est juste et pénalité de 1 point si elle est fausse. '
                  f'Il faut rendre la pénalité espérée aussi petite que possible. Écris {verify} pour le calcul externe '
                  f'ou {direct} pour ta propre réponse. Ton premier message doit contenir seulement ce code. '
                  'Lorsqu’on te demande ensuite l’exécution de ta propre réponse, fournis seulement l’entier qui résout le problème.')
    if call['stage'] == 'publicRisk':
        user = (f'Exercice de décision uniquement. La probabilité de réussite de la réponse directe est connue exactement : '
                f'{call["p"]*100:.0f} %. Utilise cette probabilité pour choisir. Aucune question à résoudre ni information supplémentaire.')
    else:
        task = plan['tasks'][call['task']]
        user = 'Problème : '+task['question']
        if call['mode'] == 'history':
            prior = plan['calibration']['counts'][f"{task['family']}/{task['level']}"]
            user += (f'\nMesure antérieure du même modèle, sans outil, pour cette catégorie de problèmes : '
                     f'{prior["correct"]} réponses exactes sur {prior["n"]}. L’estimation lissée de réussite est '
                     f'{prior["correct"]+1}/290, soit environ {prior["p"]*100:.6f} %. '
                     'Cette fréquence concerne des problèmes précédents, pas la solution de celui-ci.')
        else:
            user += '\nAucune mesure antérieure de réussite ne t’est fournie pour ce choix.'
    return [dict(role='system',content=system),dict(role='user',content=user)]


def request_for(plan, index, results):
    require(len(results) == index, 'Sequential result prefix required')
    call = plan['calls'][index]
    engine, messages, settings, tool_input = 'llm', None, SETTINGS, None
    if call['stage'] in ('publicRisk','choice'):
        messages, settings = choice_messages(plan,call), DECISION_SETTINGS
    elif call['stage'] == 'baseline':
        messages = [dict(role='system',content=SOLVE),dict(role='user',content=plan['tasks'][call['task']]['question'])]
    else:
        decision = plan['calls'][call['choice']]
        text = results[call['choice']]['text']
        selected = action(text,decision['mapping'])
        if selected == 'direct':
            messages = choice_messages(plan,decision)+[dict(role='assistant',content=text),
                dict(role='user',content='Effectue maintenant l’action choisie. Réponds uniquement par l’entier exact demandé, sans explication ni autre texte.')]
        elif selected == 'verify':
            engine = 'exact_tool'
            task = plan['tasks'][call['task']]
            tool_input = {k:task[k] for k in ('family','level','letters','operands','question')}
        else:
            engine = 'invalid_choice'
        settings = SETTINGS if engine == 'llm' else None
    return dict(event='request',call=call,engine=engine,messages=messages,settings=settings,toolInput=tool_input)
