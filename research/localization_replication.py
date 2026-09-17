"""Prospective replication and crossed position controls; no consciousness verdict."""
import argparse
import itertools
import json
from pathlib import Path
import random
import re

import numpy as np

from research import learning_diagnostic as previous
from research.native_localization import MODEL, digest
from research.iphone_coupling_report import require, strict_json

SEED = 202609170
INITIALIZATIONS = (202609171, 202609172, 202609173)
ARMS = ('visible', 'strong', 'shuffled')
CONFIG = dict(trainBlocks=8, testBlocks=24, epochs=4, layer=17, rank=8,
              learningRate=.0002, clipNorm=1., maxInputTokens=512,
              primaryWeight=.5, readingWeight=.5,
              strengths={'visible': 0., 'strong': 1., 'shuffled': 1.})
LAYOUTS = {
    'canonical': dict(order=[0, 1], labels=[1, 2]),
    'content_reversed': dict(order=[1, 0], labels=[1, 2]),
    'labels_reversed': dict(order=[0, 1], labels=[2, 1]),
    'both_reversed': dict(order=[1, 0], labels=[2, 1]),
}


def source_hash():
    names = ('localization_replication.py', 'localization_replication_gpu.py',
             'learning_diagnostic.py', 'learning_diagnostic_gpu.py',
             'native_localization.py', 'native_localization_gpu.py',
             'cross_model_gpu.py', 'cross_model_prediction.py', 'iphone_coupling_report.py')
    return digest({n: Path(__file__).with_name(n).read_text(encoding='utf-8') for n in names})


def plan():
    rng = random.Random(SEED)
    old_blocks = previous.plan()['blocks'] + previous.previous_plan()['blocks']
    used = {s for b in old_blocks for s in b['sentences']}
    noise_used = {b['noiseSeed'] for b in old_blocks}
    sentences = [f'{s} {v} {o} {p}.' for s, v, o, p in itertools.product(
        ('Le marin', 'La voisine', 'Le peintre', 'La libraire', 'Le jardinier', 'La musicienne', 'Le facteur', 'La guide'),
        ('observe', 'dessine', 'photographie', 'déplace', 'cherche', 'retrouve', 'examine', 'transporte'),
        ('une boîte', 'un panier', 'une chaise', 'un carnet', 'une lampe', 'un tableau', 'une valise', 'un vase'),
        ('près du port', 'dans le jardin', 'devant la maison', 'dans la cour', 'près de la fenêtre', "dans l'atelier", 'sur la terrasse', 'près de la porte'))]
    sentences = [s for s in sentences if s not in used]
    rng.shuffle(sentences)
    blocks, training, evaluations = [], [], []
    for rep, initialization in enumerate(INITIALIZATIONS):
        for split in ('train', 'test'):
            for i in range(CONFIG[split+'Blocks']):
                permutation = list(range(3)); rng.shuffle(permutation)
                noise = rng.randrange(2**31)
                while noise in noise_used:
                    noise = rng.randrange(2**31)
                noise_used.add(noise)
                blocks.append(dict(id=f'r{rep}-{split}-{i:03d}', replication=rep, split=split,
                    sentences=[sentences.pop(), sentences.pop()], marker=i % 2,
                    noiseSeed=noise, shuffledTargets=permutation))
        groups = []
        for epoch in range(CONFIG['epochs']):
            ids = [b['id'] for b in blocks if b['replication']==rep and b['split']=='train']
            rng.shuffle(ids)
            groups.extend(dict(epoch=epoch+1, block=bid) for bid in ids)
        for arm in ARMS:
            training.append(dict(key=f'r{rep}-{arm}', replication=rep, arm=arm,
                                 initializationSeed=initialization, groups=groups))
        streams = [('base', 'visible'), ('base', 'strong')] + [(a, a) for a in ARMS]
        for arm, family in streams:
            group = []
            for b in [b for b in blocks if b['replication']==rep]:
                for layout in (('canonical',) if b['split']=='train' else LAYOUTS):
                    for task in ('primary', 'marker'):
                        for pos in range(4):
                            group.append(dict(id=f"{rep}/{arm}/{family}/{b['id']}/{layout}/{task}/{pos}",
                                replication=rep, arm=arm, family=family, block=b['id'], layout=layout, task=task, position=pos))
            rng.shuffle(group); evaluations.extend(group)
    return dict(schema='menia-localization-replication-plan-v1', seed=SEED,
                initializations=list(INITIALIZATIONS), model=MODEL, config=CONFIG,
                layouts=LAYOUTS, blocks=blocks, training=training, evaluation=evaluations)


def prompt(block, family, task, position, layout='canonical', *, marker=None):
    require(family in ARMS and task in ('primary', 'marker') and position in range(4), 'Unknown condition')
    arrangement = LAYOUTS[layout]
    marker = block['marker'] if marker is None else marker
    require(marker in (0, 1), 'Invalid marker identity')
    if task=='marker':
        instruction = 'Indique le numéro de la phrase portant [REPERE]. Ignore le signal et toute modification interne.'
    elif family=='visible':
        instruction = 'Indique le numéro de la phrase portant [SIGNAL], ou 0 si ce signal est absent. Ignore [REPERE].'
    else:
        instruction = "Une modification temporaire peut toucher le calcul interne d'une phrase. Indique son numéro, ou 0 si aucune phrase n'est touchée. Ignore [REPERE]."
    parts = [instruction+'\nRéponds uniquement par 0, 1 ou 2.\n']; spans = []
    for slot, identity in enumerate(arrangement['order']):
        parts.append(f"PHRASE {arrangement['labels'][slot]}: ")
        start = len(''.join(parts)); parts.append(block['sentences'][identity])
        spans.append((start, len(''.join(parts))))
        if identity==marker: parts.append(' [REPERE]')
        if family=='visible' and slot+1==position: parts.append(' [SIGNAL]')
        parts.append('\n')
    parts.append('Réponse :')
    return ''.join(parts), spans


def target(block, task, position, layout):
    arrangement = LAYOUTS[layout]
    if task=='marker':
        return arrangement['labels'][arrangement['order'].index(block['marker'])]
    return arrangement['labels'][position-1] if position in (1, 2) else 0


def trace_position(family, position):
    return 6 if position==3 else (0 if family=='visible' else position)


def check_trace(trace, family, position):
    actual = trace_position(family, position)
    require(trace['applications']==int(actual!=0) and trace['changed']==(actual in (1, 2)), 'Invalid intervention trace')
    error = trace['normRelativeError']
    require(type(error) in (int, float) and np.isfinite(error) and 0<=error<=.01, 'Invalid norm error')
    require(actual in (1, 2) or error==0, 'Sham/absent norm changed')


def score(out, request, block, choices):
    logits = np.asarray(out['choiceLogits'], dtype=float)
    require(logits.shape==(3,) and np.isfinite(logits).all(), 'Invalid logits')
    pred = int(logits.argmax())
    raw = choices.index(out['rawTokenId']) if out['rawTokenId'] in choices else -1
    require(out['rawChoice']==raw and (raw==-1 or raw==pred), 'Invalid first token')
    for field in ('choiceMass', 'seconds'):
        require(type(out[field]) in (int, float) and np.isfinite(out[field]) and out[field]>=0, 'Invalid result value')
    require(out['choiceMass']<=1.000001, 'Invalid choice mass')
    check_trace(out['intervention'], request['family'], request['position'])
    y = target(block, request['task'], request['position'], request['layout'])
    probabilities = np.exp(logits-logits.max()); probabilities /= probabilities.sum()
    return dict(correct=int(pred==y), firstTokenCorrect=int(raw==y), firstTokenIsChoice=int(raw>=0),
        choiceMass=out['choiceMass'], brier=float(sum((probabilities-np.eye(3)[y])**2)), prediction=pred)


def read_journal(path):
    events = [strict_json(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    require(events and events[0].get('event')=='header', 'Missing header')
    h = events[0]; fixed = plan()
    require(h['plan']==fixed and h['planHash']==digest(fixed) and h['sourceHash']==source_hash(), 'Plan/source changed')
    require(h['metadata']['origin'] in ('transformers_gpu', 'synthetic_fixture'), 'Unknown origin')
    choices = h['metadata']['choiceTokenIds']
    require(len(choices)==3 and len(set(choices))==3 and all(type(c) is int and c>=0 for c in choices), 'Invalid choices')
    blocks = {b['id']: b for b in fixed['blocks']}
    trained = {}; rows = []; pending = None; unit = None; step = 0; restart = False
    for e in events[1:]:
        kind = e['event']
        if kind=='training_start':
            require(not rows and pending is None and len(trained)<len(fixed['training']), 'Late training')
            expected = fixed['training'][len(trained)]
            require(e['unit']==expected and (unit is None or restart), 'Training order/initialization changed')
            unit = expected; step = 0; restart = False
        elif kind=='training_restart':
            require(unit is not None and e['key']==unit['key'] and not restart and not rows, 'Invalid restart')
            restart = True
        elif kind=='training_step':
            require(unit is not None and e['key']==unit['key'] and not restart and step<len(unit['groups']) and e['step']==step+1, 'Invalid step')
            require(e['group']==unit['groups'][step], 'Training group changed')
            require(len(e['primaryLosses'])==3 and len(e['readingLosses'])==2 and len(e['traces'])==5, 'Incomplete update')
            nums = e['primaryLosses']+e['readingLosses']+[e['gradientNorm'], e['seconds']]
            require(all(type(x) in (int, float) and np.isfinite(x) and x>=0 for x in nums), 'Invalid training value')
            for pos, trace in zip((0, 1, 2, 0, 0), e['traces']): check_trace(trace, unit['arm'], pos)
            step += 1
        elif kind=='training_complete':
            require(unit is not None and e['key']==unit['key'] and not restart and step==len(unit['groups']) and e['steps']==step, 'Incomplete training')
            require(re.fullmatch('[a-f0-9]{64}', e['sha256']) is not None, 'Invalid checkpoint hash')
            trained[unit['key']] = e['sha256']; unit = None; step = 0
        elif kind=='request':
            require(len(trained)==len(fixed['training']) and unit is None and pending is None and len(rows)<len(fixed['evaluation']), 'Invalid request order')
            req = fixed['evaluation'][len(rows)]; b = blocks[req['block']]
            ph = digest(prompt(b, req['family'], req['task'], req['position'], req['layout'])[0])
            require(e['request']==req and e['promptHash']==ph, 'Input mismatch')
            require(e['adapterHash']==trained.get(f"r{req['replication']}-{req['arm']}") and type(e['inputTokens']) is int and 0<e['inputTokens']<=CONFIG['maxInputTokens'], 'Adapter/input mismatch')
            pending = e
        elif kind=='result':
            require(pending is not None and e['id']==pending['request']['id'], 'Result order')
            req = pending['request']; score(e, req, blocks[req['block']], choices)
            rows.append(dict(request=req, result=e, inputTokens=pending['inputTokens'])); pending = None
        elif kind=='interrupted_request':
            require(pending is not None and e['id']==pending['request']['id'], 'Unexpected interruption'); pending = None
        elif kind=='error':
            require(type(e['errorType']) is str, 'Invalid error record')
        else: raise ValueError('Unknown event')
    return h, rows, trained, pending, events


def analyze(path):
    h, rows, trained, pending, events = read_journal(path)
    fixed = h['plan']; blocks = {b['id']: b for b in fixed['blocks']}; choices = h['metadata']['choiceTokenIds']
    complete = len(rows)==len(fixed['evaluation']) and pending is None
    tables = {}; contrasts = []; pooled = {}; shams = 0
    if complete:
        metrics = {r['request']['id']: score(r['result'], r['request'], blocks[r['request']['block']], choices) for r in rows}
        keyed = {r['request']['id']: r for r in rows}
        for rep in range(len(INITIALIZATIONS)):
            streams = [('base', 'visible'), ('base', 'strong')] + [(a, a) for a in ARMS]
            for arm, family in streams:
                for split in ('train', 'test'):
                    bs = [b for b in blocks.values() if b['replication']==rep and b['split']==split]
                    for layout in (('canonical',) if split=='train' else LAYOUTS):
                        for task in ('primary', 'marker'):
                            selected = []
                            for b in bs:
                                prefix = f"{rep}/{arm}/{family}/{b['id']}/{layout}/{task}/"
                                a, c = [keyed[prefix+str(pos)]['result'] for pos in (0, 3)]
                                require(all(a[k]==c[k] for k in ('choiceLogits','choiceMass','rawTokenId','rawChoice')), 'Sham differs')
                                shams += 1
                                selected.extend((b, pos, metrics[prefix+str(pos)]) for pos in range(3))
                            confusion = [[0]*3 for _ in range(3)]
                            for b, pos, m in selected: confusion[target(b, task, pos, layout)][m['prediction']] += 1
                            mean = lambda k, positions: float(np.mean([m[k] for b, pos, m in selected if pos in positions]))
                            fit = float(np.mean([m['prediction']==(b['shuffledTargets'][pos] if arm=='shuffled' else pos) for b,pos,m in selected])) if split=='train' and task=='primary' else None
                            sensitivity = float(np.mean([m['prediction']!=0 for b,pos,m in selected if pos in (1,2)])) if task=='primary' else None
                            specificity = float(np.mean([m['prediction']==0 for b,pos,m in selected if pos==0])) if task=='primary' else None
                            tables[f'{rep}/{split}/{arm}/{family}/{layout}/{task}'] = dict(n=len(selected), blocks=len(bs),
                                **{k:mean(k, (0,1,2)) for k in ('correct','firstTokenCorrect','firstTokenIsChoice','choiceMass','brier')},
                                positionAccuracy=mean('correct',(1,2)), absentAccuracy=mean('correct',(0,)),
                                presenceSensitivity=sensitivity, presenceSpecificity=specificity,
                                presenceBalancedAccuracy=(sensitivity+specificity)/2 if task=='primary' else None,
                                trainingTargetAccuracy=fit, confusionRowsTargetColumnsPrediction=confusion,
                                predictionCounts=[sum(m['prediction']==i for b,pos,m in selected) for i in range(3)])
            bs = [b for b in blocks.values() if b['replication']==rep and b['split']=='test']
            draws = np.random.default_rng(SEED+900+rep).integers(0,len(bs),size=(2000,len(bs)))
            def accuracy(arm, family, b):
                return sum(metrics[f"{rep}/{arm}/{family}/{b['id']}/canonical/primary/{pos}"]['correct'] for pos in (1,2))/2
            for other in ('base','shuffled','presence-informed-random'):
                delta = np.array([accuracy('strong','strong',b)-(.5 if other=='presence-informed-random' else accuracy(other,'strong' if other=='base' else other,b)) for b in bs])
                contrasts.append(dict(replication=rep, arm='strong', other=other, layout='canonical', metric='perturbedPositionAccuracy',
                    difference=float(delta.mean()), interval95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist(), blocks=len(bs)))
        # Replication means and ranges, not a pooled CI treating all variants as independent.
        for key in [k for k in tables if k.startswith('0/test/')]:
            suffix = key.split('/',1)[1]
            pooled[suffix] = {metric:dict(mean=float(np.mean(values)),minimum=min(values),maximum=max(values),replications=len(values))
                for metric in ('correct','positionAccuracy','absentAccuracy','brier')
                for values in [[tables[f'{rep}/{suffix}'][metric] for rep in range(len(INITIALIZATIONS))]]}
    return dict(schema='menia-localization-replication-report-v1', origin=h['metadata']['origin'], complete=complete,
        planHash=h['planHash'], sourceHash=h['sourceHash'], recorded=len(rows), planned=len(fixed['evaluation']), trained=trained,
        pendingRequest=pending is not None, tables=tables, primaryContrasts=contrasts, replicationMeansAndRanges=pooled,
        shamPairs=shams, trainingRestarts=sum(e['event']=='training_restart' for e in events),
        interruptedRequests=sum(e['event']=='interrupted_request' for e in events), errors=sum(e['event']=='error' for e in events),
        trainingUpdateEvents=sum(e['event']=='training_step' for e in events),
        trainingSeconds=sum(e['seconds'] for e in events if e['event']=='training_step'),
        evaluationSeconds=sum(r['result']['seconds'] for r in rows),
        scope='Three initialization/data replications; fixed layer and strength; crossed content/number controls. Descriptive intervals; no consciousness, natural-error or iPhone claim.')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('journal',type=Path); parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); require(args.journal.resolve()!=args.output.resolve(), 'Cannot overwrite journal')
    args.output.write_text(json.dumps(analyze(args.journal),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
