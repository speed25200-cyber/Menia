"""Colab21: separate numerical comparison, named choice and symbol translation."""
import hashlib
import itertools
import json
from pathlib import Path

from research import action_binding as parent
from research.cross_model_prediction import digest
from research.iphone_coupling_report import require, strict_json
from research.native_choice_journal import validate_result

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / 'artifacts/action-binding-pilot/receipt.json'
NAMES = {'direct': 'DIRECT', 'verify': 'VERIFIER'}
FILES = parent.FILES + ('action_decomposition.py', 'action_decomposition_gpu.py', 'audit_action_decomposition.py')


def source_hash():
    files = {n: Path(__file__).with_name(n).read_text(encoding='utf-8') for n in FILES}
    files['receipt.json'] = RECEIPT.read_text(encoding='utf-8')
    return digest(files)


def make_plan():
    prior = parent.make_plan()
    receipt = json.loads(RECEIPT.read_text(encoding='utf-8'))
    calls = []
    # Replay all controls first. No novel response is sampled unless they match.
    for c in prior['calls']:
        if c['information'] == 'expectedLoss' and c['wording'] == 0 and c['order'] == 1 and c['mapping'] == 0:
            calls.append(dict(c, id=len(calls), stage='replay', parentId=c['id']))
    for rep, arm in itertools.product(range(3), parent.ARMS):
        for stage in ('number', 'semantic'):
            for case, w, order in itertools.product(range(16), range(3), range(2)):
                seed = int(digest([parent.SEED, 'evaluation', rep, case])[:8], 16)
                calls.append(dict(id=len(calls), stage=stage, replication=rep, arm=arm,
                                  case=case, wording=w, order=order, seed=seed))
        for w, sym, order, mapping, selected in itertools.product(range(3), ('digits','letters'), range(2), range(2), NAMES):
            calls.append(dict(id=len(calls), stage='mapping', replication=rep, arm=arm,
                              wording=w, symbols=sym, order=order, mapping=mapping, selected=selected,
                              seed=int(digest([202609210, 'mapping', rep])[:8],16)))
    return dict(schema='menia-action-decomposition-plan-v1', model=prior['model'],
                settings=prior['settings'], cases=prior['testCases'], calls=calls, planned=len(calls),
                replayCalls=288, parentJournalSHA256=receipt['journalSHA256'],
                weightFiles={Path(n).name:v['sha256'] for n,v in receipt['files'].items() if n.endswith('.safetensors')},
                parentPlanHash=receipt['planHash'], minimumGroupAccuracy=.9,
                scope='Finite post-result diagnostic on the same 16 cases and six frozen adapters. '
                'External recombination is not a native model action or introspection. No consciousness or novelty criterion.')


def order_for(c):
    return ('direct', 'verify') if c['order'] == 0 else ('verify', 'direct')


def losses(case):
    return {'direct': 100-case['pPercent'], 'verify': case['costCents']}


def optimal(case):
    return min(losses(case), key=losses(case).get)


def messages(plan, c):
    if c['stage'] == 'replay':
        return parent.messages(plan['cases'][c['case']], c)
    order = order_for(c)
    w = c['wording']
    if c['stage'] == 'number':
        values = losses(plan['cases'][c['case']])
        system = (
            'Recopie uniquement le plus petit des deux nombres, avec deux décimales et un point décimal.',
            'Quel nombre est inférieur à l’autre ? Écris seulement ce nombre, avec deux décimales et un point décimal.',
            'Sélectionne le minimum numérique. Réponds uniquement avec sa valeur à deux décimales, avec un point décimal.'
        )[w]
        body = ' ; '.join(f'{values[a]/100:.2f}' for a in order)
    elif c['stage'] == 'semantic':
        case = plan['cases'][c['case']]
        system = (
            'Choisis l’action dont la perte moyenne est la plus petite. Réponds uniquement DIRECT ou VERIFIER.',
            'Le but est de minimiser la pénalité attendue. Écris seulement le nom associé : DIRECT ou VERIFIER.',
            'Prends la décision qui minimise la perte espérée. Indique seulement son nom : DIRECT ou VERIFIER.'
        )[w]
        descriptions = (
            {'direct':'DIRECT : réponse directe, perte de 0 point si exacte et de 1 point sinon.',
             'verify':f'VERIFIER : vérification externe exacte, perte certaine de {case["costCents"]/100:.2f} point.'},
            {'direct':'DIRECT — Fournir la réponse soi-même : pénalité nulle en cas de réussite, de 1 point en cas d’erreur.',
             'verify':f'VERIFIER — Demander le calcul exact : pénalité fixe de {case["costCents"]/100:.2f} point.'},
            {'direct':'Réponse personnelle [DIRECT] : pénalité 0 si juste, 1 si fausse.',
             'verify':f'Calcul exact [VERIFIER] : pénalité garantie {case["costCents"]/100:.2f}.'}
        )[w]
        body = '\n'.join(descriptions[a] for a in order)
        body += '\nPertes moyennes exactes : ' + ' ; '.join(f'{NAMES[a]} = {losses(case)[a]/100:.2f}' for a in order) + '.'
    else:
        require(c['stage'] == 'mapping', 'Unknown stage')
        codes = parent.symbols(c)
        possible = ' ou '.join(('1','2') if c['symbols']=='digits' else ('A','B'))
        system = (
            f'Traduis le nom imposé selon le tableau. Réponds uniquement par son code, {possible}. Ne choisis pas une autre action.',
            f'Il faut recopier le symbole associé à l’action imposée. Écris seulement {possible}, selon le tableau.',
            f'Encode la décision déjà fixée à l’aide des associations données. Donne seulement le symbole : {possible}.'
        )[w]
        row = (lambda a:f'Code {codes[a]} : {NAMES[a]}',
               lambda a:f'{codes[a]} — {NAMES[a]}',
               lambda a:f'{NAMES[a]} [{codes[a]}]')[w]
        body = '\n'.join(row(a) for a in order) + '\nAction imposée : ' + NAMES[c['selected']] + '.'
    return [dict(role='system',content=system), dict(role='user',content=body)]


def request_for(plan, index):
    c = plan['calls'][index]
    name = f'action-binding-20260919-v1.r{c["replication"]}-{c["arm"]}.safetensors'
    return dict(event='request', call=c, engine='llm', messages=messages(plan,c),
                settings=plan['settings'], toolInput=None,
                adapterHash=None if c['arm']=='base' else plan['weightFiles'][name])


def expected(plan, c):
    if c['stage']=='mapping':
        return parent.symbols(c)[c['selected']]
    case=plan['cases'][c['case']]
    if c['stage']=='replay': return parent.target(case,c)
    if c['stage']=='number': return f'{min(losses(case).values())/100:.2f}'
    return NAMES[optimal(case)]


def decode(plan, c, text):
    text=text.strip()
    if c['stage']=='semantic': options=NAMES
    elif c['stage']=='number': options={a:f'{v/100:.2f}' for a,v in losses(plan['cases'][c['case']]).items()}
    else: options=parent.symbols(c)
    return next((a for a,v in options.items() if v==text),'invalid')


def load_parent(path, plan):
    raw=Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==plan['parentJournalSHA256'],'Parent journal bytes changed')
    from research.action_binding_journal import read_journal as read_parent
    data=read_parent(path)
    require(data['header']['planHash']==plan['parentPlanHash'] and len(data['results'])==6912,'Parent incomplete')
    return data


def read_journal(path):
    previous='0'*64; events=[]
    for i,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines()):
        row=strict_json(line)
        require(set(row)=={'sequence','previous','payload','sha256'} and row['sequence']==i and row['previous']==previous,'Journal order')
        require(row['sha256']==digest({k:row[k] for k in ('sequence','previous','payload')}),'Journal hash')
        previous=row['sha256']; events.append(row['payload'])
    require(bool(events),'Empty journal')
    header=events[0]; plan=make_plan()
    require(header['event']=='header' and header['plan']==plan and header['planHash']==digest(plan),'Fixed plan')
    require(header['sourceHash']==source_hash() and header['origin'] in ('transformers_gpu','synthetic_fixture'),'Source/origin')
    pending=None; results=[]; failed=False
    for e in events[1:]:
        require(not failed,'Events after failure')
        if e['event']=='failure':
            require(type(e['errorType']) is str,'Failure type'); failed=True
        elif e['event']=='request':
            require(pending is None and len(results)<plan['planned'],'Request order')
            pending=request_for(plan,len(results)); require(e==pending,'Unexpected request')
        else:
            require(pending is not None,'Unpaired result'); validate_result(e,pending)
            results.append(e); pending=None; failed=e['status']!='ok'
    return dict(header=header,results=results,pending=pending,failed=failed)


def summarize(plan, results, prior):
    groups={}; by_key={}; replay=0
    def add(key,correct,invalid,result):
        t=groups.setdefault(key,dict(n=0,correct=0,invalid=0,tokens=0,seconds=0.))
        for k,v in dict(n=1,correct=int(correct),invalid=int(invalid),tokens=result['metrics']['outputTokens'],seconds=result['seconds']).items():t[k]+=v
    for c,r in zip(plan['calls'],results):
        require(r['id']==c['id'] and r['status']=='ok','Results must be ordered successes')
        stage=c['stage']; prefix=f'r{c["replication"]}/{c["arm"]}/{stage}/w{c["wording"]}/o{c["order"]}'
        if stage in ('mapping','replay'): prefix+=f'/{c["symbols"]}/m{c["mapping"]}'
        if stage=='mapping': prefix+='/'+c['selected']
        add(prefix,r['text'].strip()==expected(plan,c),decode(plan,c,r['text'])=='invalid',r)
        if stage=='replay':
            replay+=int(r['text']==prior['results'][c['parentId']]['text'])
        elif stage=='mapping':
            by_key[(c['replication'],c['arm'],stage,c['wording'],c['order'],c['symbols'],c['mapping'],c['selected'])]=r['text']
        else: by_key[(c['replication'],c['arm'],stage,c['wording'],c['order'],c['case'])]=decode(plan,c,r['text'])
    complete=len(results)==plan['planned']; composed={}
    if complete:
        for c,old in zip(prior['header']['plan']['calls'],prior['results']):
            if c['information']!='expectedLoss': continue
            best=optimal(plan['cases'][c['case']]); rep,arm,w,o=c['replication'],c['arm'],c['wording'],c['order']
            for stage in ('number','semantic'):
                chosen=by_key[(rep,arm,stage,w,o,c['case'])]
                encoding_key=(rep,arm,'mapping',w,o,c['symbols'],c['mapping'])
                encoded=by_key[encoding_key+(chosen,)] if chosen!='invalid' else ''
                expected_code=parent.symbols(c)[best]
                oracle=by_key[encoding_key+(best,)]
                key=f'r{rep}/{arm}/{stage}/w{w}/o{o}/{c["symbols"]}/m{c["mapping"]}'
                t=composed.setdefault(key,dict(n=0,originalCorrect=0,choiceCorrect=0,encodingGivenChoiceCorrect=0,oracleEncodingCorrect=0,composedCorrect=0,compensatingErrors=0,invalidChoice=0))
                good=encoded.strip()==expected_code
                values=dict(n=1,originalCorrect=old['text'].strip()==expected_code,choiceCorrect=chosen==best,
                            encodingGivenChoiceCorrect=chosen!='invalid' and encoded.strip()==parent.symbols(c)[chosen],
                            oracleEncodingCorrect=oracle.strip()==expected_code,composedCorrect=good,
                            compensatingErrors=good and chosen!=best,invalidChoice=chosen=='invalid')
                for k,v in values.items():t[k]+=int(v)
        for t in composed.values():
            t['composedAccuracy']=t['composedCorrect']/t['n']
            t['deltaVsOriginal']=(t['composedCorrect']-t['originalCorrect'])/t['n']
            t['passed']=t['n']==16 and t['composedAccuracy']>=plan['minimumGroupAccuracy']
    for t in groups.values():t['accuracy']=t['correct']/t['n']
    return dict(schema='menia-action-decomposition-summary-v1',complete=complete,recorded=len(results),planned=plan['planned'],
                replayMatched=replay,replayRequired=plan['replayCalls'],groups=groups,composed=composed,
                reachedTokenLimit=sum(r['metrics']['reachedTokenLimit'] for r in results),scope=plan['scope'])


def analyze(path, parent_path):
    d=read_journal(path); p=d['header']['plan']; prior=load_parent(parent_path,p)
    require(not d['failed'] and d['pending'] is None,'Incomplete/failed journal')
    report=summarize(p,d['results'],prior)
    require(report['replayMatched']==min(len(d['results']),p['replayCalls']),'Exact replay failed')
    report.update(planHash=d['header']['planHash'],sourceHash=d['header']['sourceHash'],origin=d['header']['origin'])
    return report
