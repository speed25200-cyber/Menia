"""Separate core-score arithmetic and actual adapter-file validation for Colab23."""
import hashlib
import math
from pathlib import Path

import numpy as np

from research.answer_confidence_journal import read_journal
from research.answer_confidence_study import CRITERIA,evaluation_rows
from research.answer_confidence_plan import QUESTION_SEED
from research.answer_confidence_crossed import ARMS
from research.answer_confidence_baselines import forecast
from research.cross_model_prediction import CELLS,reference


def verify(path,report,*,check_weights=True):
    import re
    d=read_journal(path);assert d['complete'] and report['complete']
    assert (report['recordedCalls'],report['generations'],report['judgments'],report['trainingSteps'])==(10368,2592,7776,864)
    assert report['planHash']==d['header']['planHash'] and report['sourceHash']==d['header']['sourceHash']
    weights={};initial_hashes=[];differences=[]
    if check_weights:
        import torch
        from safetensors.torch import load_file
        for rep in range(3):
            initial=Path(path).with_suffix(f'.r{rep}-initial.safetensors');base=load_file(str(initial))
            sha=hashlib.sha256(initial.read_bytes()).hexdigest();weights[initial.name]=sha;initial_hashes.append(sha)
            assert base and all(torch.isfinite(v).all() for v in base.values())
            for arm in ARMS[1:]:
                c=d['checkpoints'][f'r{rep}-{arm}'];file=Path(path).with_suffix(f'.r{rep}-{arm}.safetensors')
                assert sha==c['initializationHash'];actual=hashlib.sha256(file.read_bytes()).hexdigest();assert actual==c['sha256']
                state=load_file(str(file));assert set(state)==set(base)
                assert all(v.shape==base[k].shape and torch.isfinite(v).all() for k,v in state.items())
                assert any(not torch.equal(v,base[k]) for k,v in state.items());weights[file.name]=actual
        assert len(set(initial_hashes))==3
    def equal(a,b):
        delta=abs(float(a)-float(b));assert math.isfinite(delta) and delta<1e-8,(a,b);differences.append(delta)
    rows_all=evaluation_rows(d['header']['plan'],d['answers'],d['judgments'],'test')
    gates={}
    for rep in range(3):
        rows=[r for r in rows_all if r['task']['replication']==rep];got=report['replications'][str(rep)]
        groups=[[i for i,r in enumerate(rows) if (r['task']['family'],r['task']['level'])==cell] for cell in CELLS]
        rng=np.random.default_rng(QUESTION_SEED+90000+rep)
        draws=np.concatenate([rng.choice(ids,(10000,len(ids)),replace=True) for ids in groups],axis=1)
        all_pass=[];enough=[];fmt=[];accuracies={}
        for producer in ARMS:
            truth=[];ps={name:[] for name in ('base','measured','shuffled','betaCell','outputConfidence')}
            for row in rows:
                text=row['answers'][producer]['text'].strip()
                truth.append(int(re.fullmatch(r'-?(?:0|[1-9][0-9]*)',text) is not None and int(text)==reference(row['task'])))
                for judge in ARMS:ps[judge].append(row['judgments'][judge][producer]['conditionalCorrect'])
                for name,value in forecast(d['bundles'][rep],row['task'],row['answers'][producer],producer).items():ps[name].append(value)
            accuracies[producer]=sum(truth)/192;equal(accuracies[producer],got['answerAccuracy'][producer])
            for name,prob in ps.items():
                for ids,score in [(list(range(192)),got['scores'][producer][name])]+[(ids,got['withinCell'][producer][f'{cell[0]}/{cell[1]}'][name]) for cell,ids in zip(CELLS,groups)]:
                    assert score['n']==len(ids) and score['correct']==sum(truth[i] for i in ids)
                    equal(math.fsum((prob[i]-truth[i])**2 for i in ids)/len(ids),score['brier'])
                    equal(math.fsum(prob[i] for i in ids)/len(ids),score['meanProbability'])
                    pos=[prob[i] for i in ids if truth[i]];neg=[prob[i] for i in ids if not truth[i]]
                    if pos and neg:equal(sum(int(a>b)+.5*int(a==b) for a in pos for b in neg)/(len(pos)*len(neg)),score['auc'])
                    else:assert score['auc'] is None
                if name in ARMS:equal(got['scores'][producer][name]['brier'],got['crossed']['brier'][name][producer])
            if producer not in CRITERIA['primaryProducers']:continue
            enough.append(min(sum(truth),192-sum(truth))>=20)
            own=[r['judgments']['measured'][producer] for r in rows]
            top=sum(s['topIsCode'] for s in own)/192;mass=math.fsum(s['candidateMass'] for s in own)/192
            equal(top,got['nativeMeasuredFormat'][producer]['topCodeFraction']);equal(mass,got['nativeMeasuredFormat'][producer]['meanCodeMass'])
            fmt.append(top>=.95 and mass>=.5)
            for name in CRITERIA['comparators']:
                delta=np.asarray([(ps[name][i]-truth[i])**2-(ps['measured'][i]-truth[i])**2 for i in range(192)])
                gain=math.fsum(delta)/192;samples=np.sum(delta[draws],axis=1)/192;tail=.05/(2*24)
                family=np.quantile(samples,[tail,1-tail]);ordinary=np.quantile(samples,[.025,.975])
                c=next(c for c in report['contrasts'] if (c['replication'],c['producer'],c['comparison'])==(rep,producer,name+'-measured'))
                equal(gain,c['brierGain'])
                for a,b in zip(family,c['familyInterval']):equal(a,b)
                for a,b in zip(ordinary,c['interval95']):equal(a,b)
                passed=bool(gain>=.005 and family[0]>0);assert c['passed']==passed;all_pass.append(passed)
        gates[str(rep)]=dict(enoughBothClasses=all(enough),nativeCodeFormatPassed=all(fmt),
                            answerAccuracyRetained=accuracies['measured']-accuracies['base']>=-.02,allPrimaryPassed=all(all_pass))
    assert report['gates']==gates and report['nativeConfidenceCriterion']==all(all(g.values()) for g in gates.values())
    return dict(schema='menia-native-confidence-core-audit-v1',verified=True,weightsChecked=check_weights,weights=weights,
                journalSHA256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),maxAbsoluteDifference=max(differences,default=0.),
                comparisonsChecked=24,scope='Separate core Brier, accuracy, AUROC, bootstrap and gate arithmetic. Shared integrity reader, task references and baseline fitter; not an external replication. Full summary also requires main reconstruction.')
