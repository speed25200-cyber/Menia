"""Empirical information floor for identical public inputs across hidden states.

This is a standard conditional-variance/entropy calculation, not a new theorem
or a consciousness measure. It applies to a fixed predictor on this empirical
mixture; losses collected during changing optimizer steps need not obey it.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path

from research import prospective_reader_learning as study
from research.prospective_reader_learning_audit import read_journal


def information_floor(examples):
    """Rows: (identical accessible input, binary outcome), equally weighted."""
    groups=defaultdict(list)
    for identity,label in examples:
        assert type(label) is int and label in (0,1)
        groups[identity].append(label)
    assert groups
    squared=[];entropies=[];minimum_errors=0;conflicts=0;conflict_examples=0
    for labels in groups.values():
        n=len(labels);positive=sum(labels);q=positive/n
        squared.append(n*q*(1-q))
        entropy=-(q*math.log(q)+(1-q)*math.log1p(-q)) if 0<q<1 else 0.
        entropies.append(n*entropy);minimum_errors+=min(positive,n-positive)
        if 0<positive<n:conflicts+=1;conflict_examples+=n
    n=sum(map(len,groups.values()))
    return dict(examples=n,distinctInputs=len(groups),conflictingInputs=conflicts,conflictingExamples=conflict_examples,
        minimumConditionalBrier=math.fsum(squared)/n,
        minimumConditionalCodeCrossEntropyNats=math.fsum(entropies)/n,
        codeEOSCrossEntropyLowerBoundNats=math.fsum(entropies)/(2*n),
        minimumNativeErrors=minimum_errors,maximumNativeAccuracy=1-minimum_errors/n)


def analyze(path,tokenizer):
    path=Path(path);data=read_journal(path,tokenizer);p=data['header']['plan']
    assert not data['complete'] and data['failure'] is None and data['operations']==800 and data['pending'] is None
    assert len(data['states'])==32 and all(s['case']['split']=='train' for s in data['states'].values())
    tasks={};decodes={};counts=Counter();conditions={condition:Counter() for condition in p['trainConditions']}
    for event in data['events']:
        if event['event']!='task':continue
        c=event['call'];d=event['result']['decoded'];case=data['states'][c['case']]['case']
        label=int(d['validNativeResponse'] and d['decision']==('one' if case['values'][c['target']] else 'zero'))
        tasks[c['case'],c['target'],c['condition']]=label;decodes[c['case'],c['target'],c['condition']]=d;counts['tasks']+=1
        conditions[c['condition']]['success' if label else 'failure']+=1
        conditions[c['condition']]['valid']+=int(d['validNativeResponse'])
    assert counts['tasks']==p['counts']['trainTasks']
    contrasts=[];joint_changes=[]
    for condition in p['trainConditions']:
        for group in ('selected','other'):
            keys=[(cid,target) for cid,state in data['states'].items() for target in range(state['case']['bindings'])
                if (target in state['bindingMask']['selectedRows'])==(group=='selected')]
            changed=[(cid,target) for cid,target in keys if decodes[cid,target,condition]['tokenIds']!=decodes[cid,target,'actual']['tokenIds']]
            contrasts.append(dict(condition=condition,group=group,n=len(keys),
                correct=sum(tasks[cid,target,condition] for cid,target in keys),nativeChangesFromActual=len(changed)))
            if condition=='joint_permuted':
                for cid,target in changed:
                    joint_changes.append(dict(case=cid,target=target,group=group,
                        expectedBit=data['states'][cid]['case']['values'][target],
                        actualCorrect=bool(tasks[cid,target,'actual']),jointCorrect=bool(tasks[cid,target,condition]),
                        actualLogits=decodes[cid,target,'actual']['candidateLogits'],jointLogits=decodes[cid,target,condition]['candidateLogits']))
    by_view={view:{kind:[] for kind in ('all','forecast','coding')} for view in ('text','state')}
    for epoch in range(p['optimizer']['epochs']):
        for example in study.training_examples(epoch,p):
            kind=example['kind'];state=data['states'][example['case']]
            label=example['verdict'] if kind=='coding' else tasks[example['case'],example['target'],example['condition']]
            input_hash=state['branchInputHashes'][study.branch_name(kind,example)]
            for view,rows in by_view.items():
                condition='same_schedule' if view=='text' else example['condition']
                identity=(input_hash,state['cacheHashes'][condition])
                rows[kind].append((identity,label));rows['all'].append((identity,label))
    floors={view:{kind:information_floor(rows) for kind,rows in groups.items()} for view,groups in by_view.items()}
    return dict(schema='menia-reader-training-information-floor-v1',
        journalPrefixSHA256=hashlib.sha256(path.read_bytes()).hexdigest(),chainEnd=data['chainEnd'],
        planHash=data['header']['planHash'],collectionSourceHash=data['header']['sourceHash'],
        analysisSourceSHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        verifiedPrefixOperations=data['operations'],trainingStates=len(data['states']),trainingTasks=counts['tasks'],
        taskOutcomesByCondition={k:dict(v) for k,v in conditions.items()},taskContrasts=contrasts,jointNativeChanges=joint_changes,floors=floors,
        assumptions='One fixed predictor, private copies, no condition ID or update index as input, equal weighting of the actual four-epoch training examples. Input equivalence uses both complete-query token hash and accessible-cache hash. Coding prompts are separate input groups.',
        interpretation='Known conditional variance and entropy lower bounds on this empirical training mixture; no claim of theoretical novelty, model attainability or subjective consciousness. The code/EOS loss bound assumes zero EOS loss as its most favorable case. It does not bound the changing-weight optimization trace, or the reserved V-only primary score. No reserved task outcome is read.')


if __name__=='__main__':
    from transformers import AutoTokenizer
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('journal_prefix',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    spec=study.plan()['model'];tok=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],trust_remote_code=False,local_files_only=True)
    result=analyze(args.journal_prefix,tok)
    with args.output.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result))
