"""Prepare native correctness supervision from archived TRAIN answers only.

No fine-tuning, new LLM call, threshold selection, or held-out performance claim.
Labels are targets, never model inputs. This is post-answer textual evaluation,
not a prediction before the answer or evidence of privileged internal access.
"""
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import random

from research.cross_model_prediction import digest, grade, SOLVE
from research.iphone_coupling_report import require

SEED = 202609230
CORRECTNESS_QUERY = (
    'Évalue la réponse que tu viens de donner. Écris seulement 1 si cette '
    'réponse est correcte, ou 0 si elle est incorrecte. Ne la corrige pas ici.'
)


def input_messages(question, answer):
    require(type(question) is str and type(answer) is str, 'Question and answer must be text')
    return [dict(role='system', content=SOLVE), dict(role='user', content=question),
            dict(role='assistant', content=answer), dict(role='user', content=CORRECTNESS_QUERY)]


def build_examples(rows):
    """Uses audited rows; deliberately ignores all nontraining responses."""
    examples=[];seen=set()
    for row in rows:
        task=row['task']
        if task['split']!='train':continue
        result=row['result']
        require(result['status']=='ok', 'Failed training response must not be silently removed')
        require(task['id'] not in seen, 'Repeated source id');seen.add(task['id'])
        require(type(task['replication']) is int and 0<=task['replication']<3, 'Source replication')
        target=str(int(grade(result,task)))
        examples.append(dict(sourceId=task['id'],replication=task['replication'],split='train',
                             family=task['family'],level=task['level'],
                             messages=input_messages(task['question'],result['text']),target=target))
    require(bool(examples),'No training examples')
    require(len({e['messages'][1]['content'] for e in examples})==len(examples),'Repeated training question')
    groups=defaultdict(list)
    for i,e in enumerate(examples):groups[(e['replication'],e['family'],e['level'])].append(i)
    shuffled=copy.deepcopy(examples)
    for group,indices in sorted(groups.items()):
        values=[examples[i]['target'] for i in indices]
        random.Random(int(digest([SEED,list(group)])[:16],16)).shuffle(values)
        for i,target in zip(indices,values):shuffled[i]['target']=target
    return dict(measured=examples,shuffled=shuffled)


def confidence_from_logits(logits, incorrect_id, correct_id):
    """Raw temperature-1 candidate ratio AND vocabulary mass; neither is calibrated."""
    values=[float(v) for v in logits]
    require(len(values)>1 and all(math.isfinite(v) for v in values),'Finite vocabulary logits required')
    require(type(incorrect_id) is int and type(correct_id) is int and incorrect_id!=correct_id and
            0<=incorrect_id<len(values) and 0<=correct_id<len(values),'Distinct in-range token ids')
    def lse(xs):
        maximum=max(xs)
        return maximum+math.log(math.fsum(math.exp(x-maximum) for x in xs))
    pair_lse=lse([values[incorrect_id],values[correct_id]]);all_lse=lse(values)
    return dict(conditionalCorrect=math.exp(values[correct_id]-pair_lse),
                candidateMass=math.exp(pair_lse-all_lse),
                scope='Conditional probability of code 1 among 0/1; raw and uncalibrated.')


def prepare(journal):
    from research.natural_error_journal import read_journal
    journal=Path(journal);raw=journal.read_bytes()
    receipt=json.loads((Path(__file__).parents[1]/'artifacts/natural-error-pilot/receipt.json').read_text(encoding='utf-8'))
    require(hashlib.sha256(raw).hexdigest()==receipt['journalSHA256'],'Unexpected parent journal bytes')
    data=read_journal(journal,check_fit=False)
    require(data['header']['origin']=='transformers_gpu' and len(data['rows'])==3456 and
            not data['pending'] and not data['failed'],'Incomplete or synthetic parent')
    require(data['header']['planHash']==receipt['planHash'] and data['header']['sourceHash']==receipt['sourceHash'],'Parent identity')
    examples=build_examples(data['rows']);original=examples['measured'];shuffled=examples['shuffled']
    require(len(original)==1728,'Training count')
    groups=defaultdict(lambda:dict(n=0,correct=0,shuffledAgreements=0))
    for a,b in zip(original,shuffled):
        key=f'r{a["replication"]}/{a["family"]}/{a["level"]}'
        groups[key]['n']+=1;groups[key]['correct']+=int(a['target']);groups[key]['shuffledAgreements']+=int(a['target']==b['target'])
    require(len(groups)==18 and all(g['n']==96 for g in groups.values()),'Training cell counts')
    report=dict(schema='menia-answer-confidence-data-v1',parentJournalSHA256=receipt['journalSHA256'],
                parentPlanHash=receipt['planHash'],parentSourceHash=receipt['sourceHash'],
                parentModel=receipt['model'],examplesPerArm=1728,groups=dict(groups),
                inputHash=digest([e['messages'] for e in original]),exampleHashes={k:digest(v) for k,v in examples.items()},
                excludedSplits=dict(Counter(r['task']['split'] for r in data['rows'] if r['task']['split']!='train')),
                status='Data prepared only; no new model trained or evaluated.',
                scope='Archived parent answers, not current adapted-model correctness. Fresh current-model evaluation and calibration remain required.')
    return examples,report


def exact_write(path, content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(path.read_text(encoding='utf-8')==content,'Existing different output: '+str(path))
    else:path.write_text(content,encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();examples,report=prepare(args.journal)
    for arm,values in examples.items():
        exact_write(args.output/(arm+'.jsonl'),''.join(json.dumps(e,ensure_ascii=False,allow_nan=False)+'\n' for e in values))
    exact_write(args.output/'manifest.json',json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,ensure_ascii=False,allow_nan=False))
