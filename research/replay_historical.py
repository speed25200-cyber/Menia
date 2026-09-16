"""Exploratory backtest on already inspected perturbation data, never a new test.

Original journals and grades remain unchanged. Old monitors predicted strict
success; the new restricted parser and policy search are applied after the fact.
Verification outcomes are re-executed now from the public question.
"""
import argparse
import hashlib
import json
from pathlib import Path

from menia.replay_control import Policy, candidates, improve, replay
from research import perturbation_monitor as old
from research.replay_controller import POOLS, episode, verify_question


def analyze_historical(path):
    original=old.analyze(path)
    if not original['complete']:
        raise ValueError('A complete validated perturbation journal is required')
    header,rows,pending,bundle,gate=old.read_journal(path)
    pools={'validation':[],'test':[]}
    tool_calls=0
    for row in rows:
        split=row['task']['split']
        if split not in pools:
            continue
        p=old.forecast(bundle,rows,row['task'],row['state'])
        verification=verify_question(row['task']['question'])
        tool_calls+=1
        converted=dict(task=row['task'],predictions={k:p[k] for k in ('betaCell','inputOnly','internal','shuffledLabels')},
                       result=dict(text=row['result']['text'],verification=dict(text=verification)))
        pools[split].append(episode(converted))
    selected,tables={},{}
    for name,sources in POOLS.items():
        policy,table=improve(pools['validation'],candidates(sources),Policy())
        selected[name]=policy
        tables[name]=dict(policy=policy.payload(),candidateCount=table['candidates'],
                         selectionLoss=table['selectedLoss'],diagnosticTest=replay(policy,pools['test']))
    baseline={name:replay(p,pools['test']) for name,p in (
        ('fixedBeta',Policy()),('fixedInternal',Policy(source='internal')),
        ('alwaysDirect',Policy(kind='direct')),('alwaysVerify',Policy(kind='verify')),
        ('alwaysAbstain',Policy(kind='abstain')))}
    # Keep only aggregates; no private raw answers, prompts, state vectors or IDs.
    for table in [*(t['diagnosticTest'] for t in tables.values()),*baseline.values()]:
        table.pop('losses')
    with Path(path).open('rb') as f:
        source=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(schema='menia-historical-replay-v1',origin=header['origin'],sourceJournalSha256=source,
                exploratory=True,prospectiveValidation=False,newLLMCalls=0,verificationAuditCallsNow=tool_calls,
                selectionExamples=len(pools['validation']),diagnosticExamples=len(pools['test']),
                independentDiagnosticQuestions=original['independentTestQuestions'],policies=tables,baselines=baseline,
                limitations=['Test answers were inspected before this method was designed.',
                             'Historical monitors were fit on strict correctness; new labels use the restricted numeric parser.',
                             'Verification is re-executed now; alternative online outcomes beyond these actions are unavailable.',
                             'Four variants per question are dependent; no new consciousness or performance claim.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal',type=Path); parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.journal.resolve()==args.output.resolve(): raise ValueError('Cannot overwrite input journal')
    report=analyze_historical(args.journal)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
