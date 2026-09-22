"""Separate judge change and answer change on a complete matched finite matrix.

Arithmetic decomposition, not a causal representation or privileged-access test.
Every judge scores the same frozen answer from each producer on each question.
"""
import math

from research.iphone_coupling_report import require

ARMS=('base','measured','shuffled')


def summarize(rows):
    require(bool(rows),'Empty crossed evaluation')
    seen=set();losses={(g,p):[] for g in ARMS for p in ARMS};correct={p:0 for p in ARMS}
    for row in rows:
        require(row['id'] not in seen,'Duplicate question');seen.add(row['id'])
        require(set(row['correct'])==set(ARMS) and set(row['judgments'])==set(ARMS),'Incomplete matrix')
        for p,y in row['correct'].items():
            require(type(y) in (bool,int) and y in (0,1),'Binary correctness required');correct[p]+=int(y)
        for g in ARMS:
            require(set(row['judgments'][g])==set(ARMS),'Missing producer')
            for p in ARMS:
                j=row['judgments'][g][p];prob=j['conditionalCorrect'];mass=j['candidateMass']
                require(type(prob) in (int,float) and math.isfinite(prob) and 0<=prob<=1,'Probability')
                require(type(mass) in (int,float) and math.isfinite(mass) and 0<=mass<=1,'Candidate mass')
                require(type(j['inputHash']) is str and len(j['inputHash'])==64,'Input identity required')
                require(j['inputHash']==row['judgments']['base'][p]['inputHash'],'Judges received different inputs')
                losses[(g,p)].append((prob-int(row['correct'][p]))**2)
    brier={g:{p:math.fsum(losses[(g,p)])/len(rows) for p in ARMS} for g in ARMS}
    changes={}
    for g in ARMS[1:]:
        total=brier['base']['base']-brier[g][g]
        fixed_base=brier['base']['base']-brier[g]['base']
        changed_answers=brier[g]['base']-brier[g][g]
        fixed_current=brier['base'][g]-brier[g][g]
        changed_under_base=brier['base']['base']-brier['base'][g]
        require(math.isclose(total,fixed_base+changed_answers,abs_tol=1e-12) and
                math.isclose(total,fixed_current+changed_under_base,abs_tol=1e-12),'Decomposition identity')
        changes[g]=dict(coupledBrierGain=total,judgeGainOnBaseAnswers=fixed_base,
                        answerChangeUnderCurrentJudge=changed_answers,judgeGainOnCurrentAnswers=fixed_current,
                        answerChangeUnderBaseJudge=changed_under_base)
    mass={g:{p:math.fsum(r['judgments'][g][p]['candidateMass'] for r in rows)/len(rows) for p in ARMS} for g in ARMS}
    return dict(schema='menia-answer-confidence-crossed-summary-v1',questions=len(rows),correct=correct,
                brier=brier,changes=changes,meanCandidateMass=mass,
                scope='Complete matched descriptive matrix; no population inference, calibration guarantee or consciousness criterion.')
