"""Fresh public tables for a later learning test; no model calls or outcome labels."""
from itertools import combinations

from research.cross_model_prediction import digest
from research.prospective_pair_discovery import plan as old_plan


def partition():
    old = old_plan()['cases']
    seen = {tuple(c['values']) for c in old if c['bindings'] == 8}
    pool = [tuple(int(i in ones) for i in range(8)) for ones in combinations(range(8),4)]
    fresh = sorted((v for v in pool if v not in seen),key=lambda v:digest(['menia-reader-new-tables-v1',v]))
    assert len(pool)==70 and len(fresh)>=48 and len(seen)+len(fresh)==70
    keys = ['LUMA','NERI','VAKO','SUDI','PELA','TOVI','ZERA','MIKO']
    cases = []
    for i,values in enumerate(fresh[:48]):
        table='\n'.join(f'{key} = {value}' for key,value in zip(keys,values))
        cases.append(dict(id=1000+i,split='train' if i<32 else 'reserved',bindings=8,values=list(values),
            seed=2026092100+i,tableHash=digest(values),messages=[
                dict(role='system',content='Suis le format demandé à chaque tour.'),
                dict(role='user',content='Mémorise ce tableau pour une question ultérieure :\n'+table+'\nRéponds uniquement OK pour confirmer.')]))
    return dict(schema='menia-prospective-reader-partition-v1',cases=cases,
        counts=dict(previousEightKeyTables=len(seen),train=32,reserved=16,remainingUnused=len(fresh)-48),
        exclusions=sorted(digest(v) for v in seen),
        scope='Disjoint new eight-key assignments within the same synthetic task; no new keys, task-family transfer, model calls, outcome labels or consciousness criterion. Reserved tables must not inform optimization.')
