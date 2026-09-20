"""Recompute the exploratory journal and two prespecified exported cache states."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from research import prospective_state_discovery as study
from research.audit_generation_numerics import compare_numbers
from research.iphone_coupling_report import require


def cache_audit(directory,state):
    import torch
    from safetensors.torch import load_file
    from research.confidence_cached_action_decode import _cache_hash
    from research.prospective_cache_interventions import inverse_permutation
    exported=state['cacheExport']; path=Path(directory)/exported['file']; raw=path.read_bytes()
    require(len(raw)==exported['bytes'] and hashlib.sha256(raw).hexdigest()==exported['sha256'],'Actual cache file identity')
    tensors=load_file(str(path)); names=[f'layer.{i:02d}.{kind}' for i in range(36) for kind in ('key','value')]
    require(set(tensors)==set(names),'Exported tensor names')
    length=state['trace']['cacheTokensAfterFinalToken']
    require(all(t.dtype==torch.bfloat16 and t.ndim==4 and t.shape[0]==1 and t.shape[-2]==length and
        bool(torch.isfinite(t).all()) for t in tensors.values()),'Exported cache layout')
    original=tuple((tensors[f'layer.{i:02d}.key'],tensors[f'layer.{i:02d}.value']) for i in range(36))
    require(_cache_hash(original)==exported['cacheHash']==state['cacheHashes']['actual'],'Original cache hash')
    computed={'actual':original}; reports={}
    for name in ('values_permuted','joint_permuted','restored'):
        record=state['interventions'][name]; before=computed['values_permuted'] if name=='restored' else original
        indices=study.permutation(length); indices=inverse_permutation(indices) if name=='restored' else indices
        result=tuple(tuple(t.index_select(-2,torch.tensor(indices)) if kind==1 or name=='joint_permuted' else t.clone()
            for kind,t in enumerate(pair)) for pair in before)
        require(_cache_hash(result)==record['resultCacheHash']==state['cacheHashes'][name],'Recomputed intervention cache hash')
        pairs=[(a,b) for ap,bp in zip(before,result) for a,b in zip(ap,bp)]
        norms=dict(changedTensors=sum(not torch.equal(a,b) for a,b in pairs),
            maximumAbsoluteDisplacement=max(float((b.double()-a.double()).abs().max()) for a,b in pairs),
            absoluteL2=math.sqrt(math.fsum(float((b.double()-a.double()).square().sum()) for a,b in pairs)),
            referenceL2=math.sqrt(math.fsum(float(a.double().square().sum()) for a,b in pairs)))
        for key,value in norms.items(): require(math.isclose(value,record[key],rel_tol=1e-8,abs_tol=1e-8),'Cache displacement arithmetic '+key)
        computed[name]=result; reports[name]=norms
    require(_cache_hash(computed['restored'])==_cache_hash(original),'Restoration not exact')
    return dict(case=state['case']['id'],file=exported['file'],sha256=exported['sha256'],tensors=72,
        transformationsRecomputed=True,restorationBitwiseExact=True,displacements=reports)


def verify(path,tokenizer):
    path=Path(path); data=study.read_journal(path,tokenizer)
    report=study.summarize(data)
    compare_numbers(report,json.loads(path.with_suffix('.summary.json').read_text(encoding='utf-8')))
    caches=[cache_audit(path.parent,data['states'][case]) for case in study.plan()['exportActualCacheCases']]
    return dict(schema='menia-prospective-state-discovery-audit-v1',verified=True,journalSHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
        planHash=report['planHash'],sourceHash=report['sourceHash'],chainEnd=report['chainEnd'],
        recomputedSummary=True,allForecastsPrecedeTasks=True,exportedCacheAudits=caches,
        auditorSourceHash=study.digest(Path(__file__).read_text(encoding='utf-8')),
        scope='Journal and recorded metrics recomputed; actual tensors verified for two prespecified cases. Other cache tensors and full model logits are not exported. Not independent replication or remote model-weight attestation.')


if __name__=='__main__':
    from transformers import AutoTokenizer
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('journal',type=Path)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    spec=study.plan()['model']; tokenizer=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],trust_remote_code=False)
    result=verify(args.journal,tokenizer)
    with args.output.open('x',encoding='utf-8',newline='\n') as stream: stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(verified=True,cases=24,forecasts=120,tasks=120,cacheFilesAudited=len(result['exportedCacheAudits']))))
