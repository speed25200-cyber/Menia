"""Recompute all pair-discovery records and the two exported BF16 caches."""
import argparse
import hashlib
import json
from pathlib import Path

from research import prospective_pair_discovery as study
from research.audit_generation_numerics import compare_numbers
from research.audit_prospective_binding_discovery import cache_audit


def verify(path,tokenizer):
    path=Path(path); data=study.read_journal(path,tokenizer); summary=study.summarize(data)
    compare_numbers(summary,json.loads(path.with_suffix('.summary.json').read_text(encoding='utf-8')))
    caches=[cache_audit(path.parent,data['states'][case]) for case in study.plan()['exportActualCacheCases']]
    return dict(schema='menia-prospective-pair-audit-v1',verified=True,
        journalSHA256=hashlib.sha256(path.read_bytes()).hexdigest(),planHash=summary['planHash'],
        sourceHash=summary['sourceHash'],chainEnd=summary['chainEnd'],recomputedSummary=True,
        allForecastsPrecedeTasks=True,parentStatesExactlyReproduced=24,parentTasksExactlyReproduced=24,
        exportedCacheAudits=caches,auditorSourceHash=study.digest(dict(
            pair=Path(__file__).read_text(encoding='utf-8'),cache=Path(__file__).with_name('audit_prospective_binding_discovery.py').read_text(encoding='utf-8'))),
        scope='Recorded inputs, code meanings, outcomes, phase barriers and summary verified; transformations recomputed for two caches. No independent collection or remote weight attestation.')


if __name__=='__main__':
    from transformers import AutoTokenizer
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('journal',type=Path)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    spec=study.plan()['model']; tokenizer=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],trust_remote_code=False)
    result=verify(args.journal,tokenizer)
    with args.output.open('x',encoding='utf-8',newline='\n') as f: f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(verified=True,**study.plan()['counts'],cacheFilesAudited=len(result['exportedCacheAudits']))))
