"""Same frozen Qwen backend, without adapters or activation interventions."""
import argparse
import json
from pathlib import Path

from research.native_choice_gpu import NativeChoiceBackend
from research.risk_presentation import analyze,source_hash,DECISION_SETTINGS
from research.risk_presentation_journal import collect


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('journal',type=Path);args=parser.parse_args()
    backend=NativeChoiceBackend()
    backend.metadata.update(riskPresentationSourceHash=source_hash(),settings={'decision':DECISION_SETTINGS})
    collect(args.journal,backend)
    report=analyze(args.journal)
    args.journal.with_suffix('.summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('complete','recorded','planned','statuses')}),flush=True)
