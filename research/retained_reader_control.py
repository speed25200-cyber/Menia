"""Fixed hardware control for the suffix-reader boundary, on two already seen states."""
import hashlib
import json
from pathlib import Path

from research.cross_model_prediction import digest
from research.prospective_pair_discovery import plan as parent_plan, FILES as PARENT_FILES


ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'artifacts/prospective-pair-pilot/prospective-pair-20260920-v1.jsonl'
PREPARATION=ROOT/'artifacts/retained-reader-preparation/control.json'
FILES=tuple(dict.fromkeys(PARENT_FILES+('native_localization_gpu.py','native_localization.py',
    'retained_state_reader.py','retained_reader_control.py','retained_reader_control_gpu.py')))


def source_hash():
    return digest({name:Path(__file__).with_name(name).read_text(encoding='utf-8') for name in FILES})


def plan():
    p=parent_plan()
    return dict(schema='menia-retained-reader-control-plan-v1',model=p['model'],
        parentJournalSHA256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),cases=[0,8],
        selection='first selected row and first unselected row of each fixed parent mask',
        adapter=dict(rank=32,scale=1.0,seed=2026092070),
        optimizer=dict(type='AdamW',lr=5e-5,betas=[.9,.999],eps=1e-8,weight_decay=0,steps=1,clipNorm=1.0),
        order='Reproduce two parent states, eight producer tasks and eight forecasts; compare zero reader; one update on the eight measured forecast labels; evaluate the same eight; reload saved weights; verify unchanged producer tasks.',
        counts=dict(states=2,producerTasksBefore=8,forecastsBefore=8,trainingExamples=8,updates=1,forecastsAfter=8,producerTasksAfter=8),
        controls='Stop on state/parent reproduction or native-token disagreement of the zero reader. Measure score drift separately without asserting bitwise reader logits. Preserve any failed attempt.',
        scope='Technical backward/serialization/producer-preservation control on previously seen cases. One update is not a training-budget comparison, held-out result, calibrated self-model or consciousness result.')


def read_events(path):
    events=[]; previous='0'*64
    for i,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines()):
        row=json.loads(line)
        assert row['sequence']==i and row['previous']==previous
        assert row['sha256']==digest({k:row[k] for k in ('sequence','previous','payload')})
        previous=row['sha256']; events.append(row['payload'])
    assert events[0]['event']=='header' and events[0]['plan']==plan()
    assert events[0]['planHash']==digest(plan()) and events[0]['sourceHash']==source_hash()
    return events
