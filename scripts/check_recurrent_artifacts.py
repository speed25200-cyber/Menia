"""Verify published checksums and rerun the checkpoint evaluation."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.recurrent import RecurrentMemory
from research.train_recurrent import evaluate
root=Path(__file__).resolve().parents[1]/'artifacts/recurrent-memory'
report=json.loads((root/'report.json').read_text())
for run in report['runs']:
    path=root/f"memory-seed-{run['seed']}.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==run['checkpoint_sha256']
    measured=evaluate(RecurrentMemory.load(path),seed=report['test_seed'])
    for name,values in run['evaluation'].items():
        for key,value in values.items():
            np.testing.assert_allclose(measured[name][key],value,rtol=1e-6,atol=1e-8)
fixture=json.loads((root/'mobile-fixture.json').read_text())
model=RecurrentMemory.load(root/fixture['checkpoint'])
outputs,_=model.forward(np.asarray(fixture['inputs'])[:,None,:])
np.testing.assert_allclose(outputs[:,0,:],fixture['probabilities'],rtol=1e-6,atol=1e-8)
print('Checkpoint hashes, published evaluations and mobile reference vectors OK')
