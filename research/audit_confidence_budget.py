"""Audit the complete budget journal, frozen prefix, arithmetic and real weights.

The two metric implementations share the reader and labels. This is a local
verification, not independent replication or remote base-weight attestation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from research import confidence_budget_diagnostic as study
from research.audit_composition_diagnostic import compare
from research.iphone_coupling_report import require


def file_identity(path):
    raw = Path(path).read_bytes()
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def weight_difference(reference, candidate):
    import torch
    require(reference.keys() == candidate.keys(), 'Adapter tensor keys')
    squares = []; changed = 0; maximum = 0.
    for name,x in reference.items():
        y = candidate[name]
        require(x.shape == y.shape and x.dtype == y.dtype == torch.float32, 'Adapter shape or precision')
        require(bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all()), 'Nonfinite weights')
        d = y.double()-x.double()
        squares.append(float(d.square().sum())); maximum = max(maximum,float(d.abs().max()))
        changed += int(not torch.equal(x,y))
    return dict(changedTensors=changed,maximumAbsoluteDifference=maximum,l2=math.sqrt(math.fsum(squares)))


def verify(journal, report, freeze, parent_directory):
    import torch
    from safetensors.torch import load_file
    journal = Path(journal); parent_directory = Path(parent_directory)
    data = study.read_journal(journal)
    require(data['complete'] and not data['failed'], 'Complete successful journal')
    require(data['header']['origin'] == 'transformers_gpu', 'Actual GPU run required')
    require(freeze['schema'] == 'menia-confidence-budget-training-freeze-v1' and
            freeze['postTrainingJudgmentsInPrefix'] == 0, 'Training freeze receipt')
    for key in ('planHash','sourceHash','callPlanHash','origin'):
        require(freeze[key] == data['header'][key], 'Freeze identity '+key)
    require(len(data['steps']) == freeze['trainingSteps'] == 1296 and len(data['judgments']) == 8640, 'Counts')
    require(freeze['completedReplications'] == 3 and freeze['baselineJudgments'] == 2880 and
            freeze['baselineMaxAbsoluteDifference'] == data['maxBaselineDifference'], 'Freeze baseline')
    raw_prefix = []; completed = 0; last = None
    for raw in journal.read_bytes().splitlines(keepends=True):
        last = json.loads(raw); raw_prefix.append(raw)
        completed += int(last['payload']['event'] == 'training_complete')
        if completed == 3: break
    require(completed == 3, 'Training-complete prefix missing')
    prefix = b''.join(raw_prefix)
    require(len(prefix) == freeze['prefixBytes'] and hashlib.sha256(prefix).hexdigest() == freeze['prefixSHA256'] and
            last['sha256'] == freeze['chainEnd'], 'Frozen prefix changed')
    require(set(freeze['checkpointFiles']) == {c['checkpoint'] for c in data['checkpoints'].values()}, 'Frozen checkpoints')
    require(set(freeze['unchangedParents']) == {c['checkpoint'] for c in data['header']['plan']['initial']}, 'Frozen parents')
    parents = {}; parent_files = {}; weights = {}; tensors = {}
    for initial in data['header']['plan']['initial']:
        path = parent_directory/initial['checkpoint']; identity = file_identity(path)
        require(identity == freeze['unchangedParents'][path.name] == {k:initial[k] for k in identity}, 'Original checkpoint changed')
        parents[initial['replication']] = load_file(str(path)); parent_files[path.name] = identity
    for key,event in data['checkpoints'].items():
        path = journal.parent/event['checkpoint']; identity = file_identity(path); frozen = freeze['checkpointFiles'][path.name]
        require(frozen['key'] == key and identity == {k:event[k] for k in identity} == {k:frozen[k] for k in identity}, 'Checkpoint identity')
        state = load_file(str(path)); rep = int(key[1]); reference = parents[rep]
        require(len(state) == 144 and sum(t.numel() for t in state.values()) == 2949120, 'LoRA capacity')
        require(all(t.dtype == torch.float32 and t.ndim == 2 and min(t.shape) == 8 for t in state.values()), 'Rank or dtype')
        difference = weight_difference(reference,state)
        require(difference['changedTensors'] > 0, 'Continuation did not change adapter')
        weights[path.name] = dict(**identity,key=key,tensors=len(state),parameters=sum(t.numel() for t in state.values()),
            versusEpoch2=difference)
        tensors[key] = state
    successive = {f'r{rep}':weight_difference(tensors[f'r{rep}-e4'],tensors[f'r{rep}-e8']) for rep in range(3)}
    # Report rather than impose a post-hoc behavioral criterion on displacement.
    recomputed = study.summarize(data)
    difference = compare(recomputed,report)
    return dict(schema='menia-confidence-budget-verification-v1',verified=True,
        journalSHA256=file_identity(journal)['sha256'],chainEnd=data['chainEnd'],
        planHash=data['header']['planHash'],sourceHash=data['header']['sourceHash'],callPlanHash=data['header']['callPlanHash'],
        recordedCalls=len(data['judgments']),trainingSteps=len(data['steps']),checkpointFiles=weights,
        unchangedParentFiles=parent_files,epoch4To8Displacements=successive,trainingPrefixSHA256=freeze['prefixSHA256'],
        baselineMaxAbsoluteDifference=data['maxBaselineDifference'],mainRecomputeMaxAbsoluteDifference=difference,
        separateArithmeticMaxDifference=recomputed['separateArithmeticMaxDifference'],
        auditorSourceHash=study.digest(Path(__file__).read_text(encoding='utf-8')),
        scope='Exact requests, update schedule, frozen prefix and actual local checkpoint tensors checked. Numerical implementations share the strict reader and labels; not an independent replication or an attestation of remote base weights.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal',type=Path); parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--freeze',type=Path,required=True); parser.add_argument('--parents',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); args = parser.parse_args()
    result = verify(args.journal,json.loads(args.summary.read_text(encoding='utf-8')),
        json.loads(args.freeze.read_text(encoding='utf-8')),args.parents)
    with args.output.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('verified','recordedCalls','trainingSteps','mainRecomputeMaxAbsoluteDifference','separateArithmeticMaxDifference')}))
