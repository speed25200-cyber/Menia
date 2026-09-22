"""Audit the numerical diagnostic's sequence, raw decisions and comparisons.

This checks exported records. Weights and widened cache tensors are not
exported; the collector's value-preservation checks are not independently
re-executed by this auditor.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from research import reader_permutation_numerics as study
from research.audit_generation_numerics import validate_decode, compare_numbers, branch_comparison
from research.cross_model_prediction import digest
from research.iphone_coupling_report import require


def verify(path, tokenizer):
    path = Path(path)
    raw = path.read_bytes()
    events = []; previous = '0'*64
    for i, line in enumerate(raw.decode('utf-8').splitlines()):
        row = json.loads(line)
        require(set(row) == {'sequence','previous','payload','sha256'} and
                row['sequence'] == i and row['previous'] == previous, 'Journal sequence')
        require(row['sha256'] == digest({k:row[k] for k in ('sequence','previous','payload')}), 'Journal hash')
        previous = row['sha256']; events.append(row['payload'])
    prepared = study.preparation()
    require(events and events[0]['event'] == 'header', 'Initial header required')
    header = events[0]
    require(header['origin'] == 'transformers_gpu' and
            all(header[k] == v for k,v in prepared.items()), 'Prepared numerical experiment identity')
    meta = header['metadata']
    require(meta['deterministicAlgorithms'] is True and meta['tf32'] is False and
            meta['cublasWorkspace'] == ':4096:8', 'Numerical execution settings')
    require(hashlib.sha256(study.PREFIX.read_bytes()).hexdigest() == study.PREFIX_HASH, 'Parent prefix identity')
    parent = [json.loads(line)['payload'] for line in study.PREFIX.read_text(encoding='utf-8').splitlines()]
    original = next(e['result'] for e in parent if e['event'] == 'state' and e['call']['case'] == 1008)
    tasks = {(e['call']['condition'],e['call']['target']):e['result']['decoded']
             for e in parent if e['event'] == 'task' and e['call']['case'] == 1008}
    spec = prepared['plan']['model']
    identity = spec['id']+'@'+spec['revision']+'/base-bfloat16-sdpa'
    order = [dict(mode=m,repeat=r,condition=c,target=t) for m in study.MODES
             for r in range(2) for c in study.CONDITIONS for t in range(8)]
    records = []; pending = None; reproduced = None; converted = None; terminal = None; failure = None; summary = None
    fp32_hashes = {}
    for event in events[1:]:
        require(terminal is None, 'Event after terminal record')
        kind = event['event']
        if kind == 'failure':
            terminal = 'failed'; failure = event
            continue
        if kind == 'reproduction':
            require(reproduced is None and not records and pending is None, 'Reproduction chronology')
            require(set(event['checks']) == {'text','metrics','trace','cache'} and
                    all(type(v) is bool for v in event['checks'].values()), 'Reproduction flags')
            require(re.fullmatch('[a-f0-9]{64}',event['actualCacheHash']) is not None, 'Reproduction cache digest')
            require(event['checks']['cache'] == (event['actualCacheHash'] == original['cacheHashes']['actual']), 'Reproduction cache comparison')
            reproduced = event
            continue
        require(reproduced is not None and all(reproduced['checks'].values()), 'Successful reproduction before decoding')
        if kind == 'precision_import':
            require(converted is None and len(records) == 128 and pending is None, 'Precision conversion chronology')
            c = event['conversion']; capture = event['capture']
            require(c['allValuesPreserved'] is True and c['parameters'] > 0 and c['buffers'] >= 0 and
                    re.fullmatch('[a-f0-9]{64}',c['originalValueHash']) is not None, 'Recorded conversion checks')
            require(c['targetPrecision'] == 'float32' and capture['precisionImport'] == dict(
                sourceIdentity=identity,sourceCacheHash=original['cacheHashes']['actual'],
                originalCacheRecomputed=False,operation='BF16 values widened to FP32'), 'Explicit retained-history import')
            require(re.fullmatch('[a-f0-9]{64}',event['cacheHash']) is not None, 'FP32 cache digest')
            converted = event
            continue
        if kind == 'request':
            require(pending is None and len(records) < len(order) and event == dict(event='request',**order[len(records)]),
                    'Diagnostic request order')
            require((converted is not None) == (event['mode'] == 'fp32-math'), 'Precision mode boundary')
            pending = order[len(records)]
            continue
        if kind == 'decode':
            require(pending is not None and all(event[k] == v for k,v in pending.items()), 'Unpaired diagnostic result')
            d = event['decoded']; mode = pending['mode']; condition = pending['condition']; target = pending['target']
            validate_decode(d,study.learning.plan()['taskMapping'],tokenizer)
            require(d['schema'] == 'menia-cached-native-branch-greedy-decode-v1' and d['snapshotUnchanged'] is True,
                    'Native cached decoder record')
            original_decode = tasks['actual' if condition == 'restored' else condition,target]
            require(d['prefixHash'] == original_decode['prefixHash'] and d['settings'] == original_decode['settings'] and
                    d['passes'] == original_decode['passes'][:len(d['tokenIds'])], 'Original prefix and decoding schedule')
            require(d['modelStateId'] == identity+('/exact-value-fp32-suffix' if mode == 'fp32-math' else ''), 'Reader numerical identity')
            if mode != 'fp32-math':
                require(d['snapshotCacheHash'] == original['cacheHashes'][condition], 'Original BF16 cache identity')
            else:
                expected = fp32_hashes.setdefault(condition,d['snapshotCacheHash'])
                require(re.fullmatch('[a-f0-9]{64}',expected) is not None and d['snapshotCacheHash'] == expected, 'Stable FP32 cache identity')
                if condition in ('actual','restored'):
                    require(expected == converted['cacheHash'], 'Restoration of the widened original cache')
            if mode == 'bf16-default':
                require(event['originalExactlyEqual'] == (d == original_decode), 'Original output comparison')
                # A mismatch may appear just before the collector records its failure.
            else:
                require(event['originalExactlyEqual'] is None, 'No false exact-original claim after numerical change')
            records.append(dict(**pending,decoded=d)); pending = None
            continue
        if kind == 'complete':
            require(pending is None and len(records) == 192 and converted is not None, 'Premature completion')
            require(all(e['originalExactlyEqual'] for e in events if e['event'] == 'decode' and e['mode'] == 'bf16-default'),
                    'Completion after failed BF16 reproduction')
            keyed = {(r['mode'],r['repeat'],r['condition'],r['target']):r['decoded'] for r in records}
            contrasts = []
            for mode in study.MODES:
                for target in range(8):
                    a = keyed[mode,0,'actual',target]
                    contrasts.append(dict(mode=mode,target=target,
                        joint=branch_comparison(a,keyed[mode,0,'joint_permuted',target]),
                        values=branch_comparison(a,keyed[mode,0,'values_permuted',target]),
                        restoredExactlyEqual=a == keyed[mode,0,'restored',target],
                        repeatExactlyEqual=all(keyed[mode,0,c,target] == keyed[mode,1,c,target] for c in study.CONDITIONS)))
            summary = event['summary']
            require(summary['schema'] == 'menia-reader-permutation-numerics-summary-v1' and summary['decodes'] == 192 and
                    summary['scope'] == prepared['plan']['scope'] and summary['precision'] == prepared['plan']['precision'], 'Summary scope')
            compare_numbers(summary['contrasts'],contrasts)
            terminal = 'completed'
            continue
        raise ValueError('Unexpected numerical diagnostic event: '+kind)
    require(terminal is not None, 'Diagnostic still incomplete')
    if terminal == 'completed':
        compare_numbers(json.loads(path.with_suffix('.summary.json').read_text(encoding='utf-8')),summary)
    return dict(schema='menia-reader-permutation-numerics-audit-v1',verifiedJournal=True,status=terminal,
        journalSHA256=hashlib.sha256(raw).hexdigest(),chainEnd=previous,planHash=prepared['planHash'],
        parentPrefixSHA256=study.PREFIX_HASH,decodedBranches=len(records),pendingRequest=pending,failure=failure,
        rawNativeDecisionsAndProbabilitiesChecked=bool(records),
        originalBF16OutputsCompared=sum(r['mode'] == 'bf16-default' for r in records),
        numericalContrastsRecomputed=terminal == 'completed',cacheAndWeightWideningIndependentlyRecomputed=False,
        auditorSHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),summary=summary,
        scope='Exported sequence, identities, native reports and comparison arithmetic checked. Tensor value-preservation remains a runtime check, not an independent reconstruction from the archive.')


if __name__ == '__main__':
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal',type=Path); parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); spec = study.plan()['model']
    tokenizer = AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],trust_remote_code=False)
    report = verify(args.journal,tokenizer)
    with args.output.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('status','decodedBranches','journalSHA256','planHash')}))
