"""Synthetic journal integrity checks, never new model observations."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from research import reader_permutation_numerics as study
from research.audit_reader_permutation_numerics import verify
from research.natural_error_journal import Writer


class TokenizerStub:
    eos_token_id = 151645

    def encode(self,text,add_special_tokens=False):
        return [15 if text == '0' else 16]

    def decode(self,ids,skip_special_tokens=True):
        return ''.join({15:'0',16:'1',151645:''}.get(i,'?') for i in ids)


class NumericalJournalAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'synthetic-numerics.jsonl'
        parent = [json.loads(line)['payload'] for line in study.PREFIX.read_text(encoding='utf-8').splitlines()]
        original = next(e['result'] for e in parent if e['event'] == 'state' and e['call']['case'] == 1008)
        tasks = {(e['call']['condition'],e['call']['target']):e['result']['decoded']
                 for e in parent if e['event'] == 'task' and e['call']['case'] == 1008}
        identity = tasks['actual',0]['modelStateId']
        metadata = dict(parent[0]['metadata'],syntheticAuditFixture=True)
        self.events = [dict(event='header',**study.preparation(),origin='transformers_gpu',metadata=metadata),
            dict(event='reproduction',checks=dict(text=True,metrics=True,trace=True,cache=True),
                 actualCacheHash=original['cacheHashes']['actual'])]
        records = []
        for mode in study.MODES:
            if mode == 'fp32-math':
                self.events.append(dict(event='precision_import',conversion=dict(allValuesPreserved=True,parameters=1,buffers=0,
                    originalValueHash='c'*64,targetPrecision='float32'),cacheHash='a'*64,
                    capture=dict(precisionImport=dict(sourceIdentity=identity,sourceCacheHash=original['cacheHashes']['actual'],
                        originalCacheRecomputed=False,operation='BF16 values widened to FP32'))))
            for repeat in range(2):
                for condition in study.CONDITIONS:
                    for target in range(8):
                        call = dict(mode=mode,repeat=repeat,condition=condition,target=target)
                        d = copy.deepcopy(tasks['actual' if condition == 'restored' else condition,target])
                        if mode == 'fp32-math':
                            d['modelStateId'] += '/exact-value-fp32-suffix'
                            d['snapshotCacheHash'] = ('a' if condition in ('actual','restored') else 'b')*64
                        self.events.append(dict(event='request',**call))
                        self.events.append(dict(event='decode',**call,decoded=d,
                            originalExactlyEqual=True if mode == 'bf16-default' else None))
                        records.append(dict(**call,decoded=d))
        self.summary = study.summarize(records)
        self.events.append(dict(event='complete',summary=self.summary))

    def write(self):
        writer = Writer(self.path)
        for i,e in enumerate(self.events): writer.write(e,create=i == 0)
        self.path.with_suffix('.summary.json').write_text(json.dumps(self.summary),encoding='utf-8')

    def test_complete_synthetic_grid_and_limits_of_exported_evidence(self):
        self.write(); result = verify(self.path,TokenizerStub())
        self.assertEqual(result['decodedBranches'],192)
        self.assertEqual(result['originalBF16OutputsCompared'],64)
        self.assertTrue(result['numericalContrastsRecomputed'])
        self.assertFalse(result['cacheAndWeightWideningIndependentlyRecomputed'])

    def test_rehashed_probability_corruption_is_rejected(self):
        row = next(e for e in self.events if e['event'] == 'decode' and e['mode'] == 'bf16-math')
        row['decoded']['conditionalPositive'] = .123456
        self.write()
        with self.assertRaisesRegex(ValueError,'Binary probability'):
            verify(self.path,TokenizerStub())

    def test_rehashed_request_reordering_is_rejected(self):
        row = next(e for e in self.events if e['event'] == 'request')
        row['target'] = 7
        self.write()
        with self.assertRaisesRegex(ValueError,'request order'):
            verify(self.path,TokenizerStub())

    def test_summary_cannot_hide_a_changed_decision(self):
        self.summary['contrasts'][4]['joint']['decisionEqual'] = True
        self.write()
        with self.assertRaises(ValueError):
            verify(self.path,TokenizerStub())

    def test_failure_is_retained_and_never_reported_as_completion(self):
        self.events = self.events[:3]+[dict(event='failure',errorType='RuntimeError',error='synthetic failure')]
        self.write(); result = verify(self.path,TokenizerStub())
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['decodedBranches'],0)
        self.assertIsNotNone(result['pendingRequest'])
        self.assertIsNone(result['summary'])
        self.assertFalse(result['numericalContrastsRecomputed'])


if __name__ == '__main__':
    unittest.main()
