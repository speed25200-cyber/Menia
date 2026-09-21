import copy
import math
from pathlib import Path
import tempfile
import unittest

from research import prospective_pair_discovery as study
from tests_language import test_prospective_binding_discovery as binding_tests
from tests_language.test_prospective_state_discovery import write_events


def recode(d, config, bit):
    codes=[ord(c) for c in config['codes']]; meanings=[config['negative'],config['positive']]
    probability=.9 if bit else .1
    d.update(candidateTokenIds=codes,candidateMeanings=meanings,tokenIds=[codes[bit],9],
        text=chr(codes[bit]),decision=meanings[bit],firstTokenMeaning=meanings[bit],
        candidateLogits=[0.,math.log(probability/(1-probability))],conditionalPositive=probability)
    return d


class PairDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binding_tests.BindingDiscoveryTests.setUpClass()
        cls.tokenizer=binding_tests.BindingDiscoveryTests.tokenizer
        cls.parent=binding_tests.BindingDiscoveryTests().read(binding_tests.BindingDiscoveryTests.events)
        p=study.plan(); cls.events=[dict(event='header',origin='synthetic_test_fixture',
            metadata=dict(model=p['model'],weightUpdates=0),plan=p,planHash=study.digest(p),sourceHash=study.source_hash())]
        states={}
        for case in p['cases']:
            e=copy.deepcopy(cls.parent['states'][case['id']]); trace=e['trace']
            mask=study.pair_swap(cls.tokenizer,case,trace['promptTokenIds'],trace['cacheTokensAfterFinalToken'])
            e['bindingMask']=mask
            for record in e['interventions'].values():
                record.update(permutation=mask['permutation'],permutationHash=mask['permutationHash'])
            branches=study.compile_branches(cls.tokenizer,trace['promptTokenIds']+trace['generatedTokenIds'],case,p)
            e['branchInputHashes']={k:study.digest(v['inputIds']) for k,v in branches.items()}
            if e['cacheExport']: e['cacheExport']['file']=e['cacheExport']['file'].replace('prospective-binding-','prospective-pair-')
            states[case['id']]=e; cls.events.append(e)
        for phase in study.PHASES:
            recorded=[]
            for call in study.calls(p,phase):
                state=states[call['case']]; case=state['case']; config=study.mapping(p,phase,call)
                if phase=='coding': bit=call['verdict']
                elif phase=='forecast': bit=1
                else:
                    altered=call['condition']=='values_permuted' and call['target'] in state['bindingMask']['selectedRows']
                    bit=case['values'][call['target']] ^ int(altered)
                d=copy.deepcopy(cls.parent['tasks'][(call['case'],'actual')]['decoded'])
                recode(d,config,bit); d['snapshotCacheHash']=state['cacheHashes'][call['condition']]
                cls.events.append(dict(event='request',phase=phase,call=call,inputHash=state['branchInputHashes'][study.branch_name(phase,call)]))
                e=dict(event=phase,call=call,decoded=d,seconds=.01)
                recorded.append(e); cls.events.append(e)
            if phase!='task':
                cls.events.append(dict(event='coding_complete' if phase=='coding' else 'forecasts_complete',
                    count=len(recorded),recordsHash=study.digest(recorded)))
        cls.events.append(dict(event='complete',**p['counts']))

    def read(self,events,actual=False):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'pair.jsonl'; write_events(path,events)
            return study.read_journal(path,self.tokenizer,require_actual=actual,fixture_parent=self.parent)

    def test_one_opposite_pair_only_and_every_key_queried(self):
        for e in self.events:
            if e['event']!='state': continue
            mask=e['bindingMask']; perm=mask['permutation']; moved=mask['movedPositions']
            self.assertEqual(len(moved),2)
            self.assertEqual({mask['rows'][i]['bit'] for i in mask['selectedRows']},{0,1})
            self.assertEqual([i for i in range(len(perm)) if perm[i]!=i],moved)
            self.assertTrue(all(perm[perm[i]]==i for i in range(len(perm))))
        p=study.plan(); self.assertEqual(len(study.calls(p,'task')),800)
        self.assertEqual(len(study.calls(p,'forecast')),1600)

    def test_roundtrip_question_specific_errors_and_code_inversion(self):
        summary=study.summarize(self.read(self.events))
        self.assertEqual(summary['mixedOutcomeStates']['values_permuted'],24)
        rows=[r for r in summary['units'] if r['condition']=='values_permuted' and r['mapping']==0]
        self.assertEqual([(r['group'],r['n'],r['correct']) for r in rows],[('selected',48,0),('other',112,112)])
        self.assertTrue(all(r['semanticAgreements']==r['n'] for r in summary['codeInversionAgreement']))
        self.assertTrue(all(r['correct']==r['n'] for r in summary['codingChecks']))

    def test_mask_and_decoded_mapping_tampering_rejected(self):
        bad=copy.deepcopy(self.events); bad[1]['bindingMask']['selectedRows']=[0,0]
        with self.assertRaisesRegex(ValueError,'Pair mask changed'): self.read(bad)
        bad=copy.deepcopy(self.events)
        next(e for e in bad if e['event']=='forecast')['decoded']['candidateTokenIds'].reverse()
        with self.assertRaisesRegex(ValueError,'Decoded branch mapping'): self.read(bad)

    def test_early_tasks_or_changed_barrier_rejected(self):
        bad=copy.deepcopy(self.events)
        next(e for e in bad if e['event']=='request')['phase']='task'
        with self.assertRaisesRegex(ValueError,'Request order or input'): self.read(bad)
        bad=copy.deepcopy(self.events)
        next(e for e in bad if e['event']=='forecasts_complete')['recordsHash']='e'*64
        with self.assertRaisesRegex(ValueError,'Phase freeze'): self.read(bad)

    def test_actual_mode_and_parent_reproduction_enforced(self):
        with self.assertRaisesRegex(ValueError,'Actual GPU collection required'): self.read(self.events,True)
        bad=copy.deepcopy(self.events); bad[0]['origin']='transformers_gpu'
        with self.assertRaisesRegex(ValueError,'Fixture parent forbidden'): self.read(bad,True)
        bad=copy.deepcopy(self.events)
        next(e for e in bad if e['event']=='task' and e['call']==dict(case=0,target=0,condition='actual'))['decoded']['candidateMass']=.8
        with self.assertRaisesRegex(ValueError,'Parent task changed'): self.read(bad)

    def test_behavioral_coding_failures_are_reported_without_exclusion(self):
        bad=copy.deepcopy(self.events); p=study.plan()
        for e in bad:
            if e['event']=='coding' and e['call']['case']==0 and e['call']['verdict']==1 and e['call']['mapping']==0 and e['call']['condition'] in ('actual','same_schedule','restored'):
                recode(e['decoded'],p['forecastMappings'][0],0)
        next(e for e in bad if e['event']=='coding_complete')['recordsHash']=study.digest([e for e in bad if e['event']=='coding'])
        result=study.summarize(self.read(bad))
        r=next(r for r in result['codingChecks'] if r['condition']=='actual' and r['mapping']==0 and r['verdict']==1)
        self.assertEqual((r['valid'],r['correct'],r['n']),(24,23,24))


if __name__=='__main__': unittest.main()
