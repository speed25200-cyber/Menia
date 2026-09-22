import copy
import json
from pathlib import Path
import tempfile
import unittest

from research import answer_confidence_study as study
from research.answer_confidence_plan import load_training,training_batches
from research.answer_confidence_baselines import fit
from research.answer_confidence_journal import read_journal,validate_judgment
from research.answer_confidence_analysis import analyze
from research.audit_answer_confidence import verify
from research.cross_model_prediction import digest,reference
from research.natural_error_journal import Writer
from tests_research.test_answer_confidence_baselines import fixture as baseline_fixture

ROOT=Path(__file__).resolve().parents[1]


def make_fixture(path):
    """Deliberately oracle-built outputs test analysis only, never performance."""
    p=study.make_plan();data,report=load_training(ROOT/'artifacts/answer-confidence-training-data')
    w=Writer(path);metadata=dict(confidenceTokenIds=[15,16],vocabularySize=100)
    w.write(dict(event='header',origin='synthetic_fixture',plan=p,planHash=digest(p),sourceHash=study.source_hash(),
                 trainingData=data,trainingReport=report,metadata=metadata),create=True)
    checkpoints={}
    for unit in p['trainingUnits']:
        key=unit['key'];initial=digest(['initial',unit['replication']])
        w.write(dict(event='training_start',key=key,initializationHash=initial,trainableParameters=8))
        for step,batch in enumerate(training_batches(data,unit['replication'],unit['arm']),1):
            w.write(dict(event='training_step',key=key,step=step,dataHash=digest(batch),losses=[.1]*8,
                         inputTokens=[32]*8,gradientNorm=1.,seconds=.1))
        event=dict(event='training_complete',key=key,steps=144,initializationHash=initial,
                   checkpoint=Path(path).with_suffix('.'+key+'.safetensors').name,sha256=digest(['trained',key]),trainingSeconds=14.4)
        checkpoints[key]=event;w.write(event)
    answers={};judgments={};template=baseline_fixture()[0]['answers']['base']
    for call in p['calls']:
        if call['id']==p['calibrationCalls']:
            rows=study.evaluation_rows(p,answers,judgments,'calibration')
            for rep in range(3):w.write(dict(event='baseline',replication=rep,bundle=fit(rows,rep)))
        request=study.request_for(p,call['id'],answers,checkpoints);w.write(request)
        task=p['tasks'][call['task']];good=task['id']%3!=0
        if call['kind']=='answer':
            event=copy.deepcopy(template)
            event.update(event='answer',id=call['id'],errorType=None,text=str(reference(task)+int(not good)))
            event['metrics']['effectiveGeneration']=copy.deepcopy(p['generation'])
            answers[(task['id'],call['producer'])]=event
        else:
            prob=(.9 if good else .1) if call['judge']=='measured' else ((.6 if good else .4) if call['judge']=='base' else .5)
            event=dict(event='judgment',id=call['id'],status='ok',seconds=.01,
                scores=dict(conditionalCorrect=prob,candidateMass=.9,inputHash=digest(request['messages']),inputTokens=32,
                            codeTokenIds=[15,16],topTokenId=16 if prob>=.5 else 15,topIsCode=True,sampled=False,
                            source='Recomputed full text prefix, raw lm_head logits at temperature 1'))
            judgments[(task['id'],call['judge'],call['producer'])]=event
        w.write(event)


class ConfidenceStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory();cls.path=Path(cls.directory.name)/'trial.jsonl'
        make_fixture(cls.path);cls.report=analyze(cls.path)
        cls.events=[json.loads(line)['payload'] for line in cls.path.read_text(encoding='utf-8').splitlines()]

    @classmethod
    def tearDownClass(cls):cls.directory.cleanup()

    def rewrite(self,events,name):
        path=Path(self.directory.name)/name
        # Keep the basename used to commit checkpoint names in the journal.
        path=path/'trial.jsonl';w=Writer(path)
        for i,event in enumerate(events):w.write(event,create=i==0)
        return path

    def test_complete_matched_trial_and_independent_core_arithmetic(self):
        r=self.report
        self.assertEqual(r['origin'],'synthetic_fixture');self.assertTrue(r['nativeConfidenceCriterion'])
        self.assertEqual((r['generations'],r['judgments'],r['trainingSteps'],r['baselineFits']),(2592,7776,864,3))
        self.assertEqual(len(r['contrasts']),24)
        for c in r['contrasts']:
            if c['comparison']=='shuffled-measured':self.assertAlmostEqual(c['brierGain'],.24)
        checked=verify(self.path,r,check_weights=False)
        self.assertTrue(checked['verified']);self.assertFalse(checked['weightsChecked'])
        self.assertLess(checked['maxAbsoluteDifference'],1e-8)
        for rep in r['replications'].values():
            self.assertAlmostEqual(rep['crossed']['changes']['measured']['answerChangeUnderCurrentJudge'],0.)

    def test_altered_core_scores_intervals_and_gates_are_rejected(self):
        a=copy.deepcopy(self.report);a['replications']['0']['scores']['base']['measured']['brier']+=.02
        b=copy.deepcopy(self.report);b['contrasts'][0]['familyInterval'][0]+=.01
        c=copy.deepcopy(self.report);c['gates']['0']['answerAccuracyRetained']=False
        for changed in (a,b,c):
            with self.assertRaises(AssertionError):verify(self.path,changed,check_weights=False)

    def test_rehashed_wrong_answer_context_and_late_baseline_are_rejected(self):
        changed=copy.deepcopy(self.events)
        next(e for e in changed if e['event']=='judgment')['scores']['inputHash']='a'*64
        with self.assertRaises(ValueError):read_journal(self.rewrite(changed,'wrong-context'))
        changed=[copy.deepcopy(e) for e in self.events if not (e['event']=='baseline' and e['replication']==2)]
        with self.assertRaises(ValueError):read_journal(self.rewrite(changed,'missing-freeze'))

    def test_tiny_candidate_mass_cannot_pass_as_native_confidence(self):
        changed=copy.deepcopy(self.events)
        for e in changed:
            if e['event']=='judgment':e['scores']['candidateMass']=.001
        # Calibration fitting does not use judgments, but its provenance hash
        # includes the complete calibration rows; reconstruct those receipts.
        p=changed[0]['plan'];answers={};judgments={}
        for e in changed:
            if e['event']=='answer':
                c=p['calls'][e['id']];answers[(c['task'],c['producer'])]=e
            elif e['event']=='judgment':
                c=p['calls'][e['id']];judgments[(c['task'],c['judge'],c['producer'])]=e
            elif e['event']=='baseline':e['bundle']=fit(study.evaluation_rows(p,answers,judgments,'calibration'),e['replication'])
        r=analyze(self.rewrite(changed,'tiny-mass'))
        self.assertTrue(all(c['passed'] for c in r['contrasts']))
        self.assertFalse(r['nativeConfidenceCriterion'])
        self.assertTrue(all(not g['nativeCodeFormatPassed'] for g in r['gates'].values()))


if __name__=='__main__':unittest.main()
