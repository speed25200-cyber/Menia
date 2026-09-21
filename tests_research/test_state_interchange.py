"""Known complete factorial fixture; never evidence about pretrained Qwen."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research import state_interchange as s


def fixture_group(pair,choices):
    outputs=[];intact={}
    for r in s.requests():
        kind=r['kind'];role=r['role'] if kind=='intact' else 'recipient'
        state=r['state'] if kind=='intact' else r['recipientState']
        code=r['code'] if kind=='intact' else r['recipientCode'];b=pair[role]
        body,_=s.previous.previous.prompt(b,'hidden',r['task'],b['positivePosition'] if state else 0,'trained',code)
        trace=dict(applications=int(state!=0),changed=bool(state),normRelativeError=.001 if state else 0.)
        common=dict(request=r,seconds=.001,promptHash=s.digest(body),inputTokens=42,intervention=trace)
        if kind=='intact':
            truth=state if r['task']=='monitor' else int(b['marker']==0)
            digit=truth^code
            hashes={str(site):s.digest([b['id'],r['task'],code,0 if site==17 else state,site]) for site in s.SITES}
            extra=dict(activationHashes=hashes)
        else:
            recipient=intact[(r['task'],'recipient',state,code)]
            donor=recipient if kind=='sham' else intact[(r['task'],'donor',r['donorState'],r['donorCode'])]
            if kind=='sham' or r['site']==17:digit=recipient['rawChoice']
            elif r['site']==35:digit=donor['rawChoice']
            else:digit=s.semantic_predictions(pair,r)['stateTransfer']
            rh=recipient['activationHashes'][str(r['site'])];dh=donor['activationHashes'][str(r['site'])]
            extra=dict(patch=dict(site=r['site'],tokenIndex=41,applications=1,recipientHash=rh,
                donorHash=dh,patchedHash=dh,recipientNorm=1.,donorNorm=1.,displacementNorm=float(rh!=dh)))
        z=[0.,0.,0.,0.];z[digit]=2.
        out=dict(common,**extra,choiceLogits=z,choiceMass=.8,rawChoice=digit,rawTokenId=choices[digit])
        outputs.append(out)
        if kind=='intact':intact[s.intact_key(r)]=out
    return outputs


class StateInterchangeTests(unittest.TestCase):
    def test_complete_known_mechanisms_and_rejected_corruption(self):
        with patch.dict(s.CONFIG,{'resamples':20}),tempfile.TemporaryDirectory() as temp:
            fixed=s.plan();pairs={p['id']:p for p in fixed['pairs']};choices=[3,4,5,6]
            self.assertEqual(fixed['plannedForwards'],19584)
            sentences=[sentence for p in fixed['pairs'] for role in ('donor','recipient') for sentence in p[role]['sentences']]
            self.assertEqual(len(sentences),len(set(sentences)))
            events=[dict(event='header',plan=fixed,planHash=s.digest(fixed),sourceHash=s.source_hash(),
                         metadata=dict(origin='synthetic_fixture',choiceTokenIds=choices))]
            for g in fixed['groups']:
                outputs=fixture_group(pairs[g['pair']],choices)
                events.extend([dict(event='group_start',group=g),dict(event='group_complete',id=g['id'],outputs=outputs)])
            path=Path(temp)/'fixture.jsonl';path.write_text('\n'.join(json.dumps(e) for e in events)+'\n',encoding='utf-8')
            report=s.analyze(path)
            self.assertTrue(report['complete']);self.assertEqual(report['recorded'],19584)
            self.assertEqual(report['selfShams'],3456);self.assertEqual(report['finalLayerCopies'],4608)
            self.assertEqual(len(report['tables']),72);self.assertEqual(len(report['contrasts']),144)
            primary=[c for c in report['contrasts'] if c['primary']];self.assertEqual(len(primary),6)
            for c in primary:self.assertEqual(c['difference'],.5);self.assertEqual(c['interval95'],[.5,.5])
            for site,metric in ((17,'recipientUnchanged'),(23,'stateTransfer'),(35,'donorAnswer')):
                self.assertEqual(report['tables'][f'0/test/prefix/monitor/{site}'][metric]['mean'],1.)
            self.assertTrue(all(v['allPassed'] for v in report['prerequisites'].values()))
            bad=copy.deepcopy(events[2]['outputs']);bad[8]['choiceMass']=.7
            with self.assertRaisesRegex(ValueError,'Self-sham differs'):s.validate_group(pairs[fixed['groups'][0]['pair']],bad,choices)
            incomplete=Path(temp)/'partial.jsonl';incomplete.write_text('\n'.join(json.dumps(e) for e in events[:2])+'\n')
            h,done,pending,_=s.read_journal(incomplete);self.assertEqual(done,[]);self.assertIsNotNone(pending)
            self.assertFalse(s.analyze(incomplete)['complete'])


if __name__=='__main__':unittest.main()
