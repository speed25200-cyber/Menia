import copy
import itertools
import math
import unittest
from unittest.mock import patch

from research import optimizer_memory as s
from research import audit_optimizer_memory as audit


def fixture():
    blocks=[dict(id=f'b{i}',replication=0,split='test',marker=i) for i in (0,1)]
    requests=[];rows=[]
    for arm,form,(family,task),mapping,b,pos in itertools.product(s.ARMS,s.FORMATS,s.STREAMS,(0,1),blocks,range(4)):
        truth=int(pos in (1,2)) if task=='monitor' else int(b['marker']==int(task=='marker_second'))
        predicted=0 if arm=='carry' and task=='monitor' else 1-truth if arm=='zero_grad' else truth
        digit=predicted^mapping;z=[0.,0.,0.,0.];z[digit]=math.log(3)
        req=dict(id=f"0/{arm}/{family}/{b['id']}/{form}/{task}/{mapping}/{pos}",replication=0,arm=arm,family=family,
            block=b['id'],format=form,task=task,mapping=mapping,position=pos)
        out=dict(choiceLogits=z,choiceMass=.8,rawTokenId=digit+10,rawChoice=digit,seconds=.1)
        # The independent arithmetic intentionally requires no intervention trace; the strict reader covers it separately.
        requests.append(req);rows.append(dict(request=req,result=out))
    h=dict(plan=dict(blocks=blocks,evaluation=requests),metadata=dict(choiceTokenIds=[10,11,12,13]))
    return h,rows,{},None,[]


class IndependentMemoryAuditTests(unittest.TestCase):
    def test_full_vocabulary_loss_balancing_inversion_and_contrasts(self):
        with patch.dict(s.CONFIG,dict(replications=1,resamples=20)),patch.object(s,'read_journal',return_value=fixture()):
            result=audit.recompute('synthetic-only')
        self.assertEqual(result['recorded'],512);self.assertEqual(result['shamPairs'],128)
        self.assertEqual(len(result['tables']),64);self.assertEqual(len(result['contrasts']),48)
        for mapping in (0,1):
            perfect=result['tables'][f'0/test/prefix/hidden/trained/monitor/{mapping}']
            self.assertEqual(perfect['nativeBalancedAccuracy'],1)
            self.assertAlmostEqual(perfect['balancedCrossEntropy'],-math.log(.4))
            self.assertAlmostEqual(perfect['choiceMass'],.8*4/6)
            self.assertAlmostEqual(perfect['brier'],.125)
            carry=result['tables'][f'0/test/carry/hidden/trained/monitor/{mapping}']
            self.assertEqual(carry['nativeBalancedAccuracy'],.5)
            self.assertEqual(carry['presenceAUROC'],.5)
            self.assertAlmostEqual(carry['balancedCrossEntropy'],(-math.log(.4)-math.log(.8/6))/2)
        contrasts=[x for x in result['contrasts'] if x['primary']]
        self.assertEqual(len(contrasts),6)
        expected={'carry-prefix':-.5,'zero_grad-prefix':-1.,'reset_m-carry':.5}
        for x in contrasts:
            self.assertEqual(x['nativeDifference'],expected[x['comparison']])
            self.assertEqual(x['nativeInterval95'],[expected[x['comparison']]]*2)
        changed=copy.deepcopy(result);changed['tables']['0/test/prefix/hidden/trained/monitor/0']['balancedCrossEntropy']+=.01
        with self.assertRaisesRegex(ValueError,'calculation mismatch'):audit.compare(result,changed)

    def test_changed_sham_is_rejected_independently(self):
        data=fixture();row=next(r for r in data[1] if r['request']['position']==3);row['result']['choiceMass']=.7
        with patch.dict(s.CONFIG,dict(replications=1,resamples=20)),patch.object(s,'read_journal',return_value=data):
            with self.assertRaisesRegex(ValueError,'sham failure'):audit.recompute('synthetic-only')


if __name__=='__main__':unittest.main()
