import copy
import math
import unittest

import numpy as np

from research.confidence_calibration_diagnostic import (
    auc_partition,fit_monotone,fit_rows,objective,predict_calibrated)
from research.cross_model_prediction import CELLS,reference


class ConfidenceCalibrationDiagnosticTests(unittest.TestCase):
    def test_auc_partition_exposes_group_ranking_and_exact_ties(self):
        truth=[1,1,1,0,1,0,0,0];p=[.9]*4+[.1]*4
        result=auc_partition(truth,p,['easy']*4+['hard']*4)
        self.assertEqual(result['pairs'],16);self.assertEqual(result['auc'],.75)
        self.assertEqual(result['within']['pairs'],6);self.assertEqual(result['within']['auc'],.5)
        self.assertEqual(result['between']['pairs'],10);self.assertEqual(result['between']['auc'],.9)
        self.assertEqual(sum(result[k]['contributionToGlobalAUC'] for k in ('within','between')),result['auc'])
        one_group=auc_partition(truth,p,['same']*8)
        self.assertEqual(one_group['within']['auc'],result['auc']);self.assertIsNone(one_group['between']['auc'])
        one_class=auc_partition([1,1],[.1,.9],['a','b'])
        self.assertEqual(one_class['pairs'],0);self.assertIsNone(one_class['auc'])

    def test_objective_gradient_and_hessian_match_finite_differences(self):
        x=np.asarray([-1.,-.2,.3,1.2]);y=np.asarray([0.,1.,0.,1.]);theta=np.asarray([.4,.7]);eps=1e-5
        _,grad,hess=objective(theta,x,y,.25)
        for i in range(2):
            shift=np.eye(2)[i]*eps
            plus,gplus,_=objective(theta+shift,x,y,.25);minus,gminus,_=objective(theta-shift,x,y,.25)
            self.assertAlmostEqual(grad[i],(plus-minus)/(2*eps),places=8)
            np.testing.assert_allclose(hess[:,i],(gplus-gminus)/(2*eps),rtol=1e-7,atol=1e-8)
        self.assertGreater(np.linalg.eigvalsh(hess).min(),0.)

    def test_positive_calibration_converges_and_preserves_ranking(self):
        y=[0,0,0,1]*12+[0,1,1,1]*12;p=[.01]*48+[.1]*48
        fit=fit_monotone(y,p);pred=predict_calibrated(fit,p)
        self.assertGreater(fit['slope'],0);self.assertLessEqual(fit['kktResidual'],1e-9)
        self.assertLess(np.mean((pred-y)**2),np.mean((np.asarray(p)-y)**2))
        self.assertEqual(auc_partition(y,p,['x']*96)['auc'],auc_partition(y,pred,['x']*96)['auc'])
        self.assertEqual(fit,fit_monotone(y,p))

    def test_boundary_constant_and_extreme_probabilities(self):
        y=[0,0,1,1];p=[.9,.8,.2,.1]
        fit=fit_monotone(y,p)
        self.assertEqual(fit['slope'],0.);np.testing.assert_allclose(predict_calibrated(fit,p),.5)
        constant=fit_monotone([0,0,0,1],[.5]*4)
        self.assertEqual(constant['slope'],0.);np.testing.assert_allclose(predict_calibrated(constant,[0,1]),.25)
        extreme=fit_monotone([0,0,1,1],[0,1e-20,1-1e-15,1])
        self.assertTrue(np.isfinite(predict_calibrated(extreme,[0,.5,1])).all())
        with self.assertRaises(ValueError):fit_monotone([0,0],[.1,.2])
        with self.assertRaises(ValueError):fit_monotone([0,1],[float('nan'),.5])

    def test_fit_does_not_read_held_out_or_other_replication_data(self):
        rows=[]
        for family,level in CELLS:
            for index in range(16):
                task=dict(id=len(rows),split='calibration',replication=0,family=family,level=level,
                          letters='A'*level,operands=[1]*level)
                correct=index%2;answer=dict(status='ok',text=str(reference(task)+(1-correct)))
                rows.append(dict(task=task,answers={'base':answer},judgments={'measured':{'base':{'conditionalCorrect':.2+.6*correct}}}))
        original=copy.deepcopy(rows);fit=fit_rows(rows,0,'base','measured')
        poisoned=[dict(task=dict(split='test',replication=0),answers=None,judgments=float('nan')),
                  dict(task=dict(split='calibration',replication=1),answers=None,judgments=None)]
        self.assertEqual(fit,fit_rows(rows+poisoned,0,'base','measured'));self.assertEqual(rows,original)
        self.assertEqual(fit['calibrationCount'],96)
        with self.assertRaises(ValueError):fit_rows(rows[:-1],0,'base','measured')
        with self.assertRaises(ValueError):fit_rows(rows[:-1]+[rows[0]],0,'base','measured')


if __name__=='__main__':unittest.main()
