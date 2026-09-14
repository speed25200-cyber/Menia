import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from research.recurrent import RecurrentMemory, episodes

class RecurrentTests(unittest.TestCase):
    def test_gradients_against_finite_differences(self):
        model = RecurrentMemory(hidden=4)
        x,y,mask = episodes(18,batch=2,length=4)
        _, grads = model.loss_and_grad(x,y,mask)
        for key, parameter in model.p.items():
            index = tuple(0 for _ in parameter.shape)
            old = parameter[index]
            parameter[index] = old + 1e-5
            plus,_ = model.loss_and_grad(x,y,mask)
            parameter[index] = old - 1e-5
            minus,_ = model.loss_and_grad(x,y,mask)
            parameter[index] = old
            with self.subTest(parameter=key):
                self.assertAlmostEqual(grads[key][index],(plus-minus)/2e-5,places=7)

    def test_causal_no_future_leak(self):
        model = RecurrentMemory()
        x,_,_ = episodes(19,batch=2,length=10)
        before,_ = model.forward(x)
        changed = x.copy()
        changed[7:] = 0
        after,_ = model.forward(changed)
        np.testing.assert_array_equal(before[:7],after[:7])

    def test_roundtrip_and_reset(self):
        model = RecurrentMemory(hidden=4)
        x,_,_ = episodes(20,batch=1,length=8)
        p,_ = model.forward(x)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.json'
            model.save(path,{})
            loaded=RecurrentMemory.load(path)
            q,_=loaded.forward(x)
            np.testing.assert_array_equal(p,q)
            malformed=json.loads(path.read_text())
            malformed['parameters']['b']=[float('nan')]*4
            path.write_text(json.dumps(malformed))
            with self.assertRaises(ValueError): RecurrentMemory.load(path)
        np.testing.assert_allclose(p.sum(-1),1)

    def test_mask_excludes_observed_steps(self):
        x,y,mask = episodes(21)
        np.testing.assert_array_equal(mask,1-x[:,:,4])
        self.assertFalse(mask[0].any())
        self.assertTrue(mask[-1].all())

    def test_shape_and_finite_guards(self):
        model=RecurrentMemory()
        with self.assertRaises(ValueError): model.step(np.zeros((1,4)), model.zero())
        with self.assertRaises(ValueError): model.step(np.full((1,5), np.nan), model.zero())

if __name__ == '__main__': unittest.main()
