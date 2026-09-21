"""Independent small numerical check of the zero-gradient closed formula."""
import unittest
import numpy as np
from research.audit_optimizer_displacement import zero_gradient_displacement


class OptimizerDisplacementTests(unittest.TestCase):
    def test_matches_recursive_adam_and_zero_steps(self):
        m0=np.array([.8,-.3,0.]);v0=np.array([.12,.7,0.])
        m=m0.copy();v=v0.copy();theta=np.zeros(3)
        for t in range(49,65):
            m*=.9;v*=.999
            theta-=.0002*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
        predicted=zero_gradient_displacement(m0,v0,step=48,steps=16,lr=.0002,betas=(.9,.999),eps=1e-8)
        np.testing.assert_allclose(predicted,theta,rtol=2e-14,atol=1e-18)
        np.testing.assert_array_equal(m0,[.8,-.3,0.])
        np.testing.assert_array_equal(zero_gradient_displacement(m0,v0,step=48,steps=0,
            lr=.0002,betas=(.9,.999),eps=1e-8),np.zeros(3))
        with self.assertRaises(ValueError):
            zero_gradient_displacement(m0,-v0,step=48,steps=16,lr=.0002,betas=(.9,.999),eps=1e-8)


if __name__=='__main__':unittest.main()
