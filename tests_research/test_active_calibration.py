import json
import math
import unittest

import numpy as np

from research.active_calibration import (AdaptiveBindingAgent, SensorReading,
                                         WindowRegression, predictive_nll)
from research.audit_active_calibration import Regime, ambiguity_audit, compare
from research.binding_control import Observation


class ActiveCalibrationTests(unittest.TestCase):
    def test_posterior_against_independent_matrix_identity(self):
        model = WindowRegression()
        samples = [(-1., -.8), (1., 1.3), (0., .1), (1., 1.2)]
        for u, y in samples:
            model.update(u, y)
        x = np.array([[1., u] for u, _ in samples])
        y = np.array([v for _, v in samples])
        m0 = np.array([0., 1.])
        v0 = 100*np.eye(2)
        # Gaussian conditioning via the observation covariance, not normal equations.
        inverse = np.linalg.inv(np.eye(len(y))+x@v0@x.T)
        mean = m0+v0@x.T@inverse@(y-x@m0)
        beta = .25+.5*(y-x@m0)@inverse@(y-x@m0)
        fit = model.fit()
        np.testing.assert_allclose(fit['mean'], mean, atol=1e-12)
        self.assertAlmostEqual(fit['beta'], beta, places=12)
        predicted = model.forecast(.3)
        # Student density integrates to one; no omitted variance normalization.
        grid = np.linspace(-80, 80, 200001)
        density = np.array([math.exp(-predictive_nll(predicted, float(y))) for y in grid])
        self.assertAlmostEqual(float(np.trapezoid(density, grid)), 1., places=8)

    def test_window_forgets_exactly_and_probes_both_directions(self):
        model = WindowRegression(window=6)
        commands = []
        for _ in range(20):
            u = model.choose_probe()
            commands.append(u)
            model.update(u, -.7*u+.2)
        self.assertEqual(set(commands), {-1., 1.})
        self.assertEqual(model.fit()['rank'], 2)
        rebuilt = WindowRegression(window=6)
        for u, y in model.samples:
            rebuilt.update(u, y)
        np.testing.assert_array_equal(model.fit()['mean'], rebuilt.fit()['mean'])

    def test_zero_probes_and_missing_reference_remain_unidentified(self):
        model = AdaptiveBindingAgent(record=False)
        for _ in range(40):
            model.probe(lambda u: SensorReading(0., 1.5), action=0.)
        self.assertFalse(model.parameters()['identified_given_reference_assumption'])
        self.assertEqual(model.parameters()['gain'], 1.)
        result = ambiguity_audit()
        self.assertFalse(result['reference_absent_parameters']['identified_given_reference_assumption'])
        self.assertEqual(result['max_observation_difference'], 0.)
        self.assertEqual(result['max_learned_parameter_difference'], 0.)
        self.assertEqual(result['max_true_displacement_difference'], .5)

    def test_prediction_precedes_action_and_update(self):
        agent = AdaptiveBindingAgent()
        before = agent.parameters()
        def execute(u):
            self.assertEqual([e['type'] for e in agent.events], ['prediction'])
            self.assertEqual(agent.parameters(), before)
            return SensorReading(np.float64(-u), np.float64(-u+1.5))
        agent.probe(execute)
        self.assertEqual([e['type'] for e in agent.events], ['prediction', 'observation', 'revision'])
        self.assertNotEqual(agent.parameters(), before)
        compare(agent.events, json.loads(json.dumps(agent.events)))
        with self.assertRaises(TypeError):
            SensorReading(reference=1., visual=1., true_gain=1.)

    def test_checkpoint_resume_and_frozen_update(self):
        agent = AdaptiveBindingAgent(record=False)
        for _ in range(12):
            agent.probe(lambda u: SensorReading(-u, -.5*u+1.))
        restored = AdaptiveBindingAgent.restore(json.loads(json.dumps(agent.snapshot())), record=False)
        self.assertEqual(agent.parameters(), restored.parameters())
        obs = Observation(0., 1.3, 1.5)
        self.assertEqual(agent.reach(obs), restored.reach(obs))
        saved = restored.snapshot()
        restored.probe(lambda u: SensorReading(100., 100.), learn=False)
        self.assertEqual(restored.snapshot(), saved)

    def test_learned_parameters_drive_existing_binding_controller(self):
        rng = np.random.default_rng(10)
        agent = AdaptiveBindingAgent(record=False)
        world = Regime(gain=-1., visual_scale=.5, visual_bias=1.5)
        for _ in range(40):
            noise = rng.normal(size=2)
            agent.probe(lambda u: world.calibration(u, noise))
        state = agent.parameters()
        self.assertAlmostEqual(state['gain'], -1., delta=.08)
        self.assertAlmostEqual(state['visual_bias'], 1.5, delta=.2)
        self.assertAlmostEqual(state['visual_scale'], .5, delta=.2)
        decision = agent.reach(Observation(0., state['visual_bias'], 1.5))
        self.assertLess(decision['command'], 0.)
        self.assertAlmostEqual(decision['estimated_position'], 0.)
        self.assertIsNotNone(decision['binding'])

    def test_replay_checks_nested_schema_and_values(self):
        compare({'a': [1., True]}, {'a': [1.+1e-12, True]})
        with self.assertRaises(AssertionError):
            compare({'a': [1., True]}, {'a': [1.1, True]})
        with self.assertRaises(AssertionError):
            compare({'a': [1., True]}, {'a': [1., False]})

    def test_affine_probe_design_does_not_identify_nonlinearity(self):
        result = ambiguity_audit()
        self.assertTrue(result['geometric_equals_alternating'])
        differences = result['nonlinear_vs_bias_mean_difference_by_probe']
        self.assertEqual(differences['-1.0'], 0.)
        self.assertEqual(differences['1.0'], 0.)
        self.assertEqual(differences['0.0'], .15)


if __name__ == '__main__':
    unittest.main()
