import math
import unittest

import numpy as np

from research.binding_control import (BindingControlAgent, BindingController,
                                     Observation, ordinary_conditional_mean)
from research.causal_binding import logistic, logit
from research.audit_binding_control import assert_report_equal, simulate_batch


class BindingControlTests(unittest.TestCase):
    def test_conditional_moments_against_independent_quadrature(self):
        controller = BindingController(.3, visual_sigma=.7)
        observation = Observation(-.4, 2.3, 3.)
        decision = controller.decide(observation)
        x = np.linspace(-14.4, 13.6, 60001)
        prior = np.exp(-.5*(x+.4)**2)/math.sqrt(2*math.pi)
        residual = observation.visual-x
        likelihood = (.3*np.exp(-.5*residual**2/.49)/math.sqrt(2*math.pi*.49)
                      +.7*np.exp(-.5*residual**2/16.49)/math.sqrt(2*math.pi*16.49))
        density = prior*likelihood
        density /= np.trapezoid(density, x)
        mean = np.trapezoid(x*density, x)
        variance = np.trapezoid((x-mean)**2*density, x)
        self.assertAlmostEqual(decision.estimated_position, mean, places=11)
        self.assertAlmostEqual(decision.predicted_error_variance, variance, places=11)
        for offset in (-.3, .3):
            loss = np.trapezoid((x+decision.displacement+offset-3)**2*density, x)
            self.assertAlmostEqual(loss-variance, offset**2, places=11)

    def test_report_intervention_and_compensated_prior(self):
        observation = Observation(0., 1.5, 3.)
        base = BindingController(.5).decide(observation)
        report_shift = BindingController(.5).decide(observation, report_threshold=math.log(4))
        removed = BindingController(.5).decide(observation, report_enabled=False)
        compensated = BindingController(logistic(logit(.5)+math.log(4))).decide(
            observation, report_threshold=math.log(4))
        self.assertNotEqual(base.report, report_shift.report)
        self.assertEqual(base.displacement, report_shift.displacement)
        self.assertEqual(base.displacement, removed.displacement)
        self.assertIsNone(removed.report)
        self.assertEqual(base.report, compensated.report)
        self.assertNotEqual(base.displacement, compensated.displacement)

    def test_graft_changes_action_and_response_without_changing_observation(self):
        controller = BindingController(.5)
        observation = Observation(0., 2., 3.)
        first = controller.decide(observation, binding_override=0.)
        second = controller.decide(observation, binding_override=1.)
        self.assertEqual(first.inferred_binding, second.inferred_binding)
        self.assertNotEqual(first.displacement, second.displacement)
        self.assertNotEqual(first.report, second.report)

    def test_ordinary_regression_reproduces_control(self):
        for prior in (.1, .5, .9):
            for noise in (.25, 1., 2.):
                controller = BindingController(prior, visual_sigma=noise)
                for y in (-30., -3., 0., 1., 5., 30.):
                    observation = Observation(-.7, y, 3.)
                    estimate = ordinary_conditional_mean(observation, prior, 1., noise, 4.)
                    self.assertAlmostEqual(estimate, controller.decide(observation).estimated_position, places=12)

    def test_action_is_logged_before_private_world_outcome(self):
        agent = BindingControlAgent(BindingController(.5))
        private_position = .7
        def execute(action):
            self.assertEqual([e['type'] for e in agent.events], ['observation', 'decision'])
            self.assertNotIn('position', agent.events[0])
            return (private_position+action-3)**2
        decision, cost = agent.step(Observation(0., 1.3, 3.), execute)
        self.assertEqual(cost, (private_position+decision.displacement-3)**2)
        self.assertEqual(agent.events[-1]['type'], 'outcome')
        with self.assertRaises(TypeError):
            Observation(anchor=0., visual=1., target=3., position=.7)

    def test_invalid_inputs_and_zero_disparity(self):
        controller = BindingController(.5)
        observation = Observation(1., 1., 3.)
        for q in (0., .5, 1.):
            self.assertEqual(controller.decide(observation, binding_override=q).displacement, 2.)
        for q in (-1., 2., math.nan):
            with self.assertRaises(ValueError):
                controller.decide(observation, binding_override=q)
        with self.assertRaises(ValueError):
            Observation(0., math.inf, 3.)
        with self.assertRaises(ValueError):
            BindingController(.5, visual_sigma=0.)

    def test_small_paired_simulation_preserves_invariants(self):
        metrics, error = simulate_batch(.5, 1., 716000, trials=64)
        self.assertLess(error, 1e-12)
        self.assertEqual(metrics['report_shift']['cost_difference'], 0.)
        self.assertEqual(metrics['report_removed']['action_change_rate'], 0.)
        self.assertEqual(metrics['prior_compensated']['report_disagreement'], 0.)

    def test_replay_rejects_schema_and_numeric_changes(self):
        assert_report_equal({'x': [1., None, True]}, {'x': [1.+1e-13, None, True]})
        for bad in ({'x': [1.1, None, True]}, {'x': [1., None]}, {'x': [1, None, True]}):
            with self.assertRaises(AssertionError):
                assert_report_equal({'x': [1., None, True]}, bad)


if __name__ == '__main__':
    unittest.main()
