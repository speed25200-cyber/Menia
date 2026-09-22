import itertools
import unittest

import numpy as np

from research.capacity_planning import (Model, CompiledPlanner, CapacityAgent,
                                       lower_envelope, recursive_costs, update_belief)


class CapacityPlanningTests(unittest.TestCase):
    def test_bayes_and_ignored_diagnostic(self):
        self.assertAlmostEqual(update_belief(.3, 1, .8), .24/.38)
        self.assertAlmostEqual(update_belief(.3, 0, .8), .06/.62)
        for q in np.linspace(0, 1, 11):
            for z in (0, 1):
                self.assertEqual(update_belief(q, z, .85, False), q)
                self.assertAlmostEqual(update_belief(q, z, .5), q)

    def test_two_steps_against_enumerated_state_observation_trees(self):
        model = Model()
        for use in (False, True):
            planner = CompiledPlanner(model, 2, use)
            trees = []
            for root, child0, child1 in itertools.product(range(3), repeat=3):
                if not use and child0 != child1:
                    continue
                conditional = []
                for state in (0, 1):
                    total = model.costs()[root, state]
                    for next_state, z in itertools.product((0, 1), repeat=2):
                        observation = (.85 if next_state == z else .15)
                        child = child1 if z else child0
                        total += (model.transition(root)[state, next_state]*observation
                                  *model.costs()[child, next_state])
                    conditional.append(total)
                trees.append((root, np.array(conditional)))
            for q in np.linspace(0, 1, 201):
                b = np.array([1-q, q])
                expected = [min(b@v for root, v in trees if root == action) for action in range(3)]
                np.testing.assert_allclose(planner.evaluate(q, 2)[1], expected, atol=1e-12, rtol=0)

    def test_envelope_against_all_lines(self):
        rng = np.random.default_rng(761)
        for _ in range(10):
            vectors = rng.normal(size=(100, 2))
            vectors = np.concatenate([vectors, vectors[:5], [[1, 1], [0, 1], [1, 0]]])
            q = np.linspace(0, 1, 1001)
            b = np.stack([1-q, q])
            np.testing.assert_allclose(np.min(vectors@b, axis=0),
                                       np.min(lower_envelope(vectors)@b, axis=0), atol=1e-12, rtol=0)

    def test_free_information_cannot_raise_optimal_expected_cost(self):
        model = Model()
        informed = CompiledPlanner(model, 4, True)
        ignored = CompiledPlanner(model, 4, False)
        for q in np.linspace(.01, .99, 51):
            p_one = q*.85+(1-q)*.15
            for depth in range(1, 5):
                without = min(ignored.evaluate(q, depth)[1])
                with_info = sum(prob*min(informed.evaluate(update_belief(q, z, .85), depth)[1])
                                for z, prob in [(0, 1-p_one), (1, p_one)])
                self.assertLessEqual(with_info, without+1e-12)

    def test_useless_maintenance_and_terminal_horizon(self):
        for use in (False, True):
            planner = CompiledPlanner(Model(effective_maintenance=False), 4, use)
            for q in np.linspace(0, 1, 101):
                self.assertNotEqual(planner.evaluate(q, 4)[0], 2)
            agent = CapacityAgent(CompiledPlanner(Model(), 4, use))
            agent.observe(0, 0)
            action, decision = agent.decide(1)
            self.assertEqual(decision['horizon'], 1)
            self.assertNotEqual(action, 2)

    def test_compiled_vs_recursive_with_reversed_reliability(self):
        # Also allow genuinely anti-informative diagnostics in the supplied model.
        for d in (.15, .5, .85):
            model = Model(diagnostic_accuracy=d)
            for use in (False, True):
                planner = CompiledPlanner(model, 4, use)
                for q in np.linspace(0, 1, 31):
                    for depth in range(1, 5):
                        expected, _ = recursive_costs(model, q, depth, use)
                        np.testing.assert_allclose(planner.evaluate(q, depth)[1], expected,
                                                   atol=1e-11, rtol=0)


if __name__ == '__main__':
    unittest.main()
