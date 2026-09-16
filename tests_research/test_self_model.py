import tempfile
import unittest
from pathlib import Path
import numpy as np
from research.recurrent import RecurrentMemory
from research.self_model import SelfMonitor, features
from research.evaluate_self_model import worlds, evaluate
from research.session import CognitiveSession


def constant_monitor(model, p=.75):
    width = model.hidden + 5
    return SelfMonitor(model, np.zeros(width), np.ones(width), np.zeros((width,32)),
                       np.zeros(32), np.zeros(32), np.log(p/(1-p)))


class SelfModelTests(unittest.TestCase):
    def test_public_counterfactuals_identical_private_states_change(self):
        model = RecurrentMemory()
        a, b = (worlds(model,33,96,f) for f in ('intact','noise'))
        np.testing.assert_array_equal(a['observer'], b['observer'])
        np.testing.assert_array_equal(a['target'], b['target'])
        self.assertGreater(np.linalg.norm(a['own']-b['own']), 0)
        np.testing.assert_array_equal(a['own'], a['observer'])

    def test_forecast_does_not_read_evaluation_labels(self):
        model = RecurrentMemory()
        data = worlds(model,22,96,'noise')
        monitor = constant_monitor(model)
        before = monitor.predict_features(data['own'])
        data['target'][:] = 3
        data['outcome'][:] = 0
        data['condition'][:] = -123
        np.testing.assert_array_equal(before, monitor.predict_features(data['own']))

    def test_runtime_forecasts_before_assessment_and_weights_remain_frozen(self):
        model = RecurrentMemory()
        monitor = constant_monitor(model)
        original = {k:v.copy() for k,v in model.p.items()}
        session = CognitiveSession(model,self_monitor=monitor)
        self.assertIsNone(session.observe(2)['forecast_recall_correct'])
        event = session.observe()
        self.assertAlmostEqual(event['forecast_recall_correct'],.75)
        event['forecast_recall_correct'] = 0
        result = session.assess(2)
        self.assertAlmostEqual(result['forecast_brier'],(.75-result['correct'])**2)
        with self.assertRaises(ValueError):
            session.assess(2)
        for k,v in original.items():
            np.testing.assert_array_equal(v, model.p[k])
        session.stop()
        with self.assertRaises(RuntimeError):
            session.observe()

    def test_saved_monitor_rejects_different_or_mutated_memory(self):
        model = RecurrentMemory()
        monitor = constant_monitor(model)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'monitor.json'
            monitor.save(path,{})
            loaded = SelfMonitor.load(path,model)
            np.testing.assert_array_equal(loaded.w1,monitor.w1)
            with self.assertRaises(ValueError):
                SelfMonitor.load(path,RecurrentMemory(seed=18))
        session = CognitiveSession(model,self_monitor=monitor)
        model.p['bo'][0] += .1
        with self.assertRaises(ValueError):
            session.observe(2)
        self.assertEqual(session.sequence,0)

    def test_metric_sign_and_decision_cost(self):
        model = RecurrentMemory()
        data = worlds(model,2,64,'intact')
        data['outcome'][:] = 1
        result = evaluate(constant_monitor(model,.99),constant_monitor(model,.5),data)
        self.assertGreater(result['brier_advantage'],0)
        self.assertEqual(result['own']['decision_cost'],0)
        self.assertAlmostEqual(result['observer']['decision_cost'],.2)

    def test_invalid_features(self):
        with self.assertRaises(ValueError):
            features(np.zeros((1,32)),np.full((1,4),.25),0)
        with self.assertRaises(ValueError):
            features(np.zeros((1,32)),np.full((1,4),np.nan),1)


if __name__ == '__main__':
    unittest.main()
