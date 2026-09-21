import unittest
import numpy as np
import torch
from menia.source_monitor import SourceMonitor
from research.train_source_monitor import torch_forward


class SourceTrainingTests(unittest.TestCase):
    def test_training_and_exported_inference_match_all_architectures(self):
        x = np.random.default_rng(72).normal(size=(9, 4, 5))
        for kind in ("recurrent", "reset", "window"):
            model = SourceMonitor(kind, 17)
            parameters = {k: torch.tensor(v, dtype=torch.float32) for k, v in model.p.items()}
            expected = torch.sigmoid(torch_forward(parameters, torch.tensor(x, dtype=torch.float32), kind)).numpy()
            state, actual = model.zero(4), []
            for frame in x:
                state, q = model.step(frame, state)
                actual.append(q)
            np.testing.assert_allclose(actual, expected, atol=2e-7)

    def test_future_features_do_not_change_earlier_predictions(self):
        x = torch.tensor(np.random.default_rng(73).normal(size=(9, 4, 5)), dtype=torch.float32)
        for kind in ("recurrent", "reset", "window"):
            model = SourceMonitor(kind, 17)
            parameters = {k: torch.tensor(v, dtype=torch.float32) for k, v in model.p.items()}
            first = torch_forward(parameters, x, kind)
            changed = x.clone()
            changed[5:] += 100
            second = torch_forward(parameters, changed, kind)
            torch.testing.assert_close(first[:5], second[:5], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
