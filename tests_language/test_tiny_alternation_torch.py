import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from research import tiny_alternation as A
from research import tiny_wager as X


class TinyAlternationTorchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_workspace_gradient_and_forced_reader(self):
        net = A.model(7, 8, True)
        h = torch.randn(3, 64)
        noise = torch.rand((3, 8, A.UNITS), generator=torch.Generator().manual_seed(0))
        pa, pw, zs = net.workspace(h, noise=noise)
        self.assertTrue(((zs == 0) | (zs == 1)).all())  # drawn units are exactly binary going forward
        self.assertTrue(torch.allclose(pa.sum(-1), torch.ones(3)) and torch.allclose(pw.sum(-1), torch.ones(3)))
        pw[:, 1].sum().backward()
        self.assertGreater(float(net.units.weight.grad.abs().sum()), 0)  # the gradient passes the draws
        force = (torch.rand((1, 8, A.UNITS), generator=torch.Generator().manual_seed(1)) < 0.5).float()
        _, pw_forced, z_forced = net.workspace(h[:1], force=force)
        self.assertTrue(torch.equal(z_forced, force))
        r = torch.zeros(1, 64)
        for k in range(8):  # the reader sees the imposed sequence and nothing else
            r = net.reader(force[:, k], r)
        self.assertTrue(torch.allclose(torch.softmax(net.wager(r), -1), pw_forced))

    def test_resume_is_exact_and_runs_are_recomputable(self):
        old = A.UPDATES, A.KEEP_EVERY
        try:
            A.UPDATES, A.KEEP_EVERY = 2, 1
            with tempfile.TemporaryDirectory() as d:
                straight, _ = A.train("S8", 5, lambda m: None)
                A.UPDATES = 1
                A.train("S8", 5, lambda m: None, checkpoint=Path(d) / "c.pt")
                A.UPDATES = 2
                A.train("S8", 5, lambda m: None, checkpoint=Path(d) / "c.pt")
                state = torch.load(Path(d) / "c.pt", weights_only=False)
                self.assertEqual(state["update"], 2)
                for k, v in straight.state_dict().items():
                    self.assertTrue(torch.equal(v, state["net"][k]), k)
        finally:
            A.UPDATES, A.KEEP_EVERY = old
        for arm in ("S8", "D8", "cut"):
            net = A.model(5, *A.ARMS[arm]).eval()
            x = A.evaluate(net, arm, count=12)
            self.assertGreater(len(x["cases"]), 0)
            row = {"arm": arm, "seed": 5, "micro": 8, "sample": A.ARMS[arm][1], "cases": x["cases"],
                   "untrained": {"reward": x["reward"], "cases": x["cases"]}}
            m = A.measures(json.loads(json.dumps(row)))
            self.assertEqual(set(m), {"trained", "untrained"})
            self.assertEqual(set(m["trained"]["per_step"]["controls"]), {"average", "level", "uncoordinated"})
            c = x["cases"][0]
            self.assertEqual(np.asarray(c["a_steps"]).shape, (8, A.UNITS))
            self.assertEqual(len(c["forced"]), 4)
            self.assertEqual("forced_continuous" in c, arm == "D8")
        cut = A.model(5, 8, True).eval()
        kept = A.episodes(cut, "cut", [X.W.world_rng(A.PILOT_WORLDS, i) for i in range(4)], X.greedy,
                          lambda: torch.rand((4, 8, A.UNITS)), keep=True)[-1]
        for state, _, _ in kept.values():
            self.assertTrue(torch.equal(state, cut.start))  # the cut only ever sees the initial vector


if __name__ == "__main__":
    unittest.main()
