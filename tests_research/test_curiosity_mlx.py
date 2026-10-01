import os
import unittest
import numpy as np
from research import curiosity_world as CW


@unittest.skipUnless(os.environ.get("NEED_TINY_MODEL"), "needs mlx and a tiny local model (NEED_TINY_MODEL)")
class MLXCuriosityTests(unittest.TestCase):
    """docs/LLM_CURIOSITY_PROTOCOL.md: the knowledge learns and is off while choosing; the hook acts on the choice."""

    def test_knowledge_learns_and_does_not_touch_the_choice(self):
        from research import curiosity_mlx as CM
        mind = CM.Mind(os.environ["NEED_TINY_MODEL"], block=14)
        mind.fresh(0)
        text = CW.dashboard({"mots": [(2.0, 1.0)], "base": [(3.0, 2.9)]})
        p0, mass = mind.choose(text)
        self.assertAlmostEqual(float(p0.sum()), 1.0, places=5)
        self.assertTrue(0 < mass <= 1)
        before = mind.exam("mots")
        losses = mind.study("mots", CW.session_documents(0, "mots", 1))
        self.assertEqual(len(losses), CW.ITERS)
        self.assertTrue(all(np.isfinite(losses)))
        self.assertNotAlmostEqual(mind.exam("mots"), before, places=6)
        p1, _ = mind.choose(text)
        self.assertTrue(np.allclose(p0, p1, atol=1e-6))  # the knowledge is off while choosing
        d = mind.model.model.layers[0].self_attn.q_proj.weight.shape[1]
        pushed, _ = mind.choose(text, add=np.full((3, d), 3.0, np.float32))
        self.assertGreater(np.abs(pushed - p0).max(), 1e-4)
        u = np.zeros((3, d, 1), np.float32)
        u[:, 0, 0] = 1.0
        lesioned, _ = mind.choose(text, project=(u, np.zeros((3, 1), np.float32)))
        self.assertTrue(np.all(np.isfinite(lesioned)))
        mind.fresh(0)
        self.assertAlmostEqual(mind.exam("mots"), before, places=4)  # a fresh life starts again


if __name__ == "__main__":
    unittest.main()
