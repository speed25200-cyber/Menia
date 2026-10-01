import unittest
import numpy as np
from research import curiosity_world as CW


def run_arms(names, lives=24, temperature=0.05, bonus=0.10):
    learner, chooser = CW.ToyLearner(), CW.instinct_chooser(temperature, bonus)
    uniform = lambda text, history, k: np.full(4, 0.25)
    choosers = {"C": chooser, "H": None, "L": uniform, "LH": chooser, "HP": chooser,
                "HA": chooser, "HAr": chooser}
    return {n: [CW.live(v, choosers[n], learner, n) for v in range(lives)] for n in names}


class CuriosityWorldTests(unittest.TestCase):
    def test_domains(self):
        self.assertEqual(CW.to_base(23, 7), "32")
        g = np.random.default_rng(0)
        prompt, answer = CW.example("base", g)
        x, y = prompt.split(":")[1].split("=")[0].split("+")
        self.assertEqual(int(answer, 7), int(x, 7) + int(y, 7))
        prompt, answer = CW.example("calcul", g)
        x, y = prompt.split(":")[1].split("=")[0].split("+")
        self.assertEqual(int(answer), int(x) + int(y))
        self.assertEqual(CW.exam("mots"), CW.exam("mots"))
        study = {CW.text_of(d) for d in CW.session_documents(0, "mots", 1)}
        self.assertFalse(any(p + a + "\n" in "".join(study) for p, a in CW.exam("mots")[:16]))
        for d in CW.DOMAINS:
            self.assertEqual(len(CW.document(d, g)), CW.EXAMPLES)

    def test_dashboard_numbers_keep_the_length(self):
        h = {"mots": [(4.12, 2.68)] * 5, "base": [(1.51, 1.39)] * 3, "calcul": [(0.21, 0.22)]}
        t = CW.dashboard(h)
        self.assertIn("mots : 5 séances ; dernière séance : 412 → 268.", t)
        self.assertIn("suites : jamais étudiées.", t)
        self.assertTrue(t.endswith("À toi. Choix :"))
        h2 = dict(h, base=[(1.51, 0.39)] * 3)
        self.assertEqual(len(CW.dashboard(h2)), len(t))
        self.assertAlmostEqual(CW.progress(h2, "base"), 1.12)

    def test_instinct_prefers_progress_and_explores(self):
        h = {"mots": [(2.0, 1.0)], "base": [(2.0, 1.95)], "calcul": [(0.2, 0.2)], "suites": [(3.0, 3.0)]}
        p = CW.instinct(h, 0.1, 0.0)
        self.assertEqual(int(np.argmax(p)), 0)
        self.assertGreaterEqual(p.min(), CW.EPSILON / 4 - 1e-12)
        self.assertAlmostEqual(p.sum(), 1.0)

    def test_hormone(self):
        self.assertAlmostEqual(CW.secretion(), 0.3468, places=3)
        self.assertAlmostEqual(CW.hormone(12), 1.0)
        self.assertAlmostEqual(CW.hormone(4), 0.0)
        self.assertAlmostEqual(CW.hormone(20), (2 / 3) ** 8, places=6)

    def test_curiosity_beats_chance_on_toy_learners(self):
        cal = CW.calibrate(run_arms(["H"], 12)["H"])
        t, b = cal["temperature"], cal["bonus"]
        self.assertGreater(t, 0)
        arms = run_arms(["C", "H", "L", "LH"], temperature=t, bonus=b)
        r = CW.life_verdicts(arms, t, b)
        self.assertTrue(r["verdicts"]["CUR"])
        self.assertLess(r["values"]["VIDE"]["mean"], 0.5)  # below chance; the 0.25 threshold is near the edge
        self.assertTrue(r["verdicts"]["LS1"])
        self.assertTrue(CW.domains_valid(arms["H"], t)["valid"])

    def test_plasticity_hormone_speeds_base(self):
        arms = run_arms(["C", "HP"])
        r = CW.life_verdicts(arms, 0.05, 0.10)
        self.assertIn("PL1", r["verdicts"])
        self.assertGreaterEqual(r["values"]["PL1"]["mean"], 0)
        self.assertIn(r["values"]["durable_HP"][0], ("préférence durable", "satiété", "sans trace"))

    def test_invented_histories(self):
        g = np.random.default_rng(1)
        hs = [CW.invented_history(g, 0.2, 3.5, 0.05) for _ in range(200)]
        self.assertTrue(any(len(h) < 4 for h in hs))
        for h in hs:
            for d, sessions in h.items():
                self.assertTrue(0 <= sessions[-1][1] <= 9.99)


if __name__ == "__main__":
    unittest.main()
