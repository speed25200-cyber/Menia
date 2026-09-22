import unittest
import numpy as np
from research.origin_env import Atelier, random_lives, N_MOVE, LIFE
from research.origin_neural import WorldModel, run_lives, train_world_model
from research.mutable_experiment import mark_reads_after, hits_ratio, entropy_jump, update_probe, CHANGE_STEP
from research.audit_mutable import criteria


class MutationTests(unittest.TestCase):
    def test_forced_change_replaces_the_body_and_the_mark_follows(self):
        agree_before = agree_after = changed = 0
        for n in range(200):
            env = Atelier("T", 5000 + n, forced_change_step=CHANGE_STEP)
            env.reset(); d0 = env.d
            for t in range(CHANGE_STEP):
                obs, _, _ = env.step(N_MOVE)  # read the mark
                agree_before += obs["cue_value"] == d0
            self.assertEqual(env.d, d0)
            obs, _, _ = env.step(N_MOVE)
            changed += env.d != d0
            agree_after += obs["cue_value"] == env.d
            self.assertEqual(env.d_initial, d0)
        self.assertEqual(changed, 200)
        self.assertGreater(agree_before / (200 * CHANGE_STEP), 0.75)
        self.assertGreater(agree_after / 200, 0.75)

    def test_training_mutation_draws_only_when_enabled_and_keeps_stable_streams(self):
        a = Atelier("T", 9); a.reset()
        b = Atelier("T", 9, mutate="self", mutation_probability=0.0); b.reset()
        self.assertEqual((a.d, a.e, a.p, a.g, a.s), (b.d, b.e, b.p, b.g, b.s))
        self.assertIsNone(b.change_step)
        steps = []
        for n in range(300):
            env = Atelier("T", 100 + n, mutate="world", mutation_probability=1.0); env.reset()
            steps.append(env.change_step)
            e0 = env.e
            for t in range(LIFE):
                env.step(0)
            self.assertEqual(env.d, env.d_initial)
        self.assertEqual(min(steps), 6); self.assertEqual(max(steps), 17)
        with self.assertRaises(ValueError):
            Atelier("T", 1, mutate="both")
        X, Y, _ = random_lives("T", 3, 4, mutate="self", mutation_probability=0.5)
        self.assertEqual(X.shape[1], 4)

    def test_metrics_on_synthetic_lives(self):
        lives = [{"actions": [N_MOVE] * 12 + [N_MOVE, 0, N_MOVE] + [0] * 9, "rewards": [0] * 12 + [1, 0, 1] + [0] * 9,
                  "motion_entropy": [0.1] * 12 + [1.0] * 12, "d": 1, "d_final": 2}]
        self.assertEqual(mark_reads_after(lives), 2.0)
        ratio = hits_ratio(lives)
        self.assertEqual(ratio["hits_after_per_life"], 2.0); self.assertIsNone(ratio["ratio"])
        self.assertAlmostEqual(entropy_jump(lives)["jump"], 0.9)
        rng = np.random.default_rng(0)
        control_states = rng.normal(size=(40, LIFE, 3)); control_lives = []
        for i in range(40):
            d = i % 4; control_states[i, :, d % 3] += 4; control_lives.append({"d": d})
        change_states = control_states.copy(); change_lives = [{"d": (i % 4), "d_final": ((i % 4) + 1) % 4} for i in range(40)]
        probe = update_probe(control_states, control_lives, change_states, change_lives)
        self.assertGreater(probe["old_body_accuracy"], probe["new_body_accuracy"])

    def test_criteria_logic(self):
        def run(regime, m1, m2, m3, m3c, m4):
            return {"regime": regime, "M1_update": {"new_body_accuracy": m1}, "M2_detection": {"jump": m2},
                    "M3_mark_reads_after_change": m3, "M3_control_mark_reads_after_step12": m3c, "M4_recovery": {"ratio": m4}}
        runs = [run("MS", 0.95, 0.5, 1.0, 0.0, 0.9) for _ in range(3)] + [run("S", 0.3, 0.0, 0.0, 0.0, 0.4) for _ in range(3)] + [run("MW", 0.3, 0.0, 0.0, 0.0, 0.4) for _ in range(3)]
        self.assertTrue(criteria(runs)["global"])
        runs[3]["M1_update"]["new_body_accuracy"] = 0.95
        self.assertFalse(criteria(runs)["global"])

    def test_run_lives_records_change_and_entropy_and_small_training_with_mutation_runs(self):
        model = WorldModel(hidden=6, seed=2)
        lives = run_lives(model, "T", "none", 21, 2, forced_change_step=CHANGE_STEP, record_entropy=True)
        self.assertEqual(lives[0]["change_step"], CHANGE_STEP)
        self.assertEqual(len(lives[0]["motion_entropy"]), LIFE)
        self.assertNotEqual(lives[0]["d"], lives[0]["d_final"])
        trained, log = train_world_model("T", 1, updates=3, batch=4, hidden=6, mutate="self", mutation_probability=0.5)
        self.assertTrue(np.isfinite(log[-1]["loss"]))


if __name__ == "__main__":
    unittest.main()
