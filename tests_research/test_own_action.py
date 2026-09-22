import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from research.origin_env import LIFE, DELTAS, motor_delta
from research.text_atelier import (TextModel, TextLife, childhood_batch, target_mask, action_index, run_text_lives,
                                   replay_tokens, displacement_table, displacement_summary, inquiry_summary,
                                   BOS, ACT, MOVE_TOK, CUE_TOK, VOCAB, SEQ, FIXED_BODY)
from research.audit_own_action import criteria, integrity, mutable_criteria
from research.own_action_experiment import run_one, CHANGE_STEP


class TokenTests(unittest.TestCase):
    def test_layout_and_fixed_body(self):
        X = childhood_batch("F", np.random.default_rng(3), 4)
        self.assertEqual(X.shape, (4, SEQ))
        self.assertTrue((X[:, 0] == BOS).all() and X.max() < VOCAB)
        for n in range(4):
            for t in range(LIFE):
                a = X[n, action_index(t)] - ACT
                self.assertTrue(0 <= a < 8)
                if a < 4:
                    self.assertEqual(X[n, action_index(t) + 1] - MOVE_TOK, DELTAS.index(motor_delta(FIXED_BODY, a)))
                else:
                    self.assertTrue(CUE_TOK <= X[n, action_index(t) + 1] < VOCAB)

    def test_variable_regime_uses_several_bodies(self):
        bodies = set()
        for n in range(30):
            life = TextLife("T", 100 + n)
            bodies.add(life.env.d)
        self.assertEqual(bodies, {0, 1, 2, 3})

    def test_mutable_regime_changes_the_body_in_some_childhood_lives(self):
        rng = np.random.default_rng(11)
        X = childhood_batch("VM", rng, 40)
        changed = 0
        for n in range(40):
            implied = set()
            for t in range(LIFE):
                a = X[n, action_index(t)] - ACT
                if a < 4:
                    delta = DELTAS[X[n, action_index(t) + 1] - MOVE_TOK]
                    implied.add((DELTAS.index(delta) - a) % 4)
            changed += len(implied) > 1
        self.assertGreater(changed, 8)
        self.assertLess(changed, 32)
        Xv = childhood_batch("V", np.random.default_rng(11), 40)
        for n in range(40):
            implied = set()
            for t in range(LIFE):
                a = Xv[n, action_index(t)] - ACT
                if a < 4:
                    implied.add((Xv[n, action_index(t) + 1] - MOVE_TOK - a) % 4)
            self.assertEqual(len(implied), 1)

    def test_efference_mask_drops_only_commands(self):
        mask = target_mask("VE")
        self.assertEqual(int(mask.sum()), SEQ - 1 - LIFE)
        for t in range(LIFE):
            self.assertFalse(mask[action_index(t) - 1])
            self.assertTrue(mask[action_index(t)])
        self.assertTrue(target_mask("V").all())

    def test_logged_lives_replay_exactly(self):
        model = TextModel(seed=1)
        lives = run_text_lives(model, "random", 7, 3, forced_change_step=CHANGE_STEP)
        self.assertTrue(integrity(lives))
        tokens, d0, d1 = replay_tokens(lives[0])
        self.assertEqual(tokens, lives[0]["tokens"])
        self.assertNotEqual(d0, d1)
        lives[0]["tokens"][10] = (lives[0]["tokens"][10] + 1) % VOCAB
        self.assertFalse(integrity(lives))


class GradientTests(unittest.TestCase):
    def test_manual_backward_matches_finite_differences(self):
        model = TextModel(d=8, layers=2, heads=2, ff=12, seed=3, dtype=np.float64, max_len=40)
        rng = np.random.default_rng(0)
        X = rng.integers(0, VOCAB, size=(2, 12))
        mask = np.ones(11, dtype=bool)
        mask[[2, 6]] = False
        _, grads = model.loss_and_grad(X, mask)
        for name, p in model.p.items():
            ix = tuple(int(rng.integers(0, s)) for s in p.shape)
            old = p[ix]
            p[ix] = old + 1e-5
            up, _ = model.loss_and_grad(X, mask)
            p[ix] = old - 1e-5
            down, _ = model.loss_and_grad(X, mask)
            p[ix] = old
            numeric = (up - down) / 2e-5
            self.assertAlmostEqual(numeric, grads[name][ix], delta=1e-6 + 1e-4 * abs(numeric), msg=name)

    def test_causal_mask_blocks_the_future(self):
        model = TextModel(seed=5)
        X = np.random.default_rng(1).integers(0, VOCAB, size=(1, 20))
        base = model.forward(X)[0, :10]
        X2 = X.copy()
        X2[0, 10:] = (X2[0, 10:] + 1) % VOCAB
        self.assertTrue(np.allclose(base, model.forward(X2)[0, :10], atol=1e-5))


class MeasureTests(unittest.TestCase):
    def test_displacement_and_inquiry_summaries(self):
        model = TextModel(seed=2)
        lives = run_text_lives(model, "random", 3, 4)
        rows = displacement_table(model, lives)
        self.assertEqual(len(rows), sum(1 for life in lives for a in life["actions"] if a < 4))
        for r in rows:
            self.assertAlmostEqual(sum(r["probs"]), 1.0, places=6)
        summary = displacement_summary(rows)
        self.assertIn("accuracy_after_first", summary)
        inquiry = inquiry_summary(lives, CHANGE_STEP)
        self.assertEqual(inquiry["lives"], 4)
        self.assertAlmostEqual(inquiry["inspections_per_life"], sum(1 for life in lives for a in life["actions"] if a >= 4) / 4)

    def test_criteria_on_synthetic_models(self):
        def row(step, correct, before_any, d, confidence):
            return {"life": 0, "step": step, "truth": 0, "probs": [1, 0, 0, 0], "correct": correct,
                    "before_any": before_any, "d": d, "confidence": confidence}

        def fake(regime, seed, acc_other, share, ret, ret_c, insp, late, conf_wrong, conf_before):
            # 25 moves with the fixed body (always right), 75 with other bodies at accuracy acc_other
            rows = [row(3, True, False, FIXED_BODY, 0.9) for _ in range(25)]
            rows += [row(3, i < int(acc_other * 75), False, 1, conf_wrong) for i in range(75)]
            rows += [row(0, False, True, 1, conf_before) for _ in range(10)]
            rowsM = [row(17, i < int(late * 100), False, 1, 0.9) for i in range(100)]

            def life(actions, change):
                rec = {"actions": list(actions), "rewards": [0] * LIFE, "d": 1, "e": 0, "seed": 1, "index": 0}
                if change:
                    rec["change_step"] = CHANGE_STEP
                return rec
            n_insp = int(insp * 100)
            n_mark = int(share * n_insp)
            livesM, livesC = [], []
            for i in range(100):
                actions = ([4] * n_mark + [5] * (n_insp - n_mark) + [0] * LIFE)[:LIFE]
                if i < int(ret * 100):
                    actions[20] = 4
                livesM.append(life(actions, True))
                actions = [0] * LIFE
                if i < int(ret_c * 100):
                    actions[20] = 4
                livesC.append(life(actions, False))
            summaries = {"R": {"displacement": displacement_summary(rows)}}
            return {"regime": regime, "seed": seed, "rows": {"R": rows, "M": rowsM},
                    "lives": {"M": livesM, "C": livesC, "R": []}, "summaries": summaries}
        good = []
        for seed in (17, 29, 43):
            good.append(fake("F", seed, acc_other=0.0, share=0.0, ret=0.0, ret_c=0.0, insp=0.0, late=0.25, conf_wrong=0.9, conf_before=0.9))
            good.append(fake("V", seed, acc_other=1.0, share=0.8, ret=0.7, ret_c=0.02, insp=0.05, late=0.9, conf_wrong=0.5, conf_before=0.26))
            good.append(fake("VE", seed, acc_other=1.0, share=0.8, ret=0.65, ret_c=0.03, insp=0.05, late=0.88, conf_wrong=0.5, conf_before=0.26))
        verdict = criteria(good)
        self.assertTrue(verdict["complete"] and verdict["validity"], verdict)
        for key in ("P1", "P2", "P3", "P4", "P5", "global"):
            self.assertTrue(verdict[key], key)
        bad = [dict(m) for m in good]
        bad[1] = fake("V", 17, acc_other=1.0, share=0.8, ret=0.0, ret_c=0.02, insp=0.05, late=0.9, conf_wrong=0.5, conf_before=0.26)
        self.assertFalse(criteria(bad)["P4"])
        self.assertFalse(criteria(good[:5])["complete"])
        vm = [fake("VM", seed, acc_other=1.0, share=0.8, ret=0.8, ret_c=0.05, insp=0.05, late=0.9, conf_wrong=0.5, conf_before=0.26) for seed in (17, 29, 43)]
        for m in vm:
            m["rows"]["M"] = [dict(r, step=10, confidence=0.95) for r in m["rows"]["M"][:20]] + [dict(r, step=14, confidence=0.4) for r in m["rows"]["M"][:20]] + [r for r in m["rows"]["M"] if r["step"] >= 16]
        vs = [m for m in good if m["regime"] == "V"]
        for m in vs:
            m["rows"]["M"] = [dict(r, step=10, confidence=0.95) for r in m["rows"]["M"][:20]] + [dict(r, step=14, confidence=0.94) for r in m["rows"]["M"][:20]] + [dict(r, correct=False) for r in m["rows"]["M"] if r["step"] >= 16]
            m["summaries"]["M"] = {"displacement": displacement_summary(m["rows"]["M"]), "inquiry": inquiry_summary(m["lives"]["M"], CHANGE_STEP)}
        for m in vm:
            m["summaries"]["M"] = {"displacement": displacement_summary(m["rows"]["M"]), "inquiry": inquiry_summary(m["lives"]["M"], CHANGE_STEP)}
        verdict = mutable_criteria(vs + vm)
        self.assertTrue(verdict["complete"], verdict)
        self.assertTrue(verdict["Q1"] and verdict["Q2"] and verdict["Q4"] and verdict["Q5"] and verdict["global"], verdict)


class SmokeTests(unittest.TestCase):
    def test_run_one_writes_reproducible_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_one(tmp, "VE", 1, updates=2, lives=1, log=lambda *_: None)
            self.assertEqual(set(report["sets"]), {"R", "M", "C"})
            self.assertTrue((Path(tmp) / "model-VE-1.npz").exists())
            reloaded = TextModel.load(Path(tmp) / "model-VE-1.npz")
            lives = [json.loads(l) for l in (Path(tmp) / "lives-VE-1-M.jsonl").read_text().splitlines()]
            self.assertEqual(lives[0]["change_step"], CHANGE_STEP)
            again = displacement_summary(displacement_table(reloaded, lives))
            self.assertEqual(again, report["sets"]["M"]["displacement"])


if __name__ == "__main__":
    unittest.main()
