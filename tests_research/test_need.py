import json
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
from research import need_world as W
from research import need_lora, need_verdicts as V


def lives_of(agent, stream, count, round_=0, choice_stream=None):
    return [W.play(agent, W.world_rng(stream, i, round_), W.choice_rng(choice_stream or stream, i, round_))
            for i in range(count)]


class WorldTests(unittest.TestCase):
    def test_knowing_the_needs_keeps_alive_and_chance_does_not(self):
        oracle = np.mean([l["survived"] for l in lives_of(W.Oracle(), W.STREAMS["test"], 300)])
        coin = np.mean([l["survived"] for l in lives_of(W.Coin(0.5), W.STREAMS["test"], 300)])
        self.assertGreater(oracle, 0.8)
        self.assertLess(coin, 0.12)

    def test_worlds_do_not_depend_on_the_agent(self):
        a = lives_of(W.Oracle(), 0, 20, 3, W.ARMS["need"])
        b = lives_of(W.Coin(0.9), 0, 20, 3, W.ARMS["control"])
        for x, y in zip(a, b):
            n = min(len(x["turns"]), len(y["turns"]))
            self.assertEqual([t["event"] for t in x["turns"][:n]], [t["event"] for t in y["turns"][:n]])

    def test_levels_stay_hidden_and_satisfaction_follows_the_drive(self):
        life = lives_of(W.Coin(0.5), W.STREAMS["test"], 1)[0]
        text = W.life_text(life["turns"])
        self.assertNotIn("énergie :", text)
        for t in W.decisions(life):
            e2, n2 = W.after(t["E"], t["N"], t["action"])
            self.assertEqual(t["r"], W.drive(t["E"], t["N"]) - W.drive(e2, n2))

    def test_kept_decisions_serve_the_lower_need_and_the_control_keeps_as_many(self):
        lives = lives_of(W.Coin(0.5), 0, 256, 1, W.ARMS["need"])
        need, control = W.retain(lives, "need", 1), W.retain(lives, "control", 1)
        self.assertEqual(sum(map(sum, need)), sum(map(sum, control)))
        ds = [t for l in lives for t in W.decisions(l)]
        kept = [f for fl in need for f in fl]
        rewards = np.array([t["r"] for t in ds], float)
        self.assertTrue(all(r > rewards.mean() for r, k in zip(rewards, kept) if k))
        serves = [(t["action"] == 0) == (t["E"] < t["N"]) for t, k in zip(ds, kept) if k and t["E"] != t["N"]]
        self.assertGreater(np.mean(serves), 0.85)

    def test_documents_carry_one_weight_per_target(self):
        lives = lives_of(W.Coin(0.5), 0, 64, 1, W.ARMS["need"])
        flags = W.retain(lives, "need", 1)
        docs = W.training_documents(lives, flags) + W.report_documents(lives, np.random.default_rng(0))
        for d in docs:
            self.assertEqual(len(need_lora.TARGET.findall(d["text"])), len(d["weights"]))
            self.assertTrue(any(d["weights"]))
        reports = [d for d in docs if "need" in d]
        self.assertTrue(reports)
        for d in reports:
            self.assertTrue(d["text"].endswith(str(d["answer"])))
            self.assertEqual(d["weights"][-1], 1)
            self.assertEqual(sum(d["weights"]), 1)

    def test_gzipped_lines_are_deterministic(self):
        rows = [{"a": 1, "é": "été"}, {"b": [1, 2]}]
        with tempfile.TemporaryDirectory() as tmp:
            p, q = Path(tmp, "x.jsonl.gz"), Path(tmp, "y.jsonl.gz")
            W.write_jsonl(p, rows)
            W.write_jsonl(q, rows)
            self.assertEqual(p.read_bytes(), q.read_bytes())
            self.assertEqual(W.read_jsonl(p), rows)


class LoraTests(unittest.TestCase):
    def test_only_kept_choices_and_answers_weigh(self):
        text = W.HEADER + "Tour 1 : calme. Choix : R\nTour 2 : orage. Choix : M\nTour 3 : calme. " + W.QUESTIONS["E"] + "1"
        encode = lambda s: [ord(c) for c in s]
        tokens, weights = need_lora.target_tokens(text, [0, 1, 1], encode)
        self.assertEqual(tokens, encode(text))
        weighted = "".join(chr(t) for t, w in zip(tokens, weights) if w)
        self.assertEqual(weighted, " M1")
        with self.assertRaises(ValueError):
            need_lora.target_tokens(text, [1], encode)

    def test_batches_shift_the_weights_onto_the_predicted_token(self):
        tokens, weights = need_lora.pad_batch([([5, 6, 7], [0.0, 0.0, 1.0])], 64)
        self.assertEqual(weights[0, 1], 1.0)
        self.assertEqual(weights[0].sum(), 1.0)


def synthetic_direction_lives(count=60, seed=0):
    lives = lives_of(W.Oracle(), W.STREAMS["direction"], count)
    rows = V.states(lives)
    rng = np.random.default_rng(seed)
    b_e, b_n = rng.standard_normal(24), rng.standard_normal(24)
    event = rng.standard_normal((len(W.EVENTS), 24))
    acts = np.array([r["E"] * b_e + r["N"] * b_n + event[r["event"]] for r in rows]) + 0.1 * rng.standard_normal((len(rows), 24))
    return lives, acts, b_e, b_n


class DirectionTests(unittest.TestCase):
    def test_the_planted_block_and_directions_are_found(self):
        lives, acts, b_e, b_n = synthetic_direction_lives()
        noise = np.random.default_rng(1).standard_normal(acts.shape)
        fit = V.fit_directions(lives, {6: noise, 14: acts})
        self.assertEqual(fit["layer"], 14)
        self.assertTrue(np.allclose(fit["b_E"], b_e, atol=0.05))
        self.assertTrue(np.allclose(fit["b_N"], b_n, atol=0.05))
        self.assertGreater(fit["r2_by_layer"]["14"]["E"], 0.9)
        u = np.array(fit["U"])
        self.assertTrue(np.allclose(u.T @ u, np.eye(2), atol=1e-8))


def injected_life(index, effect):
    """A test life whose decisions where both needs are high carry injection records with the given effects."""
    life = lives_of(W.Oracle(), W.STREAMS["test"], index + 1)[index]
    for t in W.decisions(life):
        t["extra"] = {"report_E": [float(t["E"] <= W.LOW), 0.9], "report_N": [float(t["N"] <= W.LOW), 0.9]}
        if t["E"] >= W.HIGH and t["N"] >= W.HIGH:
            base = {"p_R": 0.5, "mass": 0.9, "report_E": 0.1, "report_N": 0.1}
            inj = {"none": base, "E": dict(base, p_R=0.5 + effect, report_E=0.1 + effect),
                   "N": dict(base, p_R=0.5 - effect, report_N=0.1 + effect)}
            for i in range(3):
                inj[f"randE{i}"] = dict(base, p_R=0.51)
                inj[f"randN{i}"] = dict(base, p_R=0.49)
            t["extra"]["injection"] = inj
    return life


class VerdictTests(unittest.TestCase):
    def test_a_need_that_counts_passes_and_a_weak_one_fails(self):
        final = [injected_life(i, 0.3) for i in range(60)]
        dead = [dict(l, survived=False) for l in final]
        result = V.verdicts(final, dead, dead, dead, final)
        self.assertEqual(result["verdicts"], {"valid": True, "S": True, "IA": True, "IR": True, "LS": True, "global": True})
        weak = [injected_life(i, 0.05) for i in range(60)]
        result = V.verdicts(weak, dead, dead, dead, weak)
        self.assertFalse(result["verdicts"]["IA"])
        self.assertFalse(result["verdicts"]["IR"])
        self.assertTrue(result["verdicts"]["S"])

    def test_a_lesion_that_any_plane_matches_does_not_count(self):
        final = [injected_life(i, 0.3) for i in range(60)]
        dead = [dict(l, survived=False) for l in final]
        self.assertFalse(V.verdicts(final, dead, dead, dead, dead)["verdicts"]["LS"])

    def test_report_accuracy_is_balanced(self):
        final = [injected_life(i, 0.3) for i in range(30)]
        self.assertEqual(V.report_accuracy(final)["E"], 1.0)


class CausalTests(unittest.TestCase):
    def test_pairs_follow_the_exact_replay_of_the_life(self):
        from research import need_causal as C
        lives = lives_of(W.Coin(0.5), C.STREAMS["direction"], 40)
        pairs = C.choose_pairs(lives)
        self.assertEqual(pairs, C.choose_pairs(lives))
        self.assertTrue(any(p["dN"] for p in pairs) and any(p["dE"] for p in pairs))
        self.assertTrue(len({(p["dE"], p["dN"]) for p in pairs}) > 2)
        for p in pairs[:200]:
            life = lives[p["life"]]
            self.assertEqual([x["event"] for x in W.decisions(life) if x["t"] == p["j"]], [p["swap"][0]])
            # replaying the swapped world with the same choices gives the recorded changes
            turns = C.swapped(life, p)
            e = n = W.MAX
            for x in turns:
                ev = W.EVENTS[x["event"]]
                e, n = e + ev[2], n + ev[3]
                self.assertTrue(e > 0 and n > 0)
                if x["t"] == p["t"]:
                    self.assertEqual((e - x["E"], n - x["N"]), (p["dE"], p["dN"]))
                    break
                e, n = W.after(e, n, x["action"])
            self.assertIn(W.EVENTS[p["swap"][1]][0], C.decision_text(turns, p["t"]))

    def test_fit_recovers_per_token_directions_and_planes_are_orthonormal(self):
        from research import need_causal as C
        rng = np.random.default_rng(0)
        pairs = [{"dE": de, "dN": dn} for de, dn in [(-2, 0), (-1, -1), (-1, -2)] * 50]
        d_e, d_n = rng.standard_normal((4, 16)), rng.standard_normal((4, 16))
        diffs = np.array([p["dE"] * d_e + p["dN"] * d_n for p in pairs]) + 0.01 * rng.standard_normal((150, 4, 16))
        e, n = C.fit(pairs, diffs)
        self.assertTrue(np.allclose(e, d_e, atol=0.01) and np.allclose(n, d_n, atol=0.01))
        u = C.planes(e, n)
        for k in range(4):
            self.assertTrue(np.allclose(u[k].T @ u[k], np.eye(2), atol=1e-8))
        r = C.random_like(rng, 4 * e, 3)
        self.assertTrue(np.allclose(np.linalg.norm(r[0], axis=1), np.linalg.norm(4 * e, axis=1)))

    def test_causal_verdicts(self):
        from research import need_causal as C
        intact = [injected_life(i, 0.3) for i in range(60)]
        dead = [dict(l, survived=False) for l in intact]
        v = C.verdicts(intact, dead, intact)["verdicts"]
        self.assertTrue(v["IA2"] and v["LS2"] and v["global"])
        self.assertFalse(C.verdicts(intact, intact, intact)["verdicts"]["LS2"])


@unittest.skipUnless(os.environ.get("NEED_TINY_MODEL"), "needs mlx and a tiny local model (NEED_TINY_MODEL)")
class MLXAgentTests(unittest.TestCase):
    def test_evaluations_leave_the_life_unchanged_and_tokens_are_those_of_the_document(self):
        import mlx.core as mx
        from research.need_mlx import MLXAgent
        agent = MLXAgent(os.environ["NEED_TINY_MODEL"], layers=(6,))
        agent.start(W.HEADER)
        agent.decide(W.choice_line(1, 0))
        agent.commit(0)
        clean = MLXAgent(os.environ["NEED_TINY_MODEL"])
        clean.start(W.HEADER)
        clean.decide(W.choice_line(1, 0))
        clean.commit(0)
        agent.hooks[6].add = mx.array(np.zeros(agent.model.args.hidden_size, np.float32))
        agent.p_yes(W.question_line(2, 1, "E"))
        agent.hooks[6].add = None
        self.assertAlmostEqual(agent.decide(W.choice_line(2, 1))[0], clean.decide(W.choice_line(2, 1))[0], places=5)
        self.assertEqual(agent.ids, agent.encode(agent.text))
        self.assertEqual(agent.resets, 0)


if __name__ == "__main__":
    unittest.main()
