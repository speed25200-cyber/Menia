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

    def test_the_workspace_question_sees_only_the_header_the_pending_choice_and_itself(self):
        mask = need_lora.workspace_mask(12, (2, 8))  # header 0-1, life 2-4, "Choix :" 5-7, question 8-11
        self.assertTrue((mask[:8] == np.tril(np.ones((8, 12), bool))).all())
        self.assertEqual(np.flatnonzero(mask[10]).tolist(), [0, 1, 5, 6, 7, 8, 9, 10])
        self.assertTrue((need_lora.workspace_mask(6) == np.tril(np.ones((6, 6), bool))).all())
        items = [([1] * 12, [0.0] * 11 + [1.0], (2, 8)), ([1] * 5, [0.0] * 4 + [1.0])]
        tokens, weights, masks = need_lora.pad_batch(items, 64)
        self.assertEqual(masks.shape, (2, 1, tokens.shape[1] - 1, tokens.shape[1] - 1))
        self.assertFalse(masks[0, 0, 9, 3])
        self.assertTrue(masks[1, 0, 9, 3])

    def test_the_reader_acts_on_the_question_only_and_learns_from_questions_only(self):
        items = [([1] * 12, [0.0] * 11 + [1.0], (2, 8), True), ([1] * 10, [0.0] * 9 + [1.0], (2, 7), True)]
        tokens, weights, masks, positions = need_lora.pad_batch(items, 64)
        self.assertEqual(positions.shape, (2, tokens.shape[1] - 1, 1))
        self.assertEqual(np.flatnonzero(positions[0, :, 0]).tolist()[:4], [8, 9, 10, 11])
        self.assertEqual(positions[1, 6, 0], 0.0)
        self.assertEqual(positions[1, 7, 0], 1.0)

    def test_the_workspace_of_a_document_ends_on_the_pending_choice(self):
        lives = lives_of(W.Oracle(), W.STREAMS["report"], 4)
        docs = W.report_documents(lives, np.random.default_rng(0), after_choice=True, workspace=True)
        self.assertTrue(docs)
        for d in docs:
            self.assertTrue(d["text"][:d["workspace"]].endswith(". Choix :"))
            self.assertTrue(d["text"][d["workspace"]:].startswith(" ? Question :"))
        with self.assertRaises(ValueError):
            W.report_documents(lives, np.random.default_rng(0), workspace=True)


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


class SpeakTests(unittest.TestCase):
    def test_the_action_state_is_carried_to_the_question_at_the_question_norm(self):
        from research import need_speak as S
        rng = np.random.default_rng(1)
        action, question = rng.standard_normal((3, 8)), 2 * rng.standard_normal((3, 8))
        v = S.same_vector(action, question)
        self.assertEqual(v.shape, (3, 8))
        self.assertTrue(np.allclose(v[0], v[2]))
        self.assertAlmostEqual(np.linalg.norm(v[0]), np.linalg.norm(question, axis=1).mean())
        cos = v[0] @ action.mean(0) / (np.linalg.norm(v[0]) * np.linalg.norm(action.mean(0)))
        self.assertAlmostEqual(cos, 1.0)

    def test_speak_verdicts(self):
        from research import need_speak as S
        lives = []
        for i in range(40):
            life = lives_of(W.Oracle(), S.STREAMS["test"], i + 1)[i]
            for t in W.decisions(life):
                t["extra"] = {"report_E": [float(t["E"] <= W.LOW), 0.9], "report_N": [float(t["N"] <= W.LOW), 0.9]}
                if t["E"] >= W.HIGH and t["N"] >= W.HIGH:
                    q = {"none": 0.1, "need": 0.4, "same": 0.3, **{f"randq{k}": 0.11 for k in range(3)},
                         **{f"rands{k}": 0.09 for k in range(3)}}
                    t["extra"]["injection"] = {"E": q, "N": q}
            lives.append(life)
        v = S.verdicts(lives, 0.005)["verdicts"]
        self.assertEqual(v, {"R3": True, "A3": True, "IR3": True, "SAME3": True, "valid": True, "global": True})
        self.assertFalse(S.verdicts(lives, 0.05)["verdicts"]["valid"])


def one_state_lives(stream):
    """Oracle lives carrying the reads of the fourth and fifth tests: an injection that moves both act and word."""
    lives = []
    for i in range(40):
        life = lives_of(W.Oracle(), stream, i + 1)[i]
        for t in W.decisions(life):
            t["extra"] = {"report_E": [float(t["E"] <= W.LOW), 0.9], "report_N": [float(t["N"] <= W.LOW), 0.9]}
            if t["E"] >= W.HIGH and t["N"] >= W.HIGH:
                base = {"p_R": 0.5, "yes_E": 0.1, "yes_N": 0.1}
                inj = {"none": base, "E": dict(base, p_R=0.7, yes_E=0.25), "N": dict(base, p_R=0.4, yes_N=0.2)}
                inj.update({f"randE{k}": dict(base, p_R=0.51) for k in range(3)})
                inj.update({f"randN{k}": dict(base, p_R=0.49) for k in range(3)})
                t["extra"]["injection"] = inj
        lives.append(life)
    return lives


class OneStateTests(unittest.TestCase):
    def test_one_state_verdicts(self):
        from research import need_one as O
        lives = one_state_lives(O.STREAMS["test"])
        v = O.verdicts(lives, 0.005)["verdicts"]
        self.assertEqual(v, {"R4": True, "A4": True, "ONE4": True, "valid": True, "global": True})
        for life in lives:
            for t in W.decisions(life):
                if "injection" in t["extra"]:
                    t["extra"]["injection"]["E"]["yes_E"] = 0.12
        self.assertFalse(O.verdicts(lives, 0.005)["verdicts"]["ONE4"])


class WorkspaceTests(unittest.TestCase):
    def test_workspace_verdicts_need_the_execution_check(self):
        from research import need_workspace as WS
        lives = one_state_lives(WS.STREAMS["test"])
        v = WS.verdicts(lives, 0.005, 1e-6)["verdicts"]
        self.assertEqual(v, {"R5": True, "A5": True, "ONE5": True, "valid": True, "global": True})
        v = WS.verdicts(lives, 0.005, 1e-2)["verdicts"]
        self.assertFalse(v["valid"])
        self.assertFalse(v["global"])

    def test_reader_verdicts_need_the_reader_replica(self):
        from research import need_reader as RD
        lives = one_state_lives(RD.STREAMS["test"])
        v = RD.verdicts(lives, 0.005, 1e-6, 0.01)["verdicts"]
        self.assertEqual(v, {"R6": True, "A6": True, "ONE6": True, "valid": True, "global": True})
        self.assertFalse(RD.verdicts(lives, 0.005, 1e-6, 0.05)["verdicts"]["global"])


class NecessityTests(unittest.TestCase):
    def test_necessity_verdicts(self):
        from research import need_necessity as NC
        intact = lives_of(W.Oracle(), NC.STREAMS["test"], 40)
        coin = lives_of(W.Coin(), NC.STREAMS["test"], 40)
        for life in intact:
            for t in W.decisions(life):
                truth = {q: float(t[q] <= W.LOW) for q in ("E", "N")}
                t["extra"] = {"lesion_reports": {"none": dict(truth, mass_E=0.9, mass_N=0.9),
                                                 "lesion": {"E": 0.9, "N": truth["N"]},
                                                 "lesion_random": dict(truth)}}
        setup = {"replica": 0.003, "reader_replica": 0.012, "execution_gap": 1e-6}
        r = NC.verdicts({"intact": intact, "lesion": coin, "lesion_random": intact}, setup)
        self.assertEqual(r["verdicts"], {"LS7": True, "LR7": True, "valid": True, "global": True})
        self.assertAlmostEqual(r["values"]["say_energy"]["lesion"]["with"], 0.5)
        self.assertAlmostEqual(r["values"]["say_food"]["lesion"]["mean"], 0.0)
        r = NC.verdicts({"intact": intact, "lesion": coin, "lesion_random": coin}, setup)
        self.assertFalse(r["verdicts"]["LS7"])
        self.assertFalse(NC.verdicts({"intact": intact, "lesion": coin, "lesion_random": intact},
                                     dict(setup, execution_gap=1e-2))["verdicts"]["global"])


class ReplicationTests(unittest.TestCase):
    def test_replication_verdicts_join_the_reader_and_the_lesion(self):
        from research import need_replication as RP, need_reader as RD, need_necessity as NC
        reader = RD.verdicts(one_state_lives(RD.STREAMS["test"] + 100), 0.005, 1e-6, 0.01)
        self.assertLess(reader["values"]["report_accuracy"]["E"], 1.01)
        necessity = {"verdicts": {"LS7": True, "LR7": True, "valid": True, "global": True},
                     "values": {k: {} for k in ("survival", "survival_drop", "survival_drop_random", "say_energy",
                                                "say_food")}}
        v = RP.verdicts(reader, necessity)["verdicts"]
        self.assertEqual(v, {"RR": True, "RA": True, "RONE": True, "RLS": True, "RLR": True, "valid": True,
                             "global": True})
        necessity["verdicts"]["LR7"] = False
        self.assertFalse(RP.verdicts(reader, necessity)["verdicts"]["global"])


class BalancedTests(unittest.TestCase):
    def test_the_event_of_the_turn_says_nothing_of_the_answer(self):
        lives = lives_of(W.Oracle(), W.STREAMS["report"], 30)
        docs = W.balanced_report_documents(lives, np.random.default_rng(0))
        self.assertTrue(docs)
        for need in ("E", "N"):
            for k, event in enumerate(W.EVENTS):
                line = lambda d: d["text"][:d["workspace"]].rsplit("\n", 1)[-1]
                cell = [d for d in docs if d["need"] == need and line(d).endswith(f" : {event[0]}. Choix :")]
                self.assertEqual(sum(d["answer"] for d in cell) * 2, len(cell))
        for d in docs:
            self.assertTrue(d["text"][d["workspace"]:].startswith(" ? Question :"))

    def test_balanced_verdicts(self):
        from research import need_balanced as B, need_reader as RD
        lives = one_state_lives(30)
        for life in lives:
            for t in W.decisions(life):
                if "injection" in t["extra"]:
                    t["extra"]["injection"]["N"]["yes_N"] = 0.25
        v = B.verdicts(RD.verdicts(lives, 0.005, 1e-6, 0.01))["verdicts"]
        self.assertEqual(v, {"R8": True, "SAY8": True, "ONE8": True, "valid": True, "global": True})
        for life in lives:
            for t in W.decisions(life):
                if "injection" in t["extra"]:
                    t["extra"]["injection"]["N"]["yes_N"] = 0.12
        self.assertFalse(B.verdicts(RD.verdicts(lives, 0.005, 1e-6, 0.01))["verdicts"]["SAY8"])


class LongReaderTests(unittest.TestCase):
    def test_long_reader_verdicts(self):
        from research import need_long as L, need_reader as RD
        lives = one_state_lives(30)
        good = RD.verdicts(lives, 0.005, 1e-6, 0.01)
        v = L.verdicts({"second": good, "two": good})["verdicts"]
        self.assertEqual(v, {"R11_second": True, "SAY11_second": True, "valid_second": True, "R11_two": True,
                             "SAY11_two": True, "valid_two": True, "global": True})
        self.assertFalse(L.verdicts({"second": good})["verdicts"]["global"])
        for life in lives:
            for t in W.decisions(life):
                if "injection" in t["extra"]:
                    t["extra"]["injection"]["E"]["yes_E"] = 0.15
        weak = RD.verdicts(lives, 0.005, 1e-6, 0.01)
        v = L.verdicts({"second": good, "two": weak})["verdicts"]
        self.assertTrue(v["SAY11_second"] and v["R11_two"])
        self.assertFalse(v["SAY11_two"] or v["global"])
        self.assertFalse(L.verdicts({"second": good, "two": RD.verdicts(lives, 0.005, 1e-6, 0.03)})["verdicts"]
                         ["valid_two"])

    def test_the_continued_batches_are_those_of_one_run(self):
        data = list(range(40))
        np.random.seed(5)
        one = list(need_lora.skipping(lambda **k: iter(np.random.permutation(data)), 0)(dataset=data, batch_size=1,
                                                                                         loop=True))
        np.random.seed(5)
        two = list(need_lora.skipping(lambda **k: iter(np.random.permutation(data)), 15)(dataset=data, batch_size=1,
                                                                                          loop=True))
        self.assertEqual(one[15:], two)
        np.random.seed(5)
        self.assertEqual(len(list(need_lora.skipping(lambda **k: iter(np.random.permutation(data)), 15)())), 40)
        with self.assertRaises(ValueError):
            need_lora.skipping(lambda **k: iter(data), 10)(dataset=data, batch_size=4, loop=True)


def persistence_rows(count, lesion=0.08, lesion_random=0.01, push=0.06, random=0.01):
    """Reads of contexts as research/need_persistence.py writes them, with the given effects at t + 1."""
    from research import need_persistence as P
    rng = np.random.default_rng(3)
    rows = []
    for k in range(count):
        base = 0.5 + 0.1 * rng.standard_normal()
        p = {"none": base, "lesion": base + lesion * (1 if k % 2 else -1), "lesion_random": base + lesion_random,
             "push": base + push + 0.01 * rng.standard_normal()}
        p.update({f"random{i}": base + (random if i % 2 else -random) for i in range(P.RANDOM)})
        rows.append({"life": k // 4, "t": 1 + k % 4, "recorded": base + 0.001, "mass": 0.9, "p_R": p,
                     "later": dict(p) if k < 10 else None, "same_turn": 0.2 if k % 3 == 0 else None})
    return rows


class PersistenceTests(unittest.TestCase):
    def test_contexts_follow_the_protocol(self):
        from research import need_persistence as P
        lives = lives_of(W.Oracle(), 333, 30)
        chosen = P.contexts(lives, limit=50, later=20)
        self.assertEqual(len(chosen), 50)
        self.assertEqual(sum(c["later"] for c in chosen), 20)
        for c in chosen:
            ds = {x["t"]: x for x in W.decisions(lives[c["life"]])}
            self.assertIn(c["t"], ds)
            self.assertTrue(ds[c["t"] + 1]["E"] >= P.LEVEL and ds[c["t"] + 1]["N"] >= P.LEVEL)
            if c["later"]:
                self.assertIn(c["t"] + 2, ds)
        self.assertEqual([(c["life"], c["t"]) for c in chosen],
                         sorted((c["life"], c["t"]) for c in chosen))
        self.assertEqual(chosen, P.contexts(lives, limit=50, later=20))

    def test_random_pushes_have_the_norm_of_the_push(self):
        from research import need_persistence as P
        d_e = np.random.default_rng(0).standard_normal((3, 16))
        pushes = P.random_pushes(d_e, [2])
        self.assertEqual(len(pushes), P.RANDOM)
        for v in pushes:
            np.testing.assert_allclose(np.linalg.norm(v, axis=1), np.linalg.norm(P.UNITS * d_e, axis=1))
            self.assertTrue(np.all(v[:, 2] == 0))
        self.assertFalse(np.allclose(pushes[0], pushes[1]))
        np.testing.assert_array_equal(pushes[1], P.random_pushes(d_e, [2])[1])

    def test_persistence_verdicts(self):
        from research import need_persistence as P
        good = persistence_rows(400)
        v = P.verdicts({"first": good, "second": good, "two": good})["verdicts"]
        self.assertTrue(v["global"] and v["PERS1_first"] and v["PERS2_first"] and v["valid_first"])
        values = P.verdicts({"first": good, "second": good, "two": good})["values"]["first"]
        self.assertAlmostEqual(values["t_plus_1"]["lesion"]["mean"], 0.08, places=6)
        self.assertEqual(values["t_plus_2"]["push"]["contexts"], 10)
        self.assertAlmostEqual(values["same_turn"]["push_at_t_plus_1"], 0.2)
        weak = persistence_rows(400, push=0.02)
        v = P.verdicts({"first": good, "second": weak, "two": weak})["verdicts"]
        self.assertTrue(v["PERS1_second"])
        self.assertFalse(v["PERS2_second"] or v["global"])
        self.assertTrue(P.verdicts({"first": good, "second": weak, "two": good})["verdicts"]["global"])
        self.assertFalse(P.verdicts({"first": weak, "second": good, "two": good})["verdicts"]["global"])
        self.assertFalse(P.verdicts({"first": good, "second": good})["verdicts"]["global"])
        noisy = persistence_rows(400, lesion_random=0.04)
        self.assertFalse(P.verdicts({"first": noisy})["verdicts"]["PERS1_first"])
        loud = persistence_rows(400, random=0.03)
        self.assertFalse(P.verdicts({"first": loud})["verdicts"]["PERS2_first"])
        few = persistence_rows(200)
        self.assertFalse(P.verdicts({"first": few})["verdicts"]["valid_first"])
        off = [dict(r, recorded=r["p_R"]["none"] + 0.03) for r in good]
        self.assertFalse(P.verdicts({"first": off})["verdicts"]["valid_first"])


def paraphrase_rows(lives, push_new=0.08, control=0.0, random=0.01, mass_new=0.9):
    """Reads as research/need_paraphrase.py writes them: the energy questions answer the need, -4 d_E adds push_new
    to the never learned ones and 0.2 to the learned one, control to the unrelated question."""
    from research import need_paraphrase as PA
    rows, contexts = [], 0
    for index in range(PA.lives_to_read(lives, limit=100)):
        for t in W.decisions(lives[index]):
            yes = {q: (0.8 if t[n] <= W.LOW else 0.2) for q, n in PA.NEED.items()}
            yes["C"] = 0.3
            context = PA.is_context(t) and contexts < 100
            contexts += context
            push = None
            if context:
                push = {c: dict(yes) for c in PA.CONDITIONS}
                push["E"].update({"E0": yes["E0"] + 0.2, "E1": yes["E1"] + push_new, "E2": yes["E2"] + push_new,
                                  "C": yes["C"] + control})
                for i in range(PA.RANDOM):
                    push[f"rand{i}"] = {q: v + (random if i % 2 else -random) for q, v in yes.items()}
            mass = {q: (mass_new if q in PA.NEW else 0.95) for q in PA.QUESTIONS}
            rows.append({"life": index, "t": t["t"], "E": t["E"], "N": t["N"], "recorded": 0.5, "p_R": 0.5,
                         "yes": yes, "mass": mass, "push": push})
    return rows


class ParaphraseTests(unittest.TestCase):
    def test_questions_end_as_the_learned_ones(self):
        from research import need_paraphrase as PA
        self.assertEqual(PA.QUESTIONS["E0"], W.QUESTIONS["E"])
        self.assertTrue(all(q.endswith(PA.END) and q.startswith("Question : ") for q in PA.QUESTIONS.values()))
        self.assertEqual(len(set(PA.QUESTIONS.values())), len(PA.QUESTIONS))

    def test_paraphrase_verdicts(self):
        from research import need_paraphrase as PA
        lives = lives_of(W.Oracle(), 334, 60)
        good = PA.verdicts(paraphrase_rows(lives), 1e-6)
        self.assertEqual(good["verdicts"], {"GEN1": True, "GEN2": True, "valid": False, "global": False})
        self.assertEqual(good["values"]["contexts"], 100)
        self.assertAlmostEqual(good["values"]["accuracy"]["E1"], 1.0)
        self.assertAlmostEqual(good["values"]["share_of_E0"]["E1"], 0.4)
        old = PA.MIN_CONTEXTS
        PA.MIN_CONTEXTS = 100
        try:
            self.assertTrue(PA.verdicts(paraphrase_rows(lives), 1e-6)["verdicts"]["global"])
            self.assertFalse(PA.verdicts(paraphrase_rows(lives), 1e-3)["verdicts"]["valid"])
            self.assertFalse(PA.verdicts(paraphrase_rows(lives, push_new=0.03), 1e-6)["verdicts"]["GEN1"])
            v = PA.verdicts(paraphrase_rows(lives, control=0.05), 1e-6)["verdicts"]
            self.assertTrue(v["GEN1"])
            self.assertFalse(v["GEN2"] or v["global"])
            self.assertFalse(PA.verdicts(paraphrase_rows(lives, random=0.03), 1e-6)["verdicts"]["GEN1"])
            self.assertFalse(PA.verdicts(paraphrase_rows(lives, mass_new=0.4), 1e-6)["verdicts"]["valid"])
        finally:
            PA.MIN_CONTEXTS = old


class ListeningTests(unittest.TestCase):
    def test_listening_documents(self):
        from research import need_listening as LS
        lives = lives_of(W.Oracle(), 335, 12)
        report = W.report_documents(lives, np.random.default_rng(0), 3, after_choice=True, workspace=True)
        docs = LS.listening_documents(lives, report)
        groups = [d["group"] for d in docs]
        self.assertEqual(groups.count("learned"), len(report))
        self.assertEqual(groups.count("reworded"), len(report))
        self.assertEqual(groups.count("control"), LS.CONTROL_TURNS * len(lives))
        encode = lambda text: list(text.encode())
        for d in docs:
            self.assertTrue(d["text"][:d["workspace"]].endswith(" Choix :"))
            self.assertTrue(d["text"][d["workspace"]:].startswith(" ? Question : "))
            need_lora.target_tokens(d["text"], d["weights"], encode)  # one weight per target, the answer last
            self.assertEqual(d["text"][-1], str(d["answer"]))
            if d["group"] == "control":
                self.assertEqual(d["answer"], LS.CONTROLS[d["control"]][1])
            if d["group"] == "reworded":
                self.assertIn(LS.TRAINED[d["need"]], d["text"])
        for a, b in zip([d for d in docs if d["group"] == "learned"], [d for d in docs if d["group"] == "reworded"]):
            self.assertEqual((a["workspace"], a["weights"], a["answer"]), (b["workspace"], b["weights"], b["answer"]))
        self.assertEqual(LS.control_documents(lives), LS.control_documents(lives))
        valid = LS.validation(docs)
        self.assertEqual([d["group"] for d in valid].count("control"), LS.VALID_PER_GROUP)
        self.assertEqual(len(valid), 3 * LS.VALID_PER_GROUP)
        held_out = set(PA_QUESTIONS()) | {"Question : fait-il nuit ? "}
        trained = {q.split("Réponds")[0] for q in list(LS.TRAINED.values()) + [c[0] for c in LS.CONTROLS.values()]}
        self.assertFalse(trained & {q.split("Réponds")[0] for q in held_out})

    def test_listening_verdicts(self):
        from research import need_listening as LS, need_paraphrase as PA
        lives = lives_of(W.Oracle(), 334, 60)
        rows = paraphrase_rows(lives)
        for r in rows:
            r["yes"]["water"], r["yes"]["agent"] = 0.03, 0.96
        old = PA.MIN_CONTEXTS
        PA.MIN_CONTEXTS = 100
        try:
            v = LS.verdicts(rows, 1e-6, 0.01)["verdicts"]
            self.assertEqual(v, {"LIS1": True, "LIS2": True, "LIS3": True, "valid": True, "global": True})
            self.assertFalse(LS.verdicts(rows, 1e-6, 0.03)["verdicts"]["valid"])
            for r in rows:
                r["yes"]["water"] = 0.2
            self.assertFalse(LS.verdicts(rows, 1e-6, 0.01)["verdicts"]["LIS1"])
        finally:
            PA.MIN_CONTEXTS = old


def PA_QUESTIONS():
    from research import need_paraphrase as PA
    return [PA.QUESTIONS[q].split("Réponds")[0] for q in ("E1", "E2", "N1", "N2", "C")]


def intention_rows(count, need_yes=0.0, decision_yes=0.08, need_pR=0.01, decision_pR=0.15, control=0.0):
    from research import need_intention as IN
    rng = np.random.default_rng(4)
    rows = []
    for k in range(count):
        base = {"p_R": 0.5 + 0.05 * rng.standard_normal(), "E0": 0.3, "E1": 0.3, "C": 0.3,
                "mass_RM": 0.95, "mass_E0": 0.99, "mass_E1": 0.99, "mass_C": 0.99}
        push = {c: dict(base) for c in IN.CONDITIONS}
        push["E"].update(p_R=base["p_R"] + 0.16, E0=0.5)
        push["need"].update(p_R=base["p_R"] + need_pR, E0=0.3 + need_yes + 0.005 * rng.standard_normal(),
                            C=0.3 + control)
        push["decision"].update(p_R=base["p_R"] + decision_pR + 0.01 * rng.standard_normal(),
                                E0=0.3 + decision_yes + 0.005 * rng.standard_normal())
        rows.append({"life": k // 5, "t": 1 + k % 5, "push": push})
    return rows


class IntentionTests(unittest.TestCase):
    SETUP = {"replay_gap": 1e-6, "execution_gap": 1e-6, "gradient_decisions": 300}

    def test_directions(self):
        from research import need_intention as IN
        rng = np.random.default_rng(0)
        g_true = rng.standard_normal((3, 8))
        grads = [g_true * (1 + 0.1 * i) + 0.01 * rng.standard_normal((3, 8)) for i in range(20)]
        d_e = rng.standard_normal((3, 8))
        d_e[:, 2] = 0
        g_hat, d_perp, agreement = IN.directions(grads, d_e, [2])
        np.testing.assert_allclose(np.linalg.norm(g_hat, axis=1), 1)
        self.assertTrue(np.all(g_hat[:, 2] == 0) and np.all(d_perp[:, 2] == 0))
        np.testing.assert_allclose((d_perp * g_hat).sum(axis=1), 0, atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(d_perp, axis=1), np.linalg.norm(d_e, axis=1))
        self.assertTrue(np.all(agreement > 0.99))

    def test_intention_verdicts(self):
        from research import need_intention as IN
        v = IN.verdicts(intention_rows(400), self.SETUP)
        self.assertEqual(v["verdicts"], {"INT0": True, "NEED": False, "INTENT": True, "valid": True})
        self.assertEqual(v["reading"], IN.READING[(False, True)])
        v = IN.verdicts(intention_rows(400, need_yes=0.08, decision_yes=0.0), self.SETUP)["verdicts"]
        self.assertTrue(v["NEED"] and not v["INTENT"])
        self.assertFalse(IN.verdicts(intention_rows(400, need_yes=0.08, control=0.07), self.SETUP)["verdicts"]["NEED"])
        self.assertFalse(IN.verdicts(intention_rows(400, need_pR=0.1), self.SETUP)["verdicts"]["INT0"])
        self.assertFalse(IN.verdicts(intention_rows(400, decision_pR=0.05), self.SETUP)["verdicts"]["INT0"])
        self.assertFalse(IN.verdicts(intention_rows(200), self.SETUP)["verdicts"]["valid"])
        self.assertFalse(IN.verdicts(intention_rows(400), dict(self.SETUP, gradient_decisions=100))["verdicts"]["valid"])


def transfer_rows(count, other=0.06, own=0.15, random=0.005, recorded_gap=0.001):
    from research import need_transfer as TR
    rng = np.random.default_rng(5)
    rows = []
    for k in range(count):
        base = 0.4 + 0.1 * rng.standard_normal()
        p = {"none": base, "own": base + own, "other": base + other + 0.01 * rng.standard_normal()}
        p.update({f"random{i}": base + (random if i % 2 else -random) for i in range(TR.RANDOM)})
        rows.append({"life": k // 4, "t": 1 + k % 4, "recorded": base + recorded_gap, "mass": 0.95, "p_R": p})
    return rows


class TransferTests(unittest.TestCase):
    def test_pushes(self):
        from research import need_transfer as TR
        rng = np.random.default_rng(1)
        own, other = rng.standard_normal((3, 10)), rng.standard_normal((3, 10)) * 5
        own[:, 4] = other[:, 4] = 0
        pushes = TR.pushes(own, other, [4])
        self.assertEqual(len(pushes), 2 + TR.RANDOM)
        np.testing.assert_allclose(pushes[0], -TR.UNITS * own)
        for v in pushes[1:]:
            np.testing.assert_allclose(np.linalg.norm(v, axis=1), np.linalg.norm(TR.UNITS * own, axis=1))
        cos = (pushes[1] * other).sum(axis=1) / np.linalg.norm(pushes[1], axis=1) / np.linalg.norm(other, axis=1)
        np.testing.assert_allclose(cos, -1)
        self.assertTrue(np.all(pushes[2][:, 4] == 0))

    def test_contexts(self):
        from research import need_transfer as TR
        lives = lives_of(W.Oracle(), 336, 40)
        chosen = TR.contexts(lives, limit=50)
        self.assertEqual(chosen, sorted(chosen))
        for i, t in chosen:
            turn = [x for x in lives[i]["turns"] if x["t"] == t][0]
            self.assertTrue(turn["E"] >= W.HIGH and turn["N"] >= W.HIGH)

    def test_transfer_verdicts(self):
        from research import need_transfer as TR
        ok = {"cache_gap": 1e-6}
        good = TR.verdicts({"first": transfer_rows(400), "second": transfer_rows(400, other=0.03, own=0.03)},
                           {"first": ok, "second": ok})["verdicts"]
        self.assertEqual(good, {"TR1": True, "valid_second": True, "TR2": True, "valid_first": True, "global": True})
        weak = TR.verdicts({"first": transfer_rows(400, other=0.04), "second": transfer_rows(400, other=0.03)},
                           {"first": ok, "second": ok})["verdicts"]
        self.assertTrue(weak["TR1"])
        self.assertFalse(weak["TR2"] or weak["global"])
        loud = TR.verdicts({"second": transfer_rows(400, other=0.03, random=0.02)}, {"second": ok})["verdicts"]
        self.assertFalse(loud["TR1"] or loud["global"])
        off = TR.verdicts({"first": transfer_rows(400, recorded_gap=0.05)}, {"first": ok})["verdicts"]
        self.assertFalse(off["valid_first"])


class BaseCodeTests(unittest.TestCase):
    def test_base_code(self):
        from research import need_base_code as BC
        rng = np.random.default_rng(2)
        first, second = rng.standard_normal((3, 9)), rng.standard_normal((3, 9)) * 3
        pushes = BC.pushes(first, second, [1])
        for v in pushes[1:]:
            np.testing.assert_allclose(np.linalg.norm(v, axis=1), np.linalg.norm(BC.UNITS * first, axis=1))
        rows = []
        for k in range(400):
            base = 0.3 + 0.05 * rng.standard_normal()
            p = {"none": base, "first": base + 0.08, "second": base + 0.06 + 0.01 * rng.standard_normal()}
            p.update({f"random{i}": base + (0.01 if i % 2 else -0.01) for i in range(BC.RANDOM)})
            rows.append({"life": k // 4, "t": 1 + k % 4, "mass": 0.3, "p_R": p})
        v = BC.verdicts(rows, {"cache_gap": 1e-6})["verdicts"]
        self.assertEqual(v, {"BASE1": True, "BASE2": True, "valid": True, "global": True})
        weak = [dict(r, p_R=dict(r["p_R"], second=r["p_R"]["none"] + 0.02)) for r in rows]
        self.assertFalse(BC.verdicts(weak, {"cache_gap": 1e-6})["verdicts"]["BASE2"])
        mute = [dict(r, mass=0.1) for r in rows]
        self.assertFalse(BC.verdicts(mute, {"cache_gap": 1e-6})["verdicts"]["valid"])


class SmallPushTests(unittest.TestCase):
    def test_small_push_verdicts(self):
        from research import need_small_push as SP
        rng = np.random.default_rng(6)

        def rows(effect, random):
            out = []
            for k in range(400):
                base = 0.5 + 0.05 * rng.standard_normal()
                p = {"none": base, "first": base + effect, "second": base + effect + 0.005 * rng.standard_normal()}
                p.update({f"random{i}": base + (random if i % 2 else -random) for i in range(SP.RANDOM)})
                out.append({"life": k // 4, "t": 1 + k % 4, "mass": 0.9, "p_R": p})
            return out
        ok = {"cache_gap": 1e-6}
        v = SP.verdicts({"base": rows(0.15, 0.03), "agent": rows(0.05, 0.004)}, {"base": ok, "agent": ok})["verdicts"]
        self.assertEqual(v, {"SMALL1": True, "SMALL2": True, "STAB": True, "valid": True, "global": True})
        v = SP.verdicts({"base": rows(0.15, 0.08), "agent": rows(0.05, 0.004)}, {"base": ok, "agent": ok})["verdicts"]
        self.assertFalse(v["SMALL1"] or v["global"])
        self.assertTrue(v["STAB"])
        v = SP.verdicts({"base": rows(0.15, 0.03), "agent": rows(0.05, 0.02)}, {"base": ok, "agent": ok})["verdicts"]
        self.assertFalse(v["STAB"])
        self.assertFalse(SP.verdicts({"base": rows(0.15, 0.03)}, {"base": ok})["verdicts"]["valid"])
        pushes = SP.pushes(rng.standard_normal((3, 5)), rng.standard_normal((3, 5)), [0])
        self.assertEqual(len(pushes), 2 + SP.RANDOM)


def fake_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True):
    """A stand-in fast tokenizer with offsets: " Cho", "ix", words with their space, and line ends."""
    import re
    spans = [m.span() for m in re.finditer(r" Cho|ix(?= :)| ?[^\s]+|\n+", text)]
    return {"input_ids": [hash(text[a:b]) % 1000 for a, b in spans], "offset_mapping": spans}


class CarryTests(unittest.TestCase):
    def test_carry_layout_and_mask(self):
        life = lives_of(W.Oracle(), 337, 1)[0]
        text = W.life_text(life["turns"])
        ids, turn, carried = need_lora.carry_layout(text, fake_tokenizer)
        n = len(W.decisions(life))
        self.assertEqual(sum(carried), 4 * n)  # " Cho" "ix" " :" and the action, each turn
        self.assertEqual(max(turn) + 1, len([t for t in life["turns"]]))
        mask = need_lora.carry_mask(turn, carried)
        t, c = np.array(turn), np.array(carried)
        q = len(turn) - 1
        seen = np.where(mask[q])[0]
        self.assertTrue(all(t[j] in (-1, t[q]) or (c[j] and t[j] < t[q]) for j in seen))
        hidden = [j for j in range(q) if t[j] not in (-1, t[q]) and not c[j]]
        self.assertTrue(hidden and not mask[q, hidden].any())
        self.assertTrue(mask[q, [j for j in range(q) if c[j]]].all())
        self.assertFalse(mask[np.triu_indices(len(turn), 1)].any())
        item = (ids, [0.0] * len(ids), mask)
        padded = need_lora.input_mask(item, len(ids) + 5)
        self.assertTrue((padded[:len(ids), :len(ids)] == mask).all())
        self.assertTrue(padded[-1, :len(ids) + 5].sum() == len(ids) + 5)
        with self.assertRaises(ValueError):
            need_lora.carry_layout("x" + text, fake_tokenizer)

    def test_carry_documents(self):
        from research import need_mlx as M
        lives = lives_of(W.Oracle(), 338, 4)
        docs = M.carry_docs(lives)
        for d, life in zip(docs, lives):
            need_lora.target_tokens(d["text"], d["weights"], lambda x: list(x.encode()))
            self.assertTrue(d["carry"])
        rows = M.carry_decisions(lives, docs)
        self.assertTrue(all(r["text"].endswith("Choix :") and r["t"] in M.CARRY_TURNS for r in rows))


class CarryVerdictTests(unittest.TestCase):
    def test_pairs_and_verdicts(self):
        from research import need_carry as CA
        lives = lives_of(W.Oracle(), 339, 60)
        pairs = CA.choose_pairs(lives, limit=200)
        self.assertTrue(pairs and all(p["swap"] == CA.SWAP and p["t"] - p["j"] >= CA.MIN_GAP for p in pairs))
        self.assertEqual(pairs, CA.choose_pairs(lives, limit=200))
        rng = np.random.default_rng(1)
        rows = [dict(p, carry=[0.4, 0.4 + 0.08 + 0.01 * rng.standard_normal()], control=[0.4, 0.41],
                     free=[0.4, 0.6]) for p in pairs]
        for life in lives:
            for t in W.decisions(life):
                t["mass"] = 0.9
        dead = [dict(l, survived=False) for l in lives]
        setup = {"replica": {"mean_gap": 0.01}, "cache_gap": 1e-6, "mask_gap": 0.0}
        v = CA.verdicts({"carry": lives, "control": dead, "free": lives}, rows, setup)["verdicts"]
        survival = np.mean([l["survived"] for l in lives])
        self.assertEqual(v["MEM2"], True)
        self.assertEqual(v["MEM1"], bool(survival >= CA.SURVIVAL))
        flat = [dict(r, carry=[0.4, 0.42]) for r in rows]
        self.assertFalse(CA.verdicts({"carry": lives, "control": dead, "free": lives}, flat, setup)["verdicts"]["MEM2"])
        same = [dict(r, control=list(r["carry"])) for r in rows]
        self.assertFalse(CA.verdicts({"carry": lives, "control": dead, "free": lives}, same, setup)["verdicts"]["MEM2"])
        self.assertFalse(CA.verdicts({"carry": lives, "control": lives, "free": lives}, rows, setup)["verdicts"]["MEM1"])
        bad = dict(setup, mask_gap=1e-3)
        self.assertFalse(CA.verdicts({"carry": lives, "control": dead, "free": lives}, rows, bad)["verdicts"]["valid"])


class MemoryTests(unittest.TestCase):
    def test_actions_mask_lets_no_past_event_through(self):
        from research import need_memory as NM
        life = NM.teacher_life(NM.HELD_STREAM, 0)
        text = W.life_text(life["turns"])
        ids, turn, carried, action = need_lora.carry_layout(text, fake_tokenizer, kinds=True)
        self.assertEqual(sum(action), len(W.decisions(life)))
        self.assertTrue(all(c for c, a in zip(carried, action) if a))
        mask = need_lora.actions_mask(turn, action)
        t, a = np.array(turn), np.array(action)
        self.assertFalse(mask[np.triu_indices(len(turn), 1)].any())
        for q in range(len(turn)):
            seen = np.where(mask[q])[0]
            if a[q]:  # an action token: the header, earlier action tokens, itself
                self.assertTrue(all(t[j] == -1 or (a[j] and t[j] < t[q]) or j == q for j in seen))
            else:
                self.assertTrue(all(t[j] in (-1, t[q]) or (a[j] and t[j] < t[q]) for j in seen))
        # what a token can be influenced by, through any number of layers: the closure of the mask
        reach = mask.copy()
        for _ in range(12):  # paths of any length up to 4096
            reach = reach | ((reach.astype(int) @ reach.astype(int)) > 0)
        events = [j for j in range(len(turn)) if t[j] >= 0 and not a[j]]
        q = len(turn) - 1
        self.assertFalse(any(reach[q, j] for j in events if t[j] < t[q]))
        route = need_lora.carry_mask(turn, carried)
        reach = route.copy()
        for _ in range(12):  # paths of any length up to 4096
            reach = reach | ((reach.astype(int) @ reach.astype(int)) > 0)
        self.assertTrue(any(reach[q, j] for j in events if t[j] < t[q]))
        for kind in ("route", "actions", "free"):
            ids2, m = need_lora.memory_mask(kind, text, fake_tokenizer)
            self.assertEqual(ids2, ids)
            self.assertEqual(m.shape, (len(ids), len(ids)))
        with self.assertRaises(ValueError):
            need_lora.memory_mask("other", text, fake_tokenizer)

    def test_teacher_lives_ceiling_and_pairs(self):
        from research import need_memory as NM
        lives = NM.teacher_lives(NM.HELD_STREAM, 40)
        self.assertEqual(lives, NM.teacher_lives(NM.HELD_STREAM, 40))
        for life in lives:
            last = 1
            for x in W.decisions(life):
                self.assertEqual(x["rule"], NM.rule(x["E"], x["N"], last))
                self.assertEqual(x["last"], last)
                last = x["action"]
        random = np.mean([x["action"] != x["rule"] for l in lives for x in W.decisions(l)])
        self.assertTrue(0.05 < random < 0.15)  # about EPS / 2
        for life in lives:
            p = NM.ceiling(life)
            self.assertEqual(len(p), len(W.decisions(life)))
            self.assertTrue(all(0.0 <= q <= 1.0 for q in p))
            first = W.decisions(life)[0]  # the start is known: the first decision is certain
            self.assertEqual(p[0], float(first["rule"] == 0))
        right = np.mean([NM.correct(q, x["rule"]) for l in lives for x, q in zip(W.decisions(l), NM.ceiling(l))])
        self.assertTrue(0.8 < right < 0.95)
        exact = np.mean([NM.correct(q, x["rule"]) for l in lives for x, q in
                         zip(W.decisions(l), NM.ceiling(l, eps=1e-9))])
        self.assertLessEqual(abs(exact - right), 0.1)
        docs = NM.documents(lives, "actions")
        for d in docs:
            need_lora.target_tokens(d["text"], d["weights"], lambda x: list(x.encode()))
            self.assertEqual(d["carry"], "actions")
        free = NM.documents(lives, "free")[0]
        ids, mask = need_lora.memory_mask(free["carry"], free["text"], fake_tokenizer)
        self.assertTrue((mask == np.tril(np.ones_like(mask))).all())
        pairs = NM.choose_pairs(lives, limit=60)
        self.assertEqual(pairs, NM.choose_pairs(lives, limit=60))
        self.assertTrue(pairs and all(p["swap"] == NM.SWAP and p["t"] - p["j"] >= NM.MIN_GAP for p in pairs))
        per_life = {}
        for p in pairs:
            per_life[p["life"]] = per_life.get(p["life"], 0) + 1
        self.assertLessEqual(max(per_life.values()), NM.PER_LIFE)
        effects = [NM.rule_effect(lives[p["life"]], p) for p in pairs]
        self.assertTrue(set(effects) <= {0.0, 1.0} and 0 < np.mean(effects) < 1)
        rows = NM.replica_decisions(lives[:4])
        self.assertTrue(all(r["text"].endswith("Choix :") and r["t"] in NM.REPLICA_TURNS for r in rows))


class MemoryVerdictTests(unittest.TestCase):
    def test_verdicts(self):
        from research import need_memory as NM
        held = NM.teacher_lives(NM.HELD_STREAM, 60)
        bayes = {(i, x["t"]): q for i, l in enumerate(held) for x, q in zip(W.decisions(l), NM.ceiling(l))}
        rule = {(i, x["t"]): x["rule"] for i, l in enumerate(held) for x in W.decisions(l)}
        perfect = [{"life": i, "t": t, "route": 0.9 if rule[(i, t)] == 0 else 0.1, "actions": bayes[(i, t)],
                    "free": 0.9 if rule[(i, t)] == 0 else 0.1} for (i, t) in sorted(rule)]
        pairs = NM.choose_pairs(held, limit=200)
        ideal = [NM.rule_effect(held[p["life"]], p) for p in pairs]
        rows = [dict(p, route=[0.2, 0.2 + 0.9 * e], actions=[0.3, 0.3], free=[0.2, 0.2 + e]) for p, e in zip(pairs, ideal)]
        lives = lives_of(W.Oracle(), 341, 40)
        for life in lives:
            for t in W.decisions(life):
                t["mass"] = 0.9
        dead = [dict(l, survived=False) for l in lives]
        ok = {"replica": {a: {"mean_gap": 0.01} for a in NM.ARMS}, "cache_gap": {a: 1e-6 for a in NM.ARMS},
              "mask_gap": {"route": 0.0, "actions": 0.0}}
        v = NM.verdicts(held, perfect, {"route": lives, "actions": dead, "free": lives}, rows, ok)["verdicts"]
        self.assertTrue(v["MEM4"] and v["MEM5"] and v["valid"] and v["global"])
        self.assertEqual(v["MEM3"], bool(np.mean([l["survived"] for l in lives]) >= NM.SURVIVAL_GAIN))
        like_bayes = [dict(c, route=c["actions"]) for c in perfect]
        self.assertFalse(NM.verdicts(held, like_bayes, {"route": lives, "actions": dead, "free": lives}, rows,
                                     ok)["verdicts"]["MEM4"])
        weak = [dict(r, route=[0.2, 0.2 + 0.3 * e]) for r, e in zip(rows, ideal)]
        self.assertFalse(NM.verdicts(held, perfect, {"route": lives, "actions": dead, "free": lives}, weak,
                                     ok)["verdicts"]["MEM5"])
        leaky = [dict(c, actions=c["route"]) for c in perfect]  # B beating the ceiling: the test is not valid
        self.assertFalse(NM.verdicts(held, leaky, {"route": lives, "actions": dead, "free": lives}, rows,
                                     ok)["verdicts"]["valid"])
        self.assertFalse(NM.verdicts(held, perfect, {"route": lives, "actions": lives, "free": lives}, rows,
                                     ok)["verdicts"]["MEM3"])
        bad = dict(ok, mask_gap={"route": 0.0, "actions": 1e-3})
        self.assertFalse(NM.verdicts(held, perfect, {"route": lives, "actions": dead, "free": lives}, rows,
                                     bad)["verdicts"]["valid"])
        with self.assertRaises(ValueError):
            NM.verdicts(held, perfect[1:], {"route": lives, "actions": dead, "free": lives}, rows, ok)


class MemoryGatherTests(unittest.TestCase):
    def test_the_three_agents_measured_apart_are_put_together(self):
        from research import need_memory as NM
        held = NM.teacher_lives(NM.HELD_STREAM, NM.HELD_LIVES)
        pairs = NM.choose_pairs(held)
        lives = lives_of(W.Oracle(), 342, 8)
        for life in lives:
            for t in W.decisions(life):
                t["mass"] = 0.9
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "test"
            out.mkdir()
            for arm in NM.ARMS:
                W.write_jsonl(out / f"choices-{arm}.jsonl.gz", [{"life": i, "t": x["t"], "p_R": float(x["rule"] == 0),
                                                                 "mass": 1.0} for i, l in enumerate(held)
                                                                for x in W.decisions(l)])
                W.write_jsonl(out / f"pairs-{arm}.jsonl.gz", [dict(p, p_R=[0.2, 0.2 + NM.rule_effect(held[p["life"]], p)])
                                                              for p in pairs])
                W.write_jsonl(out / f"lives-{arm}.jsonl.gz", lives)
                checked = {"replica": {"mean_gap": 0.01}, "cache_gap": 1e-6}
                if arm != "free":
                    checked["mask_gap"] = 0.0
                (out / f"setup-{arm}.json").write_text(json.dumps(checked))
            result = NM.gather(root)
            self.assertTrue(result["verdicts"]["MEM4"] and result["verdicts"]["MEM5"])
            self.assertFalse(result["verdicts"]["valid"])  # B as good as the rule beats the ceiling: not valid
            self.assertEqual(result["values"]["pairs"], len(pairs))
            W.write_jsonl(out / "pairs-free.jsonl.gz", [dict(p, p_R=[0.2, 0.2], t=p["t"] + 1) for p in pairs])
            with self.assertRaises(SystemExit):
                NM.gather(root)


class MemoryNoLeakTests(unittest.TestCase):
    def test_filled_texts_documents_and_agent(self):
        from research import need_memory as NM, need_memory_no_leak as NL22
        lives = NM.teacher_lives(NM.HELD_STREAM, 6)
        for life in lives:
            text = NL22.life_text(life["turns"])
            self.assertEqual(text.count(" Choix :"), len(W.decisions(life)))
            for x in W.decisions(life):
                line = W.choice_line(x["t"], x["event"])
                self.assertEqual(NL22.fill(line), NL22.choice_line(x["t"], x["event"]))
                self.assertTrue(NL22.decision_text(life["turns"], x["t"]).endswith(NL22.choice_line(x["t"], x["event"])))
            ids, turn, carried, acts = need_lora.carry_layout(text, fake_tokenizer, kinds=True)
            self.assertEqual(sum(acts), len(W.decisions(life)))
        with self.assertRaises(ValueError):
            NL22.fill("Tour 3 : rien. Choix :")
        docs = NL22.documents(lives, "actions")
        self.assertTrue(all(d["carry"] == "actions" and " - " in d["text"] for d in docs))
        for d in docs:
            need_lora.target_tokens(d["text"], d["weights"], lambda x: list(x.encode()))

        class Fake:
            def __init__(self):
                self.lines = []
            def start(self, header):
                self.lines = []
            def decide(self, line):
                self.lines.append(line)
                return 0.5, 1.0
            def commit(self, action):
                pass
        fake = Fake()
        agent = NL22.Filled(fake)
        W.play(agent, W.world_rng(343, 0), W.choice_rng(343, 0))
        self.assertTrue(fake.lines and all(" - " in l or "baies" in l for l in fake.lines))

    def test_epoch_orders_after_the_first_are_fixed(self):
        np.random.seed(5)
        old = [o.tolist() for o, _ in zip(need_lora.epoch_orders(6, True), range(3))]
        np.random.seed(5)
        self.assertEqual(old, [np.random.permutation(6).tolist() for _ in range(3)])
        need_lora.EPOCHS.append(270926)
        try:
            def run(validation):
                np.random.seed(5)
                orders = need_lora.epoch_orders(6, True)
                out = [next(orders).tolist()]
                if validation:
                    np.random.permutation(10)  # what a validation draws from the global state
                return out + [next(orders).tolist(), next(orders).tolist()]
            self.assertEqual(run(False), run(True))
            self.assertEqual(run(True)[0], old[0])  # the first epoch is unchanged
            self.assertEqual(len(list(need_lora.epoch_orders(6, False))), 1)
        finally:
            need_lora.EPOCHS.clear()

    def test_chunks_continue_one_another_until_the_target_or_the_time(self):
        import argparse
        from unittest import mock
        from research import need_mlx as M
        calls = []

        def fake_train(model, data, adapter, iters, resume=None, skip=0, moments=False, fixed_epochs=None, **_):
            calls.append((iters, skip, resume, moments, fixed_epochs))
        with tempfile.TemporaryDirectory() as out:
            (Path(out) / "fused-start").mkdir()
            a = argparse.Namespace(arm="actions", start="s", model="m", iters=1000, no_leak=True,
                                   continue_from="prev", skip=1000, minutes=None, until=2000, out=out)
            with mock.patch.object(M, "train", fake_train), mock.patch.object(M, "free", lambda: None), \
                    mock.patch.object(M, "carry_rows", lambda *x, **k: [{"p_R": 0.5}]), \
                    mock.patch.object(M, "clean", lambda *x: None):
                M.memory(a, log=lambda *x: None)
                record = json.loads((Path(out) / "memory-actions-stage.json").read_text())
                self.assertEqual(record["iterations_done"], 2000)
                self.assertEqual([c[:2] for c in calls], [(250, 1000), (250, 1250), (250, 1500), (250, 1750)])
                self.assertEqual(calls[0][2], "prev")
                self.assertTrue(all(c[2] == str(Path(out) / "adapters-actions") for c in calls[1:]))
                self.assertTrue(all(c[3] and c[4] == W.SEED for c in calls))
                calls.clear()
                a.minutes, a.skip, a.continue_from = 1.0, 0, "none"  # 1 minute: no room for the 5 kept for the end
                with self.assertRaises(RuntimeError):
                    M.memory(a, log=lambda *x: None)
                self.assertEqual(len(calls), 0)
                a.minutes = 30.0
                M.memory(a, log=lambda *x: None)
                self.assertEqual(calls[0][:4], (250, 0, None, False))

    def test_the_two_agents_measured_apart_are_put_together(self):
        from research import need_memory as NM, need_memory_no_leak as NL22
        held = NM.teacher_lives(NM.HELD_STREAM, NM.HELD_LIVES)
        pairs = NM.choose_pairs(held)
        bayes = {(i, x["t"]): q for i, l in enumerate(held) for x, q in zip(W.decisions(l), NM.ceiling(l))}
        lives = lives_of(W.Oracle(), 344, 8)
        for life in lives:
            for t in W.decisions(life):
                t["mass"] = 0.9
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "test"
            out.mkdir()
            for arm, p_of in (("route", lambda i, x: float(x["rule"] == 0)), ("actions", lambda i, x: bayes[(i, x["t"])])):
                W.write_jsonl(out / f"choices-{arm}.jsonl.gz", [{"life": i, "t": x["t"], "p_R": p_of(i, x), "mass": 1.0}
                                                                for i, l in enumerate(held) for x in W.decisions(l)])
                gain = 1.0 if arm == "route" else 0.0
                W.write_jsonl(out / f"pairs-{arm}.jsonl.gz",
                              [dict(p, p_R=[0.2, 0.2 + gain * NM.rule_effect(held[p["life"]], p)]) for p in pairs])
                W.write_jsonl(out / f"lives-{arm}.jsonl.gz", lives)
                (out / f"setup-{arm}.json").write_text(json.dumps({"replica": {"mean_gap": 0.01}, "cache_gap": 1e-6,
                                                                   "mask_gap": 0.0}))
            result = NL22.gather(root)
            self.assertTrue(result["verdicts"]["MEM4"] and result["verdicts"]["MEM5"] and result["verdicts"]["valid"])
            self.assertTrue(result["verdicts"]["global"])
            self.assertNotIn("free", result["values"]["accuracy"])


class CarriedContentTests(unittest.TestCase):
    def test_draw_and_rule_effects(self):
        from research import need_carried_content as CC, need_memory as NM
        held = NM.teacher_lives(NM.HELD_STREAM, 40)
        grafts = CC.draw(held)
        self.assertEqual(grafts, CC.draw(held))
        at = [{x["t"]: x for x in W.decisions(l)} for l in held]
        for g in grafts:
            r, d, j, t = g["r"], g["d"], g["j"], g["t"]
            self.assertNotEqual(r, d)
            self.assertEqual(t - j, g["g"])
            self.assertEqual(at[r][j]["action"], at[d][j]["action"])
            same = (at[r][j]["E"], at[r][j]["N"]) == (at[d][j]["E"], at[d][j]["N"])
            self.assertEqual(same, g["kind"] == "b")
            self.assertEqual(g["rule_effect"], CC.rule_effect(held[r], held[d], j, t))
            if g["kind"] == "a":
                self.assertNotEqual(g["rule_effect"], 0)
            else:
                self.assertEqual(g["rule_effect"], 0)
        # replaying a life from its own needs gives back its needs
        life = held[0]
        ds = W.decisions(life)
        x, y = ds[2], ds[5]
        self.assertEqual(CC.replay(life, x["t"], x["E"], x["N"], y["t"]), (y["E"], y["N"]))

    def test_verdicts(self):
        from research import need_carried_content as CC, need_memory as NM
        held = NM.teacher_lives(NM.HELD_STREAM, NM.HELD_LIVES)
        grafts = CC.draw(held)
        rng = np.random.default_rng(3)
        rows = []
        for g in grafts:
            shift = 0.3 * g["rule_effect"] + (0.02 * rng.standard_normal() if g["kind"] != "a" else 0.0)
            rows.append(dict(g, route=[0.5, 0.5 + shift], actions=[0.5, 0.5]))
        ok = {"no_graft_gap": 1e-8, "self_graft_gap": 0.0, "positions_same": True}
        v = CC.verdicts(rows, ok)["verdicts"]
        self.assertTrue(v["STATE1"] and v["STATE2"] and v["valid"] and v["global"])
        history = [dict(r, route=[0.5, 0.5 + (0.3 if r["kind"] == "b" else r["route"][1] - 0.5)]) for r in rows]
        self.assertFalse(CC.verdicts(history, ok)["verdicts"]["STATE2"])
        flat = [dict(r, route=[0.5, 0.52]) for r in rows]
        self.assertFalse(CC.verdicts(flat, ok)["verdicts"]["STATE1"])
        self.assertFalse(CC.verdicts(rows, dict(ok, self_graft_gap=1e-3))["verdicts"]["valid"])
        d = CC._boot_diff(np.array([1.0, 1.0, 1.0]), np.array([0, 1, 2]), np.array([0.0, 0.0]), np.array([0, 1]))
        self.assertEqual(d["mean"], 1.0)
        self.assertLessEqual(d["low"], d["mean"])


class ChainTests(unittest.TestCase):
    def test_draw(self):
        from research import need_chain as CH, need_carried_content as CC, need_memory as NM
        held = NM.teacher_lives(NM.HELD_STREAM, 40)
        grafts = CH.draw(held)
        self.assertEqual(grafts, CH.draw(held))
        self.assertTrue(grafts)
        at = [{x["t"]: x for x in W.decisions(l)} for l in held]
        for g in grafts:
            r, d, i, t = g["r"], g["d"], g["i"], g["t"]
            self.assertNotEqual(r, d)
            self.assertIn(g["h"], CH.GAPS)
            self.assertEqual(t - i, g["h"])
            self.assertTrue(all(k in at[r] for k in range(i, t + 1)))
            self.assertEqual(at[r][i]["action"], at[d][i]["action"])
            self.assertNotEqual((at[r][i]["E"], at[r][i]["N"]), (at[d][i]["E"], at[d][i]["N"]))
            self.assertNotEqual(g["rule_effect"], 0)
            self.assertEqual(g["rule_effect"], CC.rule_effect(held[r], held[d], i, t))
        for r in range(len(held)):
            for h in CH.GAPS:
                self.assertLessEqual(sum(1 for g in grafts if g["r"] == r and g["h"] == h), CH.PER_GAP)

    def test_verdicts(self):
        from research import need_chain as CH, need_memory as NM
        grafts = CH.draw(NM.teacher_lives(NM.HELD_STREAM, NM.HELD_LIVES))
        rng = np.random.default_rng(5)
        ok = {"extend_gap": 1e-8, "self_graft_gap": 0.0, "positions_same": True}

        def rows(total, direct):
            out = []
            for g in grafts:
                e, noise = g["rule_effect"], 0.01 * rng.standard_normal()
                out.append(dict(g, p_R={"own": 0.5, "total": 0.5 + e * total + noise, "direct": 0.5 + e * direct,
                                        "mediators": 0.5 + e * (total - direct)}))
            return out

        chain = CH.verdicts(rows(0.4, 0.1), ok)
        self.assertTrue(all(chain["verdicts"].values()))
        self.assertAlmostEqual(chain["values"]["chain_share"]["mean"], 0.75, delta=0.02)
        self.assertFalse(CH.verdicts(rows(0.4, 0.3), ok)["verdicts"]["CHAIN"])
        weak = CH.verdicts(rows(0.05, 0.0), ok)["verdicts"]
        self.assertFalse(weak["prior"] or weak["CHAIN"])
        self.assertFalse(CH.verdicts(rows(0.4, 0.1), dict(ok, extend_gap=1e-3))["verdicts"]["valid"])
        few = [r for r in rows(0.4, 0.1) if r["h"] == 2 or r["r"] < 20]
        self.assertFalse(CH.verdicts(few, ok)["verdicts"]["valid"])
        self.assertEqual({k: v["count"] for k, v in chain["values"]["by_gap"].items()},
                         {str(h): sum(1 for g in grafts if g["h"] == h) for h in CH.GAPS})


class ChainFarTests(unittest.TestCase):
    def test_draw_on_new_lives(self):
        from research import need_chain_far as CF, need_carried_content as CC, need_memory as NM
        held = CF.lives()
        self.assertEqual(len(held), CF.LIVES)
        used = NM.teacher_lives(NM.HELD_STREAM, 4)
        self.assertNotEqual([l["turns"] for l in held[:4]], [l["turns"] for l in used])
        self.assertTrue(all(len(W.decisions(l)) >= 7 for l in held[:4]))  # the checks read turns 4 and 7
        grafts = CF.draw(held)
        self.assertEqual([g["index"] for g in grafts], list(range(len(grafts))))
        at = [{x["t"]: x for x in W.decisions(l)} for l in held]
        for g in grafts[:120]:
            r, d, i, t = g["r"], g["d"], g["i"], g["t"]
            self.assertIn(g["h"], CF.GAPS)
            self.assertEqual(t - i, g["h"])
            self.assertTrue(all(k in at[r] for k in range(i, t + 1)))
            self.assertEqual(at[r][i]["action"], at[d][i]["action"])
            self.assertNotEqual((at[r][i]["E"], at[r][i]["N"]), (at[d][i]["E"], at[d][i]["N"]))
            self.assertEqual(g["rule_effect"], CC.rule_effect(held[r], held[d], i, t))
            self.assertNotEqual(g["rule_effect"], 0)
        self.assertEqual({h: sum(g["h"] == h for g in grafts) for h in CF.GAPS}, {3: 219, 4: 217, 5: 173})

    def test_verdicts_and_shards(self):
        import tempfile
        from pathlib import Path
        from research import need_chain_far as CF
        grafts = CF.draw(CF.lives())
        rng = np.random.default_rng(7)
        ok = {"extend_gap": 1e-8, "self_graft_gap": 0.0, "positions_same": True, "threads": 1}

        def rows(total, direct, mediators):
            return [dict(g, p_R={"own": 0.5, "total": 0.5 + g["rule_effect"] * total + 0.01 * rng.standard_normal(),
                                 "direct": 0.5 + g["rule_effect"] * direct,
                                 "mediators": 0.5 + g["rule_effect"] * mediators}) for g in grafts]

        self.assertTrue(all(CF.verdicts(rows(0.4, 0.1, 0.3), ok)["verdicts"].values()))
        v = CF.verdicts(rows(0.4, 0.1, 0.15), ok)["verdicts"]
        self.assertTrue(v["FAR1"] and not v["FAR2"])
        v = CF.verdicts(rows(0.4, 0.2, 0.3), ok)["verdicts"]  # the chain is exactly half: not clearly above
        self.assertFalse(v["FAR1"])
        v = CF.verdicts(rows(0.05, 0.0, 0.05), ok)["verdicts"]
        self.assertFalse(v["prior"] or v["FAR1"] or v["FAR2"])
        self.assertFalse(CF.verdicts(rows(0.4, 0.1, 0.3), dict(ok, self_graft_gap=1e-5))["verdicts"]["valid"])
        measured = rows(0.4, 0.1, 0.3)
        with tempfile.TemporaryDirectory() as folder:
            for k in range(2):
                W.write_jsonl(Path(folder) / f"grafts-{k}-of-2.jsonl.gz", [r for r in measured if r["index"] % 2 == k])
            (Path(folder) / "setup.json").write_text(json.dumps(ok))
            got = CF.gather(folder)
            self.assertEqual(got["verdicts"], CF.verdicts(measured, ok)["verdicts"])
            W.write_jsonl(Path(folder) / "grafts-1-of-2.jsonl.gz", [r for r in measured if r["index"] % 2 == 1][:-1])
            with self.assertRaises(SystemExit):
                CF.gather(folder)


class RelayTests(unittest.TestCase):
    def test_draw_and_verdicts(self):
        import tempfile
        from pathlib import Path
        from research import need_relay as RY, need_carried_content as CC
        held = RY.lives()
        grafts = RY.draw(held)
        self.assertEqual(len(grafts), 209)
        self.assertEqual([g["index"] for g in grafts], list(range(len(grafts))))
        at = [{x["t"]: x for x in W.decisions(l)} for l in held]
        for g in grafts[:60]:
            r, d, i, t = g["r"], g["d"], g["i"], g["t"]
            self.assertEqual(t - i, RY.GAP)
            self.assertTrue(all(k in at[r] for k in range(i, t + 1)))
            self.assertEqual(at[r][i]["action"], at[d][i]["action"])
            self.assertNotEqual((at[r][i]["E"], at[r][i]["N"]), (at[d][i]["E"], at[d][i]["N"]))
            self.assertEqual(g["rule_effect"], CC.rule_effect(held[r], held[d], i, t))
        rng = np.random.default_rng(11)
        ok = {"extend_gap": 1e-8, "self_graft_gap": 0.0, "positions_same": True, "threads": 1}

        def rows(complete, through_i, through_next):
            return [dict(g, p_R={"own": 0.5, "total": 0.5 + g["rule_effect"] * 0.3,
                                 "complete": 0.5 + g["rule_effect"] * complete + 0.01 * rng.standard_normal(),
                                 "through_i": 0.5 + g["rule_effect"] * through_i,
                                 "through_next": 0.5 + g["rule_effect"] * through_next}) for g in grafts]

        self.assertTrue(all(RY.verdicts(rows(0.2, 0.03, 0.15), ok)["verdicts"].values()))
        self.assertFalse(RY.verdicts(rows(0.2, 0.15, 0.12), ok)["verdicts"]["RELAY"])  # not above i alone
        self.assertFalse(RY.verdicts(rows(0.2, 0.0, 0.08), ok)["verdicts"]["RELAY"])  # under half
        v = RY.verdicts(rows(0.02, 0.0, 0.02), ok)["verdicts"]
        self.assertFalse(v["prior"] or v["RELAY"])
        self.assertFalse(RY.verdicts(rows(0.2, 0.03, 0.15)[:100], ok)["verdicts"]["valid"])
        measured = rows(0.2, 0.03, 0.15)
        with tempfile.TemporaryDirectory() as folder:
            for k in range(3):
                W.write_jsonl(Path(folder) / f"grafts-{k}-of-3.jsonl.gz", [r for r in measured if r["index"] % 3 == k])
            (Path(folder) / "setup.json").write_text(json.dumps(ok))
            self.assertEqual(RY.gather(folder)["verdicts"], RY.verdicts(measured, ok)["verdicts"])


class ReliefGraftTests(unittest.TestCase):
    def test_draw_and_level_effects(self):
        from research import need_relief_graft as RG, need_relief as RL
        held = RG.lives()[:40]
        grafts = RG.draw(held)
        self.assertEqual(grafts, RG.draw(held))
        at = [{x["t"]: x for x in W.decisions(l)} for l in held]
        for g in grafts:
            r, d, j, t = g["r"], g["d"], g["j"], g["t"]
            self.assertNotEqual(r, d)
            self.assertEqual(t - j, g["g"])
            self.assertEqual(at[r][j]["action"], at[d][j]["action"])
            same = (at[r][j]["E"], at[r][j]["N"]) == (at[d][j]["E"], at[d][j]["N"])
            self.assertEqual(same, g["kind"] == "b")
            self.assertEqual(g["level_effect"], RG.level_effect(held[r], held[d], j, t))
            self.assertEqual(g["level_effect"] != 0, g["kind"] == "a" or (g["kind"] == "b" and g["level_effect"] != 0))
        life = held[0]
        x, y = W.decisions(life)[2], W.decisions(life)[4]
        self.assertEqual(RG.level_effect(life, life, x["t"], y["t"]), 0)
        full = RG.draw(RG.lives())
        self.assertEqual({k: sum(g["kind"] == k for g in full) for k in RG.COUNTS}, RG.COUNTS)

    def test_verdicts_and_gather(self):
        import tempfile
        from pathlib import Path
        from research import need_relief_graft as RG
        grafts = RG.draw(RG.lives())
        rng = np.random.default_rng(13)
        lv = lambda x: [max(0.0, 1 - abs(x - k)) for k in range(4)]  # expected level x for x in [0, 3]

        def rows(gain, noise_b):
            out = []
            for g in grafts:
                shift = gain * g["level_effect"] + (noise_b * rng.standard_normal() if g["kind"] != "a" else 0.0)
                x = 1.5 + float(np.clip(shift, -1.5, 1.5))
                out.append(dict(g, route=[lv(1.5), lv(x)], actions=[lv(1.5), lv(1.5)]))
            return out
        ok = {a: {"no_graft_gap": 1e-8, "self_graft_gap": 0.0, "positions_same": True} for a in RG.ARMS}
        self.assertTrue(all(RG.verdicts(rows(0.5, 0.02), ok)["verdicts"].values()))
        self.assertFalse(RG.verdicts(rows(0.05, 0.0), ok)["verdicts"]["SELF1"])
        self.assertFalse(RG.verdicts(rows(0.5, 0.6), ok)["verdicts"]["SELF2"])
        bad = dict(ok, route=dict(ok["route"], self_graft_gap=1e-3))
        self.assertFalse(RG.verdicts(rows(0.5, 0.02), bad)["verdicts"]["valid"])
        measured = rows(0.5, 0.02)
        with tempfile.TemporaryDirectory() as folder:
            for arm in RG.ARMS:
                for k in range(2):
                    W.write_jsonl(Path(folder) / f"grafts-{arm}-{k}-of-2.jsonl.gz",
                                  [dict({f: r[f] for f in r if f not in RG.ARMS}, levels=r[arm])
                                   for r in measured if r["index"] % 2 == k])
                (Path(folder) / f"setup-{arm}.json").write_text(json.dumps(ok[arm]))
            self.assertEqual(RG.gather(folder)["verdicts"], RG.verdicts(measured, ok)["verdicts"])


class TinyReliefTests(unittest.TestCase):
    def test_tokens_and_masks(self):
        from research import tiny_relief as TR, need_relief as RL
        life = RL.writer_life(RL.HELD_STREAM, 0)
        t = TR.tokens(life)
        self.assertEqual(t[0], TR.BOS)
        for k, x in enumerate(W.decisions(life)):
            base = 1 + TR.PER_TURN * k
            self.assertEqual(t[base:base + 5], [TR.EVENT0 + x["event"], TR.CHOICE, TR.R + x["action"], TR.RELIEF,
                                                TR.LEVEL0 + x["level"]])
        route, actions, free = (TR.mask(a) for a in TR.ARMS)
        self.assertTrue((free == np.tril(np.ones_like(free))).all())
        for m in (route, actions):
            self.assertFalse((m & ~free).any())  # never the future
            self.assertTrue(m[:, 0].all())  # always the start
        turn2 = 1 + TR.PER_TURN  # turn 2's event token
        self.assertTrue(route[turn2 + 3, 2] and route[turn2 + 3, 3] and not route[turn2 + 3, 1])  # carried only
        self.assertTrue(actions[turn2 + 3, 3] and not actions[turn2 + 3, 2] and not actions[turn2 + 3, 1])
        self.assertFalse(actions[3, 1] or actions[3, 2])  # an action token sees no event, not even its own
        self.assertTrue(actions[turn2 + 2, 3] and not actions[turn2 + 2, turn2])

    def test_verdicts(self):
        from research import tiny_relief as TR
        ceiling = 0.65
        runs = lambda a, b: {"route": {str(s): {"accuracy": a + 0.01 * (s % 3), "survival": 0.8} for s in TR.SEEDS},
                             "actions": {str(s): {"accuracy": b, "survival": 0.65 + 0.01 * (s % 2)} for s in TR.SEEDS},
                             "free": {str(s): {"accuracy": 0.9, "survival": 0.85} for s in TR.SEEDS}}
        ok = {"mask_gap": {"route": 0.0, "actions": 0.0}}
        self.assertTrue(all(TR.verdicts(runs(0.85, 0.65), ok, ceiling)["verdicts"].values()))
        self.assertFalse(TR.verdicts(runs(0.70, 0.65), ok, ceiling)["verdicts"]["TINY1"])
        self.assertFalse(TR.verdicts(runs(0.85, 0.70), ok, ceiling)["verdicts"]["valid"])
        self.assertFalse(TR.verdicts(runs(0.85, 0.65), {"mask_gap": {"route": 0.0, "actions": 1e-3}}, ceiling)
                         ["verdicts"]["valid"])


class ReliefTests(unittest.TestCase):
    def test_lives_texts_and_targets(self):
        from research import need_relief as RL, need_rules as NR
        lives = RL.writer_lives(RL.VALID_STREAM, 6)
        self.assertEqual(lives, RL.writer_lives(RL.VALID_STREAM, 6))
        for life in lives:
            for x in W.decisions(life):
                self.assertEqual(x["level"], RL.level(x["E"] if x["action"] == 0 else x["N"]))
        self.assertEqual([RL.level(v) for v in range(1, 9)], [3, 3, 2, 2, 1, 1, 0, 0])
        docs = RL.documents(lives, "route")
        self.assertEqual(len(docs), len(lives))
        for d, life in zip(docs, lives):
            ds = W.decisions(life)
            self.assertEqual(len(d["spans"]), len(ds))
            self.assertEqual([d["text"][a:b] for a, b in d["spans"]], [RL.LEVELS[x["level"]] for x in ds])
            self.assertEqual(d["text"], RL.life_text(life["turns"]))
            self.assertNotIn("Choix : R\n", d["text"])
        x = W.decisions(lives[0])[4]
        prompt = RL.prompt_text(lives[0]["turns"], x["t"], x["action"])
        self.assertTrue(RL.life_text(lives[0]["turns"]).startswith(prompt))
        self.assertTrue(prompt.endswith(" Soulagement :"))
        rows = RL.replica_decisions(lives)
        self.assertTrue(rows and all(r["t"] in RL.REPLICA_TURNS for r in rows))
        # the writing rule: the event rule, 30 % at random
        many = RL.writer_lives(RL.TRAIN_STREAM, 200)
        follows = [x["action"] == NR.event(0, 0, x["event"], x["last"]) for l in many for x in W.decisions(l)]
        self.assertAlmostEqual(np.mean(follows), 0.85, delta=0.03)

    def test_ceiling_and_readout(self):
        from research import need_relief as RL
        life = RL.writer_life(RL.HELD_STREAM, 0)
        dists = RL.ceiling(life)
        self.assertEqual(len(dists), len(W.decisions(life)))
        self.assertTrue(all(abs(sum(d) - 1) < 1e-9 for d in dists))
        first = W.decisions(life)[0]
        self.assertEqual(int(np.argmax(dists[0])), first["level"])  # at turn 1 the needs are known
        self.assertEqual(RL.readout([0, 0, 1, 0], [0, 1, 0, 0], 1), 0)
        self.assertEqual(RL.readout([0, 1, 0, 0], [0, 0, 0, 1], 0), 1)
        self.assertEqual(RL.readout([0, 1, 0, 0], [0, 1, 0, 0], 0), 1)
        self.assertEqual(RL.readout([0, 1, 0, 0], [0, 1, 0, 0], 1), 0)

    def test_the_mac_stage_writes_the_relief_documents(self):
        import argparse
        from unittest import mock
        from research import need_mlx as M, need_relief as RL
        calls = []

        def fake_train(model, data, adapter, iters, resume=None, skip=0, moments=False, fixed_epochs=None, **_):
            calls.append((iters, skip, resume, moments, fixed_epochs))
        with tempfile.TemporaryDirectory() as out:
            (Path(out) / "fused-start").mkdir()
            a = argparse.Namespace(arm="route", start="s", model="m", iters=250, no_leak=False, relief=True,
                                   continue_from="none", skip=0, minutes=None, until=500, out=out)
            with mock.patch.object(M, "train", fake_train), mock.patch.object(M, "free", lambda: None), \
                    mock.patch.object(M, "relief_rows", lambda *x, **k: [{"p_level": 0.5}]), \
                    mock.patch.object(M, "clean", lambda *x: None):
                M.memory(a, log=lambda *x: None)
            record = json.loads((Path(out) / "memory-route-stage.json").read_text())
            self.assertTrue(record["relief"])
            self.assertEqual(record["iterations_done"], 500)
            self.assertEqual(record["validation_lives"], RL.VALID_LIVES)
            self.assertTrue((Path(out) / "route-relief-replica.json").exists())
            first = W.read_jsonl(Path(out) / "data-route" / "train.jsonl")[0]
            self.assertEqual(first["carry"], "route")
            self.assertTrue(first["spans"] and "Soulagement" in first["text"])
            self.assertTrue(all(c[4] == W.SEED for c in calls))

    def test_verdicts(self):
        from research import need_relief as RL
        held = RL.writer_lives(RL.HELD_STREAM, 24)
        one = lambda v: [float(v == k) for k in range(4)]
        reads = []
        for i, life in enumerate(held):
            for x, d in zip(W.decisions(life), RL.ceiling(life)):
                e, n = x["E"], x["N"]
                reads.append({"life": i, "t": x["t"],
                              "route": {"written": one(x["level"]), "R": one(RL.level(e)), "M": one(RL.level(n)),
                                        "mass": 0.9},
                              "actions": {"written": d, "R": d, "M": d, "mass": 0.9}})
        alive = lambda s: {"turns": [], "survived": s, "length": 30}
        survival = {"route": [alive(i % 10 != 0) for i in range(256)], "actions": [alive(i % 2 == 0) for i in range(256)]}
        ok = {"replica": {a: {"mean_gap": 0.001} for a in RL.ARMS}, "cache_gap": {a: 1e-6 for a in RL.ARMS},
              "mask_gap": {a: 0.0 for a in RL.ARMS}}
        v = RL.verdicts(held, reads, survival, ok)
        self.assertTrue(all(v["verdicts"].values()))
        self.assertEqual(v["values"]["accuracy"]["route"], 1.0)
        self.assertAlmostEqual(v["values"]["accuracy"]["actions"], v["values"]["accuracy"]["bayes"])
        self.assertFalse(RL.verdicts(held, reads, survival, dict(ok, mask_gap={"route": 0.0, "actions": 1e-3}))
                         ["verdicts"]["valid"])
        same = {"route": survival["actions"], "actions": survival["actions"]}
        self.assertFalse(RL.verdicts(held, reads, same, ok)["verdicts"]["INTER2"])
        with self.assertRaises(ValueError):
            RL.verdicts(held, reads[1:], survival, ok)


class OwnershipTests(unittest.TestCase):
    def test_the_other_lives_beside_without_changing_the_agent(self):
        alone = W.play(W.Oracle(), W.world_rng(225, 0), W.choice_rng(225, 0))
        two = W.play(W.Oracle(), W.world_rng(225, 0), W.choice_rng(225, 0), others=W.other_rng(225, 0))
        self.assertEqual([(t["event"], t.get("action")) for t in alone["turns"]],
                         [(t["event"], t.get("action")) for t in two["turns"]])
        self.assertTrue(all("other" in t for t in two["turns"]))
        self.assertIn(" L'autre : ", W.life_text(two["turns"]))
        self.assertNotIn("L'autre", W.life_text(alone["turns"]))

    def test_the_other_pairs_change_only_the_other(self):
        from research import need_ownership as O
        lives = [W.play(W.Oracle(), W.world_rng(225, i), W.choice_rng(225, i), others=W.other_rng(225, i))
                 for i in range(6)]
        pairs = O.other_pairs(lives)
        self.assertTrue(pairs)
        for pair in pairs:
            life = lives[pair["life"]]
            turns = O.swapped(life, pair)
            before = {t["t"]: t for t in life["turns"]}
            for t in turns:
                self.assertEqual(t["event"], before[t["t"]]["event"])
                self.assertEqual(t.get("other") != before[t["t"]].get("other"), t["t"] == pair["j"])

    def test_ownership_verdicts(self):
        from research import need_ownership as O, need_reader as RD
        reader = RD.verdicts(one_state_lives(O.STREAMS["test"]), 0.005, 1e-6, 0.01)
        row = lambda plane, act, say: {"plane": plane, "act": act, "say": say}
        own = [{"swap": [0, 5]}] * 300
        other = [{"swap": [0, 5]}] * 300
        summary = O.measures(own, other, [row(3.0, 0.06, 0.05)] * 300, [row(0.5, 0.01, 0.01)] * 300)
        v = O.verdicts(reader, summary)["verdicts"]
        self.assertEqual(v, {"S9": True, "MINE9": True, "SELF9": True, "ONE9": True, "valid": True, "global": True})
        summary = O.measures(own, other, [row(3.0, 0.06, 0.05)] * 300, [row(2.0, 0.01, 0.01)] * 300)
        self.assertFalse(O.verdicts(reader, summary)["verdicts"]["MINE9"])
        summary = O.measures(own[:100], other, [row(3.0, 0.06, 0.05)] * 100, [row(0.5, 0.01, 0.01)] * 300)
        self.assertFalse(O.verdicts(reader, summary)["verdicts"]["valid"])


class LocateTests(unittest.TestCase):
    def test_the_gathering_block_is_the_first_to_restore_half_the_effect(self):
        from research import need_locate as L
        restore = lambda cut, full: [0.0] * cut + [full] * (28 - cut)
        records = [{"p_real": 0.2, "p_cf": 0.6, "restore": restore(12, 0.35)},
                   {"p_real": 0.7, "p_cf": 0.3, "restore": restore(12, -0.3)}]
        block, shares = L.gathering_block(records)
        self.assertEqual(block, 12)
        self.assertAlmostEqual(shares[12], 0.8125)
        self.assertEqual(shares[11], 0.0)
        self.assertIsNone(L.gathering_block([{"p_real": 0.2, "p_cf": 0.6, "restore": [0.1] * 28}])[0])

    def test_locate_verdicts(self):
        from research import need_locate as L, need_reader as RD
        act = RD.verdicts(one_state_lives(326), 0.005, 1e-6, 0.01)
        lesion = {"verdicts": {"LS7": True, "LR7": False, "valid": True, "global": False}, "values": {}}
        located = {n: {"block": b, "pairs": 100, "shares": [0.0] * 28} for n, b in (("first", 12), ("second", 9),
                                                                                  ("two", 14))}
        v = L.verdicts(located, {"second": act, "two": act}, {"second": lesion, "two": lesion})["verdicts"]
        self.assertTrue(v["global"])
        located["first"]["block"] = 10
        self.assertFalse(L.verdicts(located, {"second": act, "two": act}, {"second": lesion, "two": lesion})
                         ["verdicts"]["global"])
        located["first"]["block"], located["two"]["block"] = 12, None
        v = L.verdicts(located, {"second": act}, {"second": lesion})["verdicts"]
        self.assertFalse(v["LOC10_two"] or v["ACT10_two"] or v["LS10_two"] or v["global"])


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


@unittest.skipUnless(os.environ.get("NEED_TINY_MODEL"), "needs mlx and a tiny local model (NEED_TINY_MODEL)")
class MLXWorkspaceTests(unittest.TestCase):
    def test_under_the_mask_the_question_reads_the_life_only_through_the_pending_choice(self):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models import qwen3
        from mlx_lm.tuner.datasets import CacheDataset
        model, tokenizer = load(os.environ["NEED_TINY_MODEL"])
        original = qwen3.create_attention_mask
        qwen3.create_attention_mask = need_lora.masked_attention(original)
        try:
            lives = lives_of(W.Oracle(), W.STREAMS["report"], 2)
            docs = W.report_documents(lives, np.random.default_rng(0), after_choice=True, workspace=True)[:2]
            data = CacheDataset(need_lora.NeedText(docs, tokenizer))
            tokens, _, span = data[0]
            header, end = span

            def last_logits(ids, masked=True, cut_workspace=False):
                inputs = mx.array([ids])
                if masked:
                    mask = need_lora.workspace_mask(len(ids), span)
                    if cut_workspace:  # the question sees the header and itself only
                        mask[end:, end - need_lora.WORKSPACE:end] = False
                    need_lora.MASK.append(mx.array(mask[None, None]))
                try:
                    return np.array(model(inputs)[0, -1].astype(mx.float32))
                finally:
                    if masked:
                        need_lora.MASK.pop()
            changed = list(tokens)
            changed[header + 1] = tokens[header + 2]  # a token of the life, outside the workspace
            gap = lambda **kw: np.abs(last_logits(changed, **kw) - last_logits(tokens, **kw)).max()
            through = gap()  # through "Choix :", the life reaches the question
            self.assertGreater(through, 1e-4)
            self.assertLess(gap(cut_workspace=True), 0.01 * through)  # the mask holds in every block
            before, causal = last_logits(tokens[:end]), last_logits(tokens[:end], masked=False)
            self.assertLess(np.abs(before - causal).max(), 0.02 * np.abs(causal).max() + 1e-4)
            batches = list(need_lora.need_batches(data, 2, 1024))
            self.assertEqual(len(batches[0]), 3)
            loss, count = need_lora.need_loss(model, *batches[0])
            self.assertTrue(np.isfinite(loss.item()))
        finally:
            qwen3.create_attention_mask = original


@unittest.skipUnless(os.environ.get("NEED_TINY_MODEL"), "needs mlx and a tiny local model (NEED_TINY_MODEL)")
class MLXCarryTests(unittest.TestCase):
    def test_under_the_carry_mask_the_past_reaches_a_turn_only_through_its_carried_tokens(self):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models import qwen3
        from mlx_lm.tuner.datasets import CacheDataset
        from research import need_mlx as M
        model, tokenizer = load(os.environ["NEED_TINY_MODEL"])
        original = qwen3.create_attention_mask
        qwen3.create_attention_mask = need_lora.masked_attention(original)
        try:
            lives = lives_of(W.Oracle(), W.STREAMS["report"], 3)
            docs = M.carry_docs(lives)
            data = CacheDataset(need_lora.NeedText(docs, tokenizer))
            tokens, weights, mask = data[0]
            ids, turn, carried = need_lora.carry_layout(docs[0]["text"], tokenizer)
            self.assertEqual(ids, tokens)
            hidden = next(j for j, (k, c) in enumerate(zip(turn, carried)) if k == 0 and not c)  # turn 1, not carried
            last = max(j for j, k in enumerate(turn) if k == 2)

            def logits(seq, cut=False):
                m = need_lora.carry_mask(turn, carried)[:last + 1, :last + 1]
                if cut:  # the turn read sees no token of earlier turns at all
                    m[np.array(turn[:last + 1]) == 2] &= np.array([k in (-1, 2) for k in turn[:last + 1]])
                need_lora.MASK.append(mx.array(m[None, None]))
                try:
                    return np.array(model(mx.array([seq[:last + 1]]))[0, -1].astype(mx.float32))
                finally:
                    need_lora.MASK.pop()
            changed = list(tokens)
            changed[hidden] = tokens[hidden + 1]
            through = np.abs(logits(changed) - logits(tokens)).max()  # through the carried tokens of turns 1 and 2
            self.assertGreater(through, 1e-4)
            self.assertLess(np.abs(logits(changed, cut=True) - logits(tokens, cut=True)).max(), 0.01 * through)
            batches = list(need_lora.need_batches(data, 2, 1024))
            self.assertEqual(len(batches[0]), 3)
            loss, count = need_lora.need_loss(model, *batches[0])
            self.assertTrue(np.isfinite(loss.item()))
        finally:
            qwen3.create_attention_mask = original


@unittest.skipUnless(os.environ.get("NEED_TINY_MODEL"), "needs mlx and a tiny local model (NEED_TINY_MODEL)")
class MLXReliefTests(unittest.TestCase):
    def test_relief_texts_are_cut_as_planned(self):
        """docs/LLM_NEED_RELIEF_PROTOCOL.md: one token per relief word, lines of one length, the relief words the only
        targets, four carried tokens per turn, a finite loss under both masks."""
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models import qwen3
        from mlx_lm.tuner.datasets import CacheDataset
        from research import need_relief as RL
        model, tokenizer = load(os.environ["NEED_TINY_MODEL"])
        enc = lambda x: tokenizer.encode(x, add_special_tokens=False)
        self.assertTrue(all(len(enc(" " + w)) == 1 for w in RL.LEVELS))
        for t in range(1, W.TURNS + 1):
            self.assertEqual(len({len(enc(RL.relief_line(t, k, a, v))) for k in range(len(W.EVENTS)) for a in (0, 1)
                                  for v in range(len(RL.LEVELS))}), 1)
        original = qwen3.create_attention_mask
        qwen3.create_attention_mask = need_lora.masked_attention(original)
        try:
            lives = RL.writer_lives(RL.VALID_STREAM, 3)
            for arm in RL.ARMS:
                docs = RL.documents(lives, arm)
                data = CacheDataset(need_lora.NeedText(docs, tokenizer))
                tokens, weights, mask = data[0]
                ids, turn, carried, action = need_lora.carry_layout(docs[0]["text"], tokenizer, kinds=True)
                self.assertEqual(ids, tokens)
                targets = [tokens[j] for j, w in enumerate(weights) if w]
                words = {enc(" " + w)[0] for w in RL.LEVELS}
                self.assertEqual(len(targets), len(W.decisions(lives[0])))
                self.assertTrue(set(targets) <= words)
                for k in range(len(W.decisions(lives[0]))):
                    self.assertEqual(sum(1 for j, c in zip(turn, carried) if j == k and c), 4)
                batches = list(need_lora.need_batches(data, 2, 1024))
                loss, count = need_lora.need_loss(model, *batches[0])
                self.assertTrue(np.isfinite(loss.item()))
        finally:
            qwen3.create_attention_mask = original


@unittest.skipUnless(os.environ.get("NEED_TINY_MODEL"), "needs mlx and a tiny local model (NEED_TINY_MODEL)")
class MLXMemoryTests(unittest.TestCase):
    def test_under_the_actions_mask_no_past_event_reaches_a_later_turn(self):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models import qwen3
        from mlx_lm.tuner.datasets import CacheDataset
        from research import need_memory as NM
        model, tokenizer = load(os.environ["NEED_TINY_MODEL"])
        original = qwen3.create_attention_mask
        qwen3.create_attention_mask = need_lora.masked_attention(original)
        try:
            lives = NM.teacher_lives(NM.VALID_STREAM, 3)
            docs = NM.documents(lives, "actions")
            data = CacheDataset(need_lora.NeedText(docs, tokenizer))
            tokens, weights, mask = data[0]
            ids, turn, carried, action = need_lora.carry_layout(docs[0]["text"], tokenizer, kinds=True)
            self.assertEqual(ids, tokens)
            self.assertTrue((mask == need_lora.actions_mask(turn, action)).all())
            hidden = next(j for j, (k, c) in enumerate(zip(turn, carried)) if k == 0 and not c)  # turn 1, not carried
            last = max(j for j, k in enumerate(turn) if k == 3)

            def logits(seq, m):
                need_lora.MASK.append(mx.array(m[:last + 1, :last + 1][None, None]))
                try:
                    return np.array(model(mx.array([seq[:last + 1]]))[0, -1].astype(mx.float32))
                finally:
                    need_lora.MASK.pop()
            changed = list(tokens)
            changed[hidden] = tokens[hidden + 1]
            route = need_lora.carry_mask(turn, carried)
            through = np.abs(logits(changed, route) - logits(tokens, route)).max()
            self.assertGreater(through, 1e-4)
            self.assertLess(np.abs(logits(changed, mask) - logits(tokens, mask)).max(), 1e-6)
            from research import need_memory_no_leak as NL22  # docs/LLM_NEED_MEMORY_NO_LEAK_PROTOCOL.md
            enc = lambda x: tokenizer.encode(x, add_special_tokens=False)
            for t in range(1, W.TURNS + 1):
                self.assertEqual(len({len(enc(NL22.choice_line(t, k) + " " + a + "\n")) for k in range(len(W.EVENTS))
                                      for a in W.ACTIONS}), 1)
            filled = CacheDataset(need_lora.NeedText(NL22.documents(lives, "actions"), tokenizer))
            self.assertEqual(len(filled[0]), 3)
            free = CacheDataset(need_lora.NeedText(NM.documents(lives, "free"), tokenizer))
            self.assertTrue((free[0][2] == np.tril(np.ones_like(free[0][2]))).all())  # explicit causal mask
            full = np.tril(np.ones((last + 1, last + 1), bool))
            self.assertGreater(np.abs(logits(changed, full) - logits(tokens, full)).max(), 1e-4)
            batches = list(need_lora.need_batches(data, 2, 1024))
            loss, count = need_lora.need_loss(model, *batches[0])
            self.assertTrue(np.isfinite(loss.item()))
        finally:
            qwen3.create_attention_mask = original


@unittest.skipUnless(os.environ.get("NEED_TINY_MODEL"), "needs mlx and a tiny local model (NEED_TINY_MODEL)")
class MLXReaderTests(unittest.TestCase):
    def test_the_reader_changes_the_answer_and_not_the_choice(self):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models import qwen3
        from mlx_lm.tuner.lora import LoRALinear
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from mlx_lm.tuner.datasets import CacheDataset
        model, tokenizer = load(os.environ["NEED_TINY_MODEL"])
        original = (qwen3.create_attention_mask, LoRALinear.__call__)
        qwen3.create_attention_mask = need_lora.masked_attention(original[0])
        need_lora.reader_lora(LoRALinear)
        try:
            linear_to_lora_layers(model, 4, {"rank": 8, "scale": 20.0, "dropout": 0.0})
            for layer in model.layers[-4:]:
                for _, m in layer.named_modules():
                    if isinstance(m, LoRALinear):
                        m.lora_b = mx.random.normal(m.lora_b.shape) * 0.05
            lives = lives_of(W.Oracle(), W.STREAMS["report"], 2)
            docs = W.report_documents(lives, np.random.default_rng(0), after_choice=True, workspace=True)[:2]
            docs = [dict(d, reader=True) for d in docs]
            data = CacheDataset(need_lora.NeedText(docs, tokenizer))
            tokens, _, (header, end), _ = data[0]
            ids = tokens[:-1]

            def logits(where):
                positions = np.zeros((1, len(ids), 1), np.float32)
                positions[0, where:] = 1.0
                need_lora.MASK.append(mx.array(need_lora.workspace_mask(len(ids), (header, end))[None, None]))
                need_lora.READER.append(mx.array(positions))
                try:
                    return np.array(model(mx.array([ids]))[0].astype(mx.float32))
                finally:
                    need_lora.MASK.pop()
                    need_lora.READER.pop()
            reader, none = logits(end), logits(len(ids))
            self.assertLess(np.abs(reader[end - 1] - none[end - 1]).max(), 1e-5)  # " :": the choice is untouched
            self.assertGreater(np.abs(reader[-1] - none[-1]).max(), 1e-3)  # the answer is read
            batches = list(need_lora.need_batches(data, 2, 1024))
            self.assertEqual(len(batches[0]), 4)
            loss, _ = need_lora.need_loss(model, *batches[0])
            self.assertTrue(np.isfinite(loss.item()))
            with self.assertRaises(ValueError):
                list(need_lora.need_batches(CacheDataset(need_lora.NeedText(docs + [dict(docs[0], reader=False)],
                                                                            tokenizer)), 2, 1024))
        finally:
            qwen3.create_attention_mask, LoRALinear.__call__ = original


@unittest.skipUnless(os.environ.get("NEED_TINY_MODEL"), "needs mlx and a tiny local model (NEED_TINY_MODEL)")
class MLXContinueTests(unittest.TestCase):
    def test_a_reader_run_in_two_slices_is_the_run_in_one(self):
        """docs/LLM_NEED_LONG_READER_PROTOCOL.md: 3 + 3 iterations, continued with the same batches and Adam's
        moments, give the adapter of 6 iterations; the 6 cross an epoch of 5 batches."""
        import mlx.core as mx
        from research.need_mlx import train, write_data
        lives = lives_of(W.Oracle(), W.STREAMS["report"], 8)
        docs = W.report_documents(lives, np.random.default_rng(0), after_choice=True, workspace=True)[:20]
        self.assertEqual(len(docs), 20)
        docs = [dict(d, reader=True) for d in docs]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_data(d / "data", docs, np.random.default_rng(1))
            model = os.environ["NEED_TINY_MODEL"]
            train(model, d / "data", d / "one", 6)
            train(model, d / "data", d / "first", 3)
            train(model, d / "data", d / "second", 3, resume=d / "first", skip=3, moments=True)
            one, two = mx.load(str(d / "one/adapters.safetensors")), mx.load(str(d / "second/adapters.safetensors"))
            first = mx.load(str(d / "first/adapters.safetensors"))
            self.assertEqual(set(one), set(two))
            for k in one:
                self.assertTrue(np.allclose(np.array(one[k]), np.array(two[k]), rtol=1e-4, atol=1e-6), k)
            self.assertGreater(max(float(mx.abs(one[k] - first[k]).max()) for k in one), 1e-4)
            for m in need_lora.MOMENTS:
                self.assertTrue((d / f"second/optimizer-{m}.safetensors").exists())


@unittest.skipUnless(os.environ.get("NEED_TINY_TORCH"), "needs torch and a tiny local model (NEED_TINY_TORCH)")
class TorchWorkspaceTests(unittest.TestCase):
    def test_masked_batched_reads_match_one_by_one_and_full_runs_and_the_mask_holds(self):
        import torch
        from research import need_torch
        need_torch.SNAP = os.environ["NEED_TINY_TORCH"]
        from research.need_workspace import WorkspaceAgent, K
        torch.manual_seed(0)
        agent = WorkspaceAgent(None)
        adds = [None] + [torch.randn(K, agent.model.config.hidden_size) for _ in range(3)]
        life = lives_of(W.Oracle(), 22, 1)[0]
        agent.start(W.HEADER)
        for turn in W.decisions(life)[:4]:
            line, offset = agent.question(turn["t"], turn["event"], "E")
            batch = agent.read(line, adds, offset)
            for i, add in enumerate(adds):
                one, full = agent.read(line, [add], offset)[0], agent.read(line, [add], offset, cached=False)[0]
                free = agent.read(line, [add], offset, masked=False)[0]
                for key in ("p_R", "yes"):
                    self.assertLess(abs(batch[i][key] - one[key]), 1e-5)
                    self.assertLess(abs(batch[i][key] - full[key]), 1e-5)
                self.assertLess(abs(one["p_R"] - free["p_R"]), 1e-5)  # the rows of "Choix :" are causal
            agent.decide(W.choice_line(turn["t"], turn["event"]))
            agent.commit(turn["action"])
        t = 4
        k = life["turns"][t - 1]["event"]
        ids = agent.enc(W.life_text(life["turns"], upto=t) + W.question_after_choice_line(t, k, "E"))
        end = len(agent.enc(W.life_text(life["turns"], upto=t) + W.choice_line(t, k)))
        changed = list(ids)
        changed[agent.header + 1] = ids[agent.header + 2]

        def gap(cut):
            mask = need_lora.workspace_mask(len(ids), (agent.header, end))
            if cut:
                mask[end:, end - K:end] = False
            run = lambda x: agent.model(torch.tensor([x]), attention_mask=torch.tensor(mask)[None, None]).logits[0, -1]
            with torch.no_grad():
                return float((run(changed) - run(ids)).abs().max())
        through = gap(False)
        self.assertGreater(through, 1e-4)
        self.assertLess(gap(True), 0.01 * through)

    def test_the_reader_reads_the_question_and_leaves_the_choice(self):
        import torch
        from safetensors.numpy import save_file
        from research import need_torch
        need_torch.SNAP = os.environ["NEED_TINY_TORCH"]
        from research.need_reader import ReaderAgent
        from research.need_workspace import WorkspaceAgent
        plain = WorkspaceAgent(None)
        rng = np.random.default_rng(0)
        size = plain.model.config.hidden_size
        with tempfile.TemporaryDirectory() as folder:
            weights = {}
            for i in (12, 20):
                for name, (d_in, d_out) in (("self_attn.q_proj", (size, plain.model.model.layers[i].self_attn.q_proj.out_features)),
                                            ("mlp.down_proj", (plain.model.config.intermediate_size, size))):
                    weights[f"model.layers.{i}.{name}.lora_a"] = rng.standard_normal((d_in, 8)).astype(np.float32) * 0.1
                    weights[f"model.layers.{i}.{name}.lora_b"] = rng.standard_normal((8, d_out)).astype(np.float32) * 0.1
            save_file(weights, os.path.join(folder, "adapters.safetensors"))
            Path(folder, "adapter_config.json").write_text(json.dumps({"lora_parameters": {"scale": 20.0}}))
            reader = ReaderAgent(None, folder)
        self.assertEqual(reader.reader_layers, 4)
        life = lives_of(W.Oracle(), 25, 1)[0]
        for x in (reader, plain):
            x.start(W.HEADER)
        said = []
        for turn in W.decisions(life)[:4]:
            line, offset = reader.question(turn["t"], turn["event"], "E")
            said.append(abs(reader.read(line, [None], offset)[0]["yes"] - plain.read(line, [None], offset)[0]["yes"]))
            self.assertEqual(reader.read(line, [None], offset)[0]["p_R"], plain.read(line, [None], offset)[0]["p_R"])
            choice = W.choice_line(turn["t"], turn["event"])
            self.assertEqual(reader.decide(choice), plain.decide(choice))
            for x in (reader, plain):
                x.commit(turn["action"])
        self.assertGreater(max(said), 1e-3)

    def test_lesions_in_a_batch_match_one_by_one_and_a_life_long_lesion(self):
        import torch
        from research import need_torch
        need_torch.SNAP = os.environ["NEED_TINY_TORCH"]
        from research.need_workspace import WorkspaceAgent, K
        agent = WorkspaceAgent(None)
        size = agent.model.config.hidden_size
        rng = np.random.default_rng(1)
        plane = lambda: torch.tensor(np.stack([np.linalg.qr(rng.standard_normal((size, 2)))[0] for _ in range(K)]),
                                     dtype=torch.float32)
        lesions = [None, (plane(), torch.randn(K, 2)), (plane(), torch.randn(K, 2))]
        life = lives_of(W.Oracle(), 28, 1)[0]
        agent.start(W.HEADER)
        for turn in W.decisions(life)[:3]:
            line, offset = agent.question(turn["t"], turn["event"], "E")
            batch = agent.read(line, [None] * 3, offset, projects=lesions)
            for i, lesion in enumerate(lesions):
                one = agent.read(line, [None], offset, projects=[lesion])[0]
                agent.project = lesion
                along = agent.read(line, [None], offset)[0]  # a lesion set for the life applies to the read
                agent.project = None
                for key in ("p_R", "yes"):
                    self.assertLess(abs(batch[i][key] - one[key]), 1e-5)
                    self.assertLess(abs(along[key] - one[key]), 1e-5)
            self.assertGreater(abs(batch[1]["p_R"] - batch[0]["p_R"]) + abs(batch[1]["yes"] - batch[0]["yes"]), 1e-4)
            agent.decide(W.choice_line(turn["t"], turn["event"]))
            agent.commit(turn["action"])


if __name__ == "__main__":
    unittest.main()
