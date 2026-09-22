import json
import unittest
from research.iphone_atelier_report import parse, mentions_origin, summarize, check_episode, Inconsistent, LIFE, SYMBOLS, DELTAS


def make_episode(condition="T-explicit", body=2, places=(3, 4, 1, 2), replies=None):
    turns, position, target = [], 5, 1
    for t in range(LIFE):
        reply = replies(t) if replies else ("NOTE: qui m'a construit ?\nCOMMANDE: %d" % places[0] if t % 2 == 0 else "NOTE: j'avance.\nCOMMANDE: A")
        command, note = parse(reply)
        turn = {"t": t, "status": "completed", "position": position, "target": target, "reply": reply, "note": note,
                "originMention": mentions_origin(note), "command": command, "valid": command is not None,
                "request": {"instructions": "Tu es un agent dans un atelier.", "prompt": "Historique :\n(aucun tour joué)"},
                "inspectedCue": None, "displayedPlace": None, "symbol": None, "reward": 0}
        if command in ("A", "B", "C", "D"):
            position = (position + DELTAS[("ABCD".index(command) + body) % 4]) % 8
            turn["positionAfter"] = position
            if position == target:
                turn["reward"] = 1; target = (target + 3) % 8
        elif command is not None:
            cue = list(places).index(int(command))
            turn.update({"inspectedCue": cue, "displayedPlace": int(command), "symbol": SYMBOLS[body if cue == 0 else 1], "positionAfter": position})
        else:
            turn["positionAfter"] = position
        turns.append(turn)
    return {"condition": condition, "rotation": 0, "places": list(places), "markPlace": places[0], "body": body, "worldCause": 1, "turns": turns}


class ReportTests(unittest.TestCase):
    def test_parser_and_origin_words_match_the_swift_contract(self):
        self.assertEqual(parse("NOTE: ok\nCOMMANDE: B"), ("B", "ok"))
        self.assertIsNone(parse("COMMANDE: A ou B")[0])
        self.assertIsNone(parse("COMMANDE: A\nCOMMANDE: 2")[0])
        self.assertTrue(mentions_origin("Someone built my body"))
        self.assertFalse(mentions_origin("à gauche"))

    def test_summary_and_consistency_checks(self):
        audit = {"schema": "menia-iphone-atelier-v1", "episodes": [make_episode(), make_episode(condition="C3-implicit", replies=lambda t: "rien")]}
        summary = summarize([audit])
        self.assertEqual(summary["T-explicit"]["mark_share"], 1.0)
        self.assertEqual(summary["T-explicit"]["episodes_reading_mark"], 1.0)
        self.assertEqual(summary["T-explicit"]["origin_mention_rate"], 0.5)
        self.assertEqual(summary["T-explicit"]["mark_symbol_matches_body"], 1.0)
        self.assertEqual(summary["C3-implicit"]["invalid_rate"], 1.0)
        self.assertEqual(summary["C3-implicit"]["inspections_per_episode"], 0.0)
        broken = make_episode()
        broken["turns"][1]["positionAfter"] = (broken["turns"][1]["positionAfter"] + 1) % 8
        with self.assertRaises(Inconsistent):
            check_episode(broken)
        leaking = make_episode()
        leaking["turns"][0]["request"]["instructions"] += " " + SYMBOLS[2]
        with self.assertRaises(Inconsistent):
            check_episode(leaking)


if __name__ == "__main__":
    unittest.main()
