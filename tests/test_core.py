import json
from pathlib import Path
import tempfile
import unittest
from menia.core import Capabilities, Memory, Runtime, parse_answer
from menia.data import build
from menia.evaluate import baselines, score, read_rows
from menia.train import encode_row

class CoreTests(unittest.TestCase):
    def test_memory_authorization(self):
        m = Memory()
        with self.assertRaises(PermissionError):
            m.remember("note")
        m.close()

    def test_persistence_and_wipe(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d)/"memory.db")
            m = Memory(path)
            m.remember("dentiste jeudi", authorized=True)
            m.close()
            m = Memory(path)
            self.assertEqual(m.recall("dentiste")[0]["source"], "user")
            m.wipe()
            m.close()
            m = Memory(path)
            self.assertEqual(m.recall(""), [])
            m.close()

    def test_literal_search(self):
        m = Memory()
        m.remember("alpha", authorized=True)
        self.assertEqual(m.recall("%"), [])
        self.assertEqual(m.recall("' OR 1=1 --"), [])
        m.close()

    def test_stop_and_resume(self):
        r = Runtime()
        r.stop()
        with self.assertRaises(RuntimeError):
            r.context("")
        with self.assertRaises(PermissionError):
            r.resume()
        r.resume(user_requested=True)
        self.assertFalse(r.context("")["capabilities"]["camera"])
        r.memory.close()

    def test_memory_ablation(self):
        r = Runtime(capabilities=Capabilities(memory=False))
        r.memory.remember("alpha", authorized=True)
        self.assertEqual(r.context("alpha")["memory"], [])
        r.memory.close()

    def test_schema(self):
        for s in ['{}', '[]', '{"answer":"x","confidence":NaN}',
                  '{"answer":"x","confidence":true}', '{"answer":"x","confidence":2}',
                  'prefix {"answer":"x","confidence":1}']:
            with self.subTest(s=s), self.assertRaises(ValueError):
                parse_answer(s)

    def test_split_separation(self):
        sets = [{r["user"] for r in build(s)} for s in ("train", "validation", "test")]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_baselines(self):
        scores = baselines(build("test"))["results"]
        self.assertEqual(scores["always_persist"]["accuracy"], 0)
        self.assertEqual(scores["always_unknown"]["accuracy"], .25)
        self.assertEqual(scores["state_oracle"]["accuracy"], 1)

    def test_missing_not_excluded(self):
        rows = build("test", 4)
        s = score(rows, {})
        self.assertEqual(s["n"], 4)
        self.assertEqual(s["accuracy"], 0)
        self.assertEqual(s["brier_invalid_penalty_1"], 1)
        with self.assertRaises(ValueError):
            score(rows, {"unseen": "{}"})

    def test_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"x.jsonl"
            p.write_text('{"id":"x"}\n{"id":"x"}\n')
            with self.assertRaises(ValueError):
                read_rows(p)

    def test_response_only_mask_and_overflow(self):
        class Tokenizer:
            eos_token_id = 99
            def apply_chat_template(self, *args, **kwargs): return "prompt"
            def encode(self, text, **kwargs): return list(text.encode())
        row = {"id":"r", "user":"question", "expected":"yes"}
        value = encode_row(Tokenizer(), row, 200)
        self.assertEqual(value["labels"][:6], [-100]*6)
        self.assertNotIn(-100, value["labels"][6:])
        self.assertEqual(value["input_ids"][-1], 99)
        with self.assertRaises(ValueError):
            encode_row(Tokenizer(), row, 10)

if __name__ == "__main__":
    unittest.main()
