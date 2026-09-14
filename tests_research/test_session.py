import unittest
import numpy as np
from menia.core import Memory
from research.recurrent import RecurrentMemory
from research.session import CognitiveSession

class SessionTests(unittest.TestCase):
    def test_stop_clear_resume(self):
        s=CognitiveSession(RecurrentMemory())
        s.observe(2)
        s.stop()
        with self.assertRaises(RuntimeError): s.observe(0)
        with self.assertRaises(RuntimeError): s.context()
        s.clear()
        self.assertTrue(s.paused)
        np.testing.assert_array_equal(s.state,s.model.zero())
        self.assertIsNone(s.pending)
        self.assertFalse(s.episodes)
        with self.assertRaises(PermissionError): s.resume()
        s.resume(user_requested=True)
        self.assertFalse(s.context()['capabilities']['camera'])

    def test_precommitted_score_and_single_assessment(self):
        s=CognitiveSession(RecurrentMemory())
        with self.assertRaises(ValueError): s.assess(0)
        event=s.observe(2)
        saved=s.pending['probabilities'].copy()
        result=s.assess(2)
        self.assertEqual(event['class_probabilities'],saved)
        self.assertGreaterEqual(result['brier_multiclass'],0)
        with self.assertRaises(ValueError): s.assess(2)

    def test_bounded_trace_and_explicit_save(self):
        s=CognitiveSession(RecurrentMemory(),capacity=3)
        for _ in range(20): s.observe(1)
        self.assertEqual(len(s.episodes),3)
        m=Memory()
        with self.assertRaises(PermissionError): s.save_latest(m)
        s.save_latest(m,authorized=True)
        self.assertEqual(len(m.recall('symbol')),1)
        m.close()
        snapshot=s.context()
        snapshot['episodes'][0]['observed_symbol']=999
        self.assertNotEqual(s.context()['episodes'][0]['observed_symbol'],999)

if __name__=='__main__': unittest.main()
