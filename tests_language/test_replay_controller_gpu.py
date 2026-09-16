"""Run the routing loop through a tiny random Qwen; no pretrained results."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from research import replay_controller as experiment
from research.replay_controller_gpu import ReplayBackend, sample_with_states
from tests_language import test_cross_model_gpu as fixtures
from tests_research.test_replay_controller import SMALL


class ReplayBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        fixtures.CrossModelBackendTests.setUpClass()

    def test_actual_qwen_forward_collects_before_answer_and_keeps_weights(self):
        class Tiny(ReplayBackend):
            origin='synthetic_fixture'
            def __init__(self):
                self.model=fixtures.CrossModelBackendTests.model
                self.tokenizer=fixtures.CrossModelBackendTests.tokenizer
                self.metadata={'fixture':'tiny random Qwen, not pretrained performance'}
            def generate(self,task,capture):
                return sample_with_states(self.model,self.tokenizer,task,capture,
                                          settings=dict(experiment.SETTINGS,max_new_tokens=3))
        backend=Tiny()
        before={k:v.detach().clone() for k,v in backend.model.state_dict().items()}
        with patch.dict(experiment.COUNTS,SMALL,clear=True),tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.jsonl'
            with contextlib.redirect_stdout(io.StringIO()): experiment.collect(path,backend)
            result=experiment.analyze(path)
            self.assertTrue(result['complete']); self.assertEqual(result['updates'],['round1','round2','round3'])
        for key,value in backend.model.state_dict().items(): self.assertTrue(torch.equal(value,before[key]))

    def test_failed_durable_capture_removes_hooks_and_prevents_answer(self):
        model=fixtures.CrossModelBackendTests.model
        tokenizer=fixtures.CrossModelBackendTests.tokenizer
        head_calls=[]
        handle=model.lm_head.register_forward_pre_hook(lambda *args:head_calls.append(1))
        before=[len(m._forward_hooks) for m in model.modules()]
        try:
            def fail(state): raise OSError('Simulated journal write failure')
            with self.assertRaises(OSError):
                sample_with_states(model,tokenizer,dict(question='hello hello hello hello',seed=41),fail,
                                   settings=dict(experiment.SETTINGS,max_new_tokens=3))
            self.assertEqual(head_calls,[])
            self.assertEqual(before,[len(m._forward_hooks) for m in model.modules()])
        finally:
            handle.remove()


if __name__=='__main__': unittest.main()
