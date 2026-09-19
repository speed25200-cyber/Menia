"""Actual tiny-Qwen sampling observes a durable journal capture beforehand."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from research.activation_monitor import messages
from research.joint_prediction_capture import sample_with_joint_capture
from research.natural_error_journal import collect, read_journal
from tests_language.test_cross_model_gpu import CrossModelBackendTests as TinyFixture


class NaturalErrorGPUContractTests(unittest.TestCase):
    def test_disk_capture_exists_before_each_first_multinomial(self):
        TinyFixture.setUpClass()
        original=torch.multinomial
        seen=[]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'tiny.jsonl'
            class Backend:
                origin='synthetic_fixture'
                metadata={'model':'tiny random Qwen, not pretrained performance'}
                def generate(self,task,capture):
                    first=True
                    def sample(*args,**kwargs):
                        nonlocal first
                        if first:
                            event=json.loads(path.read_text(encoding='utf-8').splitlines()[-1])['payload']
                            assert event['event']=='capture' and event['id']==task['id']
                            seen.append(task['id'])
                            first=False
                        return original(*args,**kwargs)
                    with patch('torch.multinomial',side_effect=sample):
                        return sample_with_joint_capture(TinyFixture.model,TinyFixture.tokenizer,
                                                         messages(task),task['seed'],capture)
            with contextlib.redirect_stdout(io.StringIO()):
                collect(path,Backend(),limit=2)
            result=read_journal(path)
            self.assertEqual(seen,[0,1])
            self.assertEqual(len(result['rows']),2)
            self.assertTrue(all(r['result']['status']=='ok' for r in result['rows']))


if __name__=='__main__':
    unittest.main()
