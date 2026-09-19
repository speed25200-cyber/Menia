import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from research import composition_diagnostic as s
from research import composition_diagnostic_gpu as runtime
from research.native_localization_gpu import install_adapters, adapter_state, file_hash
from tests_language.test_learning_diagnostic_gpu import model, tokenizer
from tests_research.test_composition_diagnostic import calibration_fixture


class FrozenDiagnosticRuntimeTests(unittest.TestCase):
    def test_frozen_resume_noop_and_tamper_on_tiny_qwen(self):
        with tempfile.TemporaryDirectory() as d, contextlib.ExitStack() as stack, contextlib.redirect_stdout(io.StringIO()):
            torch.set_num_threads(1)
            stack.enter_context(patch.dict(s.CONFIG, dict(blocksPerReplication=2, replications=1, resamples=10, rank=2)))
            parent, composition = Path(d)/'parent.jsonl', Path(d)/'composition.jsonl'
            parent.write_bytes(b'fixture parent'); composition.write_bytes(b'fixture composition')
            frozen = {}
            for i, arm in enumerate(s.ARMS[1:]):
                net = model(36, 8); mods = install_adapters(net, rank=2)
                with torch.no_grad():
                    for m in mods.values(): m.b.fill_((i+1)*.002)
                key = 'r0-'+arm; path = runtime.checkpoint_path(parent, composition, key)
                save_file(adapter_state(mods), str(path)); frozen[key] = file_hash(path)
            before = {k: runtime.checkpoint_path(parent, composition, k).read_bytes() for k in frozen}
            stack.enter_context(patch.object(s, 'PARENT_SHA256', file_hash(parent)))
            stack.enter_context(patch.object(s, 'COMPOSITION_SHA256', file_hash(composition)))
            stack.enter_context(patch.object(s, 'checkpoints', return_value=frozen))
            stack.enter_context(patch.object(s, 'calibration', side_effect=lambda: calibration_fixture(frozen)))
            stack.enter_context(patch('research.cross_model_gpu.environment', return_value={'gpu':'CPU synthetic fixture'}))
            tok = tokenizer(); tok.add_tokens(['3'])
            stack.enter_context(patch('transformers.AutoTokenizer.from_pretrained', return_value=tok))
            loader = stack.enter_context(patch('transformers.AutoModelForCausalLM.from_pretrained', side_effect=lambda *a, **k: model(36,8).to(torch.bfloat16)))
            stack.enter_context(patch('torch.cuda.synchronize'))
            path = Path(d)/'out.jsonl'; summarize = runtime.summarize_logits; calls = 0
            def stopped(*a, **k):
                nonlocal calls
                calls += 1
                if calls == 3: raise RuntimeError('Interrupted forward')
                return summarize(*a, **k)
            with patch.object(runtime, 'summarize_logits', side_effect=stopped), self.assertRaises(RuntimeError):
                runtime.run(path, parent, composition)
            runtime.run(path, parent, composition, resume=True); report = s.analyze(path)
            self.assertTrue(report['complete']); self.assertEqual(report['recorded'], 384)
            self.assertEqual(report['interruptedRequests'], 1); self.assertEqual(report['shamPairs'], 96)
            clean = Path(d)/'clean.jsonl'; runtime.run(clean, parent, composition)
            _, a, _, _ = s.read_journal(path); _, b, _, _ = s.read_journal(clean)
            self.assertEqual([r['result']['choiceLogits'] for r in a], [r['result']['choiceLogits'] for r in b])
            for key, data in before.items(): self.assertEqual(runtime.checkpoint_path(parent, composition, key).read_bytes(), data)
            n = loader.call_count; unchanged = path.read_bytes(); path.with_suffix('.summary.json').unlink()
            runtime.run(path, parent, composition, resume=True)
            self.assertEqual(loader.call_count, n); self.assertEqual(path.read_bytes(), unchanged)
            self.assertTrue(path.with_suffix('.summary.json').exists())
            runtime.checkpoint_path(parent, composition, 'r0-composed').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checkpoint'): runtime.run(path, parent, composition, resume=True)


if __name__ == '__main__': unittest.main()
