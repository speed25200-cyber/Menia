import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from colab_activation_launcher import launch_activation
from make_learning_diagnostic_notebook import make_diagnostic_notebook
from tests_bootstrap.test_activation_launcher import FakeRunner,REVISION


class DiagnosticRunner(FakeRunner):
    def run(self,stage,command,cwd=None):
        result=super().run(stage,command,cwd)
        command=[str(x) for x in command]
        if 'research.learning_diagnostic_gpu' in command:
            path=Path(command[command.index('research.learning_diagnostic_gpu')+1])
            path.write_text('{"fixture":true}\n',encoding='utf-8')
            for arm in ('visible','weak','strong','shuffled'):
                path.with_suffix('.'+arm+'.safetensors').write_bytes(b'fixture')
        return result


class DiagnosticLauncherTests(unittest.TestCase):
    def test_one_block_frozen_code_and_local_profile(self):
        n=make_diagnostic_notebook(REVISION)
        cells=[c for c in n['cells'] if c['cell_type']=='code']
        self.assertEqual(len(cells),1)
        source=''.join(cells[0]['source']);compile(source,'diagnostic-colab','exec')
        self.assertIn(REVISION,source);self.assertIn("profile='diagnostic'",source);self.assertIn("storage='local'",source)

    def test_success_resume_and_all_four_adapters_exported(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            runners=[]
            def factory(path):
                r=DiagnosticRunner(path);runners.append(r);return r
            def no_drive(path): raise AssertionError('Unexpected Drive mount')
            for _ in range(2):
                result=launch_activation(REVISION,'# fixture',content=Path(tmp),profile='diagnostic',storage='local',
                    mount=no_drive,download=lambda p:None,runner_factory=factory)
                self.assertEqual(result['status'],'completed')
            self.assertIn('--resume',next(c for c in runners[-1].calls if 'research.learning_diagnostic_gpu' in c))
            with zipfile.ZipFile(Path(tmp)/'menia-diagnostic-apprentissage.zip') as z:
                self.assertEqual(sum(n.endswith('.safetensors') for n in z.namelist()),4)

    def test_failure_keeps_partial_exports(self):
        class Partial(DiagnosticRunner):
            def run(self,stage,command,cwd=None):
                result=super().run(stage,command,cwd)
                if 'research.learning_diagnostic_gpu' in [str(c) for c in command]: raise RuntimeError('Interrupted fixture')
                return result
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            result=launch_activation(REVISION,'# fixture',content=Path(tmp),profile='diagnostic',storage='local',download=lambda p:None,runner_factory=Partial)
            self.assertEqual(result['status'],'failed')
            with zipfile.ZipFile(Path(tmp)/'menia-diagnostic-apprentissage.zip') as z:
                self.assertIn('diagnostic/error.txt',z.namelist())
                self.assertTrue(any(n.endswith('.jsonl') for n in z.namelist()))
                self.assertEqual(sum(n.endswith('.safetensors') for n in z.namelist()),4)


if __name__=='__main__': unittest.main()
