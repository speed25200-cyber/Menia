import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from colab_activation_launcher import launch_activation
from make_native_notebook import make_native_notebook
from tests_bootstrap.test_activation_launcher import FakeRunner, REVISION


class NativeRunner(FakeRunner):
    def run(self,stage,command,cwd=None):
        result=super().run(stage,command,cwd)
        command=[str(c) for c in command]
        if 'research.native_localization_gpu' in command:
            path=Path(command[command.index('research.native_localization_gpu')+1])
            path.write_text('{"fixture":true}\n',encoding='utf-8')
            path.with_suffix('.aligned.safetensors').write_bytes(b'synthetic adapter fixture')
        return result


class NativeLauncherTests(unittest.TestCase):
    def test_resume_no_drive_and_adapter_export_with_old_data_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); old=root/'menia-results/perturbation-monitor-v1'; old.mkdir(parents=True)
            (old/'earlier.jsonl').write_text('preserve')
            runners=[]
            def factory(path):
                r=NativeRunner(path); runners.append(r); return r
            def forbidden(path): raise AssertionError('No Drive mount authorized by this profile')
            for _ in range(2):
                with contextlib.redirect_stdout(io.StringIO()):
                    result=launch_activation(REVISION,'# fixture',content=root,profile='native',storage='local',
                        mount=forbidden,download=lambda p:None,runner_factory=factory)
                self.assertEqual(result['status'],'completed')
            command=next(c for c in runners[1].calls if 'research.native_localization_gpu' in c)
            self.assertIn('--resume',command)
            self.assertEqual((old/'earlier.jsonl').read_text(),'preserve')
            with zipfile.ZipFile(root/'menia-localisation-native.zip') as z:
                self.assertTrue(any(n.endswith('.safetensors') for n in z.namelist()))
                self.assertNotIn('tentatives/earlier.jsonl',z.namelist())

    def test_partial_training_failure_keeps_journal_adapter_and_diagnostics(self):
        class Partial(NativeRunner):
            def run(self,stage,command,cwd=None):
                result=super().run(stage,command,cwd)
                if 'research.native_localization_gpu' in [str(c) for c in command]: raise RuntimeError('Fixture interruption')
                return result
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            result=launch_activation(REVISION,'# fixture',content=root,profile='native',storage='local',
                                    download=lambda p:None,runner_factory=Partial)
            self.assertEqual(result['status'],'failed')
            with zipfile.ZipFile(root/'menia-localisation-native.zip') as z:
                self.assertIn('diagnostic/error.txt',z.namelist())
                self.assertTrue(any(n.endswith('.jsonl') for n in z.namelist()))
                self.assertTrue(any(n.endswith('.safetensors') for n in z.namelist()))

    def test_one_self_contained_block_pins_code_and_local_profile(self):
        n=make_native_notebook(REVISION)
        cells=[c for c in n['cells'] if c['cell_type']=='code']
        self.assertEqual(len(cells),1)
        source=''.join(cells[0]['source']); compile(source,'native-colab','exec')
        self.assertIn(REVISION,source); self.assertIn("profile='native'",source)
        self.assertIn("storage='local'",source)


if __name__=='__main__': unittest.main()
