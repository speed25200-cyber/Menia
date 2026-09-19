import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from colab_activation_launcher import launch_activation
from make_localization_replication_notebook import make_replication_notebook
from tests_bootstrap.test_activation_launcher import FakeRunner, REVISION


class ReplicationRunner(FakeRunner):
    def run(self,stage,command,cwd=None):
        result=super().run(stage,command,cwd);command=[str(c) for c in command]
        if 'research.localization_replication_gpu' in command:
            path=Path(command[command.index('research.localization_replication_gpu')+1])
            path.write_text('{"fixture":true}\n',encoding='utf-8')
            for rep in range(3):
                for arm in ('visible','strong','shuffled'): path.with_suffix(f'.r{rep}-{arm}.safetensors').write_bytes(b'fixture')
        return result


class ReplicationLauncherTests(unittest.TestCase):
    def test_one_self_contained_cell_with_frozen_revision(self):
        n=make_replication_notebook(REVISION);cells=[c for c in n['cells'] if c['cell_type']=='code']
        self.assertEqual(len(cells),1);source=''.join(cells[0]['source']);compile(source,'replication-colab','exec')
        self.assertIn(REVISION,source);self.assertIn("profile='replication'",source);self.assertIn("storage='local'",source)

    def test_all_nine_adapters_exported_and_resume_uses_same_run_without_drive(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            runners=[]
            def factory(path):
                r=ReplicationRunner(path);runners.append(r);return r
            def no_drive(path): raise AssertionError('Unexpected Drive mount')
            for _ in range(2):
                r=launch_activation(REVISION,'# fixture',content=Path(tmp),profile='replication',storage='local',
                    runner_factory=factory,mount=no_drive,download=lambda p:None)
                self.assertEqual(r['status'],'completed')
            self.assertIn('--resume',next(c for c in runners[-1].calls if 'research.localization_replication_gpu' in c))
            with zipfile.ZipFile(Path(tmp)/'menia-replication-localisation.zip') as z:
                self.assertEqual(sum(n.endswith('.safetensors') for n in z.namelist()),9)

    def test_failure_preserves_diagnostics_and_available_checkpoints(self):
        class Partial(ReplicationRunner):
            def run(self,stage,command,cwd=None):
                result=super().run(stage,command,cwd)
                if 'research.localization_replication_gpu' in [str(c) for c in command]: raise RuntimeError('Fixture interruption')
                return result
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            r=launch_activation(REVISION,'# fixture',content=Path(tmp),profile='replication',storage='local',
                runner_factory=Partial,download=lambda p:None)
            self.assertEqual(r['status'],'failed')
            with zipfile.ZipFile(Path(tmp)/'menia-replication-localisation.zip') as z:
                self.assertIn('diagnostic/error.txt',z.namelist());self.assertTrue(any(n.endswith('.jsonl') for n in z.namelist()))
                self.assertEqual(sum(n.endswith('.safetensors') for n in z.namelist()),9)


if __name__=='__main__': unittest.main()
