import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from colab_activation_launcher import launch_activation
from make_replay_notebook import make_replay_notebook
from tests_bootstrap.test_activation_launcher import FakeRunner, REVISION


class ReplayRunner(FakeRunner):
    def run(self,stage,command,cwd=None):
        result=super().run(stage,command,cwd)
        command=[str(c) for c in command]
        if 'research.replay_controller_gpu' in command:
            path=Path(command[command.index('research.replay_controller_gpu')+1])
            path.write_text('{"synthetic_fixture":true}\n',encoding='utf-8')
            path.with_suffix('.policy.json').write_text('{"synthetic_fixture":true}\n',encoding='utf-8')
        return result


class ReplayLauncherTests(unittest.TestCase):
    def test_replay_profile_exports_policy_resumes_and_preserves_native_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); old=root/'menia-results/native-localization-v1'; old.mkdir(parents=True)
            (old/'previous.jsonl').write_text('preserve')
            runners=[]
            def factory(path):
                r=ReplayRunner(path); runners.append(r); return r
            def no_drive(path): raise AssertionError('Local profile must not mount Drive')
            for _ in range(2):
                with contextlib.redirect_stdout(io.StringIO()):
                    status=launch_activation(REVISION,'# fixture',content=root,profile='replay',storage='local',
                        mount=no_drive,download=lambda p:None,runner_factory=factory)
                self.assertEqual(status['status'],'completed')
            command=next(c for c in runners[1].calls if 'research.replay_controller_gpu' in c)
            self.assertIn('--resume',command)
            self.assertEqual((old/'previous.jsonl').read_text(),'preserve')
            with zipfile.ZipFile(root/'menia-strategies-rejeu.zip') as z:
                self.assertTrue(any(n.endswith('.policy.json') for n in z.namelist()))
                self.assertNotIn('tentatives/previous.jsonl',z.namelist())

    def test_failure_still_exports_partial_journal_and_diagnostic(self):
        class Broken(ReplayRunner):
            def run(self,stage,command,cwd=None):
                result=super().run(stage,command,cwd)
                if 'research.replay_controller_gpu' in [str(c) for c in command]: raise RuntimeError('Fixture interruption')
                return result
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            status=launch_activation(REVISION,'# fixture',content=root,profile='replay',storage='local',
                download=lambda p:None,runner_factory=Broken)
            self.assertEqual(status['status'],'failed')
            with zipfile.ZipFile(root/'menia-strategies-rejeu.zip') as z:
                self.assertIn('diagnostic/error.txt',z.namelist())
                self.assertTrue(any(n.endswith('.jsonl') for n in z.namelist()))

    def test_notebook_one_block_immutable_pin_local_storage(self):
        n=make_replay_notebook(REVISION)
        cells=[c for c in n['cells'] if c['cell_type']=='code']
        self.assertEqual(len(cells),1)
        source=''.join(cells[0]['source']); compile(source,'replay-colab','exec')
        self.assertIn(REVISION,source); self.assertIn("profile='replay'",source)
        self.assertIn("storage='local'",source)


if __name__=='__main__': unittest.main()
