"""The new profile isolates files and retains early failure diagnostics."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from colab_activation_launcher import launch_activation
from make_perturbation_notebook import make_perturbation_notebook
from tests_bootstrap.test_activation_launcher import FakeRunner,REVISION


class PerturbationRunner(FakeRunner):
    def run(self,stage,command,cwd=None):
        output=super().run(stage,command,cwd)
        command=[str(c) for c in command]
        if 'research.perturbation_monitor_gpu' in command:
            Path(command[command.index('research.perturbation_monitor_gpu')+1]).write_text('{"fixture":true}\n',encoding='utf-8')
        return output


class PerturbationLauncherTests(unittest.TestCase):
    def test_new_profile_and_resume_do_not_touch_old_experiment(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            old=root/'drive/MyDrive/Menia/activation-monitor-v1'
            old.mkdir(parents=True)
            (old/'old.jsonl').write_text('preserve',encoding='utf-8')
            runners=[]
            def factory(path):
                runner=PerturbationRunner(path)
                runners.append(runner)
                return runner
            for _ in range(2):
                with contextlib.redirect_stdout(io.StringIO()):
                    r=launch_activation(REVISION,'# fixture',content=root,mount=lambda p:None,
                        download=lambda p:None,runner_factory=factory,profile='perturbation')
                self.assertEqual(r['status'],'completed')
            self.assertEqual((old/'old.jsonl').read_text(),'preserve')
            cmd=next(c for c in runners[1].calls if 'research.perturbation_monitor_gpu' in c)
            self.assertIn('--resume',cmd)
            with zipfile.ZipFile(root/'menia-perturbations.zip') as z:
                self.assertNotIn('tentatives/old.jsonl',z.namelist())
                self.assertTrue(any(n.startswith('tentatives/') and n.endswith('.jsonl') for n in z.namelist()))

    def test_early_failure_still_exports_for_new_profile(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            r=launch_activation(REVISION,'# fixture',content=Path(d),mount=lambda p:None,
                download=lambda p:None,runner_factory=lambda p:PerturbationRunner(p,fail='3 — Installer PyTorch'),profile='perturbation')
            self.assertEqual(r['status'],'failed')
            with zipfile.ZipFile(Path(d)/'menia-perturbations.zip') as z:
                self.assertIn('diagnostic/error.txt',z.namelist())

    def test_single_cell_contains_fixed_profile_and_revision(self):
        n=make_perturbation_notebook(REVISION)
        cells=[c for c in n['cells'] if c['cell_type']=='code']
        self.assertEqual(len(cells),1)
        source=''.join(cells[0]['source'])
        compile(source,'perturbation-colab','exec')
        self.assertIn("profile='perturbation'",source)
        self.assertIn(REVISION,source)


if __name__=='__main__':
    unittest.main()
