"""Launcher integration checks, including failures before a Drive mount."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from colab_activation_launcher import LoggedRunner, launch_activation
from make_activation_monitor_notebook import make_resilient_notebook

REVISION = "3b768778cf075126c8bc15105dd2cea3a56c8664"


class FakeRunner(LoggedRunner):
    def __init__(self, directory, fail=None, gpu="NVIDIA A100-SXM4-40GB, 40960 MiB"):
        super().__init__(directory)
        self.fail, self.gpu, self.calls = fail, gpu, []

    def run(self, stage, command, cwd=None):
        self.stage = stage
        command = [str(x) for x in command]
        self.calls.append(command)
        with (self.directory / "execution.log").open("a", encoding="utf-8") as log:
            log.write(stage + "\n")
        if stage == self.fail:
            raise RuntimeError("Synthetic installation failure: missing wheel")
        if command[0] == "nvidia-smi":
            return self.gpu
        if command[-2:] == ["rev-parse", "HEAD"]:
            return REVISION
        if "clone" in command:
            Path(command[-1]).mkdir()
        if "research.activation_monitor_gpu" in command:
            Path(command[command.index("research.activation_monitor_gpu")+1]).write_text('{"fixture":true}\n', encoding="utf-8")
        if "--output" in command:
            Path(command[-1]).write_text('{"fixture":true}', encoding="utf-8")
        return ""


class ActivationLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.downloads, self.mounts, self.runners = [], [], []

    def launch(self, fail=None, gpu=None, mount=None):
        def factory(directory):
            runner = FakeRunner(directory, fail=fail, **({"gpu": gpu} if gpu else {}))
            self.runners.append(runner)
            return runner
        with contextlib.redirect_stdout(io.StringIO()):
            return launch_activation(REVISION, "# bootstrap fixture", content=self.root,
                mount=mount or self.mounts.append, download=self.downloads.append, runner_factory=factory)

    def test_installation_failure_exports_exact_stage_without_drive(self):
        result = self.launch(fail="3 — Installer PyTorch")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stage"], "3 — Installer PyTorch")
        self.assertFalse(self.mounts)
        self.assertEqual(len(self.downloads), 1)
        with zipfile.ZipFile(self.downloads[0]) as z:
            self.assertIn("diagnostic/error.txt", z.namelist())
            self.assertIn("missing wheel", z.read("diagnostic/error.txt").decode())
            self.assertEqual(json.loads(z.read("diagnostic/status.json"))["status"], "failed")

    def test_wrong_gpu_stops_before_installation_and_still_exports(self):
        result = self.launch(gpu="Tesla T4, 15360 MiB")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(self.runners[0].calls), 1)
        self.assertTrue(Path(self.downloads[0]).is_file())

    def test_drive_failure_does_not_prevent_diagnostic_export(self):
        def no_drive(path):
            raise RuntimeError("Synthetic Drive authorization unavailable")
        result = self.launch(mount=no_drive)
        self.assertEqual(result["stage"], "5 — Connecter Drive")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(self.downloads), 1)

    def test_success_preserves_old_files_and_resumes_active_journal(self):
        folder = self.root / "drive/MyDrive/Menia/activation-monitor-v1"
        folder.mkdir(parents=True)
        (folder / "earlier.jsonl").write_text("retained", encoding="utf-8")
        (folder / "active-run.json").write_text(json.dumps(dict(codeRevision=REVISION, journal="earlier.jsonl")), encoding="utf-8")
        # The synthetic collector writes its own output; archive an independent
        # earlier attempt to check preservation by the launcher.
        (folder / "another.jsonl").write_text("previous attempt", encoding="utf-8")
        result = self.launch()
        self.assertEqual(result["status"], "completed")
        call = next(c for c in self.runners[0].calls if "research.activation_monitor_gpu" in c)
        self.assertIn("--resume", call)
        self.assertEqual((folder / "another.jsonl").read_text(), "previous attempt")
        with zipfile.ZipFile(self.downloads[0]) as z:
            self.assertEqual(z.read("tentatives/another.jsonl"), b"previous attempt")

    def test_real_subprocess_stderr_is_preserved_on_nonzero_exit(self):
        runner = LoggedRunner(self.root)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "test-detail"):
                runner.run("test", [sys.executable, "-c", "import sys; print('test-detail', file=sys.stderr); sys.exit(7)"])
        self.assertIn("test-detail", (self.root / "execution.log").read_text())

    def test_notebook_has_one_self_contained_launch_cell(self):
        notebook = make_resilient_notebook(REVISION)
        cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(cells), 1)
        source = "".join(cells[0]["source"])
        compile(source, "new-colab", "exec")
        self.assertNotIn("MENIA_SETUP_OK", source)
        self.assertIn(REVISION, source)


if __name__ == "__main__":
    unittest.main()
