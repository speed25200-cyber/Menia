"""Real environment repair with no pip, plus safe notebook failure handling.

Downloads only the pinned PyPA bootstrap and pip; no model or GPU required.
"""
import ast
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.colab_bootstrap import ensure_environment, environment_python, assert_private_environment, verified_bootstrap
from scripts.make_cross_model_notebook import make_notebook


class ColabBootstrapTests(unittest.TestCase):
    def test_repair_interpreter_without_pip_preserves_existing_files_and_reuses_it(self):
        with tempfile.TemporaryDirectory(prefix="menia-bootstrap-") as tmp:
            directory = Path(tmp)/"env"
            subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(directory)], check=True)
            python = environment_python(directory)
            probe = subprocess.run([str(python), "-I", "-m", "pip", "--version"], capture_output=True)
            self.assertNotEqual(probe.returncode, 0)
            sentinel = directory/"retained.txt"
            sentinel.write_text("keep this file")
            actual = ensure_environment(directory, Path(tmp)/"cache")
            self.assertEqual(actual, python)
            self.assertEqual(sentinel.read_text(), "keep this file")
            version = subprocess.check_output([str(actual), "-I", "-m", "pip", "--version"], text=True)
            self.assertIn("pip 25.2", version)
            with patch("urllib.request.urlopen", side_effect=AssertionError("Healthy environment must not download again")):
                self.assertEqual(ensure_environment(directory), actual)

    def test_create_in_partial_directory_without_removing_its_contents(self):
        with tempfile.TemporaryDirectory(prefix="menia-bootstrap-") as tmp:
            directory = Path(tmp)/"env"
            directory.mkdir()
            (directory/"old-file.txt").write_text("retained")
            python = ensure_environment(directory, Path(tmp)/"cache")
            assert_private_environment(python, directory)
            self.assertEqual((directory/"old-file.txt").read_text(), "retained")

    def test_foreign_interpreter_and_corrupt_bootstrap_rejected(self):
        with tempfile.TemporaryDirectory(prefix="menia-bootstrap-") as tmp:
            with self.assertRaises(RuntimeError):
                assert_private_environment(sys.executable, Path(tmp)/"different-env")
            (Path(tmp)/"pip-25.2.pyz").write_bytes(b"corrupt download")
            with self.assertRaises(RuntimeError):
                verified_bootstrap(tmp)

    def test_notebook_skips_later_stages_when_setup_has_failed(self):
        notebook = make_notebook("a"*40)
        namespace = {"MENIA_SETUP_OK": False, "MENIA_CHECKS_OK": False, "MENIA_ROOT": None}
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            for index in (4,6,8):
                exec("".join(notebook["cells"][index]["source"]), namespace)
        self.assertIn("Installation incomplète", output.getvalue())
        self.assertNotIn("subprocess", namespace)
        self.assertNotIn("ROOT", namespace)


if __name__ == "__main__":
    unittest.main()
