"""One Colab entry point with diagnostics available before installation/Drive.

Embedded into the notebook. Scientific code stays at its existing immutable pin.
"""
import datetime
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import traceback
import uuid
import zipfile


class LoggedRunner:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stage = "initialisation"
        self.env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                        CUBLAS_WORKSPACE_CONFIG=":4096:8", OPENBLAS_NUM_THREADS="1",
                        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def run(self, stage, command, *, cwd=None):
        self.stage = stage
        print(f"\n--- {stage} ---", flush=True)
        pieces = []
        with (self.directory / "execution.log").open("a", encoding="utf-8") as log:
            log.write(f"\n--- {stage} ---\n")
            log.flush()
            child = subprocess.Popen([str(x) for x in command], cwd=cwd, env=self.env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace")
            try:
                for line in child.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    pieces.append(line)
                status = child.wait()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
            finally:
                child.stdout.close()
        output = "".join(pieces)
        if status:
            raise RuntimeError(f"Étape « {stage} » arrêtée (code {status}).\n" + output[-12000:])
        return output


def bundle_diagnostics(directory, root, destination):
    """Only this launch's diagnostics and this experiment's direct files."""
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for item in sorted(Path(directory).iterdir()):
            if item.is_file() and item.suffix in (".json", ".log", ".txt"):
                z.write(item, "diagnostic/" + item.name)
        if root is not None and Path(root).exists():
            for item in sorted(Path(root).iterdir()):
                if item.is_file() and item.suffix in (".json", ".jsonl", ".log"):
                    z.write(item, "tentatives/" + item.name)
    return Path(destination)


def launch_activation(code_revision, bootstrap_source, *, content=Path("/content"),
                      mount=None, download=None, runner_factory=LoggedRunner):
    if not re.fullmatch(r"[0-9a-f]{40}", code_revision):
        raise ValueError("A full immutable code revision is required")
    content = Path(content)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    base = content / "menia-activation-diagnostics"
    directory = base / stamp
    directory.mkdir(parents=True, exist_ok=False)
    runner = runner_factory(directory)
    root, archive, status = None, content / "menia-etats-internes.zip", "failed"
    info = dict(codeRevision=code_revision, python=platform.python_version(),
                freeDiskGiB=round(shutil.disk_usage(content).free / 1024**3, 2), status="starting")
    try:
        # Fail early on the actual hardware, before downloading another environment.
        gpu = runner.run("1 — GPU", ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
        info["gpu"] = gpu.strip()
        if "A100" not in gpu:
            raise RuntimeError("Ce notebook exige un A100. Choisir Exécution → Modifier le type d'exécution → GPU A100.")
        repo = content / ("menia-activation-" + code_revision[:12])
        if repo.exists():
            try:
                actual = runner.run("2 — Vérifier le code", ["git", "-C", repo, "rev-parse", "HEAD"]).strip()
            except RuntimeError:
                actual = None
            if actual != code_revision:
                # Never erase or reset an incomplete or foreign checkout.
                repo = content / ("menia-activation-" + code_revision[:12] + "-" + stamp)
        if not repo.exists():
            runner.run("2 — Télécharger le code", ["git", "clone", "--no-checkout", "https://github.com/speed25200-cyber/Menia.git", repo])
            runner.run("2 — Fixer le code", ["git", "-C", repo, "checkout", "--detach", code_revision])
        actual = runner.run("2 — Vérifier la révision", ["git", "-C", repo, "rev-parse", "HEAD"]).strip()
        if actual != code_revision:
            raise RuntimeError("Le code téléchargé ne correspond pas à la révision fixée.")
        runner.run("2 — Vérifier les sources", ["git", "-C", repo, "diff", "--exit-code", "HEAD"])
        venv = content / "menia-activation-env-v1"
        bootstrap = directory / "bootstrap.py"
        bootstrap.write_text(bootstrap_source + '\nif __name__ == "__main__":\n    ensure_environment(sys.argv[1])\n', encoding="utf-8")
        runner.run("3 — Préparer Python", [sys.executable, bootstrap, venv])
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        runner.run("3 — Installer PyTorch", [python, "-m", "pip", "install", "--progress-bar", "off",
                   "torch==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu126"])
        runner.run("3 — Installer les bibliothèques", [python, "-m", "pip", "install", "--progress-bar", "off",
                   "-r", repo / "requirements-cross-model.txt"])
        runner.run("3 — Vérifier les bibliothèques", [python, "-m", "pip", "check"])
        runner.run("4 — Vérifier CUDA", [python, "-c",
                   "from research.cross_model_gpu import environment; e=environment(); print(e['gpu'], e['requiredVersions'])"], cwd=repo)
        runner.run("4 — Tests du moniteur", [python, "-m", "unittest", "tests_research.test_activation_monitor",
                   "tests_language.test_activation_monitor_gpu", "-v"], cwd=repo)
        runner.stage = "5 — Connecter Drive"
        print("\n--- 5 — Connecter Drive ---", flush=True)
        if mount is None:
            from google.colab import drive
            mount = drive.mount
        mount(str(content / "drive"))
        folder = content / "drive/MyDrive/Menia/activation-monitor-v1"
        folder.mkdir(parents=True, exist_ok=True)
        root = folder
        active_file = root / "active-run.json"
        if active_file.exists():
            active = json.loads(active_file.read_text(encoding="utf-8"))
            if active["codeRevision"] != code_revision:
                raise RuntimeError("La tentative existante utilise une autre version. Elle est conservée dans l'archive pour examen.")
            if Path(active["journal"]).name != active["journal"]:
                raise RuntimeError("Nom de journal invalide")
            journal = root / active["journal"]
        else:
            journal = root / (stamp + ".jsonl")
            active_file.write_text(json.dumps(dict(journal=journal.name, codeRevision=code_revision)), encoding="utf-8")
        command = [python, "-m", "research.activation_monitor_gpu", journal]
        if journal.exists():
            command.append("--resume")
        runner.run("6 — Apprendre et évaluer (672 problèmes)", command, cwd=repo)
        for journal in sorted(root.glob("*.jsonl")):
            runner.run("7 — Recalculer le bilan", [python, "-m", "research.activation_monitor", journal,
                       "--output", journal.with_suffix(".summary.json")], cwd=repo)
        status = "completed"
        print("\nExécution terminée. Préparation de l'archive.", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        info["errorType"] = type(exc).__name__
        info["error"] = str(exc)
        (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"\nARRÊT À L'ÉTAPE : {runner.stage}\n{exc}\nLe diagnostic est conservé et sera téléchargé.", flush=True)
    finally:
        info.update(status=status, stage=runner.stage)
        (directory / "status.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        bundle_diagnostics(directory, root, archive)
        (base / "latest.json").write_text(json.dumps(dict(archive=str(archive), directory=str(directory), status=status)), encoding="utf-8")
        if root is not None:
            # Keep the diagnostic across runtime disconnects once Drive is mounted.
            try:
                shutil.copyfile(directory / "status.json", root / (stamp + ".launcher-status.json"))
                if (directory / "execution.log").exists():
                    shutil.copyfile(directory / "execution.log", root / (stamp + ".launcher.log"))
            except OSError:
                print("Copie du diagnostic sur Drive impossible ; l'archive locale reste disponible.", flush=True)
        if download is None:
            try:
                from google.colab import files
                download = files.download
            except ImportError:
                download = None
        if download is not None:
            try:
                download(str(archive))
            except Exception:
                print("Le téléchargement automatique n'a pas abouti. L'archive reste accessible dans le panneau Fichiers de Colab.", flush=True)
        print("Archive :", archive, flush=True)
    return info
