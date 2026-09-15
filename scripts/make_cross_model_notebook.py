"""Generate the Colab deliverable with an immutable, already committed code revision."""
import argparse
import json
from pathlib import Path
import re


def md(text):
    return dict(cell_type="markdown", metadata={}, source=text.splitlines(True))


def code(text):
    return dict(cell_type="code", metadata={}, source=text.splitlines(True), execution_count=None, outputs=[])


def make_notebook(revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("An immutable full Git commit is required")
    bootstrap = Path(__file__).with_name("colab_bootstrap.py").read_text(encoding="utf-8")
    cells = [md('''# Menia — prévoir ses propres limites : comparaison entre deux modèles

**À exécuter sur Colab, GPU A100 (40 ou 80 Go).**
**Installation corrigée : compatible avec un Python sans ensurepip, y compris Colab Python 3.13.**
Ouvrir ce nouveau notebook ; une ancienne copie sur Drive n'est pas mise à jour automatiquement.

1. **Exécution → Modifier le type d'exécution → GPU A100 → Enregistrer.**
2. **Exécution → Tout exécuter**, puis autoriser le montage de ton Drive lorsque Colab le demande.
3. À la fin, partager ici **`menia-comparaison-modeles.zip`**.

Ce pilote compare Qwen3-4B et Qwen3-8B. Chacun prévoit les réussites des deux modèles,
avant leurs réponses, avec les mêmes exemples de calibration. Les noms arbitraires sont aussi permutés.
Il teste l'auto-prévision ; il n'entraîne pas les poids et ne certifie pas une conscience.

**60 problèmes, 408 appels :** 24 problèmes de calibration × 2 modèles, puis
36 problèmes × (8 prévisions + 2 réponses). Compter environ 25 Go de poids à télécharger,
plus les bibliothèques. La durée GPU n'a pas encore été mesurée. Les calculs consomment ton quota Colab.
Les résultats et interruptions restent dans `MyDrive/Menia/cross-model-pilot-v1`.
Ce pilote Colab est distinct des trois répétitions demandées sur l'iPhone.
'''), md('''## 1. Installer dans un environnement séparé

Cette cellule fixe le code et les bibliothèques. Elle n'altère pas l'environnement Python du notebook.
Les poids publics seront téléchargés au premier appel. Aucun jeton Hugging Face n'est nécessaire.
'''), code(bootstrap + f'''
MENIA_SETUP_OK = False
MENIA_CHECKS_OK = False
MENIA_ROOT = None
from pathlib import Path
import os, subprocess, sys

CODE_REVISION = "{revision}"
REPO = Path("/content/menia-cross-" + CODE_REVISION[:12])
if not REPO.exists():
    subprocess.run(["git", "clone", "--no-checkout", "https://github.com/speed25200-cyber/Menia.git", str(REPO)], check=True)
    subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", CODE_REVISION], check=True)
actual = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
assert actual == CODE_REVISION, "Le dossier contient une autre révision : ne pas mélanger les versions."
subprocess.run(["git", "-C", str(REPO), "diff", "--exit-code", "HEAD"], check=True)
VENV = Path("/content/menia-cross-env-v2")
PYTHON = ensure_environment(VENV)
subprocess.run([str(PYTHON), "-m", "pip", "install", "torch==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu126"], check=True)
subprocess.run([str(PYTHON), "-m", "pip", "install", "-r", str(REPO / "requirements-cross-model.txt")], check=True)
subprocess.run([str(PYTHON), "-m", "pip", "check"], check=True)
MENIA_SETUP_OK = True
print("Code fixé :", actual)
'''), md('''## 2. Vérifier le GPU et les contrôles logiciels

Les tests utilisent des réponses synthétiques et un minuscule Qwen aux poids aléatoires.
Ils vérifient le logiciel ; ils ne fournissent aucun résultat scientifique sur les deux grands modèles.
'''), code('''MENIA_CHECKS_OK = False
CHECK_ENV = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", TOKENIZERS_PARALLELISM="false")
subprocess.run([str(PYTHON), "-c", "from research.cross_model_gpu import environment; e=environment(); print(e['gpu'], round(e['gpuBytes']/1024**3, 1), 'GiB')"], cwd=REPO, env=CHECK_ENV, check=True)
subprocess.run([str(PYTHON), "-m", "unittest", "tests_research.test_cross_model_prediction", "-q"], cwd=REPO, env=CHECK_ENV, check=True)
subprocess.run([str(PYTHON), "-m", "unittest", "discover", "-s", "tests_language", "-q"], cwd=REPO, env=CHECK_ENV, check=True)
MENIA_CHECKS_OK = True
'''), md('''## 3. Sauvegarder et lancer le pilote

Par défaut, cette cellule reprend le journal déjà désigné comme actif, ou en crée un au premier lancement.
Une requête dont le résultat a été perdu est conservée comme interrompue, puis on passe à la suivante.
Les erreurs techniques arrêtent le processus et restent enregistrées ; après correction, relancer la cellule.
Une sortie mal formée est conservée et évaluée selon la règle annoncée.

Laisser `NEW_ATTEMPT = False`. Après une déconnexion complète, relancer les cellules dans l'ordre :
le fichier de suivi sur Drive retrouve la tentative. Aucune tentative ne sera effacée.
'''), code('''from google.colab import drive
drive.mount("/content/drive")
import datetime, json, uuid

ROOT = Path("/content/drive/MyDrive/Menia/cross-model-pilot-v1")
ROOT.mkdir(parents=True, exist_ok=True)
MENIA_ROOT = ROOT
ACTIVE = ROOT / "active-run.json"
NEW_ATTEMPT = False
if ACTIVE.exists() and not NEW_ATTEMPT:
    active = json.loads(ACTIVE.read_text())
    assert active["codeRevision"] == CODE_REVISION, "La tentative enregistrée utilise une autre version."
    assert Path(active["journal"]).name == active["journal"], "Nom de journal invalide"
    JOURNAL = ROOT / active["journal"]
else:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    JOURNAL = ROOT / (stamp + "-" + uuid.uuid4().hex[:8] + ".jsonl")
    ACTIVE.write_text(json.dumps({"journal": JOURNAL.name, "codeRevision": CODE_REVISION}, indent=2))
print("Journal :", JOURNAL)
command = [str(PYTHON), "-m", "research.cross_model_gpu", str(JOURNAL)]
if JOURNAL.exists():
    command.append("--resume")
completed = subprocess.run(command, cwd=REPO, env=CHECK_ENV)
if completed.returncode:
    print("Exécution arrêtée. Le journal est conservé. La dernière cellule permet aussi d'exporter une tentative partielle.")
else:
    print("Collecte terminée. Consulter le bilan et télécharger l'archive ci-dessous.")
'''), md('''## 4. Voir le bilan et télécharger toutes les tentatives

Un Brier plus petit indique une meilleure prévision. `diagonalAdvantage > 0` signifie que,
dans cette matrice, les modèles se prédisent mieux eux-mêmes en moyenne ; ce score seul
ne démontre pas un accès introspectif. Les deux noms partagent les mêmes réponses cibles,
ils ne constituent pas deux répétitions indépendantes. Les coûts en points sont contrefactuels.

Le téléchargement contient les journaux et bilans, y compris les tentatives partielles.
Les poids ne sont pas inclus. Partager l'archive entière, même si un résultat est négatif ou incomplet.
'''), code('''import zipfile
from google.colab import files

journals = sorted(ROOT.glob("*.jsonl"))
assert journals, "Aucun journal trouvé : lancer d'abord la cellule précédente."
for journal in journals:
    summary = journal.with_suffix(".summary.json")
    with (ROOT / (journal.stem + ".analysis.log")).open("w") as output:
        result = subprocess.run([str(PYTHON), "-m", "research.cross_model_prediction", str(journal), "--output", str(summary)],
                                cwd=REPO, stdout=output, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        print(journal.name, "— journal à examiner ; conservé dans l'archive.")
        continue
    report = json.loads(summary.read_text())
    print(journal.name, "—", report["recordedResults"], "/", report["plannedCalls"], report["statuses"])
    for view, data in report["views"].items():
        print("  ", view, "avantage diagonal :", data["diagonalAdvantage"])
    print("  Brier :", {target: {name: metrics["systemBrier"] for name, metrics in group.items()}
                          for target, group in report["baselines"].items()})
ARCHIVE = Path("/content/menia-comparaison-modeles.zip")
with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for item in sorted(ROOT.iterdir()):
        if item.is_file() and item.suffix in (".json", ".jsonl", ".log"):
            z.write(item, item.name)
files.download(str(ARCHIVE))
''')]
    # Colab may continue Run All after a cell failure. Dependent stages stop cleanly
    # instead of generating misleading missing-variable errors or starting inference.
    guards = {
        4: ("globals().get('MENIA_SETUP_OK', False)", "Installation incomplète : corriger la première cellule avant de poursuivre."),
        6: ("globals().get('MENIA_CHECKS_OK', False)", "Les contrôles GPU et logiciels doivent réussir avant de lancer le pilote."),
        8: ("globals().get('MENIA_ROOT') is not None", "Aucun dossier de collecte ouvert ; aucune expérience n'a démarré."),
    }
    for index, (condition, message) in guards.items():
        source = ''.join(cells[index]['source'])
        cells[index]['source'] = ("if " + condition + ":\n" +
                                  ''.join("    " + line if line.strip() else line for line in source.splitlines(True)) +
                                  "else:\n    print(" + repr(message) + ")\n").splitlines(True)
    for i, cell in enumerate(cells):
        cell["id"] = f"menia-cross-{i:02d}"
    return dict(cells=cells, nbformat=4, nbformat_minor=5,
                metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                              language_info=dict(name="python"), accelerator="GPU",
                              colab=dict(name="Menia — comparaison entre modèles", provenance=[])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    destination = Path("notebooks/03_cross_model_colab.ipynb")
    destination.write_text(json.dumps(make_notebook(args.revision), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(destination)
