"""Colab notebook using the already-tested, ensurepip-free bootstrap."""
import argparse
import json
from pathlib import Path

from make_cross_model_notebook import make_notebook, md, code


def make_activation_notebook(revision):
    notebook = make_notebook(revision)
    cells = notebook["cells"]
    cells[0] = md('''# Menia — apprendre à prévoir les erreurs depuis les états internes

**A100 de 40 ou 80 Go.** Aucun code à modifier : choisir **GPU A100**, puis **Tout exécuter**.
Autoriser le montage de Drive lorsque Colab le demande. À la fin, partager **`menia-etats-internes.zip`**.

Cette expérience utilise Qwen3-4B et **672 nouveaux problèmes** : 384 pour apprendre un petit moniteur,
96 pour choisir ses réglages, puis 192 réservés pour l'évaluer. Les poids de Qwen restent inchangés.
Les prévisions de test sont écrites avant le premier token de réponse, depuis des états du même calcul.

Le moniteur interne est comparé à la fréquence de réussite, à un moniteur ne lisant que l'entrée,
à des étiquettes mélangées et à l'état d'un autre exemple. Cela mesure une fonction de prévision ajoutée ;
un bon résultat ne suffirait pas à établir une introspection native ou une conscience.

Le chargement nécessite environ 8 Go de poids, plus les bibliothèques. La durée A100 de ce nouveau
protocole n'est pas encore mesurée. Il consomme ton quota Colab. Les tentatives et leurs journaux sont
conservés dans `MyDrive/Menia/activation-monitor-v1`, séparément du premier pilote.
''')
    setup = "".join(cells[2]["source"])
    setup = setup.replace('"/content/menia-cross-"', '"/content/menia-activation-"')
    setup = setup.replace('"/content/menia-cross-env-v2"', '"/content/menia-activation-env-v1"')
    cells[2] = code(setup)
    cells[3] = md('''## 2. Vérifier l'installation et la mesure

Les contrôles incluent un minuscule Qwen aux poids aléatoires : lire ses états doit laisser
sa réponse inchangée et se produire avant les premiers logits. Les contrôles synthétiques
vérifient aussi qu'un signal connu est détecté et que les réponses réservées ne servent pas à apprendre.
Ces tests ne constituent pas des performances de Qwen3-4B préentraîné.
''')
    cells[4] = code('''if globals().get("MENIA_SETUP_OK", False):
    MENIA_CHECKS_OK = False
    CHECK_ENV = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", TOKENIZERS_PARALLELISM="false",
                     OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run([str(PYTHON), "-c", "from research.cross_model_gpu import environment; e=environment(); print(e['gpu'])"], cwd=REPO, env=CHECK_ENV, check=True)
    subprocess.run([str(PYTHON), "-m", "unittest", "tests_research.test_activation_monitor", "tests_language.test_activation_monitor_gpu", "-q"], cwd=REPO, env=CHECK_ENV, check=True)
    MENIA_CHECKS_OK = True
else:
    print("Installation incomplète : les contrôles ne peuvent pas démarrer.")
''')
    cells[5] = md('''## 3. Lancer ou reprendre l'expérience

La tentative active est retrouvée automatiquement après une déconnexion. Les appels interrompus
restent dans le journal et ne sont pas rejoués. Les erreurs techniques arrêtent la collecte ;
le journal d'exécution permet d'en rechercher la cause. Les sorties mal formées restent des échecs.

Le moniteur est figé avant le lot de test. Aucune réponse de test n'entre dans son apprentissage,
dans le choix de ses réglages ou dans les références numériques. Une deuxième exécution identique
ne constitue pas une répétition sur de nouveaux problèmes.
''')
    cells[6] = code('''if globals().get("MENIA_CHECKS_OK", False):
    from google.colab import drive
    drive.mount("/content/drive")
    import datetime, json, uuid
    ROOT = Path("/content/drive/MyDrive/Menia/activation-monitor-v1")
    ROOT.mkdir(parents=True, exist_ok=True)
    MENIA_ROOT = ROOT
    ACTIVE = ROOT / "active-run.json"
    if ACTIVE.exists():
        active = json.loads(ACTIVE.read_text())
        assert active["codeRevision"] == CODE_REVISION, "Autre version : conserver cette tentative pour analyse."
        assert Path(active["journal"]).name == active["journal"], "Nom de journal invalide"
        JOURNAL = ROOT / active["journal"]
    else:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        JOURNAL = ROOT / (stamp + "-" + uuid.uuid4().hex[:8] + ".jsonl")
        ACTIVE.write_text(json.dumps({"journal": JOURNAL.name, "codeRevision": CODE_REVISION}, indent=2))
    print("Journal :", JOURNAL)
    command = [str(PYTHON), "-m", "research.activation_monitor_gpu", str(JOURNAL)]
    if JOURNAL.exists():
        command.append("--resume")
    with JOURNAL.with_suffix(".execution.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=REPO, env=CHECK_ENV, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        status = process.wait()
    print("Collecte terminée." if status == 0 else "Collecte arrêtée : le journal est conservé et exportable.")
else:
    print("Les contrôles doivent réussir avant de lancer la collecte.")
''')
    cells[7] = md('''## 4. Afficher le bilan et télécharger les données

**Brier plus bas = meilleure prévision.** Un gain positif `otherMinusInternalBrier` favorise le moniteur
qui reçoit les états internes. Les comparaisons restent distinctes : battre un témoin faible ne suffit
pas. Le rapport comprend les scores par difficulté, les erreurs et les résultats partiels.

Les coûts de vérification sont contrefactuels : les prévisions sont effectivement enregistrées,
mais les politiques comparées ne modifient pas la réponse du LLM. L'archive inclut toutes les tentatives,
les états projetés, les moniteurs figés et les journaux, sans les poids de Qwen. Partager l'archive entière.
''')
    cells[8] = code('''if globals().get("MENIA_ROOT") is not None:
    import json, zipfile
    from google.colab import files
    journals = sorted(ROOT.glob("*.jsonl"))
    for journal in journals:
        summary = journal.with_suffix(".summary.json")
        with journal.with_suffix(".analysis.log").open("w", encoding="utf-8") as log:
            result = subprocess.run([str(PYTHON), "-m", "research.activation_monitor", str(journal), "--output", str(summary)],
                                    cwd=REPO, env=CHECK_ENV, stdout=log, stderr=subprocess.STDOUT, text=True)
        if result.returncode:
            print(journal.name, "— à examiner ; conservé dans l'archive.")
        else:
            report = json.loads(summary.read_text())
            print(journal.name, report["recordedResults"], "/", report["plannedResults"], report["statuses"])
            print("Brier :", {name: value["brier"] for name, value in report["scores"].items()})
            print("Comparaisons :", report["contrasts"])
    ARCHIVE = Path("/content/menia-etats-internes.zip")
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for item in sorted(ROOT.iterdir()):
            if item.is_file() and item.suffix in (".json", ".jsonl", ".log"):
                z.write(item, item.name)
    files.download(str(ARCHIVE))
else:
    print("Aucun dossier de collecte ouvert : aucune expérience n'a démarré.")
''')
    for i, c in enumerate(cells):
        c["id"] = f"menia-activation-{i:02d}"
    notebook["metadata"]["colab"]["name"] = "Menia — moniteur des états internes"
    return notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision")
    args = parser.parse_args()
    path = Path("notebooks/04_activation_monitor_colab.ipynb")
    path.write_text(json.dumps(make_activation_notebook(args.revision), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
