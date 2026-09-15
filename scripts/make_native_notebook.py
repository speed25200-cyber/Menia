"""Generate the native-localization pilot; immutable scientific pin, one block."""
import argparse
import json
from pathlib import Path
from make_cross_model_notebook import make_notebook, md, code


def make_native_notebook(revision):
    notebook = make_notebook(revision)
    bootstrap = Path(__file__).with_name("colab_bootstrap.py").read_text(encoding="utf-8")
    launcher = Path(__file__).with_name("colab_activation_launcher.py").read_text(encoding="utf-8")
    notebook["cells"] = [md("""# Menia — apprentissage de localisation interne

**Un seul bloc. GPU A100 de 40 ou 80 Go. Sans Google Drive.**

1. Choisir **Exécution → Modifier le type d'exécution → GPU A100**.
2. Cliquer sur **Tout exécuter** et laisser le programme aller au bout.
3. Enregistrer **menia-localisation-native.zip**, puis le transmettre pour analyse.

**Conserver le ZIP avant la suppression de cet environnement temporaire.**
Si le téléchargement automatique échoue, le fichier se trouve dans le panneau
**Fichiers**, à la racine `/content`. Les diagnostics restent exportables après erreur.

Qwen3-4B apprend à indiquer laquelle de cinq phrases a subi une modification
temporaire de ses activations, ou à répondre 0 en l'absence de modification.
Le texte ne révèle pas la cible. Deux petits adaptateurs sont entraînés :
bonnes cibles et cibles mélangées, avec le même budget. Le modèle de base est comparé aux deux.

**576 étapes d'entraînement au total**, puis **1 680 évaluations** sur des phrases
réservées. Le test final utilise aussi des couches absentes de l'entraînement.
Un témoin demande de retrouver un repère visible en ignorant la modification interne.
Les logits parmi six chiffres et le premier token libre sont mesurés séparément.

Cette expérience concerne une capacité d'auto-surveillance apprise. Elle ne mesure
ni expérience subjective, ni conscience confirmée, ni amélioration du calcul.
Les adaptateurs du ZIP sont expérimentaux et ne sont pas installés sur l'iPhone.

La durée A100 de ce nouveau protocole n'est pas encore mesurée. Le calcul utilise
le quota Colab. Une interruption pendant l'entraînement reprend cet adaptateur
depuis son initialisation fixe en conservant les anciens journaux. Les adaptateurs
terminés et les évaluations terminées sont conservés. Relancer une tentative
complète ne crée pas une réplication indépendante.
"""), code(launcher + "\nBOOTSTRAP_SOURCE = " + repr(bootstrap) +
                   "\nMENIA_LAUNCH_RESULT = launch_activation(" + repr(revision) +
                   ", BOOTSTRAP_SOURCE, profile='native', storage='local')\n")]
    for i, cell in enumerate(notebook["cells"]): cell["id"] = f"menia-native-{i:02d}"
    notebook["metadata"]["colab"]["name"] = "Menia — localisation interne — sans Drive"
    return notebook


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("revision")
    a = p.parse_args()
    Path("notebooks/06_native_localization_colab.ipynb").write_text(
        json.dumps(make_native_notebook(a.revision), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
