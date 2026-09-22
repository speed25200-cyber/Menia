"""Build the single-block Colab notebook for the in-context Atelier, pinned to a commit."""
import json
import sys
from pathlib import Path
commit = sys.argv[1]
cells = [
    {"cell_type": "markdown", "metadata": {}, "source": [
        "# Menia — Atelier en contexte pour Qwen3-4B\n",
        f"Protocole : `docs/LLM_ATELIER_PROTOCOL.md` à la révision `{commit}`.\n",
        "Runtime A100. Aucun poids n'est entraîné. Conserver `llm-atelier.zip` à la fin.\n",
        "Ce notebook teste une disposition d'un modèle préentraîné ; il ne teste ni conscience ni entraînement sans information d'origine.\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import subprocess, sys, os, json, hashlib, datetime\n",
        f"COMMIT = '{commit}'\n",
        "if not os.path.exists('/content/Menia'):\n",
        "    subprocess.run(['git', 'clone', 'https://github.com/speed25200-cyber/Menia.git', '/content/Menia'], check=True)\n",
        "os.chdir('/content/Menia')\n",
        "subprocess.run(['git', 'checkout', '--quiet', COMMIT], check=True)\n",
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numpy==2.2.6', 'transformers==4.56.2', 'accelerate==1.10.1'], check=True)\n",
        "subprocess.run([sys.executable, '-m', 'unittest', 'tests_research.test_llm_atelier', '-v'], check=True)\n",
        "import torch; assert torch.cuda.is_available(), 'Activer le GPU'\n",
        "from research.llm_atelier import HFResponder, run_plan\n",
        "from huggingface_hub import HfApi\n",
        "revision = HfApi().model_info('Qwen/Qwen3-4B').sha\n",
        "OUT = '/content/llm-atelier'\n",
        "responder = HFResponder('Qwen/Qwen3-4B', revision)\n",
        "started = datetime.datetime.now(datetime.timezone.utc).isoformat()\n",
        "summary = run_plan(responder, OUT, episodes_per_condition=48)\n",
        "receipt = {'commit': COMMIT, 'model': 'Qwen/Qwen3-4B', 'revision': revision, 'gpu': torch.cuda.get_device_name(0),\n",
        "           'started': started, 'finished': datetime.datetime.now(datetime.timezone.utc).isoformat(),\n",
        "           'episodes_sha256': hashlib.sha256(open(OUT + '/episodes.jsonl', 'rb').read()).hexdigest()}\n",
        "json.dump(receipt, open(OUT + '/receipt.json', 'w'), indent=2)\n",
        "print(json.dumps(summary, indent=2, ensure_ascii=False)); print(receipt)\n",
        "subprocess.run(['zip', '-qr', '/content/llm-atelier.zip', OUT], check=True)\n",
        "print('Télécharger /content/llm-atelier.zip')\n"]},
]
for i, cell in enumerate(cells):
    cell["id"] = f"llm-atelier-{i:02d}"
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"name": "Menia Atelier LLM", "provenance": []}},
            "nbformat": 4, "nbformat_minor": 5}
Path("notebooks/25_llm_atelier_colab.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")
print("written")
