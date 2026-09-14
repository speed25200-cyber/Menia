# Installer Menia sur iPhone 17 Pro

## Prérequis

Un Mac Apple Silicon, Xcode avec Swift 6.2, un iPhone réel, une identité de
signature Apple et le dossier HF fusionné produit par Colab.
Ce dépôt ne livre ni IPA signée ni poids entraînés. La compilation iOS doit
encore être validée sur Mac ; le présent environnement ne dispose pas de Xcode.

## Conversion sur Mac

Dans un environnement Python dédié :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mlx-lm
pip freeze > export-environment.txt
python -m mlx_lm.convert --hf-path /chemin/merged-hf --mlx-path /chemin/menia-4bit -q --q-bits 4
python -m mlx_lm.generate --model /chemin/menia-4bit --prompt "Présente-toi brièvement. /no_think" --max-tokens 128
```

Il s’agit de PTQ, pas de QAT. Conserver le HF original et le manifeste de hashes.
Le dossier doit contenir `config.json`, les poids `.safetensors`,
`tokenizer.json` et `tokenizer_config.json`. Le chargeur requiert `model_type=qwen3`
et `quantization.bits=4`. Ne pas importer le dossier `adapter` seul.

## Construire l’app

```bash
brew install xcodegen
cd ios
xcodegen generate
open Menia.xcodeproj
```

Dans Signing & Capabilities, sélectionner ton équipe et un bundle identifier
unique. Choisir l’iPhone réel comme destination, puis lancer. Les paquets MLX
Swift LM 3.31.3 et Swift Transformers 1.3.0 sont fixés dans `MeniaKit/Package.swift`.
Xcode téléchargera les dépendances de compilation ; l’app charge uniquement un
dossier local et ne possède aucun fallback serveur.

Copier `menia-4bit` dans Fichiers sur l’iPhone. Appuyer sur **Importer**, choisir
le dossier, attendre la copie, puis **Charger**. L’import copie le dossier dans
l’espace privé de l’app. Une copie externe dans Fichiers reste sous ton contrôle.
L’import d’un gros dossier peut prendre du temps ; l’arrêt de génération n’annule
pas immédiatement la copie de fichiers déjà en cours.

## Utilisation

- **Envoyer** : une requête locale, avec les cinq dernières notes.
- **Mémoriser le texte** : enregistrer explicitement le texte saisi (2 000 caractères
  max, 50 notes max). Aucun souvenir n’est extrait automatiquement des réponses.
- **Arrêter** : annulation coopérative de la génération ; les résultats tardifs
  ne sont plus affichés. Vérifier la latence d’arrêt sur appareil.
- **Effacer toutes les notes** : effacer notes et affichage, sans effacer les poids.
- **Supprimer le modèle local** : décharger et retirer le dossier des poids dans l’app.

Le passage en arrière-plan, un avertissement mémoire ou une température système
sérieuse déclenche l’arrêt. Aucun accès caméra/micro n’est demandé. Les notes
sont écrites avec protection complète des fichiers et le dossier est exclu des
sauvegardes automatiques. Ce n’est pas une garantie d’effacement forensique.

Le noyau Python et l’app Swift partagent les principes, pas une base SQLite commune.
L’app utilise des notes JSON. Aucun historique complet de conversation n’est conservé.
Un nouveau contexte est créé pour chaque demande, ce qui borne le cache.

## Mesures requises avant de parler de compatibilité confirmée

Sur appareil réel en Release, consigner : modèle et hashes, version d’iOS,
version d’app, nombre exact de tokens de prompt/sortie, état thermique initial,
pic mémoire observé via Instruments, délai du premier résultat, durée totale,
latence d’arrêt et consommation pendant une session de 10 minutes.
Effectuer 3 démarrages à froid et 5 à chaud, puis un essai en mode avion.

| Critère | Objectif de développement, non mesuré |
|---|---|
| Contexte total | ≤ 2 048 tokens |
| Sortie | ≤ 192 tokens |
| Pic mémoire | Viser < 3 Go, à mesurer et adapter |
| Stabilité | Aucun arrêt mémoire pendant 10 minutes |
| Arrêt | Aucun nouveau texte affiché après l’action |
| Effacement | Aucun rappel des notes après relance |

Le nombre de morceaux de texte du streaming n’est pas un nombre de tokens.
L’interface affiche une durée, pas un faux débit en tokens/s.
