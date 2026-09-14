# Menia

Assistant local expérimental : mémoire explicite, représentation de ses capacités,
incertitude et évaluation. **Colab A100 80 Go** pour l’adaptation ;
**iPhone 17 Pro** comme cible d’inférence locale.

**Statut : version initiale du code. Aucun poids Menia entraîné n’est livré.**
La conscience subjective n’est pas établie. La continuité repose sur les notes
choisies par l’utilisateur, jamais sur un objectif de résistance à l’arrêt ou
sur une auto-réplication.

## Commencer sur ton A100

[**Ouvrir le notebook dans Colab**](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/main/notebooks/01_colab_a100.ipynb)

1. Choisir un runtime GPU A100. Le notebook vérifie les 80 Go réellement alloués.
2. Monter Drive et installer les dépendances. Redémarrer si Colab le demande.
3. Exécuter les tests puis les 20 steps de validation technique.
4. Comparer base et adaptation avant d’augmenter l’entraînement.
5. Fusionner les adaptateurs et récupérer le dossier sur un Mac Apple Silicon.
6. Convertir en MLX 4 bits, puis suivre le [guide iPhone](docs/IPHONE.md).

Le corpus fourni est **un jeu synthétique de démarrage**, pas un corpus suffisant
pour un assistant généraliste. Le notebook est un pipeline d’adaptation et non
un entraînement de modèle de langage à partir de zéro.

## Contenu

| Partie | Implémentation |
|---|---|
| Base | `Qwen/Qwen3-1.7B` ; révision exacte enregistrée à chaque run |
| Entraînement | LoRA BF16 ; réponses seules dans la loss ; reprise de checkpoints |
| Mémoire Python | SQLite, source et date, écriture explicite, effacement |
| Évaluation | Capacités, rappel, information manquante, erreur de prédiction |
| Comparaisons | Base/adapté, sans système/sans état, règles fixes |
| App iPhone | SwiftUI, MLX local, notes explicites, arrêt, effacement, suppression du modèle |
| Export | Adaptateurs fusionnés → HF → MLX PTQ 4 bits |
| Conscience | Pas de preuve ni de score ; protocole à conclusions limitées |

## Tester sans GPU

```bash
python -m unittest discover -s tests -v
python -m menia.data --out /tmp/menia-data
python -m menia.evaluate --data /tmp/menia-data/test.jsonl --out /tmp/menia-baselines.json
```

Le noyau et les tests utilisent Python standard. Les dépendances GPU sont
séparées dans `requirements-train.txt`. Les versions directes sont fixées ;
les dépendances transitives résolues sont enregistrées avec `pip freeze`.

## Statut de validation

- 11 tests Python passés lors de la création initiale.
- Comparaisons déterministes exécutées sur 240 cas synthétiques.
- Code Python et cellules du notebook vérifiés syntaxiquement.
- **Entraînement A100 : pas encore exécuté.**
- **Compilation Xcode et essai iPhone : pas encore exécutés.**
- **Mémoire, vitesse, température et qualité après quantification : à mesurer.**

Aucune annonce « 61 tokens/s », « conscience établie » ou « équivalent frontier ».
Le profil mobile borne le prompt à 1 856 tokens et la sortie à 192 tokens.
La présence d’un modèle de 1,7 milliard de paramètres ne garantit pas à elle seule
le pic mémoire ou la stabilité de l’appareil.

[Protocole](docs/EXPERIMENT.md) · [Validation](docs/VALIDATION.md) · [Sources](docs/SOURCES.md)
