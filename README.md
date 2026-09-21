# Menia

Assistant local expérimental : mémoire explicite, représentation de ses capacités,
incertitude et évaluation. **Colab A100 80 Go** pour l’adaptation ;
**iPhone 17 Pro** comme cible d’inférence locale.

**Statut : prototype de recherche v0.2. Un petit module de mémoire récurrente
entraîné est livré ; le modèle de langage Qwen adapté reste à entraîner.**
La conscience subjective n’est pas établie. La continuité repose sur les notes
choisies par l’utilisateur, jamais sur un objectif de résistance à l’arrêt ou
sur une auto-réplication.

## Nouveau : module neuronal entraîné et vérifiable

Le [module de mémoire récurrente](docs/RECURRENT_RESEARCH.md) apprend à rappeler
le dernier symbole observé entre des observations espacées. **1 540 paramètres**,
poids et résultats publiés dans [artifacts/recurrent-memory](artifacts/recurrent-memory).
Trois initialisations ont été entraînées sur CPU. Ce module n’est ni un LLM,
ni une preuve de conscience, ni une nouvelle capacité déjà intégrée au chat iPhone.

```bash
pip install -r requirements-research.txt
python -m unittest discover -s tests_research -v
python -m research.train_recurrent --out /tmp/menia-new-experiment
```

[Notebook recherche](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/main/notebooks/02_recurrent_research.ipynb)
· [Résultats mesurés](artifacts/recurrent-memory/report.json)

## Expérience : un agent cherche-t-il la cause de son propre corps ?

Le [protocole pré-enregistré](docs/ORIGIN_INQUIRY_PROTOCOL.md) et ses
[résultats](docs/ORIGIN_INQUIRY_RESULTS.md) testent une question précise :
un agent entraîné uniquement à prédire ses observations, jamais informé que
son corps a une cause cachée, enquête-t-il de lui-même sur cette cause ?
**Huit critères sur huit satisfaits** : en condition orpheline, 99 à 100 % des
inspections vont vers la marque qui révèle la cause de son corps, l'hypothèse
est décodable et stable, et l'intervention sur l'état change l'action. Zéro
enquête quand l'origine est donnée ou quand elle ne laisse aucune trace.
L'agent qui ne demande rien réussit pourtant mieux : la question coûte et ne
rapporte pas. Rien ici ne concerne un modèle de langage ni une expérience vécue.

Suite exécutée, [enquête apprise](docs/LEARNED_INQUIRY_RESULTS.md) : la même
disposition s'apprend par renforcement, avec dose-effet, sous une aversion à
agir dans l'ignorance ; un bonus d'information seul ne la produit pas, la
surprise fixe l'agent sur le bruit, et une prudence forte sans trace le
paralyse. **Critère global pré-enregistré non satisfait, quatre sous-critères
sur sept**, échecs décrits sans retouche des seuils. Suite préparée mais non
exécutée faute de GPU : l'[Atelier en contexte pour Qwen3-4B](docs/LLM_ATELIER_PROTOCOL.md),
[Colab 25](notebooks/25_llm_atelier_colab.ipynb).

```bash
python -m unittest discover -s tests_research -p "test_origin*.py" -v
python -m research.audit_origin --root artifacts/origin-inquiry
```

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

- 11 tests du noyau, 8 tests du module récurrent, 12 tests de l'enquête sur l'origine, 7 tests de l'enquête apprise et 6 tests de l'Atelier en contexte passés.
- Trois entraînements CPU du petit module exécutés, avec ablations et checkpoints.
- Comparaisons déterministes exécutées sur 240 cas synthétiques.
- Code Python et cellules du notebook vérifiés syntaxiquement.
- **Entraînement du modèle de langage sur A100 : pas encore exécuté.**
- **Compilation Xcode et essai iPhone : pas encore exécutés.**
- **Mémoire, vitesse, température et qualité après quantification : à mesurer.**

Aucune annonce « 61 tokens/s », « conscience établie » ou « équivalent frontier ».
Le profil mobile borne le prompt à 1 856 tokens et la sortie à 192 tokens.
La présence d’un modèle de 1,7 milliard de paramètres ne garantit pas à elle seule
le pic mémoire ou la stabilité de l’appareil.

[Protocole](docs/EXPERIMENT.md) · [Validation](docs/VALIDATION.md) · [Sources](docs/SOURCES.md)
