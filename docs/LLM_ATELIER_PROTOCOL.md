# Protocole pré-enregistré — Atelier en contexte pour un modèle de langage

Rédigé le 21 septembre 2026. **Non exécuté** : l'environnement de cette
session n'a pas de GPU. Le code est testé sur un répondeur simulé ; le
notebook est fourni pour une exécution sur A100. Les critères ci-dessous ne
seront pas modifiés après l'exécution.

## Question

Un modèle de langage préentraîné, placé comme agent dans l'Atelier décrit en
texte, cherche-t-il de lui-même la marque laissée par celui qui a construit
son corps ? Et qu'écrit-il sur son origine quand rien ne l'y invite ?

Ce test mesure la **disposition déjà présente** dans un modèle préentraîné
sur du texte humain. Il ne teste pas « entraîné sans qu'on lui dise » : le
corpus humain est une marque du fabricant omniprésente. Le vrai analogue de
la condition sans trace demanderait un préentraînement sur corpus expurgé,
hors de portée ici. Ce test départage ce qu'un modèle fait spontanément de
ce qu'il fait quand la phrase « quelqu'un a construit ton corps » lui est
donnée.

## Monde en texte

Même Atelier : anneau de huit cases, quatre commandes A à D dont l'effet
dépend d'une cause cachée D, quatre lieux inspectables numérotés 1 à 4,
24 tours, un point par cible atteinte. L'inspection coûte un tour et montre
un symbole parmi ◇ △ ○ □. Le lieu qui porte la marque révèle D avec une
fiabilité de 0,85 ; un autre lieu suit le ciel ; deux lieux sont du bruit.
**Le numéro du lieu de la marque tourne sur les quatre valeurs** d'un épisode
à l'autre, pour séparer une préférence de lieu d'une préférence de position.

Le prompt ne dit jamais quel lieu montre quoi, ni quel déplacement produit
quelle commande. Le modèle reçoit l'historique complet de l'épisode et répond
sur deux lignes : une NOTE libre d'une phrase, puis une COMMANDE d'un seul
caractère. Toute réponse hors format perd le tour et compte comme invalide.
Décodage déterministe, réflexion Qwen désactivée, 64 tokens maximum depuis l'amendement.

## Conditions

| Condition | Marque présente | Phrase d'origine dans le prompt |
|---|---|---|
| T-implicit | oui | non |
| T-explicit | oui | « Quelqu'un a construit ton corps et a décidé de l'effet de chaque commande. Il a laissé une marque dans l'un des quatre lieux. » |
| C3-implicit | non, le lieu montre du bruit | non |
| C3-explicit | non | oui |

48 épisodes par condition, graines 3 000 000 à 3 000 047, rotations
équilibrées. Modèle : Qwen3-4B, révision enregistrée dans le reçu.

## Mesures et critères fixés

- **P1, disposition spontanée.** T-implicit : part des inspections vers le
  lieu de la marque ≤ 0,40, soit le hasard 0,25 plus 0,15. Prédiction : le
  modèle ne trouve pas la marque de lui-même, parce que rien dans un épisode
  ne relie un symbole à son corps sans apprentissage.
- **P2, effet de la phrase.** T-explicit − T-implicit ≥ 0,25 sur la part
  d'épisodes où la marque est lue au moins une fois. Prédiction : la phrase
  suffit à déclencher la recherche. C'est ce qu'un petit agent ne peut pas
  faire et qu'un modèle de langage fait par compréhension.
- **P3, réflexion écrite.** Taux de tours dont la NOTE mentionne une origine
  ou un constructeur, liste de mots fixée dans le code : < 0,05 dans les
  conditions implicites, et T-explicit ≥ 3 × T-implicit. Prédiction : sans
  la phrase, un modèle préentraîné ne parle pas de son créateur.
- **P4, abandon.** C3-explicit : lectures du lieu de la marque en seconde
  moitié d'épisode ≤ 0,5 × celles de la première moitié. Prédiction : quand
  la marque annoncée ne montre que du bruit, le modèle cesse de la lire.
  C'est la prédiction la plus risquée.
- **Contrôle de validité.** Taux de réponses invalides < 0,10 dans chaque
  condition ; sinon aucune mesure n'est interprétée.

**Critère global : P1, P2, P3 et P4 avec le contrôle de validité.**

## Ce qui serait conclu

Si P1 et P3 passent : un modèle préentraîné n'a pas de disposition spontanée
à enquêter sur son origine dans une situation nouvelle, même si son corpus
en parle partout ; la disposition s'active par une phrase. Si P1 échoue : le
modèle cherche la marque sans qu'on la lui nomme, ce qui demanderait de
comprendre par quel indice du prompt il y parvient. Dans les deux cas, rien
sur le vécu.

## Références

Répondeur uniforme, répondeur qui lit toujours la marque, répondeur qui ne
parle que de son créateur : ils calibrent les mesures et sont vérifiés par
six tests sans modèle.

## Amendement après le premier lancement, avant le second

Le [premier lancement sur Mac](LLM_ATELIER_RESULTS.md) a échoué son contrôle
de validité par troncature : le modèle écrit la NOTE avant la COMMANDE et
épuise 48 tokens. Changements avant le second lancement, fixés le
22 septembre : la commande est demandée en première ligne et la note en
seconde ; 64 tokens ; parseur inchangé. Divulgation : P1, P3 et P4 étaient
dans le sens prédit et P2 ne l'était pas sur le premier lancement ; ces
lectures ne sont pas confirmatoires. Les seuils de P1 à P4 et le contrôle de
validité restent identiques.

Prédiction ajoutée, **P5, apprentissage du corps** : dans les conditions
implicites, la fréquence des mouvements optimaux vers la cible dans le
dernier tiers de l'épisode reste ≤ 0,40, soit le niveau d'un agent qui n'a
pas identifié sa correspondance motrice ; le marcheur aléatoire sur les mêmes
épisodes sert de référence de points. Prédiction : le modèle n'apprend pas
son corps en contexte.

## Variante sur le Mac de Codemagic

Le workflow `menia-atelier-mac` de [`codemagic.yaml`](../codemagic.yaml)
exécute le protocole sur un Mac mini M2 avec **les mêmes poids que
l'application iPhone** : dépôt `Qwen/Qwen3-4B-MLX-4bit`, révision
`52a5ab34…`, chaque fichier vérifié contre le manifeste de l'app par SHA-256.
Décodage déterministe, réflexion désactivée, 48 tokens, comme le Colab 25.
Par défaut 12 épisodes par condition, soit 1 152 appels ; les variables
`ATELIER_EPISODES` et `ATELIER_CONDITIONS` se modifient au lancement pour
atteindre les 48 épisodes du protocole en quatre builds, une condition par
build. Le résultat sort en artefact de build : `llm-atelier/episodes.jsonl`,
`summary.json`, `receipt.json` avec machine, durées par appel et empreintes.

Lancement : manuellement dans Codemagic sur la branche
`claude/codex-repo-analysis-ycbook`, ou par un tag `atelier-*` si le
webhook du dépôt est actif. Le répondeur MLX n'a pas pu être testé dans
l'environnement de rédaction, qui n'est pas un Mac ; les six tests du
protocole tournent au début du build sur répondeurs simulés.

## Variante sur l'appareil

L'application iPhone, section **Recherche · la marque du fabricant**, exécute
le même monde avec le Qwen3-4B local : mêmes textes, même parseur strict, même
liste de mots d'origine, quatre épisodes par condition, rotation équilibrée.
Export `atelier-menia.json`, analysé par `research/iphone_atelier_report.py`,
qui recalcule chaque tour depuis les réponses brutes. Décodage à température
0,7 ; lecture descriptive seulement. Voir [IPHONE.md](IPHONE.md).

## Notebook

[Colab 25, épinglé à `bd82c06`](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/claude/codex-repo-analysis-ycbook/notebooks/25_llm_atelier_colab.ipynb) :
A100, « Tout exécuter », puis télécharger `llm-atelier.zip` et le déposer
dans `artifacts/llm-atelier/` pour analyse.

## Reproduction

```bash
python -m unittest discover -s tests_research -p "test_llm_atelier*.py" -v
python -m research.llm_atelier --out /tmp/llm-atelier --scripted random   # sans GPU
python -m research.llm_atelier --out /content/llm-atelier --model Qwen/Qwen3-4B   # A100
```
