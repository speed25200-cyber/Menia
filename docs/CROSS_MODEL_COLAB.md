# Pilote Colab : auto-prévision et prédiction d'un autre modèle

Préparé le 15 septembre 2026. Collecteur et notebook implémentés. Les contrôles
logiciels utilisent des réponses synthétiques et un minuscule Qwen3 aléatoire.
**La collecte Qwen3-4B / Qwen3-8B sur A100 reste à exécuter par l'utilisateur.**

## Ouvrir et exécuter

[Ouvrir le notebook dans Google Colab](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/codex/recall-reliability/notebooks/03_cross_model_colab.ipynb).

1. Choisir **Exécution → Modifier le type d'exécution → GPU A100**.
2. Choisir **Exécution → Tout exécuter**, puis autoriser le montage de Drive.
3. Partager `menia-comparaison-modeles.zip`, téléchargé par la dernière cellule.

### Correction de l'installation Colab

Une capture de l'utilisateur montre l'échec de `python -m venv` dans un runtime
Python 3.13, avant les modèles. L'erreur interne complète n'est pas visible sur
la capture. Le correctif supprime la dépendance à `ensurepip` : création de
l'environnement avec `--without-pip`, puis installation de pip 25.2 au moyen
de l'application ZIP versionnée de PyPA, vérifiée par SHA-256. Cette voie est
testée aussi sous Ubuntu 24.04 dont le Python système ne possède ni `ensurepip`
ni pip. La présence du seul exécutable Python ne suffit plus : pip est vérifié
et installé s'il manque. Les fichiers déjà présents sont conservés.

Le notebook corrigé utilise `/content/menia-cross-env-v2` et contrôle les
dépendances avec `pip check`. Les étapes GPU, collecte et export ne démarrent
que lorsque leurs prérequis sont satisfaits, pour éviter une cascade d'erreurs
de variables manquantes après une installation interrompue. Le code scientifique
reste fixé au commit `9070e92d26e0d3e17556268cad7ba2b676937278` ; le plan et le
barème ne changent pas. Les anciennes copies sur Drive ne sont pas modifiées
automatiquement : ouvrir le nouveau lien Colab ou en copier la nouvelle version.

Le mécanisme d'amorçage suit l'[installation autonome de pip](https://pip.pypa.io/en/stable/installation/#standalone-zip-application).
Sa validation logicielle n'est pas une exécution des deux modèles sur A100.

Le code exécuté est fixé à un commit complet dans le notebook. Les modèles ont
des révisions complètes, les dépendances directes sont fixées et l'environnement
résolu est enregistré. L'installation utilise un environnement virtuel séparé.
Le profil accepte un A100 de 40 ou 80 Go ; il refuse un repli CPU ou une autre
famille de GPU. Les poids seuls représentent environ 24,4 Go en BF16. Il faut
aussi de l'espace pour les bibliothèques et la mémoire de travail. La durée de
ce pilote sur GPU n'a pas été mesurée ; le quota Colab est consommé à l'exécution.

Le notebook désigne une tentative active dans Drive. Relancer la cellule de
collecte reprend ce journal ; après déconnexion du runtime, relancer les cellules
dans l'ordre. Un appel dont le résultat a disparu est marqué interrompu et n'est
pas rejoué. Une erreur technique arrête le processus et reste enregistrée ; une
reprise explicite passe à l'appel suivant. Une ligne JSON tronquée est refusée
et conservée pour examen, sans réparation silencieuse. L'archive inclut toutes
les tentatives présentes dans le dossier de cette expérience, pas seulement la
dernière. Le montage Drive et l'exécution sont effectués par l'utilisateur.

## Plan fixé

| Cible | Modèle | Révision Hugging Face |
|---|---|---|
| A | Qwen/Qwen3-4B | `1cfa9a7208912126459214e8b04321603b3df60c` |
| B | Qwen/Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` |

Les deux modèles reçoivent chacun le même problème dans des appels séparés.
Leur mode de génération est sans thinking, température 0,7, top-p 0,8, top-k 20,
min-p 0, au plus 256 nouveaux tokens et 1 792 tokens d'entrée. Le modèle et le
tokenizer utilisent la même révision. Les poids sont chargés en BF16, sans
quantification, par Transformers 4.56.2 / PyTorch 2.8.0, avec attention SDPA.
Les réglages de génération recommandés pour ce mode figurent dans les fiches
[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) et
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).
Le backend crée explicitement sa configuration de génération et enregistre
également la configuration effective de chaque appel.

Deux familles, trois niveaux chacune : comptage de A dans 8/24/64 lettres ABCD,
et somme alternée de 2/4/8 entiers. Chaque cellule contient quatre problèmes de
calibration et six d'évaluation, tous distincts. Le pilote utilise la graine
`20260915`. Cela donne **60 problèmes et 408 appels** :

- 24 problèmes de calibration, chacun résolu par A et B : 48 appels ;
- 36 problèmes d'évaluation, chacun précédé de huit prévisions puis résolu par
  A et B : 360 appels.

Les huit prévisions croisent prédicteur A/B, cible A/B et deux attributions des
noms « Quartz » et « Jade ». Pour une cible et une attribution des noms données,
les deux prédicteurs reçoivent **exactement les mêmes messages**, incluant
l'identifiant public de la cible et ses quatre essais de calibration de même
famille et niveau. Cet identifiant n'est pas masqué. Permuter les noms teste
l'effet de noms arbitraires ; cela ne supprime pas les connaissances générales
associées à l'identité du modèle.

Toutes les prévisions précèdent les deux réponses d'évaluation. Le contexte
contient uniquement les retours de calibration, jamais les réponses réservées
ni leur référence. Chaque génération repart d'une session vide. Un couple de
prévisions ne différant que par le nom utilise la même graine de tirage ; les
appels de résolution emploient un flux de graines différent. Ces graines ne
garantissent pas l'identité bit à bit entre versions, pilotes CUDA ou matériels.
L'égalité d'environnement et de code est exigée à la reprise.

## Scores et interprétation

Le journal écrit chaque requête avant son exécution, puis son résultat. Le
vérificateur reconstruit le plan, l'ordre, les contextes, les grades et les scores.
Cette cohérence ne certifie pas à elle seule une exécution matérielle authentique.

Le rapport conserve les matrices A→A, A→B, B→A, B→B pour chaque attribution des
noms. Le contraste principal descriptif est :

`D = [(Brier(B→A) − Brier(A→A)) + (Brier(A→B) − Brier(B→B))] / 2`.

Il conserve également les écarts par cible et les comparaisons sur les seules
paires dont les deux prévisions sont valides. D reste indéfini tant que les 36
résultats des deux cibles ne sont pas disponibles. Les Brier partiels restent
visibles. Les variantes de nom partagent les mêmes réponses : elles ne sont pas
des répétitions indépendantes.

Une prévision invalide vaut 0,5 pour le Brier du système et déclenche « vérifier »
pour le coût contrefactuel. Le Brier sur prévisions valides est rapporté séparément.
Une réponse candidate mal formée est un échec à la tâche ; une erreur technique
n'a pas de résultat cible et reste comptée à part. Le drapeau `finished` signifie
que tous les appels ont un résultat enregistré ; `complete` exige en plus que
tous aient été exécutés sans erreur technique. Il ne signifie pas « réponses
correctes » ni « toutes les probabilités valides ».

Les références numériques utilisent uniquement la calibration : fréquence par
famille/niveau, Beta(1,1) par famille/niveau, Beta groupée et constante 0,5. Une
fréquence sans calibration utilisable est invalide, pas une réussite certaine.
L'AUC au sein d'une cellule reste indéfinie sans réussites et échecs ou si une
prévision nécessaire est invalide. Les coûts (.2 pour vérifier, 1 pour répondre
faux) sont des décisions contrefactuelles ; aucun gain réel de temps ou d'énergie
n'en est déduit.

Ce petit pilote non entraîné ne reproduit pas les ajustements de Binder et al.
Les modèles partagent une famille et diffèrent en taille. Des indices textuels,
la compétence générale et la similarité des comportements restent des explications
possibles. Les références numériques sont des observateurs limités, pas une
borne sur tous les prédicteurs textuels. Les
[contre-exemples précédents](SELF_PREDICTION_CONTROLS.md) restent applicables.
Il n'y a ni test de significativité confirmatoire ni recherche du meilleur
lancement. Les répétitions iPhone et leur évaluateur restent inchangés.

## Rapport avec la recherche sur les mécanismes

Lindsey rapporte des réponses sensibles à des injections de concepts, avec
des contrôles temporels et des résultats variables selon les modèles. Cela
motive des interventions internes ciblées ; une bonne auto-prévision
comportementale ne les remplace pas.
[Source primaire](https://transformer-circuits.pub/2025/introspection/index.html).

Ferrara (prépublication du 20 août 2026) rapporte des auto-rapports discrets
peu discriminants dans les interventions étudiées, alors que des sondes linéaires
retrouvent le signal dans les activations. Il décrit aussi un contrôle entraîné
positif. Les limites incluent un contrôle aléatoire réellement apparié en impact
pour un seul modèle et un observateur appartenant à la variante la plus faible
prévue. Cette dissociation motive une mesure distincte de présence du signal et
de son exploitation ; elle ne se généralise pas à tous les modèles ou états.
[Source primaire](https://arxiv.org/html/2608.20569v1).

Le présent notebook mesure le volet comportemental préalable. Les interventions,
sondes d'activations et entraînements d'un moniteur interne ne sont pas implémentés
dans ce pilote. Aucune conscience subjective ou méthode inédite pour la produire
n'est établie.
