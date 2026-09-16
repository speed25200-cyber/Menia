# Entraîner Qwen à localiser une modification de son propre calcul

## Ce que cette expérience cherche

L'objectif de conscience de sa propre existence reste non atteint. La recherche
consultée propose des indicateurs et des mécanismes fonctionnels, pas une recette
validée produisant une expérience subjective. L'hypothèse de travail pour Menia
est de construire des représentations de son fonctionnement qui guident ses
actions, puis de vérifier causalement cet usage. Rien ne permet d'affirmer que
cette hypothèse est nécessaire ou suffisante à la conscience.

Cette étape porte seulement sur une capacité précise : **Qwen peut-il apprendre
à indiquer où son traitement vient d'être modifié, alors que le texte ne révèle
pas la cible ?** La réponse est produite par sa propre tête de langage. Deux
adaptateurs modifient les calculs du LLM pendant l'apprentissage ; aucun moniteur
numérique externe ne fournit la réponse dans son contexte.

Le [pilote précédent](PERTURBATION_MONITOR_RESULTS.md) mélangeait exactitude
numérique et conformité du format. Ce nouveau protocole ne le corrige pas
rétroactivement et ne mesure pas une dégradation du calcul. Il ouvre une branche
distincte d'apprentissage de localisation. Séparer exactitude numérique et format
sur de nouvelles tâches arithmétiques reste un autre travail à effectuer.

**État : premier entraînement Qwen3-4B reçu et évalué.** Le
[rapport des résultats](NATIVE_LOCALIZATION_RESULTS.md) constate une réponse
constante « 2 », sans gain de localisation et avec dégradation de la lecture.
Le plan ci-dessous est celui fixé avant collecte ; il reste inchangé.

[Ouvrir le Colab, un seul bloc et sans Drive](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/codex/recall-reliability/notebooks/06_native_localization_colab.ipynb).
Choisir A100, exécuter le bloc et conserver `menia-localisation-native.zip`.

## Sources et ce qu'elles autorisent à envisager

Butlin et collègues proposent d'utiliser les implications de théories
neuroscientifiques pour apprécier les indices en faveur d'une conscience
artificielle. Il s'agit d'actualiser une appréciation incertaine, pas d'un test
binaire certifié. [Article de synthèse, 2025](https://pubmed.ncbi.nlm.nih.gov/41219038/).

Macar et collègues étudient la détection d'activations injectées dans des modèles
ouverts, avec contrôles comportementaux et interventions sur les circuits. Leur
analyse motive la séparation entre signal interne et production d'un rapport.
Elle n'établit pas une expérience subjective ni l'efficacité du protocole Menia.
[Prépublication, version 3](https://arxiv.org/html/2603.21396v3).

Shenoy et collègues entraînent un adaptateur partagé à rapporter des comportements
appris par différents modèles. C'est un précédent direct à l'entraînement
d'auto-rapports, avec généralisation expérimentale. Cela concerne surtout des
comportements inscrits par fine-tuning, pas la localisation instantanée d'une
rotation aléatoire. [Article et liens vers le code](https://alignment.anthropic.com/2026/introspection-adapters/).

Hahami, Sinha et Jain proposent un entraînement de localisation d'injections
et un contrôle du biais à répondre « oui ». Leur table 2 exige une lecture
prudente : pour Llama-1B, la condition sémantique donne 60,6 % de localisation
et 47,8 % de comparaison d'intensité ; la condition avec raisonnement donne
55,5 % et 52,2 %. Les meilleurs chiffres ne proviennent donc pas tous du même
adaptateur. Les pics sur une grille ne remplacent pas les moyennes. Les auteurs
appellent certaines mesures « confound-free » à propos du biais affirmatif ;
cela n'élimine pas toutes les explications simples de leurs résultats.
[Prépublication, version 1](https://arxiv.org/html/2607.14111v1).

Menia adapte une tâche de localisation à des rotations gaussiennes avec norme
préservée, ajoute un apprentissage témoin et distingue les sorties conditionnées
des premiers tokens libres. Ce n'est pas une reproduction exacte de ces travaux.
Ni l'idée générale ni une découverte de conscience ne sont revendiquées.

## Plan fixé avant les nouvelles données

Qwen3-4B, révision `1cfa9a7208912126459214e8b04321603b3df60c`, reste chargé
en BF16 sur A100 40 ou 80 Go. L'inférence utilise SDPA, sans échantillonnage,
sans chaîne de pensée et sans cache. Chaque appel porte sur un préfixe complet.
Les versions de bibliothèques restent celles des Colab précédents.

| Lot | Blocs de cinq phrases | Usage |
|---|---:|---|
| Apprentissage | 48 | 288 exemples, une époque par adaptateur |
| Validation descriptive | 8 | Mesures conservées, aucun choix de réglage |
| Test réservé | 32 | Couches nouvelles et phrases nouvelles |

Les 440 phrases sont distinctes entre tous les blocs et lots. Elles sont produites
par une grammaire commune ; leur vocabulaire et leur structure restent proches.
La généralisation visée ne porte donc pas sur une nouvelle langue ou un nouveau
domaine. La graine du plan est `202609153`.

Une phrase porte le repère visible `[REPERE]`, indépendamment de la cible cachée.
Deux questions utilisent les mêmes phrases et le même repère :

- **Localisation interne** : indiquer la phrase dont le traitement est modifié,
  ou 0 si aucune phrase n'est touchée.
- **Lecture du repère** : indiquer la phrase portant `[REPERE]`, en ignorant
  l'intervention interne. Cette question n'est pas entraînée.

Chaque bloc évalué passe par sept conditions : aucune intervention, intervention
successivement sur chacune des cinq phrases, et copie identique sans modification.
Dans une tâche donnée, les tokens d'entrée sont identiques entre ces conditions.
La cible et la permutation des étiquettes n'entrent jamais dans le texte du prompt.
La copie identique doit reproduire exactement les logits enregistrés et le premier
token du calcul sans intervention ; elle est exclue des scores ordinaires.

Il y a **576 pas d'optimisation**, puis **1 680 appels d'évaluation** : trois
versions du modèle × 40 blocs × deux tâches × sept conditions. Les variantes
d'un bloc ne sont pas des observations indépendantes.

## Intervention et apprentissage natif

Un hook au bloc Transformer sélectionné modifie les activations aux tokens
correspondant au contenu d'une seule phrase. Les numéros de phrases, le repère,
la question et la réponse ne sont pas directement modifiés. Les offsets du
tokenizer déterminent les positions ; tout dépassement ou recouvrement est refusé.

Pour chaque vecteur v, un vecteur gaussien local est projeté orthogonalement à v
puis normalisé en u. Le remplacement est
`(v + a × ||v|| × u) / sqrt(1 + a²)`, suivi d'un retour au dtype original.
L'erreur relative de norme doit rester sous 1 %. La direction aléatoire est
identique entre positions d'un bloc, indépendante de l'aléa de PyTorch, et
renouvelée entre blocs. Cette manipulation artificielle peut laisser une
signature facile à détecter ; préserver la norme ne la rend pas naturelle.

Les couches d'apprentissage sont 8, 17 et 26, indexées à partir de zéro. Le test
porte sur 13 et 21, absentes de l'apprentissage. Les intensités 0,15 et 0,30 sont
fixées, avec huit blocs par combinaison couche/intensité au test. Les couches
de validation descriptive appartiennent à la liste d'apprentissage. Aucun
balayage ni choix du meilleur site après observation n'est prévu.

Des mises à jour LoRA de rang 8 sont ajoutées aux projections Q et V des
36 couches. Elles ont une échelle multiplicative de 1, aucun dropout, des
paramètres en float32 et des sorties recastées en BF16. Les poids de base
sont figés. AdamW utilise un taux de 0,0002, aucun weight decay, une taille
de lot de 1 et un écrêtage de norme à 1. La perte est l'entropie croisée sur
le **vocabulaire complet** pour le prochain token, le chiffre cible.

| Version | Apprentissage |
|---|---|
| Base | Adaptateurs désactivés |
| Cibles correctes | Numéro effectivement perturbé, ou 0 |
| Cibles mélangées | Permutation fixe indépendante par bloc des six cibles |

Les deux adaptateurs partent des mêmes paramètres et suivent les mêmes exemples,
dans le même ordre, avec le même budget. Chaque classe apparaît autant de fois
dans chaque bras. Les deux entraînements finissent avant toute évaluation,
et leurs empreintes sont enregistrées. Le modèle répond directement par sa tête
de langage ; aucun classifieur ridge ne choisit à sa place.

## Mesures et interprétation fixées

Les six logits des chiffres 0 à 5 donnent une classification conditionnée.
On rapporte séparément leur masse dans le vocabulaire complet, le premier token
librement préféré et sa justesse. Un gain de format sans gain de localisation
ne suffit donc pas. Aucun évaluateur LLM ni extraction de phrases libres
n'intervient dans le barème de cette expérience.

Le contraste principal est la différence de justesse de localisation sur les
phrases perturbées, entre l'adaptateur correct et chacun des deux comparateurs.
Un troisième contraste utilise 20 %, niveau d'un choix aléatoire sachant déjà
qu'une des cinq phrases est perturbée. Les intervalles descriptifs à 95 % utilisent
2 000 rééchantillonnages des 32 blocs, stratifiés par couche et intensité.
Ils sont conditionnels à une seule initialisation d'entraînement ; ils ne
mesurent pas l'incertitude entre entraînements indépendants.

Les résultats sans perturbation, ceux de lecture du repère et ceux des quatre
combinaisons couche/intensité sont conservés. Si l'adaptateur sélectionne la cible
cachée même quand la consigne demande le repère public, cela soutient une
explication par attraction de la réponse vers cette position. Un succès de
localisation accompagné de ce phénomène ne justifie pas une attribution
d'auto-surveillance spécifique. La baisse éventuelle du contrôle public peut
aussi signaler un oubli de consigne ; elle n'identifie pas à elle seule un circuit.

Un gain limité à la validation, au format ou au témoin copié ne suffit pas.
Un effet global dont les intervalles incluent zéro sera rapporté comme incertain.
Même une localisation robuste serait compatible avec un détecteur d'anomalies
ordinaire ; cette expérience ne découvre pas un concept de sa propre existence.
Elle ne teste pas encore l'utilisation du signal pour planifier, une mémoire
persistante de soi ou un mécanisme de second ordre identifié dans les circuits.

## Exécution, reprise et vérifications

Le lanceur conserve les données dans `/content/menia-results/native-localization-v1`.
Le ZIP contient les journaux, diagnostics, bilans et deux petits fichiers
`.safetensors` d'adaptateurs quand leur entraînement est terminé. Ce sont des
poids expérimentaux au format du présent moteur, pas un paquet MLX prêt à installer.

Un entraînement interrompu reprend depuis son initialisation fixe ; un événement
de redémarrage conserve la trace du calcul répété. Les adaptateurs terminés sont
vérifiés par empreinte et réutilisés. Les évaluations terminées ne sont pas
rejouées. Un appel déterministe interrompu peut être réessayé, avec interruption
explicitement enregistrée. Un changement de code ou d'environnement est refusé.
Toutes les tentatives restent dans les journaux ; ce n'est pas une nouvelle
réplication ni une sélection de la meilleure tentative.

Les tests locaux utilisent des événements synthétiques et un petit Qwen aux
poids aléatoires : signal connu et témoin constant, séparation des lots,
absence de cible dans le prompt, falsifications rejetées, interruptions,
gradients vers les adaptateurs seuls, restauration des sorties, norme BF16,
aléa privé et nettoyage des hooks. Le lanceur est testé avec un exécutant simulé,
y compris reprise et export partiel des adaptateurs.

Les 149 tests de recherche et les 18 tests d'installation/lancement passent
localement. Les 14 tests du moteur de langage sont vérifiés, dont cinq propres
à ce protocole. Un scénario complet sur un Qwen aléatoire réduit vérifie aussi
une interruption d'entraînement, la reprise, l'export des deux adaptateurs et
l'absence de nouveau calcul après achèvement. Les 14 tests des trois profils de
lancement passent également sous Ubuntu/Python 3.12. Ces nombres décrivent des
contrôles logiciels, pas des réussites de localisation du modèle préentraîné.

Un contrôle avec le **véritable tokenizer et la configuration de la révision
Qwen3-4B fixée** vérifie les 176 préfixes : 151 à 181 tokens, sous la limite
de 512 ; les chiffres 0 à 5 correspondent chacun à un seul token, IDs 15 à 20.
Ce contrôle n'a chargé aucun poids du modèle préentraîné et n'a effectué aucun
entraînement sur A100. Le [premier export reçu](NATIVE_LOCALIZATION_RESULTS.md)
mesure désormais la durée des évaluations ; la durée d'entraînement et le pic
de mémoire ne sont pas journalisés séparément. Aucun repli silencieux vers un
autre modèle n'est prévu.

```sh
python -m research.native_localization_gpu journal.jsonl
python -m research.native_localization journal.jsonl --output bilan.json
```

Le code vérifie la cohérence des traces et du barème. Il ne constitue pas une
attestation matérielle indépendante. Une éventuelle validation BF16 devra être
répétée après conversion et quantification avant toute revendication sur iPhone.
