# Colab 24 — Apprendre à distinguer les erreurs au sein d'une catégorie

**Protocole fixé avant collecte ; entraînement lancé, résultats en attente.**
Le [diagnostic du Colab 23](CONFIDENCE_CALIBRATION_DIAGNOSTIC.md) montre que
son bon classement global dépend surtout des comparaisons entre catégories.
Un recalibrage externe conserve cet ordre. L'expérience suivante teste donc
un objectif de classement local, en gardant la capacité de l'adaptateur fixe.
L'effet d'une augmentation de rang reste une expérience distincte à faire.

## Hypothèse et précédents

Ajouter une perte qui favorise les réponses correctes par rapport aux erreurs
de même famille et difficulté pourrait améliorer leur classement sur des
questions nouvelles. La régularisation du classement de confiance a des
précédents, notamment la *Correctness Ranking Loss* de
[Moon et al., ICML 2020](https://proceedings.mlr.press/v119/moon20a.html).
Les comparaisons relatives de confiance ont aussi été étudiées pour les LLM
par [Shrivastava et al., 2025, §§4.1–4.2](https://arxiv.org/html/2502.01126v1).
Ce dernier travail sollicite des préférences par prompt et les agrège ;
notre recette utilise des résultats mesurés pour entraîner un score natif.
Ce n'est pas une réplication exacte de ces articles, ni une revendication de
nouveauté de la perte logistique ou de la confiance relative.

L'objectif général de conscience reste ouvert. Cette étape cherche un
prérequis fonctionnel : une estimation individuelle de l'erreur. Elle ne
teste encore ni l'accès privilégié à l'état de génération, ni son usage
causal pour décider, ni l'expérience subjective.

## Données et témoins

Seules les 1 728 réponses **train** du Colab 17 sont utilisées, avec leurs
cibles mesurées. Les anciennes validations et évaluations ne sont pas ajoutées.
Chaque répétition possède 576 exemples, 96 par catégorie. Chaque exemple est
présent exactement une fois par époque, pendant deux époques, dans tous les bras.

Dans chaque catégorie, les réussites et erreurs sont appariées sans remise.
Le nombre de paires éligibles est `min(n_correct, n_incorrect)`. Les exemples
restants sont appariés entre eux : leur perte individuelle reste présente,
leur perte auxiliaire vaut zéro. L'ordre des paires et leur orientation sont
déterministes et mélangés. Le modèle voit chaque exemple séparément ; le
partenaire et les cibles ne sont pas ajoutés à son contexte.

| Répétition | Paires éligibles par époque | Paires totales par époque |
|---|---:|---:|
| 0 | 85 | 288 |
| 1 | 92 | 288 |
| 2 | 86 | 288 |

Les sommes alternées à huit opérandes n'ont que 1, 0 et 1 réussite dans les
trois répétitions. Le groupe sans réussite ne fournit aucune paire contrastée.
Dans les deux autres, la même réussite intervient une fois par époque, avec
au total seulement trois exemples distincts éligibles. Aucune répétition
artificielle supplémentaire ne vient masquer cette pauvreté des données.
Tous ces groupes restent dans l'évaluation et dans les tableaux de résultats.
La [couverture complète](../artifacts/confidence-ranking-preparation/report.json)
donne aussi les effectifs et empreintes des plannings.

Trois bras repartent des **mêmes poids initiaux** dans chaque répétition :

- `ce` : supervision individuelle du code correct 0/1 puis EOS ;
- `rank` : même supervision, plus classement correct des paires éligibles ;
- `neutral` : même supervision, plus perte moyenne exacte des deux ordres
  équiprobables sur les mêmes paires éligibles.

`neutral` est un régularisateur actif : il tend à rapprocher les scores.
Il n'est pas un témoin supposé sans effet. C'est pourquoi dépasser aussi `ce`
est nécessaire. Aucun bras ne reçoit de cibles individuelles mélangées.
Le modèle de base sans adaptateur fournit un quatrième producteur/juge.

## Objectif et budget fixes

Qwen3-4B à la révision `1cfa9a7208912126459214e8b04321603b3df60c`, poids de
base BF16 gelés ; LoRA Q/V FP32, rang 8, échelle 1. Mode évaluation pendant
l'apprentissage pour désactiver le dropout, avec gradients actifs.
AdamW : taux `1e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, décroissance zéro,
norme des gradients limitée à 1. Deux époques, lots de quatre paires/huit
exemples, 144 mises à jour par adaptateur, neuf adaptateurs, **1 296 mises à
jour au total**. Trois états initiaux sont sauvegardés séparément.

Pour l'exemple i, `s_i = logit(1) − logit(0)` à la position qui **prédit** le
code. Le code ajouté n'est visible qu'à la position suivante, supervisée par
EOS. Pour une paire de cibles différentes, `d = s_correct − s_incorrect` :

```
rank    : softplus(−d)
neutral : [softplus(d) + softplus(−d)] / 2
ce      : 0
```

La perte du lot est la moyenne des huit pertes individuelles code/EOS,
plus la moyenne des quatre pertes de paires, de coefficient 1. Les paires
inéligibles gardent leur zéro dans ce dénominateur. Aucun coefficient,
rang, taux ou nombre d'époques n'est sélectionné après inspection des scores.
Deux graphes sont conservés à la fois, puis la paire est rétropropagée ;
les gradients des quatre paires sont accumulés avant une seule mise à jour.

## Évaluation indépendante et critère

**1 152 nouvelles questions**, 64 par catégorie et répétition, excluant les
5 950 questions recensées des anciens plans, dont celles du Colab 23. Les
neuf adaptateurs sont tous gelés avant la première réponse d'évaluation.
Les quatre producteurs répondent aux mêmes questions avec la même graine
d'échantillonnage par question. Chacun des quatre juges évalue ensuite chaque
réponse : **4 608 générations + 18 432 jugements = 23 040 appels**.

Génération : température 0,7, top-p 0,8, top-k 20, min-p 0, 256 nouveaux
tokens au maximum, 1 792 tokens d'entrée, thinking désactivé, aucun repli de
modèle ni troncature. Jugements : rapport natif `p(1)/(p(0)+p(1))`, température
1, sans échantillonnage ; la masse des codes dans le vocabulaire est conservée.
Les juges recalculent le contexte textuel entier, sans accès au KV d'origine.
Aucun calibrateur externe n'est ajusté dans cette expérience.

Le score principal est l'AUROC restreinte aux paires réussite/erreur de **même
catégorie**, pondérée par leur nombre, avec demi-crédit aux ex æquo. Une catégorie
sans les deux classes contribue zéro paire et est signalée, jamais notée comme
réussie. Les catégories fournissant davantage de paires ont plus de poids ;
ce score n'établit donc pas un gain uniforme sur les six catégories. Le Brier,
l'AUROC globale et les scores de chaque catégorie restent descriptifs.

Les réponses des producteurs `base` et `rank` sont principales. Pour chacune
et dans les trois répétitions, le juge `rank` doit dépasser :

- `ce` d'au moins 0,05 d'AUROC ;
- `neutral` d'au moins 0,05 ;
- un score constant au sein de chaque catégorie (AUROC 0,5) d'au moins 0,10.

Cela fait **18 comparaisons**, toutes requises. Leur borne inférieure doit
être positive dans un intervalle bootstrap percentile avec correction
Bonferroni (alpha familial 0,05, queues `0,05/(2×18)`). Dix mille tirages
rééchantillonnent les **questions**, séparément dans chaque catégorie, avec
les mêmes multiplicités pour tous les juges d'un producteur. Les paires ne
sont pas traitées comme indépendantes. Les probabilités natives ne sont pas
recalibrées et les seuils ne sont pas ajustés sur ces questions.

Chaque producteur principal doit aussi présenter au moins vingt réussites
et vingt erreurs par répétition, ainsi qu'au moins trois catégories avec
cinq de chaque. Pour le juge `rank`, le token le plus probable doit être un
code dans au moins 95 % des cas et la masse moyenne des deux codes atteindre
0,5. La précision du producteur `rank` ne doit pas baisser de plus de deux
points par rapport à la base. Ce dernier seuil est un contrôle sur l'estimation
ponctuelle, **pas** une démonstration statistique de non-infériorité.
Un bootstrap indéfini fait échouer le contraste concerné.

## Vérifications et traçabilité

Les treize tests nouveaux passent sur PC, dont un journal synthétique complet
et un petit Qwen initialisé aléatoirement. Ils vérifient les gradients par un
calcul indépendant, la direction du classement, les ex æquo, les multiplicités
du bootstrap, la conservation de l'exposition, l'exclusion des anciennes
questions et le refus d'évaluer avant le gel complet. Les réponses parfaites
du journal synthétique sont construites avec le corrigé : **ce sont des tests
logiciels, aucune performance de Menia**.

Le journal conserve chaque lot, perte, durée, initialisation et empreinte de
poids, puis les requêtes et résultats dans une chaîne vérifiée. Une tentative
existante n'est ni effacée ni relancée automatiquement. Les états finaux et
les résultats devront être récupérés et audités avant toute conclusion.

```sh
python -m unittest tests_research.test_confidence_ranking tests_research.test_confidence_ranking_study tests_language.test_confidence_ranking_gpu -v
python -m research.confidence_ranking_learning_gpu JOURNAL artifacts/answer-confidence-training-data
```

[Empreintes du protocole](../artifacts/confidence-ranking-preparation/design.json) ·
[État de l'objectif général](CONSCIOUSNESS_GOAL_STATUS.md)

## Lancement observé

Le 20 septembre 2026 à 01:21:52 UTC, la version scientifique
`4cbce2edea430cc7e4df52546d448e91a7c98629` a été lancée par MCP sur l'A100
40 Go. Les treize tests nouveaux et quatre contrôles existants de la trace
de génération passent dans Colab (17 tests, 27,003 secondes). Le
[reçu de lancement](../artifacts/confidence-ranking-pilot/launch.json) constate
six mises à jour du premier adaptateur à 01:22:52 UTC, aucune requête de test,
et des empreintes de source et de plan conformes. Le processus est alors actif.
Il ne s'agit pas d'un résultat final.

[Notebook épinglé](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/65b571ad7368c0459ed33f164b05f1858ee7f30f/notebooks/24_confidence_ranking_colab.ipynb).

## Audit préparé pendant la collecte

Un [second calcul](../research/audit_confidence_ranking.py) reconstruit le
classement par matrices de comparaisons réussite/erreur, sans appeler le tri
du rapport principal. Il reconstitue les multiplicités de questions, les
intervalles, la notation exacte des réponses, le Brier et les seuils de décision.
Le lecteur du journal, le plan et le générateur aléatoire NumPy restent communs :
il s'agit d'une vérification numérique, pas d'une réplication externe.

Quatre contrôles locaux passent, avec des scores synthétiques imparfaits,
des ex æquo et des intervalles non dégénérés. Les 480 000 AUROC de bootstrap
sont recalculées ; l'écart maximal entre les champs des rapports est
3,34 × 10⁻¹⁶. Les contrôles rejettent aussi un score, un intervalle ou un
critère altéré, ainsi que des poids non finis, inchangés ou de forme incorrecte,
même quand leur empreinte est recalculée. Le
[reçu logiciel](../artifacts/confidence-ranking-preparation/audit-software-check.json)
est explicitement marqué `synthetic_fixture` : il ne mesure pas Menia.
Le premier essai du test de corruption rencontrait un verrou de fichier mappé
sur Windows ; la copie des tenseurs avant remplacement corrige ce test.
Cette correction n'affecte ni le collecteur ni l'expérience Colab en cours.

Sur l'entraînement réel, un
[contrôle du préfixe du journal](../artifacts/confidence-ranking-pilot/training-prefix-check.json)
à 01:36:00 UTC valide 392 mises à jour, dont les deux premiers adaptateurs
terminés. Il vérifie la chaîne, l'exposition appariée et l'arithmétique des
pertes auxiliaires, ainsi que les empreintes des deux poids terminés et de
leur initialisation commune. Aucune requête de test n'est encore présente.
Les nombres sont un état intermédiaire daté ; ils ne constituent pas une
évaluation de performance et ne préjugent pas de la fin de l'expérience.

L'audit final devra lire les douze fichiers d'adaptateurs réels et vérifier
leurs types, dimensions, finitude, initialisations et modifications. Ces
contrôles ne constituent pas une attestation indépendante de l'intégrité de
tous les poids de base. Le résultat scientifique reste en attente.

## Fin de l'entraînement et évaluation en cours

Le [relevé des poids figés](../artifacts/confidence-ranking-pilot/training-freeze.json)
du 20 septembre à **02:06:58 UTC** vérifie les neuf entraînements terminés,
soit 1 296 mises à jour et 144 par adaptateur. Le lecteur du protocole figé
valide la chaîne du journal d'entraînement, l'exposition appariée et les pertes.
Les empreintes des neuf adaptateurs terminés et des trois états initiaux
correspondent aux fichiers présents sur Colab, chacun de 11 811 296 octets.

Le relevé est pris après le début de l'évaluation, mais la copie du journal
est arrêtée **avant sa première requête** : aucune réponse ni performance
d'évaluation n'est lue par ce contrôle. L'empreinte du préfixe est
`eb27907387de279f6dd98cb5c37f186125f4221ab2bf413e39d931c197187e5c`.
Le processus reste actif. La réception locale de l'archive, l'audit des tenseurs
et les comparaisons sur les 23 040 appels prévus restent à terminer.
La fin de l'entraînement ne constitue pas un résultat prédictif positif.

```sh
python -m unittest tests_research.test_audit_confidence_ranking -v
python -m research.audit_confidence_ranking JOURNAL SUMMARY --output VERIFICATION
```
