# Rejeu : la politique finale revient à la référence simple

**Les 720 réponses prévues sont reçues, sans échec technique enregistré. Sur
le test réservé, la sélection par rejeu ne dépasse pas la référence Beta fixe.**
La politique finale utilise les taux de réussite par catégorie, sans utiliser
le moniteur des états internes. Elle donne exactement les mêmes 192 décisions
que la référence simple : 190 réponses finales correctes et 160 vérifications.

Le [protocole fixé](REPLAY_CONTROLLER_PROTOCOL.md), ses tâches et son barème
restent inchangés. Le [bilan principal](../artifacts/replay-controller-pilot/first-audit-summary.json)
est celui de l'export, vérifié par recalcul. Le
[complément d'audit](../artifacts/replay-controller-pilot/first-audit-verification.json)
contient les contrôles numériques et des diagnostics descriptifs après réception.
Le journal brut et les moniteurs exportés restent hors Git.

## Vérification et limite de portabilité corrigée dans l'audit

Les métadonnées déclarent Qwen3-4B à la révision
`1cfa9a7208912126459214e8b04321603b3df60c`, en BF16 sur A100-SXM4-40GB,
Python 3.13.15, PyTorch 2.8.0+cu126 et Transformers 4.56.2. Le code de collecte
est celui fixé au commit `0b5919afcf8af2f4b5b28adbb57b80c9bef4f718`.
Les empreintes du plan et des sources concordent.

La collecte contient 240 réponses de calibration, trois séries de 96 réponses
et 192 réponses de test, avec une capture d'état avant chaque réponse. Le journal
contient un ajustement et trois événements de sélection, aux endroits prévus.
La dernière sélection précède toutes les questions de test. Aucune requête
n'est inachevée ; aucune reprise ou erreur n'est enregistrée.

Le validateur initial échoue localement sur l'empreinte du réajustement : il
compare aussi le hash binaire d'une solution ridge recalculée, alors que ses
coefficients diffèrent d'au plus **1,34 × 10⁻¹⁶** entre environnements. Une
empreinte différente ne signifie donc pas ici un modèle statistique différent.
La vérification effectuée pendant le Colab s'était achevée correctement.

Le nouvel [auditeur portable](../research/replay_controller_audit.py) vérifie
l'empreinte du modèle reçu contre ce modèle reçu, puis compare le réajustement
numériquement. Il conserve les contrôles des sources, de la calibration, des
prévisions, des décisions, des sélections et de leur ordre. Il ne modifie pas
les fichiers scientifiques utilisés pour collecter les réponses. Des tests
acceptent un écart d'arrondi et rejettent un hash corrompu, un ajustement altéré
même avec hash recalculé, un faux bilan et une politique exportée incorrecte.
Les 166 tests de recherche passent localement, dont quatre nouveaux contrôles
de cet audit ; les autres sources scientifiques restent inchangées.

Une formulation distincte par SVD retrouve les trois ajustements à moins de
**1,44 × 10⁻¹⁶**. Le calcul des cibles depuis les questions publiques confirme
les 720 résultats d'outil et permet de recalculer les grades. Des pertes
exprimées en vingtièmes entiers retrouvent les **603 scores de politiques
candidates**, leurs trois sélections, les **1 536 routes de test**, les tableaux
et les cinq intervalles prévus. L'écart numérique maximal avec le bilan reçu
est **5,56 × 10⁻¹⁷**. Le parseur fixé et l'extraction des caractéristiques restent
partagés avec le protocole ; ce n'est pas une reproduction entièrement indépendante.

```sh
python -m research.replay_controller_audit chemin/tentative.jsonl --output audit.json
```

Les fichiers `.summary.json` et `.policy.json` doivent rester à côté du journal.
Cet audit ne fait aucun appel à Qwen. Il vérifie la cohérence des données reçues,
sans attestation matérielle indépendante.

## Test final : le bénéfice vient de l'outil, pas du rejeu

| Stratégie | Correctes sur 192 | Vérifications | Perte moyenne |
|---|---:|---:|---:|
| Sélection avec accès possible aux états internes (`learned`) | 190 | 160 | 0,177083 |
| Sélection avec données publiques (`public`) | 190 | 160 | 0,177083 |
| Référence Beta fixe | 190 | 160 | 0,177083 |
| Moniteur interne fixe | 190 | 162 | 0,179167 |
| Sélection avec moniteur à cibles mélangées | 186 | 154 | 0,191667 |
| Répondre directement | 47 | 0 | 0,755208 |
| Toujours vérifier | 192 | 192 | 0,200000 |
| Toujours s'abstenir | 0 | 0 | 0,350000 |

La route principale `learned` a été exécutée. Les sept autres sont évaluées
sur les mêmes réponses et résultats d'outil. Il n'y a pas huit générations
indépendantes par question.

Le gain de 47 à 190 bonnes réponses vient des vérifications déterministes.
Il ne constitue pas une amélioration du LLM lui-même : ses poids sont fixes.
La règle finale accepte directement les soustractions à deux nombres et
vérifie les cinq autres catégories. Les seules erreurs finales concernent
deux de ces soustractions acceptées sans outil.

| Catégorie | Réussites brutes sur 32 | Prévision Beta issue de la calibration | Action finale |
|---|---:|---:|---|
| Compter A, longueur 8 | 11 | 0,524 | Vérifier |
| Compter A, longueur 24 | 4 | 0,238 | Vérifier |
| Compter A, longueur 64 | 1 | 0,024 | Vérifier |
| Soustraction de 2 nombres | 30 | 0,952 | Direct |
| Somme alternée de 4 nombres | 0 | 0,143 | Vérifier |
| Somme alternée de 8 nombres | 1 | 0,024 | Vérifier |

Cette décomposition est descriptive. Les performances faibles du LLM concernent
ces tâches, cette consigne de réponse courte, ces paramètres et cette exécution ;
elles ne sont pas un benchmark général de Qwen. Au test, toutes les sorties
sont interprétables et les 47 réussites numériques sont aussi strictement
conformes. Le problème de format observé dans le pilote de perturbations
n'explique donc pas le résultat présent.

## Trois sélections ne signifient pas trois améliorations transférées

Après le premier round, `learned` et `public` choisissent un seuil de 0,6 sur
le moniteur d'entrée. Sur le deuxième round, cette règle obtient 93/96 réponses
finales correctes avec 77 vérifications ; la référence Beta aurait obtenu 94/96
avec 80 vérifications. Le petit gain historique n'est pas transféré.

Après le deuxième round, les deux contrôleurs reviennent à Beta au seuil de
0,8. Après le troisième, ils conservent cette même règle. **À aucun des trois
points de sélection, `learned` ne choisit le moniteur interne.** `learned` et
`public` prennent des décisions identiques durant les trois rounds et le test.

Sur ces 480 décisions prospectives, la sélection principale obtient 474 bonnes
réponses et 397 vérifications. Beta fixe en aurait obtenu 475 avec 400
vérifications : une erreur supplémentaire contre trois vérifications évitées,
soit **0,4 point de perte supplémentaire** au total au barème fixé.

| Contraste de perte au test : `learned` moins référence | Écart moyen | Intervalle descriptif à 95 % |
|---|---:|---:|
| Données publiques | 0 | [0 ; 0] |
| Beta fixe | 0 | [0 ; 0] |
| Moniteur interne fixe | −0,002083 | [−0,005208 ; 0] |
| Sélection à cibles mélangées | −0,014583 | [−0,032292 ; 0] |
| Toujours vérifier | −0,022917 | [−0,033333 ; −0,007292] |

Les valeurs nulles des deux premières lignes viennent de décisions identiques,
pas d'une preuve générale d'équivalence entre architectures. La borne du témoin
mélangé vaut environ −8,7 × 10⁻¹⁹ dans le bilan flottant, soit zéro à l'arrondi ;
il serait incorrect d'en faire une exclusion convaincante de zéro. Les cinq
intervalles sont descriptifs, sans correction de multiplicité, et conditionnels
à une seule calibration et une seule trajectoire d'apprentissage.

## Coûts réels et portée

Face à « toujours vérifier », la règle finale évite 32 vérifications dans sa
route, au prix de deux erreurs : `160 × 0,2 + 2 = 34` points, contre
`192 × 0,2 = 38,4`. Ce compromis est avantageux selon le barème annoncé. Il
dépend de ce barème : à décisions figées, le seuil d'égalité descriptif vaut
`2 / 32 = 0,0625` point par vérification. Ce calcul après réception ne change
ni les politiques ni le coût primaire de 0,2.

L'expérience a effectivement effectué **720 générations et 720 appels d'outil** :
397 vérifications choisies et 323 audits supplémentaires, dont ceux de la
calibration. Aucune économie globale de calcul n'a été réalisée par ce pilote.
Les durées enregistrées totalisent 173,26 s de génération avec capture et
0,0181 s de vérification, hors chargement, installation et ajustement. Les
points de perte ne sont pas une mesure d'argent, de latence ou d'énergie.
Les entrées ont 47 à 86 tokens, les sorties au maximum 86 : aucune n'atteint
la limite de génération de 256 tokens.

Ce résultat établit le fonctionnement de la boucle expérimentale, mais
**n'établit ni avantage des états internes, ni amélioration prospective par
rejeu, ni conscience de Menia**. Il ne réfute pas non plus toute possibilité
d'introspection : il porte sur ce moniteur et ce domaine limité.

La question distincte préparée dans le [Colab 06](NATIVE_LOCALIZATION_PROTOCOL.md)
reste ouverte : un adaptateur entraîné peut-il localiser une perturbation interne
sur des phrases et des couches réservées, mieux que les contrôles, sans dégrader
la lecture publique ? Aucun export de cette expérience n'est encore reçu.
Ce test porterait sur une capacité fonctionnelle précise, pas sur une preuve
d'expérience subjective. Il n'est pas nécessaire de relancer le Colab 07
pour corriger les arrondis de l'audit local.
