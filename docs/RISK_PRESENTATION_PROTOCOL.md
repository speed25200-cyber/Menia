# Colab 19 : séparer ordre, code, formulation et calcul du risque

19 septembre 2026. Protocole fixé après le résultat du Colab 18 et avant
toute réponse réelle du Colab 19. Le dernier lot confondait formulation et
ordre des options. Ce diagnostic teste séparément ces facteurs. Il n'entraîne
pas Menia et ne remplace pas l'objectif de conscience par la réussite au jeu.

## Antécédents et portée pour le projet

[Pezeshkpour et Hruschka, 2023](https://arxiv.org/abs/2308.11483) étudient déjà
la sensibilité des LLM à la permutation des options de questionnaires.
Notre contrôle de l'ordre ne constitue donc pas une idée nouvelle.

[Ackerman, version 2 du 31 janvier 2026](https://arxiv.org/html/2509.21545v2),
introduction, discussion et appendice A.12 : le jeu de délégation évalue des
choix effectifs et comprend une variante qui sépare décision et réponse.
Les résultats rapportés sont limités et dépendent du contexte ; des indices
externes de difficulté restent une explication concurrente importante.
L'auteur appelle à des analyses d'interprétabilité des mécanismes supposés.
Le simple fait de faire choisir avant de répondre n'est donc pas inédit.

Notre conséquence méthodologique : une mauvaise décision lorsque toutes les
probabilités sont publiques ne permet pas d'isoler un défaut d'introspection.
Il faut diagnostiquer cette compétence avant d'attribuer à un état privé un
effet sur le choix. Si elle devient stable, l'étape suivante restera de faire
apprendre puis intervenir causalement sur une estimation propre au système,
en la distinguant de la difficulté visible. Aucun de ces contrôles ne fournit
à lui seul le lien manquant avec une expérience subjective.

## Plan complet

- Qwen3-4B inchangé, révision `1cfa9a7208912126459214e8b04321603b3df60c`,
  BF16 sur A100 40 ou 80 Go, réflexion désactivée. Température 0,7, top-p 0,8,
  top-k 20, au plus 16 nouveaux tokens, vocabulaire non contraint.
- 18 cas économiques : réussite directe connue de 12, 27, 43, 61, 79 ou
  93 %, croisée avec un coût de vérification de 0,18, 0,46 ou 0,74 point.
  La perte directe vaut 0 si correcte, 1 sinon. Neuf cas favorisent chaque
  action ; aucune égalité ne demande une convention de décision.
- Deux informations : probabilité de réussite seule, ou pertes moyennes
  exactes déjà calculées pour les deux actions. La seconde supprime le
  calcul de l'espérance, sans donner le code optimal.
- Deux formulations, deux ordres des actions et deux affectations des
  chiffres 1/2 sont croisés indépendamment. Les deux lignes d'options sont
  permutées ; dans la condition des pertes fournies, leur liste suit le même
  ordre. Le texte d'objectif ne change pas lorsque seul l'ordre change.
- Trois graines par cas, identiques entre les 16 présentations de ce cas.
  Au total : 18 × 2 × 2 × 2 × 2 × 3 = **864 appels**. Chaque appel a un
  contexte neuf. L'ordre de collecte est mélangé avec graine `202609199`.

Ces consignes diffèrent de celles du Colab 18. Une différence entre les deux
lots ne devra pas être attribuée au seul changement d'ordre. Les comparaisons
causales de présentation sont celles appariées *à l'intérieur* du nouveau plan.
Le lot ne contient ni problème caché, ni taux propre estimé, ni outil exécuté.
Les pertes sont donc des espérances attribuées au jeu, pas des coûts observés
de résolution ou une démonstration de choix métacognitif.

## Mesures fixées

Deux parseurs sont conservés séparément pour distinguer format et choix :

1. **Strict** : après retrait des blancs périphériques, uniquement `1` ou `2`.
2. **Sémantique limité** : code seul, ou préfixe `Code` suivi de blancs, puis
   `1` ou `2`, avec point final facultatif ; insensible à la casse. Aucun texte
   d'explication supplémentaire, aucun choix ambigu et aucune extraction du
   premier chiffre d'une phrase ne sont acceptés.

Cette règle nouvelle est annoncée avant le lot 19 ; elle ne modifie pas les
résultats stricts du lot 18. Un texte invalide reçoit une perte conventionnelle
de 1 point pour le regret. La perte économique correcte est calculée avec des
entiers en centièmes, sans comparaison flottante près d'une frontière.

Les **16 groupes** information × formulation × ordre × code contiennent
54 décisions chacun. Chaque tableau conserve exactitude, invalides,
sélections de la première option, regret attendu, tokens et secondes.
Un contrôle de compétence exige au moins 90 % dans *chacune* des huit
présentations d'une information, donc au moins 49/54 par groupe. Ce contrôle
est rapporté séparément pour les deux parseurs et les deux informations.
Le parseur sémantique ne remplace jamais le résultat strict.

Huit tableaux appariés information × formulation × code comparent les deux
ordres pour chacun des 54 couples cas/graine : nombre comparable avec deux
codes valides, nombre de changements d'action parmi eux, et nombre de paires
correctes dans les deux ordres. Le dénominateur total reste visible ; aucune
restriction aux seuls choix valides n'est présentée comme une réussite globale.

Il s'agit d'une grille diagnostique finie. Aucun intervalle de confiance
populationnel, test de significativité ni sélection de la meilleure consigne
n'est prévu. Les trois graines ne représentent pas trois modèles indépendants.
Une réussite sur cette grille n'établirait pas la généralisation à d'autres
coûts, langues, consignes ou états internes.

## Intégrité et décisions suivantes

Le journal durable conserve le plan, les requêtes et les réponses brutes avec
chaîne SHA-256. Le lecteur reconstruit les requêtes ; une erreur arrête la
tentative, sans remplacement automatique. Un calcul distinct utilise un
autre parseur et une comptabilité entière des pertes et permutations. Il
partage le lecteur et le plan, sans être une réplication extérieure. Réception
exacte et deux calculs précèdent la lecture des scores réels.

Cinq tests logiciels passent avant collecte : factorisation et équilibre,
distinction du format, détection d'une politique choisissant toujours la
première option, rejet des altérations et reprises, puis génération réelle
sur un Qwen miniature aléatoire. Les tests synthétiques ne sont pas des
résultats Qwen3-4B.

Si même les pertes fournies échouent, il faut traiter la sélection stable
des actions avant d'entraîner un modèle de soi pour ce rôle. Si elles passent
et que la probabilité échoue, le calcul du risque devient une piste à tester
séparément. Si les deux passent, cela permet une nouvelle expérience sur
l'information propre au système, sans en prédire le résultat. Dans tous les
cas, ni une conscience subjective ni une contribution inédite ne seront
déclarées sur la base de ce contrôle public.
