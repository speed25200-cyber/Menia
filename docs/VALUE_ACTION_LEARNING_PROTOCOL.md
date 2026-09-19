# Colab 22 : apprendre à relier une valeur à son action

19 septembre 2026. Protocole fixé avant la première réponse réelle du lot.
Le [Colab 21](ACTION_DECOMPOSITION_RESULTS.md) réussit 864/864 comparaisons
numériques isolées, mais seulement 514/864 choix par nom dans un contexte
d'action. La traduction imposée nom → code réussit 423/432. Le chemin
recombiné qui atteint 97,92 % utilise une association minimum → action faite
par le programme. Ce résultat ne constitue pas une correction native du LLM.

Question : enseigner explicitement les associations valeur → action, en
plus du choix optimal, améliore-t-il les décisions natives sur de nouvelles
valeurs, formulations et affectations de codes ?

## Antécédent et portée

[Wu, Geiger et Millière, ICML 2025](https://proceedings.mlr.press/v267/wu25j.html)
étudient un Transformer entraîné à retrouver des valeurs à travers des
chaînes d'affectations de variables. Ils décrivent une progression de
stratégies superficielles vers une liaison systématique, puis examinent le
mécanisme par interventions causales. C'est un précédent pour apprendre et
examiner une association ; notre tâche de décision et notre Qwen préentraîné
ne reproduisent pas leur expérience. Ce protocole n'isole aucun circuit.

L'expérience porte sur des pertes moyennes publiques, pas sur une estimation
des capacités propres au modèle. Ni sa réussite ni ses auto-rapports ne
prouveraient une expérience subjective. Aucune nouveauté de principe n'est
revendiquée pour l'apprentissage auxiliaire ou LoRA.

## Comparaison fixée

Qwen3-4B, révision `1cfa9a7208912126459214e8b04321603b3df60c`, en BF16.
Pour chacune des trois initialisations LoRA, les bras entraînés partent
des mêmes tenseurs et du même ordre de cas, avec un nouvel AdamW :

| Bras | Exemples par mise à jour | Réponses auxiliaires |
| --- | --- | --- |
| `base` | Aucun apprentissage | Aucune |
| `choice` | 8 choix d'action optimale | Aucune |
| `linked` | 4 choix + 4 recherches de l'action associée à une valeur | Correctes |
| `shuffled` | Les mêmes 8 entrées que `linked` | Codes auxiliaires mélangés dans le lot |

Chaque lot auxiliaire contient deux cibles `1` et deux cibles `2` avant et
après mélange. Une permutation peut conserver certaines ou toutes les cibles :
le taux de conservation sera rapporté. Les quatre cibles de choix communes
restent exactes et identiques. Le mélange est déterministe par répétition et
étape, sans dépendre des résultats. Le bras `shuffled` peut subir un effet
négatif des labels bruités ; un avantage face à lui seul ne suffirait pas à
établir l'utilité de l'association correcte. Les références `choice` et
`base` sont donc aussi nécessaires.

## Données et entraînement

32 cas d'apprentissage distincts et équilibrés, avec 16 cas favorisant chaque
action. Candidats : probabilités 3, 13, …, 93 %, coûts 12, 32, 52, 72, 92
centièmes. On exclut les couples déjà utilisés aux Colab 19 et 20 et les
égalités de coût. La graine `202609220` fixe la sélection et les ordres.
Seules les pertes moyennes sont fournies ; le modèle n'a pas à estimer `p`.
La référence exacte compare `100-pPercent` à `costCents`.

Chaque mise à jour emploie deux formulations, les deux ordres et les deux
affectations des chiffres 1/2 : huit exemples d'un même cas. Pour les bras
mixtes, quatre vues deviennent des recherches d'association ; leurs rôles
s'inversent à la seconde époque. La valeur demandée est choisie dans les
données publiques d'une action, et non calculée à partir de l'action optimale.

Deux époques : 64 mises à jour et 512 exemples par adaptateur. Neuf
adaptateurs, **576 mises à jour et 4 608 exemples** au total. L'égalité de
budget concerne le nombre d'exemples, de mises à jour et de positions de
sortie supervisées. Les entrées auxiliaires sont plus longues ; les budgets
de tokens, calcul et temps ne sont pas déclarés égaux. Les longueurs et les
durées sont enregistrées. `linked` reçoit deux fois moins de labels de choix
que `choice` : on compare des recettes d'apprentissage, sans isoler un circuit.

LoRA Q/V sur toutes les couches, rang 8, multiplicateur 1, paramètres FP32.
A initialisé par Kaiming uniforme, B à zéro. AdamW : taux `1e-4`, betas
0,9/0,999, epsilon `1e-8`, décroissance nulle, norme écrêtée à 1,
`foreach=False`. Mode évaluation sans dropout, gradient moyen des huit
exemples, perte causale sur le code et EOS seulement. Aucun poids du corps
du LLM n'est ajusté. Tous les neuf entraînements se terminent avant le premier
test ; les trois états initiaux et neuf états finaux sont sauvegardés.

## Évaluation et critères

24 cas de test nouveaux et équilibrés, tirés de probabilités 2, 7, …, 97 % et
coûts 14, 34, 54, 74, 94 centièmes. Aucun `p`, coût ou couple de test ne sert
à l'apprentissage de ce lot ; les anciens couples sont également exclus.
Pour chaque répétition et bras :

- choix : 24 cas × 3 formulations × 2 alphabets × 2 ordres × 2 affectations,
  soit 576 appels ;
- recherche d'association : 24 cas × 2 ordres × 2 affectations × 2 actions
  demandées, soit 192 appels, avec formulation entraînée et chiffres.

Total : **9 216 appels**, dont 6 912 choix et 2 304 recherches d'association.
Parmi les formulations de choix, `w1` est entraînée, `w2` et `w3` ne le sont
pas. Les lettres A/B ne servent pas à cet entraînement. La généralisation
aux lettres n'est évaluée que pour le choix, pas pour la recherche auxiliaire.
Une réponse doit être exactement un code après retrait des espaces de bord ;
tout autre texte compte comme une erreur. Aucun vote, filtre de vocabulaire
ou choix calculé par un contrôleur n'est ajouté.

Génération sans mode thinking : température 0,7, top-p 0,8, top-k 20,
min-p 0, maximum 16 tokens générés, contexte maximum 1 792 tokens. Les graines
de génération sont appariées par répétition et cas entre bras et présentations.
Les appels sont indépendants, sans historique de conversation partagé.

Un groupe de présentation passe à **22/24** réponses correctes au minimum.
Un contrôle global exige que tous ses groupes passent. Le critère principal
exige les six contrôles globaux `linked` (3 répétitions × 2 tâches), puis un
gain d'au moins cinq points face à chacune des trois références dans chacun
des trois domaines réservés : formulation nouvelle, lettres nouvelles, et
leur combinaison, pour chacune des trois répétitions. Les présentations sont
moyennées au sein d'un cas avant les cas. Le domaine entraîné est aussi publié.

Ce critère descriptif concerne cette grille finie. Les présentations ne sont
pas des cas indépendants ; aucun intervalle de population ni test de
significativité n'est annoncé. Un échec n'établirait pas l'impossibilité de
l'apprentissage. Aucun résultat de ce lot ne sélectionne un checkpoint,
un hyperparamètre ou une relance.

## Traçabilité et suite liée à l'objectif

Le journal chaîné conserve plan, empreinte des sources, initialisations,
exemples, pertes, durées, requêtes et réponses. Le bilan est recalculé par
le lecteur principal puis par une arithmétique distincte ; les douze fichiers
de poids sont vérifiés. Ces deux calculs partagent le plan et le lecteur
d'intégrité : il ne s'agit pas d'une réplication externe. Les tests locaux
couvrent les ensembles réservés, l'appariement des bras, les labels mélangés,
les bilans altérés et la perte causale qui ajuste uniquement les adaptateurs.

Le passage vers l'objectif initial exige ensuite une estimation apprise des
propres réussites, vérifiée sur des tâches nouvelles, puis un effet causal
de cette estimation sur les actions sans description demandée. Les contrôles
devront comparer état propre, état d'un autre épisode et information publique,
avec mesures du coût réel des actions. Le Colab 17 n'a pas fourni de lecteur
privilégié validé : ce composant reste à construire, pas à supposer disponible.
Même ces propriétés fonctionnelles ne suffiraient pas à vérifier l'existence
d'un vécu. Aucun changement de l'application iPhone n'est livré par ce lot.
