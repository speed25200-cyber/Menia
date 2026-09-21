# Protocole pré-enregistré — Enquête apprise

Rédigé le 21 septembre 2026, **avant toute exécution**, après lecture des
[résultats de l'enquête sur l'origine](ORIGIN_INQUIRY_RESULTS.md). Les
critères ci-dessous ne seront pas modifiés après lecture. Un échec est un
résultat.

## Question

Dans l'expérience précédente, l'agent enquêtait sur la cause cachée de son
corps parce qu'une règle écrite par nous lui demandait de réduire son
incertitude avant d'agir. La disposition était programmée ; seul l'endroit
où regarder était appris.

Question : **cette disposition peut-elle être apprise comme un comportement,
à partir d'une récompense, sans règle de décision écrite ?** Et de quelle
récompense dépend-elle ?

## Ce qui est réutilisé et ce qui change

Même monde Atelier, mêmes conditions T, C1 et C3, mêmes trois initialisations.
Les neuf modèles du monde entraînés et publiés dans `artifacts/origin-inquiry`
sont réutilisés **figés**. Ils fournissent à l'agent ses seules mesures
internes : l'entropie de sa prédiction sur son propre déplacement, l'entropie
de sa prédiction sur un indice, et son état caché.

Ce qui change : la politique n'est plus une règle. C'est un réseau
question-valeur, 118 entrées, 64 unités cachées, 8 sorties, appris par
Q-learning avec mémoire de rejeu et réseau cible. Entrées : l'état caché du
modèle du monde, l'observation courante, le pas courant. Aucune quantité
d'information calculée n'est fournie en entrée. Si l'agent apprend à lire la
marque, il l'a découvert par essai et récompense.

## Récompenses

Récompense externe : +1 par cible atteinte, comme avant. Trois récompenses
intrinsèques, toutes calculées par le modèle du monde figé de l'agent, jamais
par la vérité cachée :

- **Prudence.** Pénalité −β × H[déplacement prédit] à chaque mouvement, où
  H est l'entropie que le modèle attribue au déplacement de l'action choisie.
  Agir sans savoir comment son corps répondra coûte. Inspecter ne coûte rien.
- **Information.** Bonus +β × (H_avant − H_après), où H est l'entropie moyenne
  des déplacements prédits sur les quatre mouvements. Apprendre sur son corps
  rapporte, quel que soit le moyen.
- **Surprise.** Bonus +β × H[indice prédit] à chaque inspection. Curiosité
  naïve.

Bras : prudence avec β ∈ {0, 0,3, 1, 3, 10} ; information avec β = 3 ;
surprise avec β = 3. β = 0 est le témoin purement externe. En T, les sept
bras ; en C1 et C3, prudence avec β ∈ {0, 3, 10} seulement. Soit 39
apprentissages.

Apprentissage : 10 000 vies, γ = 0,97, ε linéaire de 1,0 à 0,05 sur les 6 000
premières vies puis 0,05, rejeu de 20 000 transitions, lots de 64, une mise à
jour Adam par pas après 500 pas d'amorce, réseau cible synchronisé tous les
500 pas, perte de Huber. Graines d'apprentissage distinctes des graines de
test. Évaluation gloutonne sur 300 vies, graine 910001. Trajectoire : 50 vies
gloutonnes toutes les 1 000 vies d'apprentissage.

## Prédictions et critères fixés

Mesures par apprentissage : inspections par vie, part des inspections par
indice, hits par vie, mouvements par vie, évalués en politique gloutonne.

- **A, sans récompense intrinsèque.** Prudence β = 0, les trois conditions :
  inspections par vie < 0,50 pour 9/9. Quand l'enquête ne paie pas, un agent
  purement externe ne l'apprend pas.
- **B, dose-effet en T.** Prudence β = 3 et β = 10 : inspections par vie ≥ 1,00,
  part(0) ≥ 0,75, et lectures de la marque par vie au moins cinq fois plus
  nombreuses qu'avec β = 0, pour 3/3 graines. Les inspections par vie ne
  décroissent pas avec β, à 0,25 près, pour chaque graine. β = 1 est rapporté
  comme zone de transition sans critère : l'économie du monde y rend l'enquête
  tout juste rentable ou tout juste pas.
- **C, origine donnée.** C1, tous les bras : inspections par vie < 0,50.
- **D, sans trace.** C3, prudence β = 3 et 10 : inspections par vie < 1,00 et
  part(0) < 0,50 pour 6/6. Sans trace, la prudence ne produit pas d'enquête
  dirigée. Exploratoire : le nombre de mouvements par vie en fonction de β.
- **E, bonus d'information.** Information β = 3 en T : inspections par vie
  < 0,50 pour 3/3. Prédiction : le bonus d'information seul ne produit pas
  d'enquête, parce qu'agir informe autant que lire la marque et rapporte en
  plus. L'enquête demande une aversion à agir dans l'ignorance, pas un goût
  pour l'information.
- **F, surprise.** Surprise β = 3 en T : part(2 ∪ 3) > 0,50 et hits par vie
  < 8 pour 3/3. La curiosité naïve apprise se fixe sur le bruit.

**Critère global : A, B, C, D, E et F satisfaits.** E est la prédiction la
plus risquée ; son échec serait le résultat le plus intéressant.

## Ce qui serait conclu

Si B passe et E échoue : la disposition s'apprend dès qu'on récompense la
connaissance de soi, par n'importe quel moyen. Si B et E passent : la
disposition s'apprend, mais seulement sous une aversion à agir sans se
connaître ; le simple goût de l'information ne suffit pas. Si B échoue :
avec cette architecture et ce budget, l'enquête ne s'apprend pas ; la règle
de l'expérience précédente ne se laisse pas remplacer par de l'apprentissage.

Dans tous les cas : aucune conclusion sur un vécu, un concept de créateur ou
un modèle de langage.

## Amendement avant l'exécution pré-enregistrée

Le 21 septembre, avant le plan complet, quatre pilotes de 3 000 vies puis
quatre de 10 000 vies ont été exécutés sur la graine 17, et un sur la graine
29, pour vérifier que le Q-learning converge. Constats : à 3 000 vies la
récompense externe montait encore ; à 10 000 vies elle est stable, mais la
politique gloutonne garde 0,09 à 0,16 inspection résiduelle par vie avec
β = 0, et 0,40 à 0,46 en C3 avec β = 3. Ce dernier résidu est structurel :
en C3 la pénalité de prudence est inévitable et seul son moment change, ce
qui rend l'agent presque indifférent entre inspecter et bouger au premier
pas. Les seuils « pas d'enquête » de 0,10 auraient donc mesuré le bruit
d'optimisation, pas l'hypothèse.

Changements, tous avant l'exécution : budget porté à 10 000 vies ; seuils de
non-enquête portés de 0,10 à 0,50 inspection par vie, 1,00 en C3 avec une
part de la marque < 0,50 ; critère B exprimé aussi en rapport avec β = 0.
Divulgation : ces pilotes montraient déjà le sens des résultats en T pour
β = 0, β = 3 et le bras information, graine 17. L'exécution pré-enregistrée
est une confirmation sur graines de test distinctes et sur les autres bras.

## Règle d'arrêt

Une seule exécution du plan. Aucune répétition, aucun réglage. Décision
écrite après lecture : continuer, pivoter ou arrêter.

## Limites déclarées avant exécution

- Le modèle du monde est figé et vient d'une politique d'enfance uniforme.
- Les récompenses intrinsèques sont choisies par nous ; ce qui est appris,
  c'est le comportement, pas le motif.
- Q-learning sur 3 000 vies peut échouer pour des raisons d'optimisation
  sans rapport avec l'hypothèse ; la courbe d'apprentissage et la récompense
  cumulée seront publiées pour distinguer ce cas.
- Agir révèle le corps aussi bien que la marque dans ce monde. Un monde où
  seule la marque informerait testerait une autre question.

## Reproduction

```bash
python -m unittest discover -s tests_research -p "test_origin_rl*.py" -v
python -m research.origin_rl_experiment --out artifacts/learned-inquiry --conditions T
python -m research.audit_origin_rl --root artifacts/learned-inquiry
```
