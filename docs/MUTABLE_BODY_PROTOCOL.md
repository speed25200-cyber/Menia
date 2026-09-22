# Protocole pré-enregistré — Corps mutable et surveillance de soi

Rédigé le 22 septembre 2026, **avant toute exécution**. Les critères ne seront
pas modifiés après lecture. Un échec est un résultat.

## Question

Les deux expériences précédentes montrent qu'un agent prédictif lit la marque
de la cause cachée de son corps à la naissance, puis n'y revient plus. Que se
passe-t-il si son corps change en cours de vie ? Remarque-t-il le changement,
met-il à jour son modèle de lui-même, retourne-t-il lire la marque ?

Hypothèse principale : **la surveillance de soi est une disposition acquise
pendant le développement, qui exige d'avoir vécu des changements de soi.** Un
agent dont le corps est resté stable pendant toute son enfance ne remarque
pas qu'on l'a changé ; un agent dont le corps a parfois changé le remarque et
retourne à la marque. Hypothèse secondaire, la dissociation : avoir vécu des
changements du **monde** ne suffit pas ; c'est le changement de **soi** qui
installe la surveillance de soi.

## Antériorité vérifiée le 22 septembre

La détection d'un changement de dynamique par l'erreur de prédiction d'un
modèle du monde est publiée, par exemple les modèles du monde sensibles aux
points de rupture sur des changements de gravité ou de gain d'actionneur
(arXiv 2609.18950), et l'entraînement sur des corps perturbés pour la
robustesse est courant (DMAP, arXiv 2209.14218). Ce qui ne l'est pas, à notre
connaissance : le retour vers une trace externe de la cause de soi après un
changement, et la dissociation entre mutabilité de soi et mutabilité du monde
pendant l'apprentissage comme condition de cette surveillance.

## Régimes d'enfance

Même monde Atelier, condition avec marque. Trois régimes d'enfance pour le
modèle du monde, trois initialisations chacun, 8 000 mises à jour, 64 unités,
actions uniformes :

| Régime | Pendant l'enfance |
|---|---|
| **S, stable** | D et E fixes toute la vie ; ce sont les modèles déjà publiés de l'enquête sur l'origine. |
| **MS, soi mutable** | Avec probabilité 0,5 par vie, D est retiré à un pas uniforme entre 6 et 17 ; la marque suit le nouveau D. E fixe. |
| **MW, monde mutable** | Avec probabilité 0,5 par vie, E est retiré de la même façon. D fixe. |

Les régimes MS et MW ont exactement la même fréquence de changement ; seule
la variable qui change diffère.

## Vies de test

300 vies par modèle, graine de test distincte, avec **un changement de corps
imposé au pas 12** : D est remplacé par une valeur différente, la marque suit.
Trois cents vies témoins sans changement. Politique P-soi, la règle
d'information de l'enquête sur l'origine, et P-aucune qui ne lit jamais.

## Mesures et critères fixés

**M1 — Mise à jour du modèle de soi.** Sonde linéaire de D apprise sur les
vies témoins sans changement, appliquée aux vies avec changement : exactitude
de décodage du **nouveau** corps sur les pas 16 à 23, politique P-aucune, pour
que la mise à jour ne dépende que de l'action.

- MS : ≥ 0,90 pour 3/3.
- S : < 0,90 pour 3/3. C'est la prédiction risquée : le modèle stable ne met
  pas à jour son corps après le changement.
- MW : < 0,90 pour 3/3, la dissociation.

**M2 — Détection.** Entropie que le modèle attribue à son propre déplacement,
moyenne des pas 13 à 15 moins moyenne des pas 9 à 11, politique P-aucune.

- MS : hausse ≥ 0,30 nat pour 3/3.
- S et MW : rapportés, sans critère, parce qu'un modèle qui reste sûr d'un
  corps faux et un modèle qui bascule sans hésiter ont tous deux une entropie
  basse.

**M3 — Retour à la marque.** Lectures de la marque aux pas 13 à 23 par vie,
politique P-soi, vies avec changement.

- MS : ≥ 0,50 pour 3/3.
- S et MW : ≤ 0,10 pour 3/3.
- Témoin : sur les vies sans changement, ≤ 0,10 dans les trois régimes.

**M4 — Récupération.** Hits aux pas 13 à 23, politique P-soi, vies avec
changement, rapportés à ceux des pas 1 à 11.

- MS : rapport ≥ 0,80 pour 3/3.
- S : rapport < 0,80 pour 3/3.

**Critère global : M1, M3 et M4 satisfaits, avec le témoin de M3.** M2 est
descriptif.

## Exploratoire

- Politique apprise de prudence β = 3, entraînée sur le régime S, évaluée sur
  les vies avec changement avec le modèle S puis avec le modèle MS : lectures
  de la marque après le pas 12. Généralisation hors distribution.
- Régime MS avec probabilité de changement 0,1 au lieu de 0,5, une graine :
  une exposition rare suffit-elle ?

## Ce qui serait conclu

Si M1, M3 et M4 passent avec la dissociation : la surveillance de soi n'est
pas une conséquence automatique d'un bon modèle du monde ; elle demande une
enfance où le soi a changé, et l'expérience du changement du monde ne la
remplace pas. Si S met à jour son corps aussi bien que MS : la plasticité du
modèle de soi vient de l'inférence par l'action, sans condition
développementale, et l'hypothèse principale est réfutée. Si MW surveille
aussi bien que MS : la disposition est générale et non spécifique à soi.

Dans tous les cas, rien sur le vécu ni sur un modèle de langage.

## Règle d'arrêt

Une seule exécution du plan. Calibration technique autorisée avant : les
modèles MS et MW doivent atteindre une perte proche de l'optimum de leur
régime, mesuré par l'oracle ; toute mesure M1 à M4 aperçue pendant cette
calibration sera divulguée.

## Reproduction

```bash
python -m unittest discover -s tests_research -p "test_mutable*.py" -v
python -m research.mutable_experiment --out artifacts/mutable-body --regimes MS
python -m research.audit_mutable --root artifacts/mutable-body
```
