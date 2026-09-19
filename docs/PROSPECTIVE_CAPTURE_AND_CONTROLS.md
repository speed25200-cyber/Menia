# Prévoir les erreurs : capture commune et comparateur de sortie enrichi

19 septembre 2026. **Préparation logicielle, pas résultat de prévision et pas
encore protocole complet de collecte.** Le Colab 16 garde ses sources figées.
La [capture de confiance seule](OUTPUT_CONFIDENCE_VALIDATION.md) est déjà
vérifiée sur quatre cas courts du Qwen3-4B préentraîné. Le présent ajout capture
aussi l'entrée et deux états internes dans la même génération.

## Une comparaison avec trois nombres ne suffit pas

Le maximum de probabilité, la marge des deux premiers tokens et l'entropie
résument fortement une distribution. Un lecteur de centaines de coordonnées
internes peut les battre simplement parce qu'il dispose d'une information
moins comprimée. Cela ne caractériserait pas nécessairement un mécanisme qui
évalue ses propres capacités.

Contre-exemple analytique, sans LLM : la réponse correcte est le token 0.
Deux distributions sont `(0,8 ; 0,2)` et `(0,2 ; 0,8)`. Maximum, marge et
entropie sont identiques ; les probabilités de réponse correcte sont pourtant
0,8 et 0,2. Si les deux cas sont équiprobables, un prédicteur limité à ces
trois résumés annonce 0,5 et a un Brier attendu de 0,25. Un lecteur du vecteur
qui produit la distribution peut annoncer 0,8 ou 0,2 et atteindre 0,16. Ce
gain de 0,09 existe sans ajouter de suivi de second ordre. La construction
emploie une réponse de référence fixée ; elle n'est pas une mesure de Menia.

De même, pour une tête linéaire `z = W h + b`, si `W` a plein rang colonne,
les logits complets déterminent `h` par
`h = (WᵀW)⁻¹ Wᵀ (z − b)`. Il s'agit d'une propriété algébrique conditionnelle.
Nous n'avons pas mesuré ici le rang de la tête Qwen ni reconstruit ses états
depuis ses logits. Cette remarque interdit seulement de qualifier toute
information située avant la tête de « privée » par définition.

## Comparateurs implémentés

`research.prospective_confidence_features` fournit des ensembles emboîtés :

| Nom | Variables disponibles avant le premier token | Dimension |
|---|---|---:|
| `inputOnly` | Caractères de la question, famille/difficulté, projection des embeddings d'entrée | 390 |
| `outputConfidence` | Famille/difficulté et les trois résumés de confiance | 9 |
| `inputConfidence` | Entrée et trois résumés | 393 |
| `finalControl` | Ensemble précédent et projection de l'état normalisé qui alimente la tête de sortie | 521 |
| `internal` | Ensemble précédent et projection de l'état intermédiaire | 649 |

Le contraste à privilégier est donc l'apport de l'état intermédiaire au-delà
de `finalControl`, en plus des comparaisons plus faibles conservées. Ce dernier
reste une projection de 128 coordonnées : il n'épuise pas l'information de la
sortie et ne constitue pas un observateur optimal. Un gain serait relatif à
ces variables et à ces lecteurs. Une référence par fréquence, les contrôles
de permutation et la variabilité d'ajustement restent à fixer dans le protocole.

L'identifiant du token le plus probable est conservé dans le journal mais n'est
pas traité comme une grandeur numérique dans ces variables : l'ordre des IDs
du vocabulaire ne mesure pas une proximité sémantique. Le token effectivement
tiré et les scores de la réponse terminée sont exclus de la prévision initiale.

## Questions et lecteurs préparés

`research.natural_error_questions` produit 3 456 questions, réparties entre
trois répétitions sur le même modèle. Chaque répétition contient, dans chacune
des six cellules famille/difficulté, 96 questions d'apprentissage, 32 de
validation et 64 de test. Les ensembles sont disjoints entre toutes les phases
et répétitions. Ils excluent 1 630 questions distinctes des anciens plans
numériques et des contrôles techniques. Le code reproductible ne fournit pas
les réponses futures du modèle ; nouveauté ici signifie absence des anciens
essais Menia, pas absence du préentraînement.

`research.natural_error_readouts` ajuste les coefficients sur l'apprentissage
seul et choisit la régularisation sur le Brier de validation, dans la grille
`0,001 / 0,01 / 0,1 / 1 / 10 / 100`. La normalisation des variables utilise
uniquement l'apprentissage. Les probabilités sont limitées à `[0 ; 1]`.
Une référence Beta par cellule et deux contrôles de correspondance complètent
les cinq ensembles de variables :

- `shuffledMiddle` apprend avec des états intermédiaires permutés dans chaque
  cellule, puis utilise un donneur d'apprentissage déterminé sans la réponse
  à la nouvelle question. Il conserve l'entrée, la confiance et l'état final
  du problème courant. La permutation conserve les marges et peut contenir
  des points fixes ; ce n'est pas une distribution nulle complète.
- `donorMiddle` applique le lecteur interne ajusté sur les vrais états à un
  état intermédiaire d'un donneur d'apprentissage de même cellule. Cela change
  l'entrée d'un lecteur externe, pas une activation du LLM lui-même.

Les partitions d'ajustement incomplètes, les doublons et les erreurs techniques
sont refusés explicitement. Trois tests synthétiques montrent qu'altérer tous
les exemples de test ou d'une autre répétition laisse l'ajustement identique,
que le signal artificiel prévu est appris, et que modifier l'état intermédiaire
courant ne change que le bras qui l'utilise. Deux autres tests vérifient les
3 456 questions, les exclusions, les graines et l'équilibre des partitions.

Ces composants sont prêts séparément. Le collecteur, les engagements de
prévision dans le journal, l'analyse complète et ses critères restent à fixer
et à tester avant de lancer ce nouveau jeu sur GPU. Aucune de ses réponses
Qwen3-4B n'a encore été collectée.

## Capture et vérifications

`research.joint_prediction_capture` observe les mêmes trois sites que l'ancien
moniteur, puis appelle la capture de confiance existante. Son callback reçoit
les états projetés et les statistiques du premier passage dans la tête, avant
le premier tirage. La trace de la réponse terminée n'est retournée qu'après
la génération. Aucun ancien collecteur n'est modifié.

Six tests passent sur un petit Qwen aléatoire : trois nouveaux tests et trois
contrôles existants du générateur. Ils vérifient l'identité du texte, des IDs,
des métadonnées, du générateur aléatoire, des états avec l'ancien collecteur et
de la confiance avec sa capture seule. Ils vérifient aussi l'instant du callback,
l'absence de la graine future dans les variables et le retrait de tous les hooks
si l'enregistrement échoue. Trois tests supplémentaires couvrent les dimensions
et l'emboîtement des comparateurs, le rejet des champs de réponse future et
l'indépendance du contrôle final aux changements de la seule projection médiane.
La capture commune n'a pas encore été exécutée sur Qwen3-4B/A100 à ce stade.

```sh
python -m unittest tests_language.test_joint_prediction_capture tests_research.test_prospective_confidence_features tests_research.test_natural_error_questions tests_research.test_natural_error_readouts -v
```

## Position par rapport à la littérature

Kadavath et al. séparent l'évaluation d'une réponse proposée, `P(True)`, de
la prévision de connaissance sans réponse proposée, `P(IK)`. Ils rapportent
aussi des difficultés de calibration sur de nouvelles tâches. Notre séparation
avant/après réponse suit donc un antécédent ; elle n'est pas une invention.
[Article](https://arxiv.org/abs/2207.05221).

Macar et al. distinguent détection d'une injection et identification de son
contenu, avec des mécanismes distribués et des effets causaux à différentes
couches. Ce résultat concerne une anomalie injectée : il ne valide pas à lui
seul une prévision d'erreur naturelle. Notre proposition de mesurer ces erreurs
reste une étape distincte, pas une reproduction de leur circuit.
[Texte consulté, version 1, §2, §4 et §5](https://arxiv.org/html/2603.21396v1).

Avant collecte, il reste à réunir ces composants dans un protocole complet,
avec contrastes et traitement des échecs fixés avant les réponses. Une amélioration
de prévision ne démontrerait ni que le LLM consulte lui-même ce lecteur, ni un
modèle de sa propre existence. Le passage à des décisions natives et ses
interventions causales resteraient à tester séparément.
