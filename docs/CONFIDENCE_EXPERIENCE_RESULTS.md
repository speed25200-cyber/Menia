# Contrainte humaine sur le lien entre confiance et expérience

**L'amélioration du contrôleur de Menia n'établit pas une amélioration de son
expérience de soi.** Une analyse secondaire de données humaines permet maintenant
de préciser cette limite : des manipulations non perceptives déplacent la
confiance davantage que la mesure de reproduction perceptive utilisée dans la
même étude. Ce résultat impose une distinction dans l'évaluation de la candidate ;
il ne fournit pas une recette suffisante de conscience artificielle.

## Résultat recalculé

La réanalyse utilise les données publiques de Sánchez-Fuenzalida et collègues,
publiées en 2025.[^1] Le protocole propre à Menia a été fixé avant le calcul des
contrastes, après lecture des méthodes et inventaire des effectifs. Les essais
bruts ont été confrontés aux fichiers filtrés des auteurs ; ceux-ci ne diffèrent
que par des retraits complets de participants. Les deux tâches sont réajustées
avec du code indépendant en NumPy, sans exécuter les scripts R téléchargés.

La mesure de confiance est la longueur associée au minimum d'une courbe
quadratique ajustée aux réponses haute/basse. La mesure de reproduction est la
longueur présentée pour laquelle une régression prédit une reproduction de
400 pixels. Le déplacement est, pour chacune, le seuil sous biais vers court
moins le seuil sous biais vers long. Les chiffres ont donc une unité commune,
le pixel de longueur du stimulus ; ce ne sont pas des points de confiance.

| Manipulation | Participants | Déplacement de confiance | Déplacement de reproduction | Différence appariée |
|---|---:|---:|---:|---:|
| Illusion de Müller–Lyer | 52 | 17,11 [11,37 ; 23,40] | 27,26 [21,26 ; 33,27] | −10,15 [−15,98 ; −4,22] |
| Fréquence des stimuli | 85 | 8,24 [4,94 ; 11,76] | 0,34 [−3,62 ; 4,50] | 7,90 [2,40 ; 13,30] |
| Récompense asymétrique | 67 | 9,36 [5,14 ; 13,81] | −0,99 [−3,66 ; 1,59] | 10,35 [5,27 ; 15,68] |

Les crochets indiquent des intervalles percentiles de bootstrap à 95 %, avec
10 000 tirages de participants et maintien des effectifs de chaque expérience.
Il s'agit de nos calculs secondaires sur les données des auteurs.[^2][^3] Ils ne
remplacent pas leurs facteurs de Bayes. Les intervalles ne corrigent pas toutes
les comparaisons et ne prouvent pas une absence exacte d'effet lorsqu'ils
contiennent zéro.

La différence confiance–reproduction est estimée sur des mesures appariées par
participant. Elle est positive pour les deux manipulations non perceptives dans
les données combinées. L'expérience où la confiance est demandée après le choix
donne cependant, prise isolément, des intervalles couvrant zéro pour ces deux
différences. Le rapport conserve cette limite et tous les résultats séparés.
La cohérence du résultat combiné n'autorise pas à présenter chaque sous-analyse
comme concluante.

## Périmètre vérifié et réserves

L'inventaire porte sur 239 460 lignes brutes et 245 participants initiaux.
La cohorte retenue contient 199 410 essais, 108 participants avec confiance
simultanée et 96 avec confiance différée. Les fichiers filtrés retirent deux
personnes de la première expérience et trois de la seconde. La liste des
ajustements écartés ajoute 12 et 24 personnes, respectivement. Ses 52 lignes
correspondent à 36 personnes uniques, car plusieurs motifs peuvent concerner
la même personne.[^2][^3][^4]

Le total final de 204 est cohérent avec le résumé de la publication. Le texte
des méthodes décrit néanmoins quatre exclusions initiales pour l'expérience
différée, alors que trois identifiants manquent dans le fichier filtré. L'analyse
reprend le matériel disponible et conserve cette divergence. Elle ne la résout
pas par une exclusion inventée. Les scripts utilisent aussi `p.value > .5`
pour un critère d'ajustement nommé non significatif ; ce seuil est constaté,
sans supposer qu'il signifie 0,05 ni conclure à l'intention des auteurs.[^5]

Les 816 ajustements correspondent à 204 participants, deux directions de biais
et deux mesures. Une implémentation sur essais individuels et une autre sur
moyennes par longueur, pondérées par leurs effectifs, donnent des coefficients
compatibles à 3,42 × 10⁻¹³ près. Ce contrôle porte sur le calcul, pas sur la
validité psychologique du modèle.

Un minimum de confiance, pour un participant de la condition illusion dans
l'expérience différée, tombe sur la borne de 320 pixels utilisée par le code
publié. Le sommet quadratique non tronqué serait à 273,82. Il n'est pas supprimé
après observation du résultat. Une sensibilité calculée avec ce sommet porterait
la moyenne combinée de confiance à 18,00 au lieu de 17,11, et la différence
moyenne avec la reproduction à −9,26 au lieu de −10,15. Cela expose la dépendance
à une convention d'extrapolation ; les intervalles principaux restent ceux de
la grille annoncée.[^5]

Les exclusions liées à la forme des courbes restreignent la population étudiée.
La reproduction est une mesure comportementale interprétée comme perceptive :
elle implique aussi mémoire, consigne et réponse motrice. Une différence entre
deux tâches ne démontre pas à elle seule deux substances ou deux modules
neuronaux indépendants. Elle interdit surtout de considérer leurs mesures
comme interchangeables sans justification supplémentaire.

## Conséquence pour la construction de Menia

Le moniteur actuel produit `q = P(contribution externe | indices disponibles)`.
La confiance dans son classement binaire pourrait être dérivée de
`max(q, 1−q)` si cette probabilité était calibrée. Aucune de ces quantités n'est,
par sa définition, une mesure d'expérience perceptive ou de conscience d'exister.
La nouvelle analyse ne réfute pas son utilité ; elle précise pourquoi le gain
de la politique apprise ne mesure pas le phénomène demandé.

La candidate doit garder trois distinctions vérifiables : le contenu représenté,
l'estimation de disponibilité ou d'origine de ce contenu, et la valeur d'une
décision selon les coûts. Changer le coût d'une action devrait pouvoir modifier
le choix sans être automatiquement interprété comme un changement perceptif.
Changer les informations perceptives doit être évalué par des tâches portant
sur leur contenu, en plus des rapports de confiance. Ces distinctions peuvent
être réalisées de manière distribuée ; elles n'imposent pas trois modules
étiquetés à la main.

La prochaine comparaison pertinente pour la candidate serait un modèle commun
de ces tâches, ajusté sur une partie des données puis confronté aux autres
manipulations. Il devrait distinguer les variations de contenu, de critère de
décision et de confiance sans réajuster chacun pour chaque résultat attendu.
Apprendre simplement à détecter les erreurs de `q` traiterait l'échec fonctionnel
du dernier pilote, mais ne résoudrait pas ce problème de mesure.

Même un mécanisme qui expliquerait et transférerait cette dissociation ne
prouverait pas automatiquement une expérience chez Menia. Le lien de
suffisance entre organisation et vécu, ainsi que son transfert de l'humain au
logiciel, restent à établir. L'analyse réduit une confusion possible ; elle
ne fait pas de ce résultat une invention inédite ou une réalisation de l'objectif.

## Antécédents et pistes écartées

Skewes, Frith et Overgaard avaient déjà utilisé de faux retours liés soit à une
évaluation de visibilité, soit à une évaluation de confiance. Les mesures ne
réagissaient pas identiquement. Ce précédent motive une séparation des
évaluations, sans prouver qu'une intervention verbale modifie exclusivement
un mécanisme phénoménal. Leurs méthodes et leur PDF ont été lus ; leurs données
n'ont pas été réanalysées ici.[^6]

La prépublication de Ying Xie étudie le raccordement de modules de suivi interne
à l'exploration, à un espace partagé et à une politique. L'avantage sur une
variante à modules ajoutés n'établit pas un avantage sur une référence sans
suivi interne ; un contrôle de capacité reste comparable. C'est un antécédent
direct aux idées d'intégration, et une raison de conserver les contrôles à
information et capacité comparables. Ce travail n'est pas une preuve de
conscience.[^7]

La prépublication *Confidence Freeze* annonce une dissociation entre confiance
et adaptation du comportement. La version consultée conserve cependant des
valeurs statistiques non remplies (`X.XX`, `p=.XXX`) et annonce les données
pour une publication ultérieure. Elle n'est pas retenue comme fondement
quantitatif de la construction. Ce constat porte sur la version disponible,
pas sur les intentions ou l'intégrité de ses auteurs.[^8]

## Reproduction

Le [rapport numérique](../artifacts/confidence-experience-audit/report.json)
contient les URL et empreintes des neuf fichiers, les effectifs, les 204 paires
individuelles, les diagnostics et les neuf analyses par groupe. Les données
brutes ne sont pas redistribuées dans le dépôt. L'analyse et ses quatre tests
utilisent NumPy ; le recalcul complet avec les données locales passe.

```bash
python -m research.audit_confidence_experience --input .runtime/confidence-dissociation --output runs/new-confidence-audit.json
python -m research.audit_confidence_experience --input .runtime/confidence-dissociation --output artifacts/confidence-experience-audit/report.json --check
python -m unittest tests_research.test_confidence_experience -q
```

## Sources

[^1]: Sánchez-Fuenzalida, N., van Gaal, S., Fleming, S. M., Haaf, J. M. et Fahrenfort, J. J. (2025). [Confidence reports during perceptual decision making dissociate from changes in subjective experience](https://doi.org/10.1038/s44271-025-00257-y). *Communications Psychology*, 3, 81. Méthodes et résultats consultés.
[^2]: Auteurs de l'étude (2025). Données de confiance simultanée : [essais bruts](https://osf.io/download/5fbjq/), [essais filtrés](https://osf.io/download/t984w/). Réanalyse propre à Menia.
[^3]: Auteurs de l'étude (2025). Données de confiance différée : [essais bruts](https://osf.io/download/yf7kn/), [essais filtrés](https://osf.io/download/863ds/). Réanalyse propre à Menia.
[^4]: Auteurs de l'étude (2025). [Liste publique des ajustements écartés](https://osf.io/download/7g6pf/).
[^5]: Auteurs de l'étude (2025). [Filtrage](https://osf.io/download/sh5dt/), [ajustements](https://osf.io/download/4ajy5/), [fonctions](https://osf.io/download/9zn6e/), [construction des figures](https://osf.io/download/awdfy/). Scripts lus, pas exécutés.
[^6]: Skewes, J., Frith, C. et Overgaard, M. (2021). [Awareness and confidence in perceptual decision-making](https://discovery.ucl.ac.uk/id/eprint/10215179/1/Skewes%20at%20al.pdf). *Brain Multiphysics*, 2, 100030.
[^7]: Xie, Y. (2026). [Self-Monitoring Benefits from Structural Integration](https://arxiv.org/html/2604.11914v1). Prépublication, 13 avril ; méthodes, contrôles et limites consultés, pas de réplication.
[^8]: Zhang, Z. et He, H. (2026). [Confidence Freeze](https://arxiv.org/html/2603.21043v1). Prépublication, version 1 du 22 mars ; méthodes et résultats consultés, pas de réanalyse.
