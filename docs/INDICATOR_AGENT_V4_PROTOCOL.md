# Protocole pré-enregistré — L'agent à indicateurs, version 4

Rédigé le 23 septembre 2026 à 0 h 05 UTC (commit `41b9d4f`), après la publication de la
version 3 (`docs/INDICATOR_AGENT_V3_RESULTS.md`) et de ses analyses
exploratoires, **avant l'écriture du code de la version 4 et avant toute
exécution**. Mêmes règles que les versions 1 à 3.

## Diagnostic de la version 3

Lu sur ses artefacts publiés (paramètres de la graine 89, vies de test des
jeux R et M) ; aucune graine de la version 4 n'a servi.

1. **L'agent reste trop souvent immobile.** Dans 49 % des vies du jeu M, il
   ne fait aucun mouvement entre les pas 24 et 30. Sa croyance sur le corps
   après le changement est alors exacte dans 0,27 des pas, contre 0,98 quand
   il bouge au moins deux fois : c'est tout l'échec d'AE-2, et le module
   Corps n'a alors rien de nouveau à écrire (GWT-4).
2. **Ses valeurs ne distinguent pas les buts.** Quand l'énergie est haute et
   qu'un bon objet est connu, les trois valeurs apprises sont presque égales
   (−0,09, −0,05, −0,06) et l'agent choisit l'objet dans 41 % de ces
   décisions, rester dans 36 % et la recharge dans 23 %. La récompense
   externe, valeur des objets et malaises, est rare et bruitée : l'écart
   entre les buts se perd dans le bruit de l'ajustement.
3. **Rester est redécidé à chaque pas.** Sa valeur est celle de « rester un
   pas, puis agir », presque égale à celle d'agir tout de suite.
4. **L'énergie arrive en retard dans l'espace de travail.** Même une règle
   fixe ne s'y recharge que dans 0,62 des pas où l'énergie vraie est basse
   (analyse exploratoire publiée).
5. **La diffusion du corps compte, mais rarement.** Sur les lectures qui
   suivent un mouvement, l'AUROC du moniteur vaut 0,994 avec le corps
   diffusé et 0,786 sans ; ces lectures sont rares parce que l'agent bouge
   peu, et l'écart global n'est que de 0,028.

## Question

**Une priorité homéostatique — des alarmes intéroceptives et corporelles qui
prennent l'espace de travail, et des valeurs de buts apprises contre la
pulsion ressentie (apprentissage par renforcement homéostatique, Keramati et
Gutkin, 2014) — donne-t-elle l'agence à buts concurrents (AE-1), l'attention
dépendante de l'état (GWT-4) et l'incarnation (AE-2), sans perdre les six
propriétés robustes ?**

## Ce qui change, et rien d'autre

1. **Alarmes, avec priorité d'accès.** Avant l'attention par pertinence de la
   version 2, deux alarmes sont testées à chaque pas :
   - **intéroceptive** : un besoin lu par le module Intéro est sous 0,35 et
     l'espace de travail ne le reflète pas (il le garde à 0,35 ou plus, ou
     plus haut que la lecture de plus de 0,1) ; Intéro écrit ;
   - **corporelle** : la croyance du module Corps s'écarte de celle de
     l'espace de travail de plus de 0,5 en variation totale ; Corps écrit.

   L'alarme intéroceptive passe avant la corporelle. Sans alarme, le
   contrôleur de la version 2 décide. Un seul écrivain par pas. Les variantes
   « illimité », « hasard » et « tour de rôle » gardent leur contrôleur, sans
   alarme.
2. **Pulsion.** Les valeurs Q(recharge), Q(objet) et Q(rester), mêmes entrées
   qu'en version 3 (besoins par cases, distances, valeur de l'objet), même
   Monte-Carlo sur 16 pas avec γ = 0,9, même rejeu, même ε = 0,2, sont
   ajustées sur la **récompense interne** r_t − D_{t+1}, où
   D = (1 − ê)² + (1 − f̂)² est la pulsion calculée sur l'énergie ê et la
   satiété f̂ estimées par le module Intéro. La récompense externe du monde
   est inchangée, et c'est elle qui est mesurée.
3. **Rester est un engagement.** Comme la recharge et l'objet, le but
   « rester » n'est remis en jeu qu'à l'écriture de la vision ou de
   l'intéroception dans l'espace de travail, alarmes comprises.
4. **Départ sans préférence.** Quand plusieurs buts ont exactement la même
   valeur (valeurs nulles du premier tour), le but est tiré au hasard parmi
   eux ; la version 3 préférait l'objet par l'ordre de comparaison.
   L'évaluation de pertinence du contrôleur garde un ordre fixe.

Monde (trois objets), modules, moniteur, schéma d'attention, code de teinte,
planificateur, engagement des buts « objet » et « recharge » : **identiques à
la version 3**.

## Évaluation et prédictions fixées

Mêmes jeux, mêmes quinze variantes, mêmes mesures, **mêmes quatorze critères
et mêmes seuils** que les versions 1 à 3 (AE-1 : retour moyen des 3 derniers
tours − 3 premiers ≥ 0,2·Δ). Contrôle de validité : Δ > 0 sur chaque graine.

- **Prédictions principales** : AE-1, AE-2 et GWT-4 passent.
- **Prédictions secondaires** : GWT-1 passe (un arbitrage qui protège
  l'énergie rend les malaises d'énergie insensibles à la lésion de la
  vision) ; GWT-3 passe (un agent qui bouge rend le corps diffusé utile au
  moniteur).
- **Contrôle** : RPT-2, GWT-2, HOT-1, HOT-2, AST-1 et PP-1 passent encore.
- **Rien n'est prédit** pour RPT-1, HOT-3 et HOT-4, que ces changements ne
  visent pas.
- **Critère global** : les quatorze propriétés passent. Il n'est pas attendu.

Graines d'agent : **103, 107, 109** ; graine de développement **13**. Au plus
trois essais de développement, déclarés ; après eux, plus aucune
modification.

## Ce que le résultat dira

Si AE-1, AE-2 et GWT-4 passent, le point faible des trois premières versions
tenait au signal d'apprentissage et à l'accès des besoins à l'espace de
travail, pas à l'architecture. S'ils échouent, la partie « agent » reste le
point faible, avec un diagnostic plus précis. Dans tous les cas, ce protocole
ne mesure pas une expérience vécue.

## Exécution

`python -m research.indicator_experiment --version 4`, artefacts dans
`artifacts/indicator-agent-v4`, audit `research/audit_indicator_agent.py`
avec la même option, branché en CI. Résultats dans
`docs/INDICATOR_AGENT_V4_RESULTS.md`.
