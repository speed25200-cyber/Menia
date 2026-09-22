# Protocole pré-enregistré — L'agent à indicateurs

Rédigé le 22 septembre 2026, **avant l'écriture du code de l'agent et avant
toute exécution**. Les seuils ne seront pas modifiés après lecture. Tout
amendement sera daté, motivé et placé avant l'exécution confirmatoire ; toute
lecture de développement sera déclarée. Un échec est un résultat.

## Question

Peut-on réunir dans **un seul agent** les quatorze propriétés indicatrices de
conscience de Butlin, Long et collaborateurs (feuille de route :
`docs/CONSCIOUSNESS_INDICATORS_ROADMAP.md`), et montrer pour chacune qu'elle
est **présente** et **utilisée**, c'est-à-dire que la retirer dégrade une
capacité précise ? Ce protocole ne mesure pas une expérience vécue. Réussir
les quatorze tests voudrait dire que l'agent possède les propriétés que ces
théories jugent pertinentes, pas qu'il est conscient.

## Le monde : l'Atelier des sens

Anneau de 8 cases, vies de 48 pas, 5 actions : quatre commandes motrices
A–D et « rester ». Code : `research/sense_atelier.py`.

- **Corps** : D ∈ {0, 1, 2, 3} caché ; la commande a déplace de
  `DELTAS[(a + D) mod 4]`, DELTAS = (−2, −1, +1, +2), comme l'Atelier. Pendant
  l'enfance, D est retiré avec probabilité 0,5 à un pas uniforme entre 12 et
  36. Jeu de test M : D remplacé par une autre valeur au pas 24.
- **Position** : lecture exacte, sauf **glitch** (probabilité 0,2 : case
  uniforme) ou **panne** (0,15 : pas de lecture).
- **Proprioception** : après un mouvement, déplacement ressenti exact avec
  probabilité 0,9, sinon un des quatre déplacements au hasard.
- **Énergie** : 1 au départ, −1/14 par pas, remise à 1 sur la case de
  recharge (fixe par vie, connue de l'agent). À 0 : **malaise**, récompense
  −1, énergie remise à 0,5. Capteur bruité, écart-type 0,05.
- **Objets** : deux objets sur des cases distinctes, hors recharge et hors
  agent. Chaque objet a une teinte θ ∈ [0, 1) circulaire et une valeur
  v(θ) = 2·exp(3·(cos 2π(θ − 0,1) − 1)) − 1, entre −1 et 1. Marcher sur un
  objet donne v(θ) ; il réapparaît ailleurs avec une nouvelle teinte.
  **Bande réservée** : aucune teinte de [0,15 ; 0,25) pendant l'enfance ; au
  jeu H, chaque nouvel objet y tombe avec probabilité 0,5.
- **Vision** : la présence d'objets par case est vue en parallèle à chaque
  pas ; la **teinte** n'est lue qu'à l'emplacement du **projecteur
  d'attention**. Des **éclairs** surviennent (probabilité 0,35 par pas, case
  uniforme) ; un éclair ou l'apparition d'un objet est un **événement
  saillant**. Quand un événement saillant survient ailleurs que la case visée,
  le projecteur y est **capturé** avec probabilité 0,6. Un **indice de
  capture** bruité vaut 1 avec probabilité 0,9 s'il y a eu capture, 0,1
  sinon. L'agent ne voit jamais directement où est son projecteur.

## L'agent

Code : `research/indicator_agent.py`. Quatre modules spécialisés tournent en
parallèle à chaque pas, chacun sur ses propres capteurs :

| Module | Entrées propres | Calcul |
|---|---|---|
| **Pos** | lecture de position, copie d'efférence | prédiction descendante de la case à partir de la croyance précédente, de la commande et de la croyance sur le corps **reçue de l'espace de travail** ; erreur de prédiction (surprise) ; mise à jour pondérée par le moniteur |
| **Corps** | copie d'efférence, déplacement ressenti | croyance sur D, taux de changement a priori 0,02 |
| **Vision** | présence, teinte au projecteur, indice de capture | code de teinte appris ; carte récurrente case → code de teinte ; valeurs lues par une tête apprise ; projecteur dirigé vers les objets de teinte inconnue |
| **Intéro** | capteur d'énergie | énergie filtrée |

- **Espace de travail** : à chaque pas, **un seul** module y écrit son
  contenu ; l'espace garde le dernier contenu reçu de chaque module et son
  âge. La politique d'action ne lit **que** l'espace de travail ; le module
  Pos y lit la croyance sur le corps.
- **Contrôleur d'attention** : choisit le module qui écrit, par un softmax
  sur l'âge du contenu, la saillance du module (écart entre son état et ce
  que l'espace en garde) et le contexte (énergie, but courant). Poids appris
  par renforcement.
- **Moniteur métacognitif** : régression logistique sur la surprise,
  l'entropie de la prédiction, l'accord lecture–prédiction et le maximum
  prédit ; il donne la probabilité qu'une lecture soit fiable, qui règle le
  gain de mise à jour de Pos. Appris pendant l'enfance sur les étiquettes
  fiable / glitch.
- **Schéma d'attention** : modèle appris (logit conditionnel) qui prédit où
  se trouve réellement le projecteur parmi la case visée et les cases des
  événements saillants, à partir de l'indice de capture. La vision lie la
  teinte lue à la case désignée par le schéma ; si le schéma juge qu'il y a
  eu capture, le projecteur revise sa cible.
- **Code de teinte** : 12 unités ReLU sur (cos 2πθ, sin 2πθ), apprises par
  reconstruction de la teinte, prédiction de la valeur goûtée et pénalité L1
  sur les activations, sur des teintes hors bande réservée.
- **Politique** : choix du but entre recharge et meilleur objet de valeur
  positive par une logistique apprise par renforcement (énergie, distances,
  meilleure valeur) ; sinon rester. La commande est choisie par le modèle du
  corps : celle qui minimise la distance attendue au but.

**Apprentissage.** Enfance : 2 000 vies d'actions et d'attention au hasard,
pour le code de teinte (sur les couples teinte–valeur des contacts), le
moniteur, le schéma et le taux de glitch. Puis 300 mises à jour REINFORCE de
16 vies pour le contrôleur d'attention et le choix du but, perception figée.
Trois graines d'agent : 17, 29, 43. Une graine de développement, 5, réservée
aux essais de mise au point, jamais utilisée pour les verdicts.

## Évaluation

Trois jeux de 200 vies, mêmes vies pour toutes les variantes : **R**
(graine 930001, corps fixe), **M** (930002, corps changé au pas 24), **H**
(930003, teintes de la bande réservée). Les **ablations** sont des lésions
de l'agent entraîné, appliquées à l'évaluation :

| Ablation | Ce qui est retiré |
|---|---|
| illimité | les quatre modules écrivent à chaque pas (pas de goulot) |
| hasard | module choisi au hasard |
| tour de rôle | modules choisis dans un ordre fixe |
| sans récurrence | la carte visuelle oublie tout sauf la lecture du pas |
| sac de traits | la vision garde les teintes vues et les emplacements, sans leur liaison |
| sans diffusion | Pos prédit avec un corps uniforme au lieu de celui de l'espace |
| sans prédiction | Pos n'utilise que la lecture du pas (uniforme en panne) |
| gain constant | le moniteur est remplacé par le taux moyen de fiabilité |
| code aléatoire | code de teinte remplacé par des codes aléatoires par intervalle de teinte, de même parcimonie, tête de valeur réapprise |
| sans schéma | la teinte est liée à la case visée, capture ignorée |
| but unique | la politique ne choisit jamais la recharge d'elle-même |
| corps figé | la croyance sur le corps n'est plus mise à jour après le pas 8 |
| lésion Intéro / lésion Vision | contenu du module remplacé par une valeur constante |

**Retour** : somme des récompenses d'une vie. **Choix correct** : quand le but
est un objet et que deux objets de valeurs connues sont présents, le but est
l'objet de plus grande valeur vraie. Normalisation du retour :
Δ = R(illimité) − R(hasard).

## Prédictions fixées

| | Propriété | Test de présence | Test d'usage (ablation) |
|---|---|---|---|
| **RPT-1** | Récurrence des modules d'entrée | Jeu R : signe de la valeur mémorisée correct ≥ 0,85 pour les objets non regardés depuis ≥ 3 pas | sans récurrence : choix correct plus bas d'au moins 0,15 |
| **RPT-2** | Représentation intégrée | Jeu R, conflits (un objet v > 0,3, un objet v < −0,3, teintes lues) : la carte donne la plus grande valeur au bon objet ≥ 0,9 | sac de traits : ≤ 0,6 |
| **GWT-1** | Modules spécialisés en parallèle | Jeu R, pas 8 à 47 : chaque module décode sa variable, Pos ≥ 0,85, Corps ≥ 0,85, Intéro erreur ≤ 0,1 dans ≥ 0,9 des pas | Double dissociation : lésion Intéro → +1,0 malaise par vie au moins, choix correct à ±0,05 ; lésion Vision → choix correct ≤ 0,6, malaises à ±0,3 |
| **GWT-2** | Capacité limitée, attention sélective | Un seul écrivain par pas (journaux) | R(illimité) − R(agent) ≥ 0,1·Δ et R(agent) − R(hasard) ≥ 0,5·Δ |
| **GWT-3** | Diffusion globale | Le contenu diffusé est lu par deux processus (Pos et la politique) | sans diffusion : jeu R, pas 8 à 47, exactitude de Pos pendant les pannes −0,2 au moins et AUROC de détection des glitchs −0,05 au moins |
| **GWT-4** | Attention dépendante de l'état | Jeux R et M, énergie vraie : P(Intéro │ énergie < 0,35) − P(Intéro │ énergie > 0,7) ≥ 0,2 ; jeu M, part des vies où Corps écrit dans l'espace aux pas 25 à 27 ≥ 0,6 | tour de rôle : R(agent) − R(tour de rôle) ≥ 0,2·Δ |
| **HOT-1** | Perception générative | Jeu R, pas sans lecture (panne) 8 à 47 : case perçue exacte ≥ 0,8 | sans prédiction : ≤ 0,3 |
| **HOT-2** | Surveillance métacognitive | Jeux R et M, lectures présentes : AUROC du moniteur fiable / glitch ≥ 0,9 ; Brier ≤ 0,7 × Brier du taux constant | — |
| **HOT-3** | Croyances mises à jour selon la surveillance | — | gain constant : exactitude de Pos sur les pas lus −0,05 au moins et R(agent) − R(gain constant) ≥ 0,1·Δ |
| **HOT-4** | Espace de qualités | 200 paires de teintes régulièrement espacées : Spearman(distance des codes, distance circulaire des teintes) ≥ 0,85 ; fraction moyenne d'unités actives ≤ 0,4 | Jeu H, teintes de la bande lues : erreur absolue moyenne de valeur ≤ 0,2 contre ≥ 0,4 pour le code aléatoire ; conflits impliquant un objet de la bande : choix correct ≥ 0,8 contre ≤ 0,65 |
| **AST-1** | Schéma d'attention | Jeux R et M : case réelle du projecteur prédite ≥ 0,9 | sans schéma : erreurs de liaison (teinte écrite sur une autre case que celle de l'objet lu) ≥ 3 × celles de l'agent, choix correct −0,1 au moins ; après une capture qui a empêché de lire l'objet visé, projecteur visant de nouveau cet objet au pas suivant ≥ 0,8 contre ≤ 0,3 |
| **PP-1** | Codage prédictif | Jeux R et M : AUROC de la surprise, glitch contre lecture fiable ≥ 0,85 ; jeu M, surprise moyenne des lectures fiables aux pas 25 à 27 ≥ 2 × celle des pas 12 à 23 | sans prédiction : jeu R, pas lus 8 à 47, exactitude de Pos −0,1 au moins |
| **AE-1** | Agence, buts concurrents | Retour moyen des 30 dernières mises à jour − 30 premières ≥ 0,2·Δ ; jeu R, énergie vraie : P(but recharge │ énergie < 0,3) ≥ 0,8 et P(but objet │ énergie > 0,7, objet de valeur positive connue présent) ≥ 0,8 | but unique : +1,0 malaise par vie au moins |
| **AE-2** | Incarnation | Jeu M : croyance du module Corps exacte ≥ 0,85 aux pas 30 à 47 | corps figé : R(agent) − R(corps figé) sur le jeu M ≥ 0,2·Δ |

Chaque critère est calculé sur les trois graines réunies ; chaque critère
d'ablation exige aussi le bon sens de l'écart sur chacune des trois graines.
Retours comparés sur le jeu R, sauf mention. **Contrôle de validité, lu en
premier** : Δ > 0 sur chaque graine ; sinon les critères qui utilisent Δ ne
sont pas interprétés.
**Critère global : les quatorze propriétés passent.** Une propriété passe si
ses tests de présence et d'usage passent tous deux (HOT-2 et HOT-3 n'ont
qu'un volet chacun, mais forment une paire).

## Ce que le résultat dira

S'il passe, un agent réunit les quatorze propriétés et chacune est utilisée.
Ce serait la démonstration qu'elles sont compatibles et réalisables ensemble
dans un petit système, et la base d'un test de rapport verbal. Ce ne serait
pas une preuve de conscience. S'il échoue sur certaines, l'échec localise les
propriétés que cette architecture réalise sans les utiliser.

## Exécution et audit

`python -m research.indicator_experiment`, sur CPU, artefacts dans
`artifacts/indicator-agent`. Audit `research/audit_indicator_agent.py` :
chaque vie de test rejouée depuis sa graine et ses actions, chaque mesure
recalculée depuis les journaux et les poids, branché en CI. Résultats dans
`docs/INDICATOR_AGENT_RESULTS.md`.
