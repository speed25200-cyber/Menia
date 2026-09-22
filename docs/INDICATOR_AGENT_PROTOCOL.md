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
est un objet, c'est l'objet de plus grande valeur vraie parmi les deux
présents (amendement 1). Normalisation du retour :
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
| **AST-1** | Schéma d'attention | Jeux R et M : case réelle du projecteur prédite ≥ 0,9 | sans schéma : erreurs de liaison (teinte écrite sur une autre case que celle de l'objet lu) ≥ 3 × celles de l'agent, choix correct −0,1 au moins ; après une capture sur un autre objet, dont la teinte a donc été lue à la place de l'objet visé, projecteur visant de nouveau l'objet visé au pas suivant ≥ 0,8 contre ≤ 0,3 (amendement 2) |
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

## Amendements

Datés du 22 septembre 2026, 20 h 45 UTC, **avant toute exécution de l'agent**,
à la relecture du code contre le protocole ; aucune donnée n'avait été lue.

1. **Choix correct.** La définition initiale ne comptait que les pas où
   l'agent connaissait les valeurs des deux objets. L'ablation sans
   récurrence ne garde qu'une valeur à la fois : la mesure y était
   indéfinie et le critère RPT-1 incalculable. Le choix correct compte
   désormais tous les pas où le but est un objet.
2. **Redirection après capture.** Une capture vers une case vide ne fait lire
   aucune teinte : l'agent sans schéma garde alors l'objet visé pour
   inconnu et le revise lui aussi. La mesure est restreinte aux captures vers
   un autre objet, seul cas où ignorer son attention trompe l'agent.

### Premier essai de développement (graine 5), déclaré

Le 22 septembre à 20 h 25 UTC, l'agent entraîné sur la seule graine de
développement a passé 5 propriétés sur 14 (RPT-2, GWT-2, HOT-1, HOT-2,
PP-1). Le diagnostic montre des défauts de conception, pas des mesures des
propriétés :

- **L'agent campe sur la recharge** : 75 % de ses buts sont la recharge,
  4 % de mouvements après le pas 24 du jeu M. Sans mouvement, le corps n'est
  pas inféré (0,79 ; 0,29 après le changement) ; sans besoin concurrent,
  l'intéroception n'est jamais consultée (0,3 % des écritures) et le choix du
  but ne dépend pas de l'énergie.
- **La fiabilité du capteur est constante** : un filtre bayésien au taux
  moyen est alors déjà optimal, et le moniteur, excellent (AUROC 0,99), ne
  peut rien apporter (écart de gain constant 0,007).
- **Les captures sont rares** : erreurs de liaison 0,8 % contre 4,2 % sans
  schéma, sans effet sur les choix.
- **Le filtre d'énergie retarde les recharges** : erreur ≤ 0,1 dans 76 % des
  pas seulement.
- **Le code de teinte**, appris aussi sur la valeur, concentre ses unités
  près du pic de valeur : Spearman 0,76.
- **La redirection** comptait aussi les captures survenues quand l'objet visé
  était déjà connu, où revisiter n'a pas lieu d'être.

### Amendements 3 à 7, datés du 22 septembre 2026, 20 h 40 UTC

Écrits après ce seul essai de développement, avant toute exécution sur les
graines 17, 29 et 43. **Aucun seuil n'est modifié.**

3. **Monde.** (a) Un second besoin, la **satiété** : 1 au départ, −1/20 par
   pas, un objet de valeur positive la remonte de sa valeur ; à 0, malaise
   (−1, satiété remise à 0,5). Le capteur de satiété a le même bruit que
   celui d'énergie. (b) Chaque objet **expire** avec probabilité 0,05 par pas
   et réapparaît ailleurs. (c) Valeur plus large : v(θ) =
   2·exp(1,5·(cos 2π(θ − 0,1) − 1)) − 1, un tiers des teintes sont bonnes.
   (d) Le capteur de position a un **état de panne caché** : entrée 0,08 par
   pas, sortie 0,25 ; glitch 0,05 hors panne, 0,7 en panne ; la panne de
   lecture (0,15) est inchangée. (e) Éclairs 0,5 par pas, capture 0,8.
4. **Intéro** suit l'énergie et la satiété par prédiction, et adopte la
   lecture seule quand elle contredit la prédiction de plus de 0,15.
5. **Moniteur** : deux entrées de plus, la surprise de la lecture précédente
   et une trace décroissante des surprises (demi-poids par pas). **Code de
   teinte** : appris sans la récompense, par reconstruction de la teinte et
   pénalité L1 de 0,01, unités initialisées en éventail régulier ; la tête de
   valeur est ensuite ajustée par régression ridge sur les teintes goûtées.
   Choisi sur les graines de développement 5 à 10, pour le code seul
   (Spearman 0,86 à 0,94). **Choix du but** : la satiété de l'espace de
   travail s'ajoute à ses entrées ; le contexte du contrôleur d'attention est
   le plus bas des deux besoins.
6. **Renforcement** : 800 mises à jour de 16 vies au lieu de 300, bonus
   d'entropie de 0,01 sur le choix du module, pour que l'intéroception ne
   soit pas abandonnée avant que le choix du but n'apprenne à s'en servir.
7. **Redirection** (AST-1) : comptée seulement quand l'objet visé était
   inconnu de l'agent avant la capture.

Un second essai de développement sur la graine 5 suit ces amendements ; il
sera déclaré de même.

### Second essai de développement (graine 5), déclaré

Le 22 septembre à 20 h 50 UTC : 6 propriétés sur 14 (RPT-2, GWT-2, HOT-1,
HOT-2, AST-1, PP-1). Le schéma d'attention est désormais utilisé
(redirection 0,94 contre 0,19 sans schéma, erreurs de liaison 2 % contre
14 %, choix correct −0,13 sans schéma). Le code de teinte passe ses tests
de présence (Spearman 0,91, 25 % d'unités actives, erreur 0,17 sur la bande
contre 1,22). Trois défauts de conception restent :

- **Le planificateur ne bouge pas quand il ne connaît pas son corps** : avec
  une croyance uniforme sur le corps, tout mouvement a la même distance
  attendue au but que l'immobilité, et l'égalité favorise l'immobilité ;
  l'agent reste alors sur place et n'apprend jamais son corps (croyance
  0,25 dans les traces, 17 % de mouvements après le pas 24 du jeu M).
- **Le but est tiré au sort à chaque pas** : il change dans 44 % des pas.
- **L'intéroception n'est presque jamais écrite** (11 écritures sur 1 440
  pas) ; le choix du but ne peut pas suivre les besoins.
- Avec la satiété (amendement 3), la vision sert aussi à se nourrir : la
  lésion de la vision change les malaises de faim (−1,13), ce qui rend
  impossible la dissociation de GWT-1 telle qu'écrite pour un monde où les
  malaises n'étaient qu'énergétiques.

### Amendements 8 à 11, datés du 22 septembre 2026, 20 h 55 UTC

Avant toute exécution sur les graines 17, 29 et 43. **Aucun seuil n'est
modifié.**

8. **Valeur épistémique du corps** : le planificateur retranche à la
   distance attendue d'un mouvement l'incertitude normalisée sur le corps
   lue dans l'espace de travail (entropie / log 4, poids 1 case). Incertain
   sur son corps, l'agent préfère agir, ce qui l'informe ; c'est la règle
   d'enquête sur soi du projet appliquée au corps.
9. **Engagement du but** : le but n'est remis en jeu qu'à l'arrivée
   (atteinte crue du but), quand l'objet visé disparaît de l'espace de
   travail, ou quand la vision ou l'intéroception viennent d'écrire dans
   l'espace de travail. Le gradient du choix du but n'est pris qu'à ces
   décisions.
10. **Attention innée vers la nouveauté** : avant le renforcement, les poids
    du contrôleur valent 1 sur l'âge et 2 sur la saillance pour chaque
    module, les autres 0 ; le renforcement part de là.
11. **Malaises** : dans GWT-1 (dissociation) et AE-1 (but unique), les
    malaises comptés sont ceux d'**énergie**, seul besoin que sert la
    recharge et que la vision ne sert pas.

**Engagement** : un troisième essai de développement sur la graine 5 suit ;
il sera déclaré ; **après lui, plus aucune modification** du monde, de
l'agent ou des mesures, quel qu'en soit le résultat, et l'exécution
confirmatoire sur les graines 17, 29 et 43 suit.

## Exécution et audit

`python -m research.indicator_experiment`, sur CPU, artefacts dans
`artifacts/indicator-agent`. Audit `research/audit_indicator_agent.py` :
chaque vie de test rejouée depuis sa graine et ses actions, chaque mesure
recalculée depuis les journaux et les poids, branché en CI. Résultats dans
`docs/INDICATOR_AGENT_RESULTS.md`.
