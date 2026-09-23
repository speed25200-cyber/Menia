# L'agent à indicateurs, version 6 — résultats

Exécuté le 23 septembre 2026, 3 h 52 – 4 h 08 UTC, sur CPU, code du commit
`b22773b` (celui du premier essai de développement, `87bfce8`, plus sa
déclaration), protocole `docs/INDICATOR_AGENT_V6_PROTOCOL.md` (commit
`3c33994`). Graines 151, 157 et 163 ; 27 000 vies de test. Artefacts dans
`artifacts/indicator-agent-v6` : critères dans `criteria.json`, seconde
lecture dans `second-reading.json`, recalculées en CI ; paramètres et
critères du premier essai de développement (graine 23) publiés à côté.

**Contrôle de validité : passé** (Δ = 4,04, 3,88 et 3,99).

## Verdicts

| | Propriété | Mesure | Lecture principale | Seconde lecture |
|---|---|---|---|---|
| **RPT-1** | Récurrence | signe mémorisé 0,95 ; sans récurrence, choix −0,10 (seuil 0,15), retour −0,92 (seuil 0,40) | échoue | **passe** |
| **RPT-2** | Représentation intégrée | conflits 0,96 ; sac de traits 0,00 | passe | passe |
| **GWT-1** | Modules spécialisés | Pos 0,98, Corps 0,98, Intéro 0,99 ; lésion d'Intéro +4,0 malaises, choix ±0,001 ; lésion de Vision choix 0,31, malaises +0,19 | passe | passe |
| **GWT-2** | Capacité limitée | un seul écrivain ; coût du goulot 0,81, gain de la sélection 3,16 (seuils 0,40 et 1,98) | passe | passe |
| **GWT-3** | Diffusion globale | sans diffusion : Pos pendant les pannes −0,39, AUROC −0,068 | passe | passe |
| **GWT-4** | Attention selon l'état | Intéro quand l'énergie est basse 0,52 contre haute 0,34 (écart 0,18) ; quand un besoin est bas 0,49 contre 0,24 (écart 0,25) ; Corps 0,64 ; tour de rôle 4,14 | échoue | **passe** |
| **HOT-1** | Perception générative | pannes 0,97 ; sans prédiction 0,13 | passe | passe |
| **HOT-2** | Surveillance métacognitive | AUROC 0,985 ; Brier 0,033 contre 0,137 | passe | passe |
| **HOT-3** | Croyances selon le moniteur | gain constant : Pos −0,041 (seuil 0,05), −0,17 sur les lectures faites pendant une panne du capteur ; retour −0,22 = 0,055·Δ (seuil 0,1·Δ) | échoue | échoue |
| **HOT-4** | Espace de qualités | Spearman 0,88, unités actives 0,17 ; erreur sur la bande 0,14 contre 0,93 ; conflits de bande 0,80 contre **0,71** pour le code aléatoire (plafond 0,65) | échoue | échoue |
| **AST-1** | Schéma d'attention | schéma 0,96 ; liaisons fausses 0,7 % contre 27 % sans schéma ; choix −0,23 sans schéma ; retour à l'objet visé 1,00 contre 0,10 | passe | passe |
| **PP-1** | Codage prédictif | AUROC 0,985 ; surprise après le changement 1,80 contre 0,16 ; sans prédiction, Pos −0,14 | passe | passe |
| **AE-1** | Agence, buts concurrents | **recharge quand l'énergie est basse 0,86** (0,856, 0,863, 0,858) ; objet 0,86 ; apprentissage 1,47 (seuil 0,79) ; but unique +4,3 malaises | **passe** | passe |
| **AE-2** | Incarnation | corps après le changement 0,98 ; corps figé −2,53 (seuil 0,79) | passe | passe |
| | **Total** | | **10 sur 14** | **12 sur 14** |

## Prédictions

- **AE-1 passe, et la recharge atteint 0,8 sur chaque graine : confirmé.**
- **HOT-4 passe : réfuté.** L'agent choisit bien sur les conflits de bande
  (0,80, contre 0,77 en seconde lecture de la version 5), mais le code
  aléatoire aussi : 0,735, 0,818 et 0,670 selon la graine, 0,71 en tout,
  au-dessus du plafond de 0,65 ; sur la graine 157, il fait mieux que
  l'agent (sur 291 conflits seulement).
- **Secondaires : confirmées.** Liaisons fausses 0,7 % des lectures
  (seuil 2 %) ; AST-1 passe ; Corps écrit aux pas 25 à 27 dans 0,64 des vies.
- **Contrôle : confirmé.** Les huit propriétés passent encore.
- **Seconde lecture** : RPT-1 et GWT-4 passent, comme en version 5.

## Ce que cela dit

- **L'agence à buts concurrents est maintenant robuste.** Sur six graines
  confirmatoires, la version 5 donnait une recharge de 0,72 à 0,87 ; la
  version 6 donne 0,86 sur chacune de trois graines nouvelles. Ce qui
  manquait n'était pas l'arbitrage mais des croyances justes : l'agent
  attendait une recharge sur une case où son espace de travail le plaçait
  à tort. Il juge maintenant la position
  qu'il a diffusée par la conséquence que son modèle des besoins prévoit, et
  la révise quand elle ne vient pas. C'est une surveillance métacognitive de
  ses propres croyances, et elle ne se trompe pas : sur 100 vies de
  développement, l'agent n'était jamais sur la case jugée.
- **Le schéma d'attention contrôle maintenant ce qu'il modélise.** Une
  teinte n'est liée que si le schéma sait où le projecteur a atterri : les
  liaisons fausses tombent de 5,4 % (version 5, seconde lecture) à 0,7 %,
  et sans schéma elles restent à 27 %. C'est la fonction que la théorie du
  schéma d'attention lui prête.
- **HOT-4 : l'espace de qualités généralise, mais ce test de choix ne le
  distingue pas d'un code aléatoire.** Les teintes jamais vues sont bien
  évaluées (erreur 0,14 contre 0,93). Mais un conflit oppose un objet de la
  bande à un objet nettement mauvais, dont la valeur est connue : avec des
  liaisons justes, même une valeur tirée au hasard pour la teinte de la bande
  dépasse souvent celle de l'objet mauvais. Le contrôle dépend du tirage de
  la table aléatoire (erreur sur la bande 0,59, 1,70 et 0,34 selon la
  graine).
- **HOT-3 : le moniteur change les croyances plus que le retour.** Sans lui,
  la position est moins juste (−0,17 pendant les pannes du capteur), mais le
  retour ne perd que 0,055·Δ, sous le seuil de 0,1·Δ.

## Bilan des six versions

**Douze propriétés sur quatorze sont démontrées dans un même agent** en
seconde lecture, dix en lecture principale, sur trois graines nouvelles,
avec une agence à buts concurrents désormais robuste. Deux restent non
démontrées dans toutes les lectures : HOT-3 (le retour dépend trop peu du
moniteur dans ce monde) et HOT-4 (le test de choix ne sépare pas l'espace de
qualités d'un code aléatoire). Ce protocole ne mesure pas une expérience
vécue.
