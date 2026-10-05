# Protocole pré-enregistré — Seulement survivre, avec un apprentissage qui marche (test 31)

Rédigé le 5 octobre 2026, **après le pilote** (`docs/TINY_SURVIVAL_PILOT_PLAN.md`)
et **avant tout apprentissage de A ou de C** avec ce réglage ; l'heure est
celle du commit. Seuils fixés, un échec est un résultat.

## D'où vient ce test (dit tel quel)

- **Test 30** (publié, non valide). De petits transformeurs appris de zéro
  n'apprenaient que de leur survie (1 par tour vécu), avec REINFORCE.
  Aucun n'a appris à survivre : le témoin B est resté à 0,405, sous le
  seuil de 0,60. Le test ne disait rien.
- **Le pilote** (publié). Quatre façons d'apprendre, réglées sur **le
  témoin B seul**, dans 256 mondes à part (flux `[270926, 73]`), graines
  100 à 102. A n'a jamais été appris. Selon la règle fixée d'avance, le
  réglage retenu est **P2, un acteur-critique** : B y survit à 0,688 en
  moyenne (0,664 ; 0,703 ; 0,695).

**Question (celle du test 30).** Un petit transformeur qui n'apprend que de
sa survie, et qui ne voit son passé qu'à travers ses propres états,
survit-il mieux qu'un témoin qui ne voit que ses actions passées, et mieux
que ce que permet l'événement du tour seul ?

## Ce qui est repris du test 30

Tout, sauf l'algorithme d'apprentissage (`docs/TINY_SURVIVAL_PROTOCOL.md`) :
- le monde, et les repères sur les 256 mondes de mesure (flux
  `[270926, 44]`) : règle « événement » 0,723 ; règle « besoins » 0,906 ;
- les tokens (3 par tour : l'événement, « Choix », l'action ; le
  soulagement n'est pas écrit) ;
- les masques A (« porte »), B (« actions »), C (« libre », publié sans
  seuil) ;
- le modèle (2 couches, dimension 64, 4 têtes) ;
- **la récompense : 1 par tour vécu après la décision, rien d'autre** ;
- les graines 0 à 9, et les flux de leurs mondes et de leurs tirages
  d'actions (`[270926, 71, graine, mise à jour, vie]`,
  `[270926, 72, graine, mise à jour]`) ;
- la mesure : survie sur les 256 mondes de mesure, l'agent prenant
  l'action la plus probable (à égalité exacte, l'autre que la dernière).

## L'apprentissage (le réglage P2 du pilote)

- **Acteur-critique.** Une tête de valeur lit le token « Choix » et prédit
  le retour.
- **Retour escompté** : la somme de γ^(k−1) sur les tours vécus après la
  décision, avec γ = 0,9.
- **Avantage** : le retour moins la valeur, normalisé dans le lot.
- **Pertes** : politique, plus 0,5 × la perte de valeur, moins 0,01 ×
  l'entropie.
- AdamW, taux 3e-4, **4 000 mises à jour de 64 vies**.

## Prédictions fixées (celles du test 30)

Intervalles à 95 % sur les 10 graines (loi de Student, 9 degrés de
liberté).

| | Prédiction | Critère |
|---|---|---|
| **SURV1** | Sans aucune cible, porter ses états le fait survivre mieux que ses seules actions | moyenne de (survie de A − survie de B, même graine) ≥ **0,08**, borne basse > 0 |
| **SURV2** | Il dépasse ce que permet l'événement du tour seul | moyenne de (survie de A − 0,723) ≥ **0,05**, borne basse > 0 |

**Critère global** : SURV1 et SURV2, test valide.

**Publié sans seuil** : C (« libre ») ; graine par graine ; les courbes
d'apprentissage ; la part des décisions où l'agent prend l'action de la
règle « besoins ».

**Validité** :
- masques : comme au test 30 (écart ≤ 1e-6, graine 0) ;
- **l'apprentissage a eu lieu** : survie moyenne de B ≥ **0,60**.

## Ce que le résultat dira

**Si SURV1 et SURV2 passent.** Sans aucune cible, sans pré-entraînement,
seulement en apprenant à survivre, un petit transformeur qui ne voit son
passé qu'à travers ses propres états survit mieux qu'un témoin qui ne voit
que ses actions, et mieux que ce que permet l'événement du tour. Il a
appris de lui-même à porter, dans ses états, quelque chose de ses besoins
cachés. Rien ne lui a nommé ces besoins. Ce que ces états portent n'est
pas mesuré ici.

**Si SURV1 passe et SURV2 échoue.** Porter ses états aide, mais il ne
dépasse pas assez ce que permet l'événement du tour.

**Si SURV1 échoue.** Seulement en apprenant à survivre, à cette taille et
avec cet apprentissage, porter ses états ne le fait pas survivre assez
mieux que ses seules actions.

**Si l'apprentissage n'a pas eu lieu** (B sous 0,60), le test n'est pas
valide.

**Ce que le résultat ne dira pas.**
- Ce n'est pas la preuve d'un ressenti.
- Il ne dira pas ce que portent les états.
- La récompense (la survie) vient du monde, qui connaît les besoins, mais
  elle ne nomme ni le besoin ni l'action à prendre.
- Le réglage a été choisi sur B seulement, dans d'autres mondes : il n'a
  pas été choisi pour favoriser A. Mais il a été choisi pour que B
  apprenne ; un réglage meilleur pour A pourrait exister.
- Ces modèles sont petits, le monde aussi.

## Exécution

- **Code** : `research/tiny_survival_ac.py` (à écrire), qui reprend le
  modèle et l'apprentissage P2 de `research/tiny_survival_pilot.py`, torch
  sur le processeur local.
- **Verdicts** : numpy seulement, vérifiés en CI.
