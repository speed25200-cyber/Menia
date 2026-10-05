# Plan pré-enregistré du pilote — Apprendre à survivre, mieux (avant le test 31)

Rédigé le 5 octobre 2026, avant tout code du pilote et toute exécution ;
l'heure est celle du commit. Au moment d'écrire, le test 30 n'est pas fini
(24 apprentissages sur 30), et ses premiers résultats bruts sont publiés.

## Pourquoi un pilote

Au test 30, l'apprentissage par renforcement (REINFORCE, 2 000 mises à
jour) n'a pas appris à survivre. Sur les apprentissages déjà finis, la
survie va de 0 à 0,47, sous la règle « événement » (0,723), pour **tous**
les masques, témoin compris. Le test sera très probablement non valide
(témoin sous 0,60) : il ne dit rien sur la question posée.

Avant de reposer la question (test 31), il faut un apprentissage qui
marche. Le régler en regardant la différence entre A et B fausserait le
test. **Le pilote ne règle donc l'apprentissage que sur le témoin B**
(« actions »), dont la seule tâche est d'apprendre ce que permet
l'événement du tour et ses actions passées. A n'est jamais appris pendant
le pilote.

## Ce que le pilote compare

Quatre réglages, tous avec la même récompense que le test 30 (1 par tour
vécu, rien d'autre), le même modèle et les mêmes masques :
- **P1** : REINFORCE du test 30, mais 6 000 mises à jour ;
- **P2** : acteur-critique. Une tête de valeur lit le token « Choix » et
  prédit le retour. Le retour est escompté (γ = 0,9). L'avantage est le
  retour moins la valeur. Perte de valeur × 0,5, entropie 0,01, taux 3e-4,
  4 000 mises à jour de 64 vies ;
- **P3** : P2 avec γ = 0,97 ;
- **P4** : P2 avec 4 passes par lot, et un rapport des probabilités borné
  à 0,2 (façon PPO).

## Règle de choix, fixée ici

- **Graines du pilote** : 100, 101 et 102 (jamais utilisées par un test).
- **Mondes du pilote** : 256 mondes du flux `[270926, 73]`, distincts des
  256 mondes de mesure (flux `[270926, 44]`).
- Pour chaque réglage, on apprend **B seulement**, sur les trois graines.
  On mesure sa survie, en prenant l'action la plus probable.
- **On retient le réglage où B survit le mieux en moyenne**, s'il atteint
  au moins **0,65**. À égalité à 0,005 près, on retient le plus simple
  (dans l'ordre P1, P2, P3, P4).
- **Si aucun réglage n'atteint 0,65**, le pilote échoue. Il est publié, et
  le test 31 n'est pas lancé sous cette forme.

## Ensuite

Le protocole du test 31 sera écrit après le pilote, avec le réglage
retenu. Il reprendra les critères du test 30 (SURV1, SURV2, validité), sur
des graines neuves (0 à 9) et les 256 mondes de mesure.

Tout est publié : réglages, survies par graine, et le choix.
