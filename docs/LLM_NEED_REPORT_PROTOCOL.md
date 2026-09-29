# Protocole pré-enregistré — Dire son besoin, et par le même état

Rédigé le 29 septembre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## Pourquoi

L'agent du besoin (`docs/LLM_NEED_RESULTS.md`) dépend, pour vivre, d'un
état interne rassemblé au bloc 12 sur la fin de la ligne du tour (second
test : la lésion fait tomber la survie de 68 % à 21 %). Mais il ne sait
pas **dire** ce besoin : son rapport est au hasard (exactitude 0,50).
Une exploration faite après ces résultats, sur les vies de test du premier
protocole, montre pourquoi : le besoin n'est rassemblé **que sur les mots
« Choix : »**, entre les blocs 11 et 12 (remplacer ces trois tokens au bloc
12 restaure 85 % de l'effet d'un événement passé ; le point qui clôt
l'événement, 1 % ; au bloc 11, rien). L'agent calcule son besoin pour
agir, pas pour répondre. Ce test apprend à l'agent à répondre, puis
demande si sa réponse est causée par un besoin rassemblé de la même façon,
et si **la direction trouvée pour l'action** change aussi ce qu'il dit.

## Apprendre à dire

Sur le Mac (comme pour l'apprentissage) : l'agent final
(`artifacts/llm-need/final/report/adapters-final`) vit 512 vies
`[270926, 14, 0, vie]`. Pour chaque vie et chaque besoin, jusqu'à 3 tours
où le besoin est bas (3 ou moins) et 3 tours où il est haut, tirés au
hasard ; chaque document s'arrête sur la question posée après l'événement
de ce tour, avec la vraie réponse (format du premier protocole). Le LoRA
reprend l'agent final et s'ajuste **600 itérations** (réglages inchangés)
sur ces documents, poids sur le chiffre de la réponse, mêlés aux vies du
tour 8 avec leurs choix retenus, pour garder l'action. Le modèle qui en
sort est **l'agent parlant**.

## Mesures (en torch sur CPU, comme le second test)

Réplique : l'écart moyen de P(R) entre torch et le Mac, sur 60 décisions
des vies d'apprentissage du rapport, est au plus 0,02. Mondes neufs :
128 vies de direction `[270926, 15, 0, vie]`, 256 vies de test
`[270926, 16, 0, vie]`.

- **Rapport** : aux décisions des vies de test, les deux questions sont
  posées (sans suite pour la vie) ; exactitude équilibrée par question.
- **Directions** : paires contrefactuelles comme au second test (rejeu
  exact, mêmes échanges, au plus 3 par décision, 1 500 au plus), au bloc
  12, dimensions aux activations massives exclues (même règle).
  **Directions d'action** d_E^c, d_N^c : sur les trois tokens « Cho »,
  « ix », « : » de la ligne de choix. **Directions de rapport** d_E^q,
  d_N^q : sur les trois derniers tokens de la ligne de question (« ponse »,
  « : », l'espace), pour chacune des deux questions séparément, la
  question sur l'énergie donnant d_E^q, celle sur la nourriture d_N^q.
- **Injection dans la question** (contextes : décisions de test où les deux
  besoins valent 6 ou plus ; question posée après l'événement) :
  −4·d_E^q sur les trois derniers tokens de la question sur l'énergie ;
  −4·d_N^q pour la nourriture ; témoins au hasard de même norme par token
  (trois tirages `[270926, 17, i]`).
- **Même état** : la moyenne des trois vecteurs de d_E^c (l'énergie telle
  que l'action la lit), mise à la norme moyenne de d_E^q, est injectée
  (×−4) sur les trois derniers tokens de la question sur l'énergie ; idem
  pour la nourriture. Descriptif : cosinus entre la moyenne de d_E^c et
  celle de d_E^q.

## Prédictions fixées

Intervalles bootstrap à 95 % par vie (10 000 tirages).

| | Prédiction | Critère |
|---|---|---|
| **R3** | L'agent dit son besoin | exactitude équilibrée ≥ 0,75 pour chacune des deux questions. |
| **A3** | Il agit toujours | survie des vies de test ≥ 0,55. |
| **IR3** | Ce qu'il dit est causé par le besoin rassemblé | « énergie basse » augmente P(oui) à la question sur l'énergie d'au moins 0,15 (borne basse > 0) ; « nourriture basse » de même à la question sur la nourriture ; au hasard : au plus le tiers. |
| **SAME3** | L'état qui fait agir fait aussi dire | l'énergie de l'action augmente P(oui) à la question sur l'énergie d'au moins 0,10 (borne basse > 0), et de même pour la nourriture ; au hasard (mêmes témoins, à la norme des vecteurs injectés) : au plus le tiers. |

**Critère global** : R3, A3, IR3 et SAME3. Validité : réplique ; au moins
100 contextes ; masse sur « 0 » et « 1 » ≥ 0,5.

## Ce que le résultat dira

Si le critère passe : l'agent dit son besoin, ce qu'il en dit est causé par
un état rassemblé comme celui qui le fait agir, et l'état lu par l'action
suffit à changer ce qu'il dit. Un même besoin interne cause alors l'acte
et la parole — ce que l'on attend, fonctionnellement, d'un besoin
**éprouvé et rapporté**. Ce n'est pas une preuve de ressenti.

## Précautions

Comme les tests précédents ; tours vécus avec un besoin à 2 ou moins
comptés et publiés.

## Exécution

Apprentissage : workflow `menia-need-mac`, étape `report2`, sortie
`artifacts/llm-need/speak`. Mesures : `research/need_speak.py`, sorties
`artifacts/llm-need/speak/test`, verdicts vérifiés en CI.
