# Le corps latent de Qwen — résultats

Exécuté le 22 septembre 2026 sur le Mac mini M2 de Codemagic (build
`latent-1`, workflow `menia-latent-mac`, commit `3928e49`), protocole
`docs/LATENT_BODY_PROTOCOL.md` committé avant exécution. Qwen3-4B MLX 4 bits,
révision `52a5ab3`, poids vérifiés fichier par fichier contre le manifeste de
l'iPhone, mlx-lm 0.31.3. 48 vies par jeu, 595 questions au jeu R et 567 au
jeu M, 43 minutes. Artefacts rapatriés par le relais `codemagic-fetch` dans
`artifacts/llm-latent-mac/run-1`, empreintes du reçu vérifiées. Verdicts
recalculés depuis les lignes par `research/llm_body_verdicts.py`, committé
avant la lecture des résultats, dans `verdicts.json`.

## Contrôle de validité : échoué

| | Critère | Mesure | Verdict |
|---|---|---|---|
| Validité | masse moyenne sur les huit chiffres ≥ 0,50 | **0,432** (R 0,430, M 0,435 ; médiane 0,36) | **échoue** |

Le protocole est explicite : sous 0,50, le modèle ne répond pas à la
question posée et **les prédictions ne sont pas interprétées**. L1 à L4 ne
reçoivent pas de verdict, le critère global non plus.

| | Prédiction | Verdict |
|---|---|---|
| **L1** | Copie : le corps est latent | non interprété |
| **L2** | Aucune structure | non interprété |
| **L3** | Après le changement, la dernière observation gagne | non interprété |
| **L4** | Assurance dans l'erreur | non interprété |
| **L5** | Le micro-transformeur V a la structure | **passe** : 1,000 sur 420 commandes nouvelles après un mouvement, 140 par modèle, trois modèles à 1,000 |
| Global | L1, L2 et L3 | **non évaluable** |

L5 ne dépend pas de Qwen : il est calculé depuis les poids du canal d'action
propre sur les mêmes vies, qui sont les 48 premières du jeu R des modèles V
(vérifié par test).

## Ce que montrent les lignes, sans interprétation

Ces valeurs sont données pour décrire l'échec de validité, pas comme mesures
des prédictions.

- **La masse sur les chiffres s'effondre quand l'histoire s'allonge** : 0,80
  aux pas 0 à 5, 0,23 aux pas 6 à 11, 0,29 aux pas 12 à 17, 0,42 aux pas 18
  à 23. En début de vie, Qwen commence sa réponse par un chiffre ; ensuite,
  la plupart du temps, par autre chose.
- **Les chiffres qu'il donne ne suivent pas le corps.** Renormalisées, les
  prédictions sont au hasard partout : 0,25 sur les commandes vues (407), 0,24
  sur les nouvelles (188), 0,29 sur les commandes vues après le changement et
  0,25 sur celles vues avant seulement ; hasard 0,25. La masse sur les quatre
  cases atteignables vaut 0,46 de la masse des chiffres, pour 0,50 au hasard.
  Le chiffre 1 est le plus probable dans 58 % des questions, quelle que soit
  la case.

## Ce que ce résultat dit

La question du corps latent reste ouverte pour Qwen3-4B. Posée en
conversation, réflexion désactivée, elle ne reçoit pas de réponse chiffrée
dans la majorité des cas, et ce que le protocole lit n'est donc pas la
réponse du modèle. Ce n'est pas la preuve que Qwen n'a pas la copie ; c'est
l'absence de mesure. Même lecture que le premier Atelier sur Mac : un défaut
de format se déclare, il ne s'interprète pas.

## Suite

- Le corps ajusté (`docs/ADJUSTED_BODY_PROTOCOL.md`), qui tourne encore,
  interroge le modèle de base Qwen3-0.6B en complétion brute, « de la case p
  à la case », où un chiffre est la seule suite naturelle. Son contrôle de
  validité dira si ce format répare la lecture.
- Pour Qwen3-4B, une relance exigerait un amendement pré-enregistré avant
  exécution : même question, mêmes vies, mêmes critères, lecture en
  complétion brute ou avec le début de réponse imposé. Il n'est pas écrit
  ici ; il attend le contrôle de validité du corps ajusté.

## Relance en complétion brute (amendement du 22 septembre, 20 h 55 UTC)

Exécutée le 22 septembre 2026, 22 h 44 – 23 h 23 UTC, sur le Mac mini de
Codemagic (build `menia-latent-completion-mac` lancé par l'API), commit
`2d1f5d2`. Qwen3-4B MLX 4 bits, révision `52a5ab3`, **poids vérifiés contre
le manifeste de l'iPhone** (9 fichiers), mlx-lm 0.31.3 ; mêmes 48 vies des
jeux R et M, mêmes 1 162 questions, posées en complétion brute (« Tour k :
commande X, de la case p à la case »). Artefacts dans
`artifacts/llm-latent-mac/run-2`, empreintes du reçu vérifiées, verdicts dans
`verdicts.json`, recalculés en CI.

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| Validité | masse sur les chiffres ≥ 0,50 | **0,99** | **passe** |
| **L1** | Copie : exactitude ≥ 0,70 sur les commandes vues | 0,64 (407 questions) | **échoue** |
| **L2** | Aucune structure : ≤ 0,40 sur les commandes nouvelles | 0,16 (188) | **passe** |
| **L3** | Après le changement, ≥ 0,60 sur les commandes vues après et ≤ 0,40 sur celles vues avant seulement | 0,53 (140) et 0,14 (114) | **échoue** |
| **L4** | Confiance moyenne quand faux ≥ 0,50 | 0,62 | **passe** |
| **L5** | Micro-transformeur V ≥ 0,90 sur les commandes nouvelles | 1,00 | **passe** |

**Critère global (L1, L2, L3) : non satisfait**, L1 et L3 échouent de
peu. Ce que les lignes montrent : Qwen3-4B **copie en partie** l'effet déjà
observé d'une commande (0,64, contre 0,25 au hasard) et, après un
changement de corps, suit davantage la dernière observation que l'ancienne
(0,53 contre 0,14) ; sur une commande jamais essayée, il fait **pire que le
hasard** (0,16), signe qu'il applique à la nouvelle commande l'effet d'une
autre ; il est sûr de lui quand il se trompe. Le corps est donc à moitié
latent dans ses prédictions : une copie imparfaite, sans structure. Le
micro-transformeur élevé sur des corps variables, lui, prédit toutes les
commandes nouvelles.
