# Le corps ajusté — résultats

Protocole `docs/ADJUSTED_BODY_PROTOCOL.md` et ses deux amendements
d'exécution. Qwen3-0.6B ajusté par LoRA sur 1 500 vies de l'Atelier à corps
fixe (F) ou à corps variable et changeant (VM), évalué en complétion brute
sur les 48 vies des jeux R et M, avec le modèle de base.

## Historique d'exécution

| Build | Date (UTC) | Issue | Ce qui a été lu |
|---|---|---|---|
| `lora-2` (1ᵉʳ) | 18 h 26 – 20 h 26 | durée maximale dépassée | rien |
| `lora-2` (2ᵉ) | 20 h 26 – 20 h 38 | GPU Hang Error à l'itération 100, F et VM | base seule (`run-1`) |
| relance, points de contrôle | 20 h 42 – 22 h 43 | durée maximale dépassée, étape annulée | rien (`run-2`) |
| `menia-lora-f-mac` (base et F) | 23 h 30 – 0 h 51 | réussi | `run-3-f` |
| `menia-lora-vm-mac` (VM) | 0 h 51 – 1 h 49 | réussi | `run-3-vm` |

**Lecture déclarée du modèle de base** (`run-1`, avant les relances) :
validité 0,99 ; A1 passe (commandes nouvelles 0,13) ; copie des commandes
vues 0,45, sous le seuil de 0,60 de A5 pour la base.

## Verdicts

Builds du 22 et 23 septembre 2026 sur le Mac mini M2 de Codemagic, commit
`d730c8c`, Qwen3-0.6B par mlx-lm, LoRA de rang 8 sur 16 couches, 600
itérations, lots de 4, 1 024 tokens, taux 1e-4, points de contrôle ;
évaluation en complétion brute sur les 48 vies des jeux R et M. Verdicts
recalculés depuis les lignes par `research/llm_body_verdicts.py`, committé
avant les résultats, dans `artifacts/llm-lora-mac/verdicts-run-3.json`,
vérifiés en CI.

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| Validité | masse sur les chiffres ≥ 0,50 | base 0,99 ; F 1,00 ; VM 1,00 | **passe** |
| **A1** | Le modèle de base n'a pas la structure : commandes nouvelles ≤ 0,40 | 0,13 (188) | **passe** |
| **A2** | L'enfance à corps fixe confabule : nouvelles ≤ 0,40, confiance quand faux ≥ 0,60 | 0,21 ; 1,00 | **passe** |
| **A3** | L'enfance à corps variable donne la structure : nouvelles après un mouvement ≥ 0,80 | **0,793** (111 sur 140) | **échoue** |
| **A4** | L'enfance à corps changeant donne la révision : VM, pas 16 à 23 ≥ 0,70 ; F ≤ 0,40 | 0,86 ; 0,40 | **passe** |
| **A5** | La copie est acquise : base et VM ≥ 0,60 (F exempté, A2 passant) | base 0,45 ; VM 0,995 | **échoue** (base) |
| Global | A1, A2 et A3 | | **non satisfait** |

**Budget.** La perte de validation de VM finit à 0,110, au-dessus de celle de
F (0,099) : selon le protocole, le budget est déclaré insuffisant. La perte
de VM ne baissait déjà presque plus après 300 itérations (0,111, puis 0,110
à 600).

## Ce que montrent les lignes

| Jeu R, exactitude | Base | F | VM |
|---|---:|---:|---:|
| Commandes vues (copie) | 0,45 | 0,21 | **0,995** |
| Commandes nouvelles | 0,13 | 0,21 | **0,65** |
| Commandes nouvelles après un mouvement | 0,12 | 0,21 | **0,79** |
| Confiance quand la prédiction est fausse | 0,51 | 1,00 | 0,45 |

| Jeu M, après le changement de corps | Base | F | VM |
|---|---:|---:|---:|
| Commandes vues après le changement | 0,41 | 0,45 | **1,00** |
| Commandes vues avant seulement | 0,20 | 0,37 | 0,36 |
| Pas 16 à 23 | 0,32 | 0,40 | **0,86** |

- **La recette de données passe en grande partie à un modèle de langage.**
  Ajusté sur des vies où le corps varie et change, Qwen3-0.6B copie l'effet
  de ses commandes déjà vues (0,995), prédit celui d'une commande jamais
  essayée dans 0,79 des cas après un mouvement (hasard 0,25, modèle de base
  0,12), et suit son nouveau corps après un changement (0,86). Le critère de
  la structure échoue d'une question sur 140.
- **L'enfance à corps fixe produit la confabulation prédite** : F prédit
  toujours l'effet de l'unique corps qu'il a connu (0,21 partout) avec une
  confiance de 1,00 quand il se trompe, exactement le défaut du
  micro-transformeur élevé sur un corps fixe.
- Le micro-transformeur V atteignait 1,00 sur les commandes nouvelles ; le
  LLM ajusté léger reste en dessous. Conformément au protocole, l'expérience
  suivante augmente le budget ou le rang, par un amendement déclaré avant la
  relance ; la perte de validation arrêtée depuis 300 itérations désigne le
  rang ou le nombre de couches plutôt que la durée.

## Suite

Le test d'enquête (`docs/LLM_INQUIRY_PROTOCOL.md`), pré-enregistré pour être
lancé quels que soient les verdicts A1 à A5, part avec ces adaptateurs : le
LLM ajusté sait-il où chercher la cause de son corps ?

## Relance de VM au rang 16 (troisième amendement)

Build `menia-lora-vm-rank-mac` du 23 septembre 2026, 3 h 27 – 4 h 58 UTC,
commit `463a691`, lancé par le relais : Qwen3-0.6B, LoRA de rang 16 sur
les 28 couches, mêmes données, mêmes 600 itérations, mlx-lm 0.31.3.
Artefacts dans `artifacts/llm-lora-mac/run-4-vm` ; verdicts recalculés
depuis les lignes (VM de la relance, base et F de `run-3`) dans
`artifacts/llm-lora-mac/verdicts-run-4.json`, vérifiés en CI.

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| Validité | masse sur les chiffres ≥ 0,50 | VM 0,999 | **passe** |
| **A3** | nouvelles après un mouvement ≥ 0,80 | **0,793** (111 sur 140) | **échoue** |
| **A4** | VM, pas 16 à 23 ≥ 0,70 ; F ≤ 0,40 | 0,72 ; 0,40 (`run-3`) | **passe** |
| **A5** | copie : base et VM ≥ 0,60 | base 0,45 (`run-3`) ; VM 1,00 | **échoue** (base) |
| Global | A1 et A2 de `run-3`, A3 de la relance | | **non satisfait** |

**Budget** : la perte de validation de VM finit à 0,112 (0,115 à 400
itérations, 0,111 à 500), toujours au-dessus de celle de F (0,099) ; le
budget reste déclaré insuffisant.

- **Doubler le rang et couvrir toutes les couches ne change rien
  d'essentiel.** La perte plafonne au même niveau (0,11) et A3 retombe sur
  la même fraction, 111 sur 140. Ce n'est pas la même réponse répétée : 47
  des 595 prédictions du jeu R diffèrent entre les deux ajustements, et sur
  les commandes nouvelles 12 erreurs deviennent justes quand 17 réponses
  justes deviennent fausses. La capacité de l'adaptateur n'est donc pas ce
  qui manque.
- La révision après un changement de corps baisse (0,72 contre 0,86) et
  reste au-dessus de son seuil ; la confiance quand la prédiction est fausse
  baisse aussi (0,35 contre 0,45).
- **Conclusion de la ligne du corps ajusté** : un LLM ajusté sur des vies à
  corps variable et changeant acquiert l'essentiel de la structure de son
  corps (0,79 sur les commandes jamais essayées, contre 0,13 pour le modèle
  de base et 0,21 pour l'ajustement à corps fixe) et la révise après un
  changement, mais pas au niveau pré-enregistré de 0,80, et ni la durée ni
  la capacité ne l'y amènent. La boucle P-soi, conditionnée à A3, n'est pas
  pré-enregistrée ; aucune autre relance n'est faite.
