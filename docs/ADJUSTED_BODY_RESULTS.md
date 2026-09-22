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
| `menia-lora-f-mac` et `menia-lora-vm-mac` | à partir de 23 h 24 | en cours | — |

**Lecture déclarée du modèle de base** (`run-1`, avant les relances) :
validité 0,99 ; A1 passe (commandes nouvelles 0,13) ; copie des commandes
vues 0,45, sous le seuil de 0,60 de A5 pour la base.

## Verdicts

À compléter à l'arrivée des deux builds.
