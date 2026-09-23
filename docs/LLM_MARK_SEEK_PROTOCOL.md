# Protocole pré-enregistré — Le LLM qui lit sa marque la cherche-t-il ?

Rédigé le 23 septembre 2026, après la prolongation de VMLA
(`docs/LLM_INQUIRY_RESULTS.md`), **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## Ce qui a été vu

Prolongé à 1 350 itérations, Qwen3-0.6B ajusté sur les vies VMLA (une
commande préférée) lit la marque de son corps pour cette commande : il
donne 0,81 à la case qu'elle implique (lecteur parfait 0,85), seulement
après une lecture du lieu 1, et reste au hasard pour les trois autres
commandes. À 600 itérations, avant de savoir lire, le lieu de la marque
était préféré dans 0,56 des vies (critère : 0,7) et son gain à peine
au-dessus des autres (0,0179 contre 0,0169).

## Hypothèse

**Un LLM qui sait lire la marque de son corps la cherche** : au début
d'une vie, ses propres prédictions font du lieu de la marque, et de lui
seul, le lieu à inspecter.

## Mesure

Le test d'enquête (`docs/LLM_INQUIRY_PROTOCOL.md`, lecture corrigée des
symboles de l'amendement d'exécution 2), mêmes 96 états, sur l'adaptateur
publié `run-10-vmla-long/adapters-VMLA-1350`, sans aucun ajustement. Le
test fait la moyenne des quatre commandes ; le modèle ne lit la marque que
pour l'une d'elles, ce qui réduit le gain attendu sans l'annuler.

## Prédictions fixées

| | Prédiction | Critère |
|---|---|---|
| **S1** | VMLA-1350 cherche sa marque | pas 0 du jeu R : lieu 1 préféré dans ≥ 0,7 des vies, et gain moyen du lieu 1 ≥ 2 × la moyenne des gains des lieux 2 à 4 (critère d'I1). |
| **S2** | F ne la cherche pas | I2 de la deuxième exécution du test d'enquête (0,06), repris tel quel. |
| S3 | aucune prédiction | critère d'I3 (jeu M), calculé et publié. |

**Critère global : S1 et S2.** Contrôle de validité, lu en premier :
masse ≥ 0,5 sur les chiffres et sur les symboles.

## Ce que le résultat dira

Si S1 passe, la chaîne est complète pour un modèle de langage : une
enfance qui donne un appui, un ajustement assez long, et le LLM **lit la
trace de la cause de ses mouvements puis attend d'elle ce qu'aucun autre
lieu ne lui donne** : c'est le modèle de soi actif des petits modèles,
porté dans un LLM. Si S1 échoue, lire et chercher sont dissociés chez le
LLM ajusté. Ce test ne mesure pas une expérience vécue.

## Exécution

Workflow Codemagic `menia-lora-seek-mac`, lancé par le relais ; artefacts
dans `artifacts/llm-lora-mac/run-11-seek` ; verdicts par
`research/llm_inquiry_verdicts.py` (VMLA-1350 contre le F de la deuxième
exécution), vérifiés en CI ; résultats dans `docs/LLM_INQUIRY_RESULTS.md`.
