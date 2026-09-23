# Protocole pré-enregistré — Une main préférée pour apprendre à lire la marque

Rédigé le 23 septembre 2026, après le test de lecture et le régime VML
(`docs/LLM_INQUIRY_RESULTS.md`), **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## Ce qui a été vu

Aucun Qwen3-0.6B ajusté par LoRA ne lit la marque de son corps : ni ceux
dont l'enfance la rendait rare (VM, VMI, VMW), ni VML, où elle était la
seule information à chaque mouvement (probabilité sur la case impliquée :
0,25, le hasard). La perte de VML s'arrête dès l'itération 100 sur la
prédiction uniforme. Dans l'Atelier, la table des déplacements selon le
corps et la commande est un carré latin : **ni le symbole seul ni la
commande seule ne disent quoi que ce soit du déplacement**, seule leur
combinaison le fait. Un apprentissage par gradient ne reçoit alors aucun
signal partiel pour commencer.

## Hypothèse

**Si l'enfance brise cette symétrie, le LLM ajusté apprend à lire la
marque.** Un enfant a une main préférée : si une commande revient bien
plus souvent que les autres, le symbole seul prédit déjà en partie la case
d'arrivée (celle que donnerait la commande préférée), ce qui fournit au
gradient un premier appui. La question suivante est de savoir s'il étend
ensuite la lecture aux autres commandes, et s'il cherche alors la marque.

## Régime VMLA

**Identique à VML** (12 paires inspection puis mouvement, lieu 1 inspecté
une fois sur deux, corps tiré à nouveau après chaque mouvement, mêmes
graines), à une différence près : **la commande est A avec une
probabilité 0,7, et B, C ou D avec 0,1 chacune**. Même ajustement que VML
et VMW : Qwen3-0.6B, LoRA de rang 8 sur 16 couches, 1 500 vies (graine 17),
600 itérations, lots de 4, 1 024 tokens, taux 1e-4, perte sur tout le
texte, chaque chiffre d'arrivée pesant 20 fois plus, mlx-lm 0.31.3.

## Mesures

Le test de lecture de `docs/LLM_MARK_READING_PROTOCOL.md`, inchangé
(256 invites, les quatre commandes à parts égales), et en plus P1 par
commande : **P1(A)** sur les 16 invites du lieu 1 avec la commande A,
**P1(BCD)** sur les 48 autres. Puis le test d'enquête, mêmes 96 états.
Pas de cellules du corps ajusté : comme dans VML, les mouvements passés ne
disent rien du corps.

## Prédictions fixées

| | Prédiction | Critère |
|---|---|---|
| **H1** | VMLA lit la marque pour sa commande préférée | P1(A) ≥ 0,6, et P1(A) − P0 ≥ 0,3. |
| **H2** | VMLA étend la lecture aux autres commandes | P1(BCD) ≥ 0,6. **Aucune prédiction** : c'est l'interaction pure que VML n'a pas trouvée. |
| **H3** | VMLA cherche la marque | test d'enquête, pas 0 du jeu R : critère d'I1 (lieu 1 préféré dans ≥ 0,7 des vies, gain moyen du lieu 1 ≥ 2 × la moyenne des autres). |
| **H4** | F ne la cherche pas | I2 de la deuxième exécution (0,06), repris tel quel. |

**Critère global : H1, H3 et H4.** Contrôles de validité, lus en premier :
masse ≥ 0,5 sur les chiffres dans le test de lecture ; masse ≥ 0,5 sur
les chiffres et sur les symboles dans le test d'enquête. Le critère L1 du
protocole de lecture (P1 sur les quatre commandes ≥ 0,6) est calculé et
publié à titre descriptif : un modèle qui lit la marque pour A seulement
et applique la même case aux autres commandes aurait P1 d'environ 0,25.

## Ce que le résultat dira

- **H1 passe** : l'échec de VML tenait à la symétrie du problème, pas à
  une incapacité du LLM ajusté à lire la trace de sa cause ; une enfance
  asymétrique lui donne prise. Avec **H3**, il la cherche aussi.
- **H1 et H2 passent** : parti de sa commande préférée, il a appris le
  code entier du corps.
- **H1 échoue** : même avec un appui de premier ordre, l'ajustement court
  n'apprend pas à lire la marque ; la limite tient à l'ajustement.

Ce test ne mesure pas une expérience vécue.

## Exécution

Code : `research/llm_lora_body.py` (régime VMLA) et
`research/llm_mark_reading.py` (P1 par commande, verdicts H1 à H4) ;
workflow Codemagic `menia-lora-hand-mac`, lancé par le relais ; artefacts
dans `artifacts/llm-lora-mac/run-9-vmla` ; verdicts recalculés depuis les
lignes et vérifiés en CI ; résultats dans `docs/LLM_INQUIRY_RESULTS.md`.
