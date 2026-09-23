# Version 6 — analyses exploratoires

Non pré-enregistrées, déclarées comme telles. Faites le 23 septembre 2026
après la publication de la version 6, sur l'agent du premier essai de
développement (graine 23, `artifacts/indicator-agent-v6/dev-params-23.json`),
sans réentraînement ; aucune graine confirmatoire n'a servi. Script
`research/indicator_v6_explorations.py`, sortie
`artifacts/indicator-agent-v6/explorations.json`.

## HOT-4 : un autre test de choix séparerait-il l'espace de qualités ?

En version 6, le code aléatoire choisit presque aussi bien que l'agent sur
les conflits de bande, parce qu'un objet nettement mauvais est facile à
écarter. Un choix entre un objet de la bande et **un autre bon objet**
(valeurs distantes d'au moins 0,2, teintes lues) demande, lui, de bien
évaluer la teinte jamais vue. 200 vies du jeu H :

| | Bande contre bon objet (écart ≥ 0,2) | (écart ≥ 0,3) | Bande contre objet mauvais (test publié) |
|---|---:|---:|---:|
| Agent | 0,76 (442 pas) | 0,77 (359) | 0,81 (1 845) |
| Code aléatoire | **0,43** (408) | **0,40** (313) | 0,51 (1 160) |
| Sans goulot | 0,82 (257) | 0,84 (191) | 0,88 (1 554) |

Ce test sépare nettement l'espace de qualités du code aléatoire (0,76
contre 0,43), mais l'agent y reste sous 0,8 : ses choix sont limités par
l'espace de travail (0,82 sans goulot). **Il ne ferait pas passer HOT-4 aux
seuils de la version 1** ; il montre en revanche que l'espace de qualités
sert au choix.

## HOT-3 : des pannes plus fréquentes rendraient-elles le moniteur utile au retour ?

150 vies du jeu R, pannes du capteur à 0,08 par pas (monde publié) puis
0,2 :

| Pannes | Pos, agent − gain constant | Retour, agent − gain constant | Recharge quand l'énergie est basse |
|---|---:|---:|---:|
| 0,08 | 0,038 | 0,094·Δ | 0,87 |
| 0,2 | **0,081** | 0,088·Δ | 0,85 |

Plus de pannes rendent le moniteur plus utile à la position (le volet Pos
de HOT-3 passerait), **pas au retour** : l'agent lui-même perd plus, et
l'écart reste sous 0,1·Δ. Changer ce paramètre du monde ne démontrerait pas
HOT-3.

## Conclusion

Aucune des deux pistes n'est retenue. HOT-3 et HOT-4 restent les deux
propriétés non démontrées ; les mécanismes sont présents et agissent (le
moniteur rend la position plus juste, l'espace de qualités évalue et fait
choisir les teintes jamais vues), mais leurs effets restent sous les seuils
fixés en version 1. Aucune de ces analyses ne mesure une expérience vécue.

## Changer le monde pour HOT-3 et HOT-4 ?

Deux changements du monde, chacun motivé par la raison pour laquelle le
test d'usage ne peut pas montrer la propriété, ont été essayés avant tout
protocole de version 7. Le code de la version 6 est entraîné de zéro dans
le monde changé (enfance, phase d'agence, évaluation complète), sur les
graines de développement 23 et 29 seulement. Script
`research/indicator_world_explorations.py`, sorties dans
`artifacts/indicator-agent-v6/world-explorations/`. (Un premier essai de la
seconde bande, lancé par erreur avec les réglages d'apprentissage de la
version 1, a été écarté ; deux autres, interrompus, n'ont rien donné.)

- **Pannes bloquées (HOT-3).** Pendant une panne du capteur, les lectures
  fausses désignaient chacune une case au hasard, et se contredisaient.
  Ici elles désignent toutes la même case pendant l'épisode : une illusion
  persistante.
- **Seconde bande (HOT-4).** Dans un conflit de bande, l'objet opposé
  était toujours d'une teinte familière et nettement mauvaise. Ici les
  teintes 0,55 à 0,65 (valeurs proches de −0,9) sont nouvelles elles aussi :
  jamais vues dans l'enfance, elles apparaissent dans le jeu H.

| Monde | Graine | Propriétés | HOT-3 : Pos ; retour | HOT-4 : Spearman ; conflits agent / code aléatoire |
|---|---|---:|---|---|
| publié (version 6, essai de développement) | 23 | 11 | 0,038 ; 0,092·Δ | 0,850 ; 0,81 / 0,51 |
| pannes bloquées | 23 | 11 | **0,061** ; 0,061·Δ | 0,850 ; 0,80 / 0,56 |
| seconde bande | 23 | 10 | 0,037 ; 0,055·Δ | **0,826** ; 0,85 / 0,60 |
| seconde bande | 29 | 10 | 0,042 ; 0,099·Δ | **0,830** (0,925 dans le monde publié) ; 0,83 / **0,73** |

- **Pannes bloquées** : le moniteur devient utile à la position (0,061,
  au-dessus du seuil de 0,05), mais toujours pas au retour (0,061·Δ pour
  0,1·Δ). L'agent sait mieux où il est ; ce qu'il gagne à le savoir reste
  petit.
- **Seconde bande** : le code aléatoire est écarté sur une graine (0,60)
  mais pas sur l'autre (0,73), et la régularité du code de teinte baisse
  sur les deux (0,850 → 0,826 ; 0,925 → 0,830), sous le seuil de 0,85 : le
  code s'apprend sur les teintes goûtées dans l'enfance, et une seconde
  bande jamais vue y fait un second trou. On gagne d'un côté de HOT-4 ce
  qu'on perd de l'autre.
- Le retour perdu sans moniteur varie fortement d'une graine à l'autre
  (0,055·Δ à 0,099·Δ) ; aucun monde essayé ne le porte nettement au-dessus
  de 0,1·Δ.

**Aucune version 7 n'est pré-enregistrée sur ces mondes** : ni l'un ni
l'autre ne démontrerait HOT-3 ou HOT-4 de façon robuste, et les essayer sur
des graines confirmatoires reviendrait à chercher le monde où les seuils
passent. Les deux propriétés restent présentes, actives, et non démontrées
aux seuils de la version 1.
