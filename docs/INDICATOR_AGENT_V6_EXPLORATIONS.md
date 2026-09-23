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
