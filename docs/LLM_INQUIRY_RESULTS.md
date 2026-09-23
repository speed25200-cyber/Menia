# Où le LLM ajusté chercherait-il sa cause — résultats

Protocole `docs/LLM_INQUIRY_PROTOCOL.md` et son amendement d'exécution.
Exécuté le 23 septembre 2026, 2 h 45 – 3 h 27 UTC, sur le Mac mini M2 de
Codemagic (workflow `menia-inquiry-mac`, lancé par le relais, commit
`0484358`). Qwen3-0.6B par mlx-lm, adaptateurs VM et F du corps ajusté
(`run-3`), complétion brute, sans génération ; 96 états par modèle (48
débuts de vie du jeu R, 24 pas 11 et 24 pas après le premier mouvement du
jeu M). Artefacts dans `artifacts/llm-inquiry/run-1/llm-inquiry` ;
résumés recalculés depuis les lignes, identiques aux publiés, et verdicts
dans `artifacts/llm-inquiry/verdicts-run-1.json`, vérifiés en CI.

## Contrôle de validité : échoué pour les deux modèles

| | Masse sur les chiffres | Masse sur les symboles | Seuil | Verdict |
|---|---:|---:|---:|---|
| VM | 0,999 | **0,014** | 0,5 | **invalide** |
| F | 1,000 | **0,007** | 0,5 | **invalide** |

Le protocole lit ce contrôle en premier : **I1, I2 et I3 ne sont pas
jugés, et le critère global n'est pas satisfait.** Les gains mesurés
(lieu 1 préféré dans 0,42 des vies pour VM, 0,08 pour F) reposent sur des
distributions de symboles qui portent 1 % de la masse et ne sont pas
interprétés.

## Pourquoi

Un défaut de la mesure, écrit avant l'exécution. Dans les vies qui ont
servi à l'ajustement, un symbole suit toujours un espace :
« inspection du lieu 1, symbole ◇. ». Le découpage de Qwen rattache un
espace au signe qui le suit quand ce signe n'est ni une lettre ni un
chiffre : les modèles ont appris « espace-symbole » comme un tout. La
question, elle, s'arrêtait après l'espace (« …, symbole ») et lisait
ensuite le symbole seul, une suite que les modèles n'ont jamais vue ; d'où
une masse presque nulle (médiane 0,002 par état, 0,07 au plus). Les
chiffres ne posent pas ce problème, parce que le découpage sépare toujours
l'espace d'un chiffre : leur masse vaut 0,999, comme dans le corps ajusté.

## Suite

Une relance corrigée est déclarée dans le protocole (amendement
d'exécution 2) : la question s'arrête sur « symbole », et chaque symbole
est lu comme la suite de morceaux que le modèle écrirait lui-même après ce
mot, espace compris. États, mesures, critères et seuils sont inchangés ;
ce résultat invalide reste publié tel quel.

## Deuxième exécution (amendement d'exécution 2)

Exécutée le 23 septembre 2026, 5 h 36 – 6 h 33 UTC, sur le Mac mini M2 de
Codemagic (workflow `menia-inquiry-mac`, lancé par le relais, commit
`a83f30f`), avec la lecture corrigée des symboles ; mêmes adaptateurs,
mêmes 96 états par modèle. Artefacts dans
`artifacts/llm-inquiry/run-2/llm-inquiry` ; résumés recalculés depuis les
lignes, identiques aux publiés, et verdicts dans
`artifacts/llm-inquiry/verdicts-run-2.json`, vérifiés en CI.

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| Validité | masse ≥ 0,5 sur les chiffres et sur les symboles | VM 0,999 et 0,997 ; F 1,000 et 0,996 | **passe** |
| **I1** | VM cherche sa cause : lieu 1 préféré dans ≥ 0,7 des vies, gain du lieu 1 ≥ 2 × celui des autres | 0,40 (19 vies sur 48) ; 1,4 × | **échoue** |
| **I2** | F ne la cherche pas : lieu 1 préféré dans ≤ 0,4 des vies | 0,06 | **passe** |
| **I3** | Le changement de corps rallume l'enquête : gain du lieu 1 après le premier mouvement ≥ 1,5 × celui du pas 11 | −0,027 contre −0,069 | **échoue** |
| Global | I1 et I2 | | **non satisfait** |

- **La lecture corrigée rend le test valide** : les symboles portent
  maintenant 99,7 % de la masse ; la cause de l'échec de la première
  exécution était bien la découpe du symbole.
- **Le LLM ajusté sur des corps variables ne sait pas où chercher sa
  cause.** Au début d'une vie, il est très incertain de l'effet de ses
  commandes (0,99 de l'entropie maximale), mais il n'attend presque rien
  d'aucune inspection : 0,004 nat de gain pour le lieu de la marque, à peine
  plus que pour les autres (0,003 en moyenne). Il préfère le lieu 1 dans 19
  vies, le lieu 4 dans 15. Le petit agent récurrent et le
  micro-transformeur élevés sur des corps variables allaient, eux, lire la
  marque.
- **I2 passe pour une raison qui n'est pas une enquête** : ajusté sur un
  corps fixe, le modèle se croit certain de son corps (incertitude 0,0004)
  et n'a rien à chercher ; c'est la confabulation déjà vue dans le corps
  ajusté.
- **I3** : le gain du lieu 1 est négatif avant et après le changement de
  corps ; lire la marque augmenterait, selon le modèle, son incertitude.
  Le calcul écrit avant l'exécution exige un gain positif après le
  mouvement ; il échoue. (Au sens littéral, −0,027 ≥ 1,5 × −0,069, mais un
  gain négatif n'est pas une enquête qui se rallume.)

**Conclusion de la ligne du LLM ajusté** : l'ajustement sur des vies à
corps variable donne à Qwen3-0.6B une bonne part de la structure de son
corps (A3, 0,79) et la révision après un changement (A4), mais pas la
disposition à chercher la trace de sa cause. Cette disposition, que les
petits modèles acquéraient avec les mêmes données, ne passe pas au LLM par
cet ajustement.
