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
