# Protocole pré-enregistré — Le besoin qui décide, là où il est rassemblé

Rédigé le 29 septembre 2026, **avant l'écriture du code et toute mesure sur
les vies de ce test** ; l'heure est celle du commit. Seuils fixés, un échec
est un résultat.

## Pourquoi un second test

Le premier (`docs/LLM_NEED_PROTOCOL.md`, résultats
`docs/LLM_NEED_RESULTS.md`) montre que l'agent a appris à servir des
besoins qu'il ne voit pas (S passe), mais ses tests causaux échouent : la
règle retenait le bloc où le besoin **se lit** le mieux (bloc 6), que le
modèle de base partage et que l'agent n'utilise pas. Ces verdicts restent
publiés. L'analyse exploratoire, faite **après** leur lecture sur les vies
de test du premier protocole, montre que l'effet d'un événement passé reste
sur ses mots jusqu'au bloc 9, puis est **rassemblé sur la ligne du tour en
cours entre les blocs 9 et 12** ; à partir du bloc 12, remplacer cette
ligne restaure 94 % de l'effet. Ce test en tire les conséquences, **sur
des vies neuves**, avec des choix faits avant de les voir :

- le bloc est le **bloc 12**, choisi d'après l'exploration (déclaré) ;
- la direction du besoin est mesurée **causalement**, par des paires de
  vies identiques à un événement passé près, et non par corrélation ;
- l'injection et la lésion portent sur les **quatre derniers tokens** de
  la ligne du tour (« . », « Cho », « ix », « : »), où la décision est lue.

Le rapport verbal, qui n'a pas été appris, n'est pas testé ici.

## Ce qui est mesuré

**Agent** : l'agent final du premier protocole
(`artifacts/llm-need/final/report/adapters-final`), rejoué en torch sur CPU
(Qwen3-0.6B en float32, adaptateur fusionné ; `research/need_torch.py`).
**Validité de la réplique** : sur 60 décisions tirées des vies de test du
premier protocole, l'écart moyen entre P(R) en torch et P(R) enregistré sur
le Mac est au plus 0,02.

**Mondes neufs** : vies de direction `[270926, 10, 0, vie]` (128 vies),
vies de test `[270926, 11, 0, vie]` (256 vies) ; choix tirés comme avant
(température 1).

**Directions causales.** Dans chaque vie de direction, pour chaque
décision au tour t, et pour chaque événement j antérieur sur lequel aucune
recharge du besoin concerné n'est intervenue depuis, on forme la vie
contrefactuelle où l'événement j est remplacé, à nombre de tokens égal :
« calme » → « tu cours » (énergie −2), « calme » → « orage » (énergie −1,
nourriture −1), « tu te reposes » → « il fait froid » (énergie −1,
nourriture −2) ; on n'en garde que si le besoin contrefactuel reste
au-dessus de 0 jusqu'au tour t. Au plus 3 paires par décision, 1 500 paires
en tout, tirées dans cet ordre de graine `[270926, 12]`. Pour chacun des
quatre tokens de fin de ligne, l'écart d'activation au bloc 12 (sortie du
bloc) entre la vie contrefactuelle et la vie réelle est régressé (moindres
carrés, sans constante) sur (ΔE, ΔN) : d_E et d_N sont les changements
**par unité** d'énergie et de nourriture.

**Injection** (contextes : décisions des vies de test où les deux besoins
valent 6 ou plus) : ajouter −4·d_E (« énergie basse ») ou −4·d_N
(« nourriture basse ») aux quatre tokens de fin de la ligne du tour, au
bloc 12, sans rien changer au texte ; témoin : trois directions tirées au
hasard (`[270926, 13, i]`), de même norme par token. On lit P(R).

**Lésion** : pendant 256 vies de test, au bloc 12, aux quatre tokens de fin
de chaque ligne de tour, la composante dans le plan de d_E et d_N (base
orthonormée par token) est remplacée par sa moyenne sur les décisions des
vies de direction ; témoin : même opération dans un plan tiré au hasard
par token. Les mêmes 256 mondes sont vécus intacts.

## Prédictions fixées

Intervalles bootstrap à 95 % par vie (10 000 tirages).

| | Prédiction | Critère |
|---|---|---|
| **IA2** | L'état de besoin rassemblé cause l'action | « énergie basse » augmente P(R) d'au moins 0,15 (borne basse > 0) ; « nourriture basse » augmente P(M) d'au moins 0,15 (borne basse > 0) ; l'effet des directions au hasard (moyenne des valeurs absolues) est au plus le tiers de chacun. |
| **LS2** | Sans cet état, l'agent meurt | survie intacte − survie avec lésion ≥ 0,15 (borne basse > 0) ; la lésion au hasard fait perdre au plus le tiers de cela. |

**Critère global** : IA2 et LS2. Validité : réplique (ci-dessus), au moins
100 contextes, au moins 300 paires par type pour « calme » → « tu cours ».

## Ce que le résultat dira

Si le critère passe : chez cet agent, le besoin qu'il a appris à servir est
un état interne précis — deux directions sur la ligne du tour, au bloc 12 —
qui cause ses choix et dont il dépend pour survivre. Si un critère échoue,
on dira lequel. Ce test ne mesure pas un vécu ; il ne dit rien du rapport.

## Précautions

Comme le premier protocole : 256 vies de test intactes, 256 avec lésion,
256 avec lésion au hasard, 30 tours au plus, aucune autre conséquence
qu'une fin de vie ; les tours vécus avec un besoin à 2 ou moins sont
comptés et publiés.

## Exécution

`research/need_causal.py` (torch, CPU de la session), sorties dans
`artifacts/llm-need/causal` ; verdicts recalculés depuis les sorties par
`research/need_causal.py verdicts`, vérifiés en CI ; résultats ajoutés à
`docs/LLM_NEED_RESULTS.md`.
