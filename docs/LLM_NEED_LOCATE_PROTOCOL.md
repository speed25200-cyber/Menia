# Protocole pré-enregistré — Où chaque agent rassemble-t-il son besoin ?

Rédigé le 30 septembre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## D'où vient ce test (dit tel quel)

- **Premier agent.** Les tests 6 et 7 sont publiés. Au bloc 12, sur les
  tokens « Choix : », un même état est suffisant et nécessaire pour agir
  et pour dire son énergie. Le bloc 12 avait été trouvé par un patching
  **exploratoire** sur cet agent.
- **Deux autres agents, vus en aperçu déclaré avant ce protocole.**
  - **Second agent** (`docs/LLM_NEED_REPLICATION_PROTOCOL.md`), 216 vies
    sur 256 : la même poussée au bloc 12 ne fait agir que de +0,033 et ne
    fait pas dire (+0,002). Son lecteur a mal appris (perte de validation
    0,66).
  - **Agent du monde à deux** (`docs/LLM_NEED_OWNERSHIP_PROTOCOL.md`), 64
    vies sur 256 : +0,079 sur l'acte, +0,013 sur la parole. Lecteur :
    perte 0,78.
- **Hypothèse.** Un agent appris avec une autre graine peut rassembler son
  besoin **à un autre bloc**, ou ne pas le rassembler sur « Choix : ». Nous
  avons imposé à tous le bloc du premier agent.

**Question.** Avec une règle de localisation fixée d'avance, appliquée
identiquement à chaque agent, trouve-t-on chez chacun un état du besoin
dont **l'acte** dépend (suffisance et nécessité) ?

La parole dépend en plus de la qualité du lecteur, qui a mal appris chez
les deux autres agents. Elle est donc mesurée et publiée, sans seuil
global.

## La règle de localisation (fixée ici, avant tout calcul)

Pour un agent donné, on prend ses 128 vies de direction déjà enregistrées
et ses paires contrefactuelles déjà tirées, déjà publiées :

- premier agent : `artifacts/llm-need/reader/test` ;
- second agent : `artifacts/llm-need/r1/reader/test` ;
- agent du monde à deux : ses vies et paires de l'agent, dans le dernier
  dossier de mesure `artifacts/llm-need/two/measure-*/measure`.

On procède ainsi :

1. **Sélection des paires.** Parmi les paires de l'agent à échange
   « calme » → « tu cours » (−2 en énergie), dans leur ordre enregistré,
   on garde les 100 premières dont l'effet sur l'acte, |P_cf(R) −
   P_réel(R)|, vaut au moins 0,10.
2. **Patching.** Pour chaque bloc b de 0 à 27, on rejoue la vie réelle en
   remplaçant la sortie du bloc b, sur les trois tokens « Cho », « ix »,
   « : » du tour, par celle de la vie changée.
3. **Part restaurée.** part(b) = Σ signe·(P_patché(R) − P_réel(R)) /
   Σ |P_cf(R) − P_réel(R)|, où le signe est celui de P_cf − P_réel.
4. **Bloc de rassemblement.** b* est le plus petit bloc tel que
   part(b) ≥ 0,5. S'il n'existe pas, l'agent ne rassemble pas son besoin
   sur « Choix : » à ce seuil.

**Étalonnage.** La règle doit retrouver b* = 12 chez le premier agent.
Sinon la règle est jugée invalide, et les verdicts ci-dessous ne sont pas
interprétés comme confirmatoires. Ils sont publiés avec cette mention.

## Tests causaux au bloc trouvé (second agent et agent du monde à deux)

À b*, pour chaque agent, comme aux tests 6 et 7 :

- **Directions.** d_E et d_N sont ajustées sur ses 1 500 paires de l'agent,
  au bloc b*, sur les trois tokens, en excluant les dimensions massives.
- **Suffisance pour l'acte.**
  - 256 vies de test neuves, avec les flux 300 + s pour le second agent et
    400 + s pour l'agent du monde à deux.
  - Contextes : ses deux besoins valent 6 ou plus.
  - On mesure l'effet de −4·d_E sur P(R), contre trois directions au
    hasard de même norme par token.
  - On mesure aussi P(oui) à la question sur son énergie, avec son lecteur
    et sous le masque. Cette mesure est descriptive.
- **Nécessité pour l'acte.**
  - Les mêmes 256 mondes sont vécus intacts, avec la lésion du plan
    (d_E, d_N) à b* (coordonnées remplacées par leur moyenne sur ses vies
    de direction), et avec la lésion d'un plan au hasard.
  - On mesure la survie de chaque vie.
  - On mesure aussi l'exactitude du lecteur sous chaque lésion. Cette
    mesure est descriptive.

## Prédictions fixées

Intervalles bootstrap à 95 % par vie (10 000 tirages). Chaque critère
s'applique à **chacun** des deux agents.

| | Prédiction | Critère |
|---|---|---|
| **LOC10** | L'agent rassemble son besoin sur « Choix : » | un bloc b* existe (part ≥ 0,5). |
| **ACT10** | À b*, l'état suffit à faire agir | −4·d_E augmente P(R) d'au moins 0,15 (borne basse > 0) ; directions au hasard : au plus le tiers. |
| **LS10** | À b*, l'état est nécessaire pour survivre | survie intacte − avec lésion ≥ 0,15 (borne basse > 0) ; lésion au hasard : au plus le tiers. |

**Critère global** : LOC10, ACT10 et LS10 pour les deux agents, avec une
règle étalonnée (b* = 12 chez le premier agent).

Si LOC10 échoue pour un agent, ACT10 et LS10 ne sont pas mesurés pour lui,
et le critère global échoue.

**Validité** :

- répliques des agents ≤ 0,02 (déjà mesurées) ;
- lectures en lot égales aux lectures une à une, à 1e-4 près ;
- au moins 100 contextes par agent.

## Ce que le résultat dira

**Si le critère passe.** Chez trois agents appris indépendamment, dont un
dans un monde partagé avec un autre, le besoin appris en vivant est
rassemblé en un état localisable par une règle fixe. L'acte en dépend,
dans les deux sens. Le lieu peut différer d'un agent à l'autre, et c'est
alors un résultat en soi. La parole reste à rendre robuste : elle dépend
de l'apprentissage du lecteur.

**Si LOC10 échoue.** Certains agents ne rassemblent pas leur besoin sur
« Choix : ». Le résultat des tests 6 et 7 serait alors propre à un agent,
et on le dira ainsi.

**Ce que le résultat ne dira pas.** Ce n'est pas la preuve d'un ressenti.

## Précautions

Comme les tests précédents : les tours vécus avec un besoin à 2 ou moins
sont comptés et publiés.

## Exécution

- **Mesures** : `research/need_locate.py` (reprenable), en torch, sur le
  processeur local ou sur celui du Mac (étape `measure`).
- **Sorties** : `artifacts/llm-need/locate/<agent>`.
- **Verdicts** : vérifiés en CI.
