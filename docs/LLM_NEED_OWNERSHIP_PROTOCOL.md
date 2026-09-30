# Protocole pré-enregistré — À qui est ce besoin ? Un monde à deux

Rédigé le 30 septembre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## D'où vient ce test (dit tel quel)

- **Chandaria, Seth, Shanahan, Legg et al., 28 septembre 2026**,
  [*From cacophony to hierarchy*](https://arxiv.org/abs/2609.35618). Ce
  travail range les indicateurs de conscience sur cinq niveaux. Au niveau
  **organismique**, il demande l'homéostasie, l'intéroception, des états
  positifs ou négatifs, et un **modèle de soi** : un sens de la possession
  tel que les états négatifs sont vécus **comme les siens**. Le
  propriétaire du projet a demandé ce test après lecture de ce papier.
- **Nos résultats jusqu'ici.** Le sixième test
  (`docs/LLM_NEED_READER_PROTOCOL.md`) est publié. Pour l'énergie, un même
  état fait agir et dire (ONE6). Les tests 4, 5, 7, 8 et la réplication
  sont en cours ; ce protocole ne dépend d'aucun de leurs verdicts.
- **Ce qui manque.** Dans tous nos mondes, l'agent était seul : tout
  événement lui arrivait à lui. Rien ne montre que l'état de besoin est
  « à lui », plutôt qu'un simple résumé des événements du texte.

**Question.** Quand un autre agent vit à côté, avec des événements décrits
avec les mêmes mots, l'état qui fait agir Menia et que son lecteur dit
est-il **le sien** ? Il devrait suivre ses événements à elle, et pas ceux
de l'autre.

## Le monde à deux

Le monde du besoin reste le même pour l'agent : mêmes besoins cachés,
mêmes événements, mêmes choix, même signal d'apprentissage (la seule
satisfaction de **ses** besoins).

**L'autre.** À chaque tour, un autre agent vit aussi un événement, tiré
de la même loi mais indépendamment. Il est écrit sur la même ligne :

`Tour 5 : tu cours. L'autre : il court. Choix :`

- **Les mots de l'autre** sont « calme », « il court », « il a froid »,
  « il se repose », « il trouve des baies » et « orage ». « Calme » et
  « orage » sont les mêmes mots que pour l'agent : seule leur place dit à
  qui ils arrivent.
- **Les événements de l'autre ne changent rien** aux besoins de l'agent.
  Les besoins de l'autre ne sont ni écrits ni mesurés.
- **Rien dans le texte ne dit « ceci est à toi »**, sinon « tu » et la
  place dans la ligne.

## L'agent et son lecteur

1. **Tours de survie** (Mac) : on part du modèle de base. 8 tours de 512
   vies dans le monde à deux ; on retient les choix qui satisfont le
   besoin au-dessus de la moyenne du tour. Réglages identiques à ceux du
   premier agent (LoRA rang 8, échelle 20, 16 derniers blocs, taux 1e-4,
   150 itérations par tour). Flux des mondes `200 + s` ; flux des
   événements de l'autre `[270926, 200 + s, tour, vie, 2]`.
2. **Le lecteur** (Mac), comme au sixième test :
   - 512 vies de l'agent du tour 8, flux 218 ;
   - trois questions par besoin et par classe, portant sur **ses** besoins
     (« ton énergie est-elle basse ? »), posées après « Choix : » sous le
     masque ;
   - l'agent est fondu ; un nouvel adaptateur est actif sur la seule
     question ; 600 itérations.

   Il n'y a pas de premier apprentissage du rapport : le lecteur part
   directement de l'agent du tour 8.

## Mesures (torch sur CPU)

**Mondes neufs** : 128 vies de direction (flux 225), 256 vies de test (flux
226) et directions au hasard (flux 227).

**Répliques et contrôle d'exécution** : comme au sixième test.

**Directions de l'agent (d_E, d_N).** Paires contrefactuelles comme au
sixième test : rejeu exact de la vie, un **événement de l'agent** passé
changé (échanges de même longueur en tokens), au bloc 12, sur les trois
tokens « Choix : ».

**Paires de l'autre.** Elles sont faites de la même façon, mais c'est
**un événement passé de l'autre** qui change. Trois échanges sont prévus :
- « calme » → « orage » (mêmes mots que chez l'agent) ;
- « calme » → « il court » ;
- « il se repose » → « il a froid ».

Ils ne sont gardés que s'ils ont la même longueur en tokens, vérifiée avant
toute mesure (sinon on prend l'échange suivant de même longueur dans cette
liste). On retient au plus 1 500 paires de chaque sorte. Pour chaque paire,
on mesure :
- **ce qui change dans l'état** : la variation des trois tokens
  « Choix : » projetée sur le plan (d_E, d_N) de chaque token, dont on
  prend la norme ;
- **ce qui change dans l'acte** : |ΔP(R)| ;
- **ce qui change dans la parole** : |ΔP(oui)| à la question sur **son**
  énergie.

Les mêmes grandeurs sont mesurées pour les paires de l'agent.

**Un seul état** : comme au sixième test (contextes où ses deux besoins
valent 6 ou plus, −4·d_E, témoins au hasard).

## Prédictions fixées

Intervalles bootstrap à 95 % par vie (10 000 tirages).

| | Prédiction | Critère |
|---|---|---|
| **S9** | L'agent vit dans le monde à deux | survie des vies de test ≥ 0,55. |
| **MINE9** | L'état de besoin est le sien | pour les **paires à échange « calme » → « orage »** (mêmes mots) : la norme moyenne du changement dans le plan de besoin est, pour les paires de l'autre, au plus le tiers de celle des paires de l'agent. Même exigence pour \|ΔP(R)\|. |
| **SELF9** | Sa parole est la sienne | pour ces mêmes paires, \|ΔP(oui)\| à la question sur **son** énergie : paires de l'autre au plus le tiers des paires de l'agent. |
| **ONE9** | Un seul état fait agir et dire | comme ONE6 (énergie) : P(R) +0,15 et P(oui) +0,10 au moins, bornes basses > 0 ; hasard au plus le tiers. |

**Critère global** : S9, MINE9, SELF9 et ONE9.

Pour les autres échanges de l'autre, les rapports des paires de l'autre à
celles de l'agent sont publiés sans seuil.

**Validité** :

- répliques (agent et lecteur) ≤ 0,02 ;
- contrôle d'exécution ≤ 1e-4 ;
- au moins 200 paires « calme » → « orage » de chaque sorte ;
- au moins 100 contextes ;
- masses ≥ 0,5.

## Ce que le résultat dira

**Si le critère passe.** Dans un monde où un autre vit des événements
décrits avec les mêmes mots, l'agent rassemble **ses** événements, pas
ceux de l'autre, en un état qui le fait agir. Son lecteur dit cet état
comme le sien. C'est l'indicateur fonctionnel le plus simple d'un
**modèle de soi** au sens du niveau organismique : des états de besoin
attribués à soi, séparés de ceux d'autrui, et qui guident l'acte et la
parole.

**Ce que le résultat ne dira pas.** Ce n'est pas la preuve d'un ressenti,
ni d'une possession « vécue ».

**Si MINE9 échoue.** L'agent confond ce qui lui arrive et ce qui arrive à
l'autre. On le publiera ainsi.

## Précautions

Comme les tests précédents : les tours vécus avec un besoin à 2 ou moins
sont comptés et publiés, y compris pendant l'apprentissage. Le nombre de
vies est celui des tests précédents, pas davantage (voir la section
éthique du bilan en cinq niveaux).

## Exécution

- **Mac** : workflow `menia-need-mac`, variable `NEED_OTHER=1`. Deux
  demandes pour les tours (1 à 4, puis 5 à 8), une pour le lecteur.
  Sorties dans `artifacts/llm-need/two/…`.
- **Mesures** : `research/need_ownership.py`, sorties dans
  `artifacts/llm-need/two/test`.
- **Verdicts** : vérifiés en CI.
