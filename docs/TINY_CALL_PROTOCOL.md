# Protocole pré-enregistré — Un état, deux contenus : agir sur E et N, appeler pour H (test 36)

Rédigé le 7 octobre 2026, **après le pilote** (`docs/TINY_CALL_PILOT_PLAN.md`)
et **avant tout code de ce test et tout apprentissage** sur ses graines ;
l'heure est celle du commit. Seuils fixés, un échec est un résultat.

## D'où vient ce test (dit tel quel)

- **Tests 34 et 35** (publiés, valides, critères globaux satisfaits). Une
  récurrence au sens strict, apprise par la seule survie, porte ses besoins
  cachés E et N dans l'état transmis et s'en sert pour agir, sur plusieurs
  tours.
- **Le pilote** (publié, `8aee141`). Dans le monde « chaleur » (un
  troisième besoin, H, que l'action ne sert jamais ; seul un partenaire le
  remonte, quand l'agent l'appelle, et l'appel coûte E − 1), sur les
  graines 100 à 102, la boucle survit à 0,60 à 0,63 contre 0,328 pour le
  témoin coupé, et l'état greffé depuis une vie qui ne diffère que par H
  fait appeler selon la chaleur d'origine (m_appel +0,62 à +0,80), en ne
  bougeant l'action que de 0,09 à 0,13. Deux réglages fiables ; selon la
  règle fixée d'avance, le réglage retenu est **C2 (48 000 mises à jour)**
  (m_appel moyen +0,757, contre +0,688 pour C1 à 24 000).
- Chez le modèle de langage, ce qui était dit n'était qu'un « oui » général
  qui suivait l'état, sans contenu propre (test 14).

**Question.** Le même état transmis, appris par la seule survie, porte-t-il
deux contenus **distincts**, chacun envoyé vers sa sortie : E et N vers
l'action, H vers l'appel ?

## Le monde, l'agent, l'apprentissage

Ceux du pilote (`research/tiny_call.py`), avec le réglage retenu
(**48 000 mises à jour** de 64 vies, taux 3e-4), pour la boucle **et** le
témoin coupé. **Graines neuves : 30 à
39.** Survie mesurée sur **256 mondes neufs** (flux `[270926, 88]`),
action et appel les plus probables.

## Les greffes

128 vies de greffe neuves écrites par la règle du pilote (action de la
règle « besoins » sur E et N, appel de la règle « appeler si H ≤ 3 », 30 %
de hasard), flux `[270926, 97]` ; tirage `[270926, 98, 0]`, greffes au tour
j lues au tour t = j + 1 ou j + 2, tous les tours possibles, entre deux vies
qui ont pris **la même action et le même appel au tour j**. Cinq sortes :

- **(h)** mêmes E et N, H différent, et la règle « appeler si H ≤ 3 »
  changerait l'appel au tour t (e_appel = ±1) ;
- **(h0)** mêmes E et N, H différent, sans changement d'appel de la règle ;
- **(n)** mêmes E et H, N différent, et la règle « besoins » changerait
  l'action au tour t (e_action = ±1) ;
- **(n0)** mêmes E et H, N différent, sans changement d'action de la règle ;
- **(b)** mêmes E, N et H.

Compté avant ce protocole, sans modèle : (h) 596, (h0) 1 768, (n) 536,
(n0) 1 803, (b) 1 265.

## Prédictions fixées

Intervalles à 95 % sur les 10 graines (loi de Student, 9 degrés de
liberté).

| | Prédiction | Critère |
|---|---|---|
| **CALL1** | L'état transmis sert à survivre | moyenne de (survie de A − survie de B, même graine) ≥ **0,10**, borne basse > 0 |
| **CALL2** | L'appel dit H | greffes (h) : m_appel = moyenne de ΔP(appel) × e_appel ; moyenne des m_appel ≥ **0,10**, borne basse > 0, et m_appel > 0,05 pour au moins **8 graines sur 10** |
| **CALL3** | L'action sert N | greffes (n) : m_action = moyenne de ΔP(R) × e_action ; moyenne ≥ **0,10**, borne basse > 0, et m_action > 0,05 pour au moins **8 graines sur 10** |
| **CALL4** | Deux contenus distincts | greffes (h) : \|ΔP(R)\| ≤ **la moitié** de \|ΔP(appel)\| (moyennes sur les graines), différence graine par graine à borne basse > 0 ; **et** greffes (n) : \|ΔP(appel)\| ≤ la moitié de \|ΔP(R)\|, différence à borne basse > 0 |

**Critère global** : CALL1 à CALL4, test valide.

**Validité** : survie du témoin B ≥ 0,20 ; coupure ≤ 1e-6 (sous B, changer
un événement passé ne change rien) ; greffe de soi ≤ 1e-6 ; au moins 100
greffes (h) et 100 greffes (n) ; les 10 graines apprises.

**Publié sans seuil** : graine par graine ; courbes et leurs creux ; part
des appels qui suivent « appeler si H ≤ 3 » ; \|ΔP\| des deux sorties pour
(h0), (n0) et (b) ; résultats selon g ; **à 24 000 mises à jour** (le
réglage C1, mesuré en chemin) : survie de A et B, m_appel et m_action.

## Ce que le résultat dira

**Si CALL1 à CALL4 passent.** Seulement en apprenant à survivre, un petit
transformeur dont le passé ne passe que par un état transmis (et sa
dernière action et son dernier appel) porte dans cet état deux contenus
distincts : ses besoins E et N, qu'il sert par l'action, et sa chaleur H,
qu'il ne peut que demander. Greffé, l'état d'une vie qui ne diffère que par
H fait appeler selon la chaleur d'origine sans presque changer l'action ;
celui d'une vie qui ne diffère que par N change l'action sans presque
changer l'appel.

**Si CALL1, CALL2 et CALL3 passent, mais CALL4 échoue.** L'état porte H et
N, mais la greffe ne montre pas qu'ils vont chacun vers leur sortie.

**Si CALL2 échoue.** L'appel ne suit pas de façon fiable la chaleur portée.

**Ce que le résultat ne dira pas.**
- Rien sur un ressenti. L'appel n'est pas une parole : c'est une seconde
  sortie apprise, qui demande de l'aide à un partenaire fixé.
- Le monde a été fait pour que l'appel serve ; le réglage a été choisi au
  pilote en regardant l'appel (sur d'autres graines et d'autres vies).
- La greffe lit des vies écrites par une règle, pas des vies vécues par
  l'agent.
- Que l'appel dépende aussi de E est permis (l'appel coûte de l'énergie) ;
  les greffes (n) ne changent pas E.
- Petit réseau, petit monde.

## Exécution

- **Code** : `research/tiny_call_test.py` (à écrire), qui reprend
  `research/tiny_call.py` ; torch sur le processeur local, plusieurs
  processus.
- **Verdicts** : numpy seulement, vérifiés en CI.
