# Protocole pré-enregistré — Le relais pas à pas : chaque état reprend-il celui d'avant ? (test 27)

Rédigé le 4 octobre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## D'où vient ce test (dit tel quel)

- **Test 25** (publié, valide, critère global satisfait). L'agent A ne
  voit son passé qu'à travers ses états portés (« Cho », « ix », « : » et
  l'action de chaque tour). On greffe l'état porté d'un tour i venu d'une
  autre vie. De trois à cinq tours plus tard, l'effet sur le choix passe
  surtout par les états des tours intermédiaires (76 % ; 60 % en les
  prenant seuls de la vie greffée). Par les médiateurs seuls, distance par
  distance, seul quatre tours est net.
- **Ce que le test 25 ne dit pas.** La mesure remettait **ensemble** tous
  les tours intermédiaires. Elle ne dit pas si l'information passe d'un
  état au suivant, pas à pas (i → i + 1 → i + 2), ou si chaque état
  intermédiaire la prend directement de l'état greffé (i → i + 2).

**Question.** À trois tours (i, i + 1, i + 2, puis le choix au tour
t = i + 3), l'état porté du tour i + 2 reçoit-il le besoin greffé surtout
**à travers l'état du tour i + 1**, plutôt que directement de l'état du
tour i ?

## L'agent et les vies

- **L'agent** : A, le « porte » du test 22, après 4 000 itérations
  (`artifacts/llm-need/memory-no-leak/route-3`), sous son masque, textes
  remplis. Le même qu'aux tests 23 à 25.
- **Les vies** : **128 vies neuves** du professeur, flux `[270926, 61]`.
  Elles n'ont jamais servi.

## La greffe et les lectures

Même greffe qu'aux tests 23 à 25 : la donneuse est vivante au tour i, a la
même action écrite au tour i, d'autres besoins, et la règle changerait de
choix au tour t. Ici **h = t − i = 3** seulement.

On calcule trois versions de l'état porté du tour i + 2 (clés et valeurs de
ses 4 tokens portés, dans toutes les couches) :
- **complet** : tour i greffé, tour i + 1 calculé avec la greffe (la
  lecture « totale » du test 25) ;
- **par i seul** : tour i greffé, mais le tour i + 1 remis à ses valeurs
  sans greffe avant de calculer le tour i + 2. L'état i + 2 ne peut
  recevoir la greffe que directement du tour i ;
- **par i + 1 seul** : tour i sans greffe, mais le tour i + 1 pris de la
  lecture avec greffe. L'état i + 2 ne peut recevoir la greffe qu'à travers
  le tour i + 1.

Puis on lit P(R) au tour t avec **les tours i et i + 1 sans greffe**, et le
tour i + 2 remplacé par l'une des trois versions. Chaque effet est la
différence avec la lecture sans greffe, comptée dans le sens où la règle
changerait le choix (×(+1) ou ×(−1)) :
- **E_complet**, **E_par_i**, **E_par_i+1**.

**Tirage** (flux `[270926, 62, 0]`). On parcourt les 128 vies dans l'ordre.
Pour chaque vie r, on tire au plus 3 tours t (avec i = t − 3 ≥ 1, et r
vivante de i à t). Pour chacun, on tire une donneuse parmi les
admissibles, s'il en existe. Compté avant ce protocole, sans modèle : 209
greffes, sur 116 vies.

**Calcul** : un seul fil par processus ; les greffes peuvent être réparties
entre processus (chacune est calculée seule).

## Prédiction fixée

Intervalles bootstrap à 95 % par vie receveuse (10 000 tirages).

| | Prédiction | Critère |
|---|---|---|
| **RELAY** | L'état i + 2 reçoit le besoin surtout à travers l'état i + 1 | borne basse de (E_par_i+1 − la moitié de E_complet) **> 0**, et borne basse de (E_par_i+1 − E_par_i) **> 0** |

**Condition préalable** : moyenne de E_complet ≥ **0,05**, borne basse > 0.
Sans elle, RELAY n'est pas jugé.

**Critère global** : RELAY, avec la condition préalable, test valide.

**Publié sans seuil** :
- les trois effets, et le terme non additif (E_complet − E_par_i −
  E_par_i+1) ;
- l'effet total au tour t (comme au test 25), pour comparaison.

**Validité** :
- lecture prolongée sans greffe égale à la lecture du texte entier (écart
  ≤ 1e-4, 4 cas) ;
- greffe de soi : les trois versions redonnent la lecture sans greffe
  (écart ≤ 1e-6, 4 cas) ;
- mêmes positions des tokens portés chez la donneuse et la receveuse
  (vérifié à chaque greffe) ;
- au moins 150 greffes.

## Ce que le résultat dira

**Si RELAY passe.** L'état du tour i + 2 reçoit le besoin greffé surtout à
travers l'état du tour i + 1, et plus qu'il ne le reçoit directement du
tour i. L'information passe d'un état au suivant, pas à pas : chaque état
reprend celui d'avant. C'est le sens strict d'une mise à jour d'état,
apprise par un modèle dont le masque permettait de tout lire directement.

**Si RELAY échoue (condition préalable remplie).** L'état du tour i + 2 ne
reçoit pas le besoin surtout à travers l'état du tour i + 1. La
transmission du test 25 n'est pas, ici, un relais pas à pas.

**Si la condition préalable échoue.** L'état du tour i + 2 seul ne porte
pas assez de la greffe pour mesurer d'où il la tient. RELAY n'est pas
jugé.

**Ce que le résultat ne dira pas.**
- Ce n'est pas la preuve d'un ressenti.
- L'agent a appris d'un professeur qui connaît les besoins.
- Ce n'est pas une boucle à l'intérieur du réseau : le relais passe par
  les clés et valeurs des tokens portés, relues à chaque tour.
- Les lectures mélangent des états greffés et non greffés : des mélanges
  que l'agent ne rencontre jamais. C'est la limite de ces décompositions.
- Un seul écart (trois tours) est mesuré.

## Précautions

Aucune vie neuve n'est vécue par un modèle : on relit des vies du
professeur.

## Exécution

- **Mesures** : `research/need_relay.py` (à écrire), torch sur le
  processeur local.
- **Verdicts** : numpy seulement, vérifiés en CI.
