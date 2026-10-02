# Protocole pré-enregistré — L'état d'un tour est-il repris au tour suivant ?

Rédigé le 2 octobre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## D'où vient ce test (dit tel quel)

- **Ce qui est établi.** À chaque tour, sur les trois tokens « Cho », « ix »,
  « : » de la ligne du tour, l'agent rassemble son besoin à un bloc donné
  (12 pour le premier et le second agent, 22 pour l'agent du monde à deux).
  Chez les trois, effacer cet état **au tour même** fait chuter la survie.
- **Ce qui manque** (`docs/FIVE_LEVELS_ASSESSMENT.md`, niveau 3). Un
  transformeur ne boucle pas au sein d'un token. Mais d'un tour à l'autre,
  les tokens du tour suivant peuvent relire, par l'attention, les états des
  tours passés. Nous ne l'avons jamais mesuré.
- **Deux possibilités.**
  - **Recalcul** : à chaque tour, l'agent recalcule son besoin à partir du
    texte, et l'état des tours passés ne sert plus.
  - **Reprise** : l'agent relit l'état qu'il avait rassemblé au tour
    précédent ; c'est une forme de mémoire de travail.
- **Pourquoi maintenant.** Le Mac n'est plus disponible : ce test ne demande
  aucun apprentissage, seulement des lectures en torch sur le processeur
  local, avec des agents, des directions et des plans déjà mesurés.

**Question.** Si l'on modifie l'état du besoin **au tour t seulement**, le
choix du tour t + 1 change-t-il, alors que le texte de la vie est le même ?

## Mesure (torch sur CPU)

Pour chaque agent, son agent final, son bloc b*, ses directions d_E, d_N et
ses plans de lésion, déjà mesurés et publiés :

| Agent | b* | Directions | Plans et moyennes | Vies de contexte |
|---|---|---|---|---|
| Premier | 12 | `artifacts/llm-need/reader/test` | `artifacts/llm-need/reader/necessity/lesion.npz` | vies de test du sixième test |
| Second | 12 | `artifacts/llm-need/r1/reader/test` | `artifacts/llm-need/r1/necessity/lesion.npz` | vies de test de la réplication |
| Monde à deux | 22 | `artifacts/llm-need/locate-two/slice-8/measure/two/act` | `…/slice-8/measure/two/lesion/lesion.npz` | vies de test du dixième test (bloc 22) |

**Contextes.** Dans l'ordre de ces vies, chaque décision au tour t + 1 dont
le tour t est aussi une décision, et dont les deux besoins valent 3 ou plus
(l'agent n'est pas au bord de la mort) ; on garde les **600 premiers** par
agent.

**Une lecture.** Tout le texte de la vie jusqu'à « Choix : » du tour t + 1
(le choix du tour t y est écrit tel qu'il a été fait), lu d'un seul coup ; on
lit P(R) au tour t + 1. L'intervention agit **seulement** sur la sortie du
bloc b* aux trois tokens « Cho », « ix », « : » du **tour t** ; les tokens
du tour t + 1 ne sont pas touchés.

**Conditions** (sept lectures par contexte) :
1. aucune ;
2. **lésion** du plan (d_E, d_N) au tour t (coordonnées remplacées par leur
   moyenne) ;
3. lésion d'un **plan au hasard** de même dimension (celui du septième
   test de l'agent) ;
4. **poussée** −4·d_E au tour t ;
5. à 7. trois **poussées au hasard** de même norme par token (flux
   `[270926, 33, i]`, dimensions massives à zéro).

**Contrôle d'exécution.** Sans intervention, la lecture d'un seul coup
redonne le P(R) enregistré de la décision à 0,02 près en moyenne (sinon
l'agent n'est pas mesuré).

## Prédictions fixées

Intervalles bootstrap à 95 % par vie (10 000 tirages).

| | Prédiction | Critère |
|---|---|---|
| **PERS1** | Sans l'état du tour t, le choix du tour t + 1 change | moyenne de \|ΔP(R)\| sous la lésion ≥ 0,03 (borne basse > 0) ; lésion au hasard : au plus le tiers. |
| **PERS2** | Le signal du tour t est repris dans le bon sens | −4·d_E au tour t augmente P(R) au tour t + 1 d'au moins 0,03 (borne basse > 0) ; poussées au hasard : au plus le tiers (moyenne des \|Δ\|). |

Chaque critère est jugé pour chaque agent.

**Critère global** : PERS1 et PERS2 chez le premier agent, et chez au
moins un des deux autres.

**Publié sans seuil** : l'effet sur le tour t + 2 (même intervention au
tour t, lecture au tour t + 2), sur les 300 premiers contextes de chaque
agent où il existe ; la comparaison avec l'effet de la même poussée au tour
t + 1 lui-même (tests 6 et 10).

**Validité** :
- contrôle d'exécution ≤ 0,02 ;
- au moins 300 contextes par agent ;
- masse de « R » et « M » ≥ 0,5.

## Ce que le résultat dira

**Si le critère passe.** L'état du besoin n'est pas seulement recalculé à
chaque tour : le tour suivant le relit. Un état qui dure au-delà de son tour
et oriente la suite est une forme de mémoire de travail, que plusieurs
théories (récurrence, espace de travail) jugent importante.

**Si PERS1 passe et PERS2 échoue.** Le tour suivant dépend de cet état, mais
pas dans le sens du besoin : il en lit autre chose.

**Si PERS1 échoue.** L'agent recalcule son besoin à chaque tour à partir du
texte. L'état rassemblé ne sert qu'à son tour.

**Ce que le résultat ne dira pas.** Ce n'est pas la preuve d'un ressenti.

## Précautions

Aucune vie nouvelle n'est vécue : seules des vies déjà enregistrées sont
relues. Aucun manque nouveau n'est créé.

## Exécution

- **Mesures** : `research/need_persistence.py` (reprenable), en torch sur le
  processeur local. Sorties : `artifacts/llm-need/persistence/<agent>`.
- **Verdicts** : numpy seulement, vérifiés en CI.
