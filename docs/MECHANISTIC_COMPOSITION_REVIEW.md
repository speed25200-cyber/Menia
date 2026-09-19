# Tester ce qui est transféré : état interne ou réponse

Note du 19 septembre 2026, préparée pendant les évaluations du Colab 12,
avant consultation de leurs scores. **Proposition conditionnelle, pas un
nouveau protocole fixé ni une expérience exécutée.** Le Colab 12 garde son
code, ses checkpoints, ses données et ses critères.

## Deux explications à distinguer

Une réponse adaptée à la consigne peut venir d'une variable d'état réutilisée,
ou d'une règle de classification directement ajustée à cette consigne. Même
une bonne lecture linéaire des activations ne départage pas ces mécanismes.

On pourrait transférer une représentation candidate d'un essai donneur vers
un destinataire dont **l'état initial et le code de réponse sont opposés**.
Le destinataire conserve sa question. Voici deux cas avec donneur en code
normal et destinataire en code inversé :

| État du donneur | Réponse du donneur | Réponse attendue si l'état est transféré | Réponse si seul le chiffre est copié |
|---|---|---|---|
| Absence | 0 | 1 | 0 |
| Présence | 1 | 0 | 1 |

Inclure aussi le sens inverse et les transferts à code identique. En notation
binaire, l'hypothèse d'état donne `y = état_donneur XOR code_destinataire` ;
l'hypothèse de copie donne `y = état_donneur XOR code_donneur`. Les états
opposés empêchent de compter comme transfert la réponse correcte que produirait
déjà le destinataire intact. Il s'agit de prédictions concurrentes, pas d'un
test de conscience.

## Précédents et faux positifs

[Geiger et al.](https://arxiv.org/html/2303.02536v4) relient variables
interprétables et représentations distribuées par des interventions
d'échange. Leur §4.4 montre aussi qu'une recherche d'alignement peut exploiter
la structure d'un réseau aléatoire suffisamment large. L'échange causal et
la recherche d'un sous-espace ont donc des précédents ; leur application à
Menia ne serait pas, à elle seule, une invention.

[Makelov et al.](https://proceedings.iclr.cc/paper_files/paper/2024/file/70b8505ac79e3e131756f793cd80eb8d-Paper-Conference.pdf)
montrent qu'une intervention dans un sous-espace peut obtenir l'effet visé en
activant une voie qui n'assurait pas le calcul initial. Faire changer une
réponse dans le sens attendu ne suffit donc pas à localiser le mécanisme
habituellement utilisé.

[Sutter et al., §6–8 et annexe I.2.2](https://arxiv.org/html/2507.08802v1)
montrent que des alignements suffisamment expressifs peuvent obtenir des scores
d'intervention très élevés même sur des réseaux aléatoires. Leur contrôle
utilisant des noms disjoints fait échouer cet alignement dans le réseau
aléatoire étudié, tout en préservant le résultat sur le réseau entraîné.
Leur conclusion ne rend pas toute analyse causale inutile ; elle exige de
préciser les hypothèses et la puissance de l'outil d'alignement.

## Contraintes pour un futur essai

- Sélectionner sites, dimensions et paramètres sur des données de développement
  distinctes. Réserver phrases **et éléments lexicaux** au test, avec contrôle
  visible de compréhension ; le Colab 12 ne réserve que les combinaisons.
- Conserver les essais intacts, les transferts sans changement d'état, les
  codes identiques et opposés, les directions témoins et la lecture publique.
  Rapporter tous les essais, sans retenir uniquement ceux dont les réponses
  initiales sont correctes.
- Limiter et déclarer la capacité d'un éventuel alignement appris ; comparer
  aux témoins non entraînés et mélangés. Un outil puissant pourrait apprendre
  lui-même la transformation recherchée.
- Combiner échanges, neutralisation et restauration, en mesurant les effets
  sur les tâches ordinaires. Ces interventions fourniraient des contraintes
  complémentaires, pas une garantie universelle contre les voies artificielles.
- Comparer les prédictions des deux explications sur un ensemble réservé,
  sans choisir après coup le meilleur site ou la meilleure répétition.

Si le Colab 12 ne conserve pas la compréhension des consignes et la lecture,
il faudra d'abord expliquer cet échec. S'il les conserve, ce contraste pourrait
chercher une représentation d'état causalement réutilisée. Les effectifs,
sites, budgets et règles de lecture restent à fixer avant tout nouvel essai.

Même un résultat favorable ne montrerait ni que l'état représente l'existence
de Menia, ni qu'il s'accompagne d'une expérience. Le lien entre mécanisme appris,
modèle de soi et conscience reste une exigence distincte de l'objectif.
