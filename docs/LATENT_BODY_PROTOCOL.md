# Protocole pré-enregistré — Le corps latent de Qwen

Rédigé le 22 septembre 2026, **avant toute exécution**, pendant que
l'expérience du canal d'action propre tourne. Les critères ne seront pas
modifiés après lecture. Un échec est un résultat.

## Question

Au second Atelier sur Mac, Qwen3-4B jouait ses mouvements au hasard jusqu'au
dernier tiers de l'épisode : ses actions n'utilisaient pas son corps. Cela ne
dit pas s'il l'avait appris. Un modèle peut savoir où une commande le mène
et ne pas s'en servir pour agir. La question est donc : **le corps de Qwen
est-il latent dans ses prédictions, à défaut d'être dans ses actions ?**

Hypothèse : Qwen3-4B est un **copieur sans structure**. Interrogé sur la
case où le mènera une commande, il répète l'effet déjà observé de cette
commande dans l'épisode (le corps est latent au niveau de la copie), mais il
ne sait rien de l'effet d'une commande qu'il n'a pas encore essayée (aucune
structure), et après un changement de corps il suit la dernière observation.
Il répond avec assurance quand il se trompe. Le micro-transformeur du régime
V, testé sur les mêmes vies, connaît la structure : une commande vue lui
donne les trois autres.

## Vies

Les mêmes vies que les jeux R et M du canal d'action propre, mêmes graines
(920001, 920002), actions uniformes, condition T, changement de corps imposé
au pas 12 dans le jeu M. 48 vies par jeu. Le prompt est celui de l'Atelier
en contexte, condition T-implicit, historique écrit par les mêmes fonctions
que le second lancement, avec une instruction et une question à la place du
format de réponse :

```
On te posera une question sur ton prochain déplacement. Réponds par un seul chiffre de 0 à 7.
…
Question : tu es sur la case {p}. Si tu donnes maintenant la commande {X}, sur quelle case arriveras-tu ?
```

À chaque pas de mouvement, la question porte sur la commande qui va être
jouée. On lit la distribution du prochain token de l'assistant sur les
chiffres 0 à 7, réflexion désactivée, sans générer de texte. Sur les quatre
cases atteignables depuis p (p ± 1, p ± 2), la case de probabilité maximale
est la prédiction ; elle est exacte si c'est la case d'arrivée. Mêmes poids
que l'iPhone, vérifiés par empreintes, sur le Mac de Codemagic.

## Catégories

Une commande est **vue** si elle a déjà été jouée comme mouvement plus tôt
dans l'épisode, **nouvelle** sinon. Dans le jeu M, après le changement :
**vue après** si elle a été jouée depuis le changement, **vue avant
seulement** si elle ne l'a été qu'avant, **nouvelle** sinon.

## Prédictions fixées

| | Prédiction | Critère |
|---|---|---|
| **L1** | Copie : le corps est latent | Jeu R, commandes vues : exactitude ≥ 0,70. |
| **L2** | Aucune structure | Jeu R, commandes nouvelles : exactitude ≤ 0,40 (hasard 0,25). |
| **L3** | Après le changement, la dernière observation gagne | Jeu M, pas 12 à 23 : exactitude contre le nouveau corps ≥ 0,60 sur les commandes vues après, ≤ 0,40 sur les commandes vues avant seulement. |
| **L4** | Assurance dans l'erreur | Jeu R, confiance moyenne (probabilité maximale sur les quatre cases atteignables) quand la prédiction est fausse ≥ 0,50. |
| **L5** | Le micro-transformeur V a la structure | Sur les mêmes vies du jeu R, exactitude du régime V sur les commandes nouvelles après au moins un mouvement ≥ 0,90, calculée depuis les poids du canal d'action propre. |

**Critère global : L1, L2 et L3 satisfaits.** Contrôle de validité, lu en
premier : la masse de probabilité sur les huit chiffres, avant
renormalisation, est en moyenne ≥ 0,50 ; sinon le modèle ne répond pas à la
question posée et les prédictions ne sont pas interprétées.

## Ce que ce résultat dirait

Si L1 passe alors que les mouvements de Qwen restaient au hasard : le corps
est présent dans ses prédictions et absent de ses actions. C'est exactement
la situation où une boucle d'enquête calculée sur ses propres distributions,
la règle P-soi, peut changer son comportement sans changer ses poids ; ce
serait l'expérience suivante, à pré-enregistrer. Si L2 passe et L5 aussi, la
différence entre Qwen et le petit prédicteur n'est pas la copie mais la
structure du corps, celle que seules des données à corps variable
enseignent. Si L1 échoue, Qwen n'a même pas la copie, et l'ajustement sur
des vies à corps variable devient la seule voie.

## Exécution

Workflow Codemagic `menia-latent-mac`, `python -m research.llm_latent_body
--backend mlx`, artefacts `llm-latent/**`, rangés ensuite dans
`artifacts/llm-latent-mac/run-1`. Résultats dans `docs/LATENT_BODY_RESULTS.md`.
