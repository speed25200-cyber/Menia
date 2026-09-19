# Préparer une intervention avant le jugement et l'action

20 septembre 2026. Instrumentation sur petit Qwen aléatoire et CPU. Aucun
nouveau résultat de Qwen3-4B. Le Colab 23 continue son protocole figé ; ces
fichiers ne modifient ni son entraînement, ni ses sources, ni ses critères.

## Blocage à traiter

Même si le [Colab 23](NATIVE_ANSWER_CONFIDENCE_PROTOCOL.md) améliore la prévision
des erreurs, il ne montrera pas encore que l'agent utilise cette estimation
pour choisir une action. Modifier son score déclaré ne suffirait pas non plus.
Une intervention directement orientée vers les tokens de sortie peut déplacer
ce score sans apprentissage de ses erreurs. Il faut distinguer l'information
prédictive, sa verbalisation et son utilisation dans une politique.

Un précédent existe déjà : [Kumaran et al., §3.5.3 et §4.5.1](https://arxiv.org/html/2603.22161v2)
interviennent sur les activations de Gemma pour modifier l'abstention, et
distinguent confiance de sortie, confiance verbale et information interne.
Pour Menia, ce résultat motive un test d'usage causal, sans autoriser à appeler
nouvelle la relation générale entre confiance et décision. La page Nature
repérée lors de cette recherche n'a pas pu être récupérée ; l'analyse repose
sur cette version du texte intégral, déjà utilisée dans le dossier Menia.

[Heimersheim et Nanda, §§2–3](https://arxiv.org/html/2404.15255v1) expliquent que
restaurer un comportement et le perturber ne testent pas des propriétés
équivalentes, et que les conclusions dépendent des contrastes entre entrées.
[Zhang et Nanda](https://arxiv.org/html/2309.16042v2) étudient la sensibilité aux
métriques et aux méthodes de perturbation. Les outils ci-dessous appliquent
ces précautions ; ils ne constituent pas un nouveau principe scientifique.

## Préfixe commun aux deux suites

La proposition est de partir de la même question et de la même réponse
candidate, puis d'ouvrir séparément deux suites : demander une estimation
d'exactitude ou une décision d'accepter/vérifier. Le verdict de confiance ne
doit pas être ajouté au contexte de la décision. Une intervention est placée
sur un token du préfixe commun, avant les deux demandes divergentes.

[`confidence_prefix_interventions.py`](../research/confidence_prefix_interventions.py)
capture ou modifie la sortie d'un bloc Qwen au dernier token du préfixe
déclaré. Il vérifie que les branches contiennent exactement ces IDs. Le
préfixe est recalculé intégralement, sans modification du cache de génération.
Le code ne choisit pas lui-même une frontière sémantique : le futur protocole
devra prouver que les IDs déclarés précèdent bien la demande de jugement ou
d'action. Une égalité de tokens ne suffit pas à prouver cette propriété.

Dans une base orthonormale U figée, l'opération est :

`h_modifié = h_receveur + U Uᵀ (h_donneur − h_receveur)`.

Elle remplace les coordonnées du sous-espace et conserve le complément à
l'arrondi numérique près. Le code refuse une base non orthonormale, une
substitution du vecteur entier, des états non finis et des entrées rembourrées.
Les autres tokens restent identiques au site d'intervention. La norme du
déplacement est enregistrée. Une telle substitution peut malgré tout produire
un état inhabituel : l'orthogonalité n'établit pas sa plausibilité pour le modèle.

## Témoin qui déplace directement la sortie

Après la normalisation finale, soit d la différence entre les deux lignes
de la tête de sortie correspondant aux codes. Ajouter `s d / ||d||²` à cet
état déplace de s la différence des logits, donc le log-rapport conditionnel
des deux codes, à l'arrondi près. Aucun label d'exactitude n'est nécessaire.
Ce témoin montre pourquoi un effet causal sur le score seul ne suffit pas
à identifier une auto-évaluation apprise. Il ne dit pas que toute intervention
sur une représentation de confiance se réduit à ce cas.

Le [reçu construit sur CPU](../artifacts/confidence-prefix-preparation/control.json)
conserve deux séquences de tokens arbitraires, avec déplacements −1, 0, +1.
Les poids sont aléatoires, les tokens ne sont pas des tâches d'exactitude, et
aucune performance de Menia n'est mesurée. Le même reçu vérifie le préfixe
commun, une substitution partielle, et un témoin remettant l'état lui-même.

Les cinq tests logiciels passent. Ils vérifient la projection et sa restauration,
l'indépendance du préfixe vis-à-vis des tokens futurs, la localisation du
changement, l'identité du témoin sans modification, la conservation des poids
et du générateur aléatoire, et le retrait des hooks même après exception.

```sh
python -m unittest tests_language.test_confidence_prefix_interventions -v
python -m research.confidence_prefix_control --output artifacts/confidence-prefix-preparation/control.json
```

## Ce qui manque avant une expérience interprétable

Aucune direction de confiance n'est encore identifiée ou ajustée. Il faudra
vérifier le résultat du Colab 23, puis utiliser des données de découverte
distinctes d'un nouveau test. Les couches, positions, rangs, donneurs,
amplitudes et critères devront être figés avant ce test. Une direction choisie
pour augmenter le score ne sera pas automatiquement une direction prédictive.

Le futur contraste devra comprendre état propre, donneur apparié, donneur
mélangé, directions de contrôle à déplacement comparable, perturbation et
restauration. Les donneurs peuvent transporter l'identité de la réponse ou
la difficulté plutôt que la confiance ; les contrôles devront les séparer.
Les deux contextes peuvent aussi utiliser des mécanismes différents, même
s'ils partagent un préfixe. Un résultat négatif sur une position ne suffirait
pas à conclure à l'absence de mécanisme ailleurs.

La prévision des erreurs, le choix natif, le coût de vérification et la capacité
de résolution doivent être mesurés séparément. Le contraste avec une simple
modification des codes de sortie est un contrôle supplémentaire, pas un test
de conscience. Le lien entre ces fonctions et une expérience subjective reste
non établi, ainsi que la nouveauté d'une éventuelle contribution Menia.
