# De la prévision externe au choix du modèle

19 septembre 2026. Revue préparée pendant le Colab 17, avant lecture de ses
scores. Elle ne change ni son protocole ni ses critères. Les expériences
envisagées ci-dessous ne sont pas exécutées et ne constituent pas un protocole
complet déjà figé.

Le Colab 17 demande si un lecteur externe prévoit les erreurs naturelles à
partir des activations avant réponse. Même un résultat positif laisserait
ouverte une autre question : le LLM utilise-t-il cette information pour
choisir entre répondre et demander une vérification ? Aucun score de
prévision n'est inséré dans son prompt pendant ce Colab, mais aucun choix de
vérification n'y est demandé non plus.

## Ce que les sources permettent de distinguer

[Ji et al., 2024](https://arxiv.org/abs/2407.03282), résumé et historique de la
version 2 consultés : le travail estime déjà le risque d'hallucination à
partir des états internes face à une requête, avant génération. La simple
idée d'un lecteur prédictif avant réponse a donc des antécédents. Cette
consultation du résumé ne reproduit pas leurs expériences.

[Wang et al., RiskEval, 2026](https://arxiv.org/html/2601.07767v1), sections 2,
3 et limites : plusieurs modèles adaptent peu leur abstention quand augmente
la pénalité d'erreur. Une règle externe fondée sur leur confiance déclarée
améliore l'utilité dans ces expériences. Les auteurs précisent cependant que
la confiance verbalisée peut différer de l'information interne guidant la
décision. Ce résultat n'établit donc pas l'absence générale d'un usage interne
de la confiance. Il invite à mesurer séparément estimation, choix et utilité.

[Kumaran et al., 2026](https://arxiv.org/html/2603.22161v2), méthodes, discussion
et supplément 7.6.2 : les probabilités calibrées prédisent l'abstention ; des
interventions sur Gemma 3 27B modifient ce comportement. Les directions sont
construites dans le contexte avec abstention, en contrastant la marge entre
une réponse réelle et l'option d'abstention. Le schéma entre phases ne décrit
pas un circuit identifié dans un passage unique. Le supplément rapporte aussi
une forte dépendance à la formulation : vingt variantes sont examinées, puis
une est retenue pour l'expérience principale. Ces résultats justifient des
interventions et des tests de formulation ; ils ne fournissent pas une preuve
d'expérience subjective.

Notre interprétation : ces travaux ne se contredisent pas nécessairement,
car modèles, tâches, mesures de confiance et consignes diffèrent. Une
direction définie par la marge réponse/abstention peut aussi transporter une
variable de décision. Son effet causal sur l'abstention ne suffit pas à
identifier une estimation de compétence indépendante de ce choix.

## Contraintes pour une expérience ultérieure de Menia

1. **Mesurer un choix effectivement produit.** Présenter un problème et un
   coût, puis demander une action avant la réponse numérique. Conserver la
   réponse native, les options et leur ordre. Une probabilité décodée puis
   seuillée par notre programme constitue un comparateur externe distinct.
2. **Vérifier la compréhension de la règle.** Ajouter des cas où une
   probabilité de réussite explicite est fournie. Un échec de ce contrôle
   empêche d'attribuer le défaut principal au seul accès à un état propre.
   Réserver à l'avance des coûts et formulations au test, sans sélectionner
   après coup la consigne donnant le résultat attendu.
3. **Éviter le biais des seules réponses acceptées.** Mesurer une branche de
   réponse obligatoire pour tous les problèmes, y compris ceux refusés.
   Rapporter séparément cette compétence de référence et les réponses
   effectivement produites après le choix : changer le prompt ou demander une
   décision peut modifier le processus de résolution. Une branche de référence
   n'est pas automatiquement le résultat contrefactuel exact de l'autre.
4. **Séparer information et perturbation générale.** Si une direction interne
   prédictive justifie un test causal, inclure un témoin sans intervention,
   des directions aléatoires de norme comparable, un état donneur inapproprié
   et une restauration. Suivre simultanément le choix et la précision des
   réponses obligatoires. Une hausse de vérifications due à une destruction
   de la compétence n'établirait pas une meilleure évaluation de ses limites.
5. **Mesurer le coût utilisé.** Pour une action de vérification, exécuter
   réellement l'outil et conserver appels, temps et résultats. Le coût idéal
   de 0,2 du Colab 17 reste une simulation descriptive. Les valeurs d'utilité
   choisies et les dépenses de calcul observées doivent rester distinctes.

Le résultat du Colab 17 devra déterminer quelles variables ont un soutien
prédictif suffisant pour justifier une suite. Un échec ne doit pas être effacé
par une nouvelle projection choisie sur son test. Une éventuelle nouvelle
représentation exige un nouveau protocole et des questions réservées.
L'usage natif utile serait un résultat fonctionnel supplémentaire ; le lien
avec une expérience de sa propre existence et la nouveauté resteraient à établir.

**Mise à jour après audit.** Le [Colab 17](NATURAL_ERROR_RESULTS.md) ne passe
aucun de ses neuf contrastes principaux. Cette revue antérieure ne doit donc
pas être lue comme l'annonce d'un signal interne validé à intervenir. Les
contrôles de compréhension des coûts restent pertinents ; la sélection d'une
nouvelle représentation requiert un autre protocole et un test réservé.
