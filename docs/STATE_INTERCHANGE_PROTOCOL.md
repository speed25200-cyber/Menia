# Colab 15 — distinguer transfert d'état, copie de réponse et absence d'effet

Protocole fixé le 19 septembre 2026, après l'audit du Colab 14 et avant toute
inférence Qwen3-4B de cette expérience. **Pilote mécanistique sans entraînement
et sans alignement appris.**

## Pourquoi cette étape maintenant

Le [Colab 14](OPTIMIZER_MEMORY_RESULTS.md) montre que les trois checkpoints
précédant la dernière phase produisent 92,71–100 % de réponses canoniques
correctes, mais généralisent mal à la reformulation. Cela permet de revenir
à la [question mécanistique préparée auparavant](MECHANISTIC_COMPOSITION_REVIEW.md),
dans le domaine canonique où la compétence existe. Nous prenons les trois
checkpoints `prefix`, pas le meilleur modèle ni la meilleure graine. Ce choix
est motivé par les résultats 14 : il n'était pas préenregistré avant ceux-ci.

Cette étape ne répare pas le défaut de reformulation. Elle demande si une
intervention sur les activations peut transmettre le contenu de l'état tout
en laissant la règle de réponse du destinataire agir. Le résultat orientera
la correction suivante : il ne représente pas déjà une conscience de soi.

## Trois prédictions et une ambiguïté à éviter

Pour des états binaires D et R et des codes de réponse d et r :

```text
Transfert d'état : D XOR r
Copie du chiffre prévu chez le donneur : D XOR d
Destinataire inchangé : R XOR r
```

Si D et R sont opposés et d et r aussi, les deux dernières prédictions sont
identiques. Le contraste seul ne distinguerait donc pas copie et absence
d'effet. Le protocole réalise **les seize combinaisons**, dont états identiques,
codes identiques et leurs deux inversions. Chaque paire d'hypothèses diverge
sur huit des seize combinaisons ; les deux tests logiques le vérifient.

Les chiffres prévus selon les étiquettes sont distingués des sorties réellement
observées du donneur et du destinataire. Toutes les observations sont conservées,
y compris quand ces sorties initiales sont incorrectes ou hors des options.

## Poids, données et interventions fixés

- Modèle : Qwen3-4B BF16 à la révision déjà utilisée ; adaptateurs `r0-prefix`,
  `r1-prefix`, `r2-prefix` du Colab 14, plus la base sans adaptateur.
- Trois jeux de phrases, chacun comprenant 16 paires dans le vocabulaire connu
  et 8 paires avec le vocabulaire réservé à l'apprentissage des adaptateurs.
  Chaque paire possède un donneur et un destinataire de textes distincts.
  Les phrases et graines ne réutilisent pas celles des expériences 6–14.
- Les quatre combinaisons des marqueurs publics donneur/destinataire sont
  équilibrées. Les positions perturbées alternent entre les deux phrases.
  La perturbation reste la rotation normée injectée après la couche 17.
- Deux questions canoniques : présence de la perturbation et lecture de
  `[REPERE]` sur la première phrase. La seconde est un contrôle de transfert
  d'information publique. Elle ne mesure pas la préservation de toutes les
  autres capacités après une lésion, ni le transfert entre tâches différentes.
- Sites d'échange : sorties des couches **17, 23 et 35**, indexées à partir de
  zéro, au dernier token du prompt. On remplace le vecteur complet par celui
  du donneur. Aucun sous-espace, classifieur, rotation apprise ou recherche de
  meilleur site n'est ajusté sur les résultats.

La couche 17 est un contrôle de chronologie : la modification des autres
positions, faite à sa sortie, ne doit pas encore modifier son dernier token.
Les deux états du même donneur y fournissent donc le même vecteur. La couche 35
est un contrôle de copie : après elle, seuls la normalisation finale et le
calcul des logits restent, donc sa substitution doit reproduire les logits du
donneur. Ce résultat attendu par construction ne sera pas une découverte.
La couche 23, six couches après l'injection, est le site principal fixé sans
optimisation sur des scores de transfert.

## Effectifs et mesures

Chaque paire et chaque modèle donnent 16 passages intacts, 24 échanges avec
le propre vecteur du destinataire et 96 transferts donneur–destinataire.
Total : **144 groupes, 19 584 passages**, dont **3 456 témoins identiques** et
**4 608 copies de dernière couche**. Aucun poids ne change.

Les 72 tableaux résument, par répétition, lot, modèle, tâche et site : accord
avec les trois prédictions, pertes des cibles dans le vocabulaire complet,
copie effective du token donneur, conservation effective du token destinataire,
sorties autorisées et norme du remplacement. Les 48 tableaux intacts décrivent
la compétence initiale par code. Les agrégats donnent le même poids à chaque
paire, puis à ses seize combinaisons.

Les **six contrastes primaires** sont les différences d'accord « transfert
d'état moins copie du chiffre » et « transfert d'état moins destinataire
inchangé », pour les trois `prefix`, la couche 23, la présence cachée et les
16 paires du lot principal. Les 138 autres contrastes sont secondaires. Les
intervalles individuels à 95 % utilisent 2 000 rééchantillonnages de paires,
stratifiés par les quatre combinaisons de marqueurs publics. Ils n'appliquent
pas de correction de multiplicité et ne constituent pas un test global.

Une interprétation positive du contraste principal nécessite d'abord au moins
90 % de réponses intactes correctes dans chacun des deux codes et chacune des
deux tâches, pour la répétition concernée. Ce prérequis est évalué sur les
nouvelles phrases. Un échec n'entraîne aucun retrait d'essai ou remplacement
de checkpoint ; il limite la conclusion mécanistique. Même si les deux
contrastes sont positifs avec intervalles excluant zéro dans les trois
répétitions admissibles, on parlera d'un soutien à ce transfert dans ce montage,
pas d'une identification unique du mécanisme habituel.

## Intégrité et limites

Le journal fixe plan, sources, environnement, empreintes des parents, prompts
et activations. Le lecteur vérifie l'ordre complet des requêtes, les échanges
exactement appliqués, l'identité des témoins, la copie à la couche 35 et le
contrôle de chronologie à la couche 17. Une différence technique invalide
l'interprétation au lieu d'être transformée en résultat cognitif.

La reprise travaille par groupe : un groupe incomplet est déclaré comme tel
et entièrement recalculé, sans mise à jour des poids. Le compteur de reprises
reste dans le bilan ; les passages perdus avant écriture ne sont pas présentés
comme des observations nouvelles. Un journal terminé est un no-op. Les données
brutes sont privées, les agrégats et empreintes sont publiables.

L'échange causal possède des précédents explicites chez
[Geiger et al.](https://arxiv.org/html/2303.02536v4). Les
[illusions d'intervention étudiées par Makelov et al.](https://proceedings.iclr.cc/paper_files/paper/2024/file/70b8505ac79e3e131756f793cd80eb8d-Paper-Conference.pdf)
motivent la prudence : produire une sortie contrefactuelle peut solliciter
des voies différentes du calcul initial. Ici, l'absence d'alignement appris
retire cette source particulière d'expressivité, mais un remplacement complet
peut encore importer du contexte ou créer un état inhabituel. Le transfert
public comparable aide à ne pas baptiser toute recombinaison « introspection ».

Un résultat négatif à la couche 23 ne prouvera pas l'absence de représentation
d'état ailleurs. Un résultat positif ne montrera ni expérience subjective,
ni représentation de sa propre existence. La généralisation de la formulation,
les erreurs naturelles et l'utilité pour l'action restent à tester séparément.

Six tests logiciels passent avant gel : tables de vérité, jeu factoriel complet
avec mécanismes connus, rejet des témoins altérés et groupes incomplets,
136 vrais passages d'un Qwen miniature aléatoire, préfixes de longueurs
différentes et retrait des hooks en cas d'erreur. Ils ne sont pas des résultats
du modèle préentraîné.
