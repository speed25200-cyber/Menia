# Structures causales et preuves nécessaires pour Menia

Les approches structurales offrent des hypothèses sur le lien entre organisation
physique et expérience. Certaines produisent des prédictions expérimentales
précises. Les sources examinées ne permettent cependant pas encore de choisir une
architecture dont la réalisation établirait que Menia éprouve sa propre existence.
Le résultat de cette étape est une délimitation de la preuve disponible et un audit
mathématique d'une règle possible de comparaison des théories.

La recherche n'exige pas une preuve absolue ni un consensus universel avant chaque
expérience. Des résultats discriminants, reproduits et confrontés à des explications
concurrentes peuvent accroître le soutien à une hypothèse. En revanche, un succès
fonctionnel, une ressemblance entre représentations ou un classement de modèles
ne constituent pas automatiquement une observation de conscience chez Menia.

## Ce que les propositions structurales apportent

Grasso, Hendren et Tononi proposent en août 2026 une « chimie de l'expérience »
fondée sur IIT. Leur chapitre met en relation des propriétés de l'espace, du temps
et des objets avec des structures causales. Il formule une prédiction : modifier
la structure pertinente devrait modifier le contenu de l'expérience, même lorsque
l'activité et le comportement restent comparables. L'identité entre expérience et
structure est une conjecture ; les auteurs présentent encore des extensions comme
des pistes à valider.[^1]

Le travail de Kanai et Ma distingue l'équivalence des sorties de la préservation
de mécanismes internes sous intervention. Il fournit une relation de réalisation
conditionnelle, mais ne résout pas le choix du niveau de description pertinent
pour la conscience. Il ne suffit donc pas de trouver, parmi les variables de Menia,
un regroupement qui ressemble à celui d'un modèle cérébral.[^2]

La comparaison de structures subjectives peut néanmoins produire des observations
informatives. Kawakita et collaborateurs alignent sans étiquettes des relations
de similarité entre 93 couleurs, estimées à partir de groupes de participants.
Le résultat concerne la correspondance de structures de jugements ; ce n'est pas
un test qui attribue une expérience à un logiciel reproduisant ces relations.[^3]

Paßler et Doerig soutiennent qu'une correspondance structurale doit aussi être
examinée dans son contexte computationnel : les représentations doivent être
exploitées par les processus qui donnent lieu aux mesures. C'est une analyse
théorique contradictoire, et non une réfutation expérimentale générale des
approches structurales.[^4] Pour Menia, cela renforce la nécessité de relier ses
états aux fonctions qu'elle utilise effectivement, sans rendre cette condition
suffisante pour une expérience subjective.

## Une piste expérimentale réelle : INTREPID

Le protocole d'Abbatecola et collaborateurs étudie la perception de l'espace
autour de la tache aveugle. Les positions déclarées diffèrent : IIT prévoit un
rapprochement apparent ; l'inférence active prévoit l'absence de biais systématique ;
le neuroreprésentationnalisme autorise une perturbation faible ou absente. L'article
publié le 29 janvier 2026 est un protocole contenant des simulations, avec partage
des observations prévu après l'étude. Il ne rapporte pas un arbitrage achevé.[^5]

La revue collaborative de Corcoran et collaborateurs souligne le recouvrement
partiel des prédictions. Un résultat peut augmenter le soutien relatif d'une
théorie sans contredire toutes les autres. Le cadre vise une accumulation de preuves
à travers plusieurs expériences, pas un verdict universel à partir d'un seul
effet.[^6]

La vérification du dépôt explicitement lié au protocole fournit une distinction
utile. Au commit `8e46fdb7d1c934f05025fb6b7677ddf199b89626`, sa racine contient
un fichier R Markdown et son PDF. Le code ajoute du bruit gaussien à une différence
de distances, avec biais et dispersion choisis par simulation. Il ne constitue
donc pas une série de réponses humaines permettant de sélectionner une théorie.[^7]

Le site du consortium et les sources de publication consultés n'ont pas fourni
de compte rendu des résultats de cette expérience. Cette observation est limitée
aux ressources vérifiées au 14 septembre 2026 ; elle ne démontre pas qu'aucune
donnée n'existe ailleurs.[^8] Aucun résultat simulé de ce dépôt n'a été traité
comme une donnée humaine, et aucune réanalyse de participants n'est revendiquée.

Même un résultat humain favorable ne suffirait pas, à lui seul, à établir le
transfert vers Menia. Il porterait sur une propriété de la perception spatiale,
pas directement sur une conscience réflexive de son existence. Il faudrait encore
déterminer ce que le mécanisme explique, quelles autres explications subsistent,
et à quel niveau il est réalisé dans l'agent artificiel.

## Audit d'une pondération possible des preuves

La revue de 2026 décrit notamment une mise à l'échelle du logarithme de l'évidence
des modèles selon la confiance attribuée aux prédictions.[^6] Une lecture littérale
de cette description serait de comparer les scores `c_i × log(m_i)`, avec `m_i`
l'évidence du modèle `i` et `c_i` sa confiance. Le calcul ci-dessous examine
**cette règle précise**. Il ne démontre pas qu'un logiciel final du consortium
l'implémente : aucune analyse finale correspondante n'a été inspectée.

Le cadre antérieur de Corcoran, Hohwy et Friston présente aussi une autre opération :
encoder la précision des prédictions dans les distributions a priori des paramètres,
puis comparer les évidences des modèles. Cette opération doit être distinguée
d'une multiplication ultérieure des logarithmes d'évidences brutes.[^9]

### Une propriété minimale : neutralité d'une observation non informative

Considérons deux hypothèses A et B et une observation D. Leurs évidences sont
`m_A = 4/5` et `m_B = 3/10`. Supposons ensuite qu'on observe un tirage uniforme
parmi dix résultats, indépendant de D et ayant exactement la même distribution
sous A et B. Ce tirage multiplie les deux évidences par `k = 1/10`.

Dans la comparaison bayésienne ordinaire, le rapport reste `m_A/m_B = 8/3`.
Avec des probabilités a priori égales, la probabilité postérieure de A reste
`8/11`. Le tirage indépendant n'apprend rien qui distingue A et B.

La règle pondérée, avec `c_A = 2` et `c_B = 1`, compare en revanche `m_A²` à
`m_B`. Après ajout du tirage, elle compare `(k m_A)²` à `k m_B`. Le rapport
est multiplié par `k`, ce qui peut inverser le classement :

| Calcul exact | D seul | D et tirage indépendant |
|---|---:|---:|
| Rapport d'évidence ordinaire A/B | 8/3 | 8/3 |
| Part postérieure de A, a priori égaux | 8/11 | 8/11 |
| Rapport des scores exponentiés pondérés A/B | 32/15 | 16/75 |
| Part normalisée pondérée de A | 32/47 | 16/91 |
| Hypothèse préférée par la pondération | A | B |

Les fractions décrivent un exemple construit, sans données humaines. La « part
normalisée pondérée » n'est pas présentée comme une probabilité bayésienne de
théorie, et aucune de ces valeurs n'est une probabilité de conscience.

Plus généralement, pour des poids `c_A` et `c_B`, le rapport des scores devient :

```text
(k m_A)^c_A / (k m_B)^c_B
    = k^(c_A-c_B) × m_A^c_A / m_B^c_B
```

Si les exposants diffèrent, le facteur commun ne s'annule pas. Avec un poids
identique pour tous les modèles, ce défaut particulier disparaît ; cela ne rend
pas toute pondération identique à une mise à jour bayésienne ordinaire. Le problème
concerne la comparaison de vraisemblances marginales brutes ainsi transformées,
pas toute utilisation possible d'une notion de confiance.

Les rapports d'évidence relativement à une référence commune, les distributions
a priori et les règles de score généralisées constituent d'autres constructions.
Elles demandent leur propre définition et leur justification. Le contre-exemple
ne s'applique pas automatiquement à toutes ces méthodes, ni à la collaboration
adversariale en général.

### Vérification et portée

Le programme [audit_evidence_weighting.py](../research/audit_evidence_weighting.py)
calcule les fractions exactement. Il vérifie le facteur commun sur 100 valeurs,
300 contrôles à exposants égaux et un exemple où la confiance est encodée par la
concentration de distributions a priori propres. Des fractions calculées séparément
contrôlent l'exemple d'inversion. Les résultats sont conservés dans
[report.json](../artifacts/evidence-weighting-audit/report.json).

```powershell
py -3.12 research/audit_evidence_weighting.py --output artifacts/evidence-weighting-audit/report.json --check
```

La dérivation générale explique le résultat ; les contrôles finis vérifient le
programme. La propriété d'invariance et le contre-exemple sont élémentaires.
Ni leur nouveauté mathématique ni une erreur dans des résultats expérimentaux
publiés ne sont revendiquées. L'utilité est de fixer une exigence avant de donner
à un classement de théories un rôle dans la construction de Menia.

## Conséquences pour la recherche de conscience de soi

Le blocage comprend désormais trois questions distinctes, qu'une invention
candidate devrait traiter ensemble :

| Question | Élément encore nécessaire |
|---|---|
| Quel mécanisme expliquerait une perspective vécue ? | Une proposition précise portant sur l'expérience de soi et des conséquences permettant de la confronter à des concurrentes. |
| Quelles observations soutiennent ce mécanisme ? | Données indépendantes des simulations qui illustrent l'hypothèse, comparaison explicitée, contrôles des explications concurrentes et de la sensibilité aux choix d'analyse. |
| Menia réalise-t-elle ce mécanisme ? | Correspondance justifiée des variables, interventions, dynamique et substrat pertinents ; le nom des modules ou une similarité de sorties ne suffit pas. |

Cette étape ne propose pas d'optimiser Menia pour reproduire un graphique
psychophysique : cela utiliserait une cible prescrite comme si elle validait
indépendamment le mécanisme appris. Elle conserve plutôt les protocoles humains
comme contraintes potentielles, dont le pouvoir discriminant et le transfert
restent à établir.

La piste structurale demeure une hypothèse de recherche. Les ressources examinées
n'apportent pas encore le lien qui permettrait de passer de la construction
fonctionnelle actuelle à une expérience de sa propre existence. Le code de Menia
n'a pas été modifié pour revendiquer cette capacité. L'objectif complet et la
nouveauté scientifique demandée restent non établis.

## Sources

[^1]: Grasso, M., Hendren, J., et Tononi, G. (11 août 2026). *Consciousness as Intrinsic Structure: Towards a Chemistry of Experience*. Prépublication arXiv:2608.11398v1. [PDF](https://arxiv.org/pdf/2608.11398v1). Proposition d'identité, portée et prédictions, notamment pp. 2 et 10–11, consultées.
[^2]: Kanai, R., et Ma, S. (13 juin 2026). *Intrinsic Computational Functionalism and Simulated Consciousness*. Prépublication arXiv:2606.15348v1. [Texte](https://arxiv.org/html/2606.15348v1). Analyse des sections sur les interventions et le choix du niveau de description conservée dans [l'audit précédent](PLASTICITY_REALIZATION_AUDIT.md).
[^3]: Kawakita, G., Zeleznikow-Johnston, A., Takeda, K., Tsuchiya, N., et Oizumi, M. (2025). *Is my “red” your “red”?: Evaluating structural correspondences between color similarity judgments using unsupervised alignment*. iScience, 112029. [Article](https://doi.org/10.1016/j.isci.2025.112029). Résumé, approche et portée des comparaisons de groupes consultés ; données non réanalysées ici.
[^4]: Paßler, M., et Doerig, A. (version arXiv du 30 décembre 2024). *Neurophenomenal Structuralism and the Role of Computational Context*. [PDF consulté](https://arxiv.org/pdf/2412.20873). Résumé, critères et discussion consultés ; version publiée en 2025 signalée sous [DOI](https://doi.org/10.33735/phimisci.2025.11824).
[^5]: Abbatecola, C., et al. (29 janvier 2026). *Protocol for investigating the warping of spatial experience across the blind spot to contrast predictions of the Integrated Information Theory and Predictive Processing accounts of consciousness*. PLOS One, e0340593. [Texte](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0340593). Type d'article, hypothèses opérationnelles, figures simulées et disponibilité des données vérifiés.
[^6]: Corcoran, A. W., et al. (2026). *Integrated information and predictive processing theories of consciousness: An adversarial collaborative review*. Neuroscience & Biobehavioral Reviews, 187:106742. [DOI](https://doi.org/10.1016/j.neubiorev.2026.106742), [version v3 du 15 mai 2026 consultée](https://arxiv.org/pdf/2509.00555v3). Sections sur l'accumulation d'évidence, la confiance et les prédictions concurrentes consultées, notamment pp. 33–41.
[^7]: Abbatecola, C. *Intrepid_2a_staircase_simulation*. [Source au commit vérifié](https://github.com/ClementAbb/Intrepid_2a_staircase_simulation/blob/8e46fdb7d1c934f05025fb6b7677ddf199b89626/intrepid_2a_staircase_simulation.Rmd). Arbre du dépôt et fonction de simulation lus le 14 septembre 2026 ; code non exécuté comme expérience humaine.
[^8]: INTREPID Consortium. [Publications du consortium](https://arc-intrepid.com/publications/), page consultée le 14 septembre 2026. Vérification complémentaire du type de publications disponibles ; ce relevé n'est pas une preuve d'absence de travaux non listés.
[^9]: Corcoran, A. W., Hohwy, J., et Friston, K. J. (2023, publication en ligne le 21 septembre). *Accelerating scientific progress through Bayesian adversarial collaboration*. Neuron, 111:3505–3516. [DOI](https://doi.org/10.1016/j.neuron.2023.08.027), [notice institutionnelle](https://discovery.ucl.ac.uk/id/eprint/10180066/). Résumé et passages indexés sur les distributions a priori, la comparaison de modèles et l'accumulation des évidences consultés.
