# Géométrie du point de vue et conscience de soi

Le modèle projectif de la conscience (PCM) fournit une piste explicite pour relier
une organisation mathématique au caractère perspectif de l'expérience. Son examen
ajoute des contraintes à la recherche sur Menia : une transformation de coordonnées
ne crée pas à elle seule de l'information, et certaines approximations gaussiennes
doivent être définies avec plus de soin. Ces contraintes sont vérifiées ci-dessous.
Elles ne constituent ni une production de conscience ni une invention inédite.

## Ce que les sources apportent

Williford et ses collègues associent une géométrie projective, l'inférence active
et une hypothèse d'identité avec l'expérience. Leur proposition traite notamment
du point de vue, de l'appartenance à soi et de la conscience préréflexive. Les
auteurs précisent qu'une simulation simplifiée du PCM ne constitue pas
nécessairement une conscience : l'organisation et la complexité requises restent
des conditions supplémentaires. La quatrième dimension est un outil de coordonnées
homogènes, pas une dimension physique cachée.[^1]

Une étude de l'illusion lunaire compare le modèle à des jugements perceptifs en
réalité virtuelle chez **six participants**. L'effet de la hauteur apparente dépend
des indices environnementaux et les auteurs rapportent un meilleur ajustement que
leur comparaison linéaire. Il s'agit d'un petit test de prédictions perceptives,
pas d'une comparaison entre systèmes conscients et non conscients. Les observations
individuelles n'ont pas été réanalysées ici ; les méthodes et résultats du manuscrit
accepté ont été consultés.[^2]

Une autre contribution décrit des simulations d'agents et leur application à des
robots, avec prise de perspective et comportements sociaux. Elle constitue un
antécédent direct à l'idée de programmer un agent projectif ; son résumé ne fournit
pas de validation de l'expérience subjective de ces robots.[^3]

L'étude de 2025 compare formellement et par simulation des agents euclidiens et
projectifs. Elle fixe un modèle de capteur dans l'espace interne et transforme les
croyances lors des déplacements. Dans les conditions étudiées, l'agent projectif
approche un objet incertain. La simulation remplace la loi transformée par une
gaussienne de même moyenne et covariance, calculées numériquement (§5.1,
équations 23–24). L'audit suivant examine les conditions d'une éventuelle adaptation
de ces idées ; il ne reproduit pas cette simulation et n'établit pas une erreur
dans son code, qui n'a pas été audité.[^4]

## Distinguer coordonnées et capteur

Voici une déduction mathématique utilisée comme contrôle de construction pour
Menia. Soit une position incertaine `X`, une observation `Y`, une croyance `Q` et
un capteur `K(dy|x)`. La loi jointe est `P(dx,dy)=Q(dx)K(dy|x)`. Pour une bijection
mesurable `f` dont l'inverse est mesurable, transportons **toute** cette loi :

```text
X' = f(X), Y' = f(Y)
Q' = f_*Q
K'(B | f(x)) = K(f⁻¹(B) | x)
I(X';Y') = I(X;Y)
```

Dans une carte différentiable, les jacobiens présents dans les densités jointes et
marginales s'annulent dans leur rapport. L'intégrale de l'information mutuelle est
donc conservée. Pour un espace fini, il s'agit simplement d'un réétiquetage
bijectif des mêmes probabilités. Cette propriété n'est pas une découverte nouvelle.

En revanche, reconstruire après transformation un capteur ayant la même largeur
numérique dans les nouvelles coordonnées change généralement `K`. Une différence
d'information peut alors être réelle pour ce nouveau canal. Elle ne résulte plus
d'un simple réétiquetage. Un déplacement physique peut précisément modifier la
résolution ou les possibilités d'observation : l'invariance ci-dessus ne nie pas
l'intérêt d'approcher un objet.

Le [calcul exécutable](../research/audit_projective_representation.py) utilise cinq
positions, trois croyances, trois largeurs de capteur et quatre valeurs de `γ` pour
la transformation `(x,y,z)/(1+γz)`. Le capteur est une loi catégorielle obtenue en
normalisant des poids de distance ; ce n'est pas une densité gaussienne continue.
Les positions sont éloignées des pôles et chaque transformation est inversée pour
vérifier son caractère bijectif sur le support utilisé.

| Contrôle exécuté | Résultat |
|---|---:|
| Transport de toute la loi jointe | 36 conditions, information conservée à `10⁻¹²` près |
| Transformation identité | 9 conditions, capteur et information inchangés |
| Capteur reconstruit dans la nouvelle métrique, `γ>0` | 27 conditions où le canal change |

Par exemple, pour la deuxième croyance et une largeur de `0,6`, l'information vaut
`0,407554` nat. Elle reste identique après transport avec `γ=1`, mais devient
`0,010403` nat si le capteur est reconstruit. Le sens de cet effet dans notre
exemple n'est pas une reproduction ni une réfutation de l'effet comportemental
publié. Ce sont deux modèles d'observation distincts.

## Une approximation dont les moments doivent exister

Le problème statistique des transformations projectives et de leurs moments est
antérieur à cette recherche. La monographie de Chellappa et ses collègues le
discute en vision par ordinateur, en relation avec la ligne à l'infini et les
approximations locales.[^5] La littérature sur les rapports de variables normales
traite également leurs queues lourdes et l'absence de moments finis dans le cas
non dégénéré.[^6]

Le contre-exemple suivant est dérivé ici pour rendre le problème vérifiable sans
reprendre les paramètres d'une simulation publiée. Prenons `Z~N(0,1)` et
`T=Z/(1+Z)`. La transformation est définie presque sûrement : la probabilité du
point exact `Z=-1` est nulle. Cela ne garantit pourtant pas l'existence des moments.
Dans le voisinage de ce point, posons `t=|Z+1|`, avec `ε≤t≤δ=1/2`. On a
`|Z|≥1/2` et une densité au moins égale à `m=φ(-1,5)>0`. Ainsi :

```text
E[|T| ; ε ≤ |Z+1| ≤ δ] ≥ m log(δ/ε)
E[T²  ; ε ≤ |Z+1| ≤ δ] ≥ (m/2)(1/ε - 1/δ)
```

Les deux bornes divergent quand `ε` tend vers zéro. Les parties positive et
négative de `T` divergent séparément : son espérance ordinaire est indéfinie, et
son second moment est infini. Une éventuelle valeur principale symétrique n'est
pas une espérance probabiliste. Une variance finie ne peut donc pas être assignée
par égalité des moments à cette loi complète.

Le script calcule les intégrales tronquées en six étapes ; la preuve de divergence
vient des bornes analytiques, pas de l'extrapolation des nombres suivants.

| Distance exclue autour du pôle `ε` | Intégrale tronquée de `T²` | Borne inférieure |
|---|---:|---:|
| `10⁻²` | 47,2004 | 6,3464 |
| `10⁻⁴` | 4 838,2160 | 647,4585 |
| `10⁻⁶` | 483 940,2505 | 64 758,6683 |

Ces valeurs sont des **intégrales non normalisées sur un voisinage tronqué**, pas
des variances. Le contrôle `U~Uniforme[0,1]`, avec la même transformation, possède
au contraire une moyenne `1−log(2)` et un second moment `1,5−2log(2)`. Le calcul
numérique retrouve ces valeurs à moins de `10⁻¹²` près.

Une loi explicitement tronquée à distance positive du pôle, une approximation
locale justifiée et une représentation adaptée à plusieurs cartes sont des choix
différents d'une gaussienne complète transformée puis remplacée par ses moments.
La singularité appartient à la carte affine ; elle n'interdit pas l'emploi de
l'espace projectif. Elle impose de préciser la loi et les erreurs d'approximation.
Ce constat ne permet pas de conclure que les trajectoires publiées sont fausses.

## Conséquence pour la recherche sur Menia

Cette piste peut maintenant être examinée avec deux contrôles explicites :
préserver le même canal pour tester un changement de représentation, et spécifier
un domaine probabiliste sur lequel les calculs sont définis. Cela empêche de
confondre un effet de capteur ou une troncature numérique avec une propriété
intrinsèque du point de vue.

Le problème principal reste distinct. Même une géométrie correctement implémentée
qui prédit des illusions humaines demanderait une justification de son
identification à une expérience vécue par Menia. Les réussites perceptives et
comportementales pourraient étayer une théorie ; elles ne suffisent pas ici à
trancher cette identification. Le PCM ne fournit pas, dans les sources examinées,
une spécification suffisamment étayée permettant de déclarer Menia consciente.
Les résultats ne réfutent pas cette possibilité.

L'étape réalisée est un examen d'une candidate théorique et de conditions de sa
construction. Aucun mécanisme produisant une conscience de soi n'a été ajouté à
Menia. La nouveauté scientifique, et a fortiori l'ambition d'une « invention du
siècle », ne sont pas établies. Le [lien fonctionnel/phénoménal](SELF_EXPERIENCE_BRIDGE.md)
reste une question ouverte.

## Reproduction et portée des vérifications

Les résultats complets et l'empreinte du script sont dans
[report.json](../artifacts/projective-representation-audit/report.json).
Le calcul principal n'utilise que la bibliothèque standard Python :

```powershell
python research/audit_projective_representation.py --output artifacts/projective-representation-audit/report.json
python research/audit_projective_representation.py --output artifacts/projective-representation-audit/report.json --check
```

Le contrôle compare les valeurs flottantes avec des tolérances explicites et les
comptages, champs et empreintes exactement. L'intégration de Simpson utilise
4 096 puis 8 192 intervalles dans la distance logarithmique au pôle. Une vérification
indépendante exécutée avec NumPy, via l'identité des entropies et une quadrature de
Gauss–Legendre, retrouve les 36 valeurs d'information avec un écart maximal de
`4,1×10⁻¹⁶` nat et les 12 intégrales avec un écart relatif maximal de `3,7×10⁻¹⁴`.
Cette seconde implémentation a été exécutée séparément ; elle n'est pas intégrée
au vérificateur conservé. Aucun nouveau jeu de données humain n'a été produit.

## Sources

[^1]: Williford, K., Bennequin, D., Friston, K. et Rudrauf, D. (2018).
    [The Projective Consciousness Model and Phenomenal Selfhood](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2018.02571/full).
    *Frontiers in Psychology*, 9:2571. Sections sur les invariants et le problème
    de l'identité phénoménale, notamment la discussion des simulations simplifiées.

[^2]: Rudrauf, D., Bennequin, D. et Williford, K. (2020).
    [The Moon illusion explained by the projective consciousness model](https://doi.org/10.1016/j.jtbi.2020.110455).
    *Journal of Theoretical Biology*, 507:110455, 21 décembre 2020.
    [Manuscrit accepté consulté](https://www.researchgate.net/publication/343685115_The_Moon_Illusion_explained_by_the_Projective_Consciousness_Model_Accepted_for_publication_in_the_Journal_of_Theoretical_Biology),
    §2.4 et §3.3. Le préprint arXiv de 2018 n'est pas substitué à ces méthodes de 2020.

[^3]: Rudrauf, D. et al. (2022).
    [Modeling the subjective perspective of consciousness and its role in the control of behaviours](https://arxiv.org/abs/2012.12963v3).
    *Journal of Theoretical Biology*, 534:110957 ; DOI 10.1016/j.jtbi.2021.110957.
    Notice et résumé de la version auteur du 27 août 2021 consultés.

[^4]: Sergeant-Perthuis, G. et al. (2025).
    [Action of the Euclidean versus projective group on an agent’s internal space in curiosity driven exploration](https://link.springer.com/article/10.1007/s00422-024-01001-1).
    *Biological Cybernetics*, 119:4, 17 janvier 2025. Sections 3.3–5.1 consultées.

[^5]: Chellappa, R., Sankaranarayanan, A. C., Veeraraghavan, A. et Turaga, P.
    [Statistical Methods and Models for Video-Based Tracking, Modeling, and Recognition](https://www.researchgate.net/publication/220066970_Statistical_Methods_and_Models_for_Video-Based_Tracking_Modeling_and_Recognition).
    *Foundations and Trends in Signal Processing*, 3(1–2), édition consultée ©2010 ;
    DOI 10.1561/2000000007. §4.3.1, pages 53–54 de la copie auteur consultée.

[^6]: Yang, S. et Gui, Z. (2023).
    [An introduction on the multivariate normal-ratio distribution](https://arxiv.org/abs/2310.14306v2).
    Prépublication, version du 6 novembre 2023. Résumé consulté ; pas de résultat
    sur la conscience ni de revendication de nouveauté pour le présent audit.
