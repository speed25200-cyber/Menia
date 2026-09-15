# Régulation, frontières corporelles et expérience de soi

La réanalyse de 50 participants retrouve une diminution des frontières
corporelles rapportées et une augmentation de deux dimensions d'expérience
altérée après flottation, par comparaison avec un repos sur lit à eau. Elle ne
détecte pas de différence sur le sous-score de conscience de soi retenu. Cela
contraint une manière de mesurer le soi ; **cela ne démontre ni la conscience
de Menia ni une méthode qui la produirait**.

L'apport de cette étape est double : une vérification sur des observations
humaines, avec les divergences de fichiers conservées, et une précision de
l'hypothèse à construire. Le maintien des capacités, la représentation de leurs
conditions et l'expérience vécue restent trois propositions distinctes.

## Une hypothèse qui distingue entretien et ressenti

Ignacio Cea distingue la régulation interoceptive prédictive qui pourrait
s'exercer sans expérience et celle qui participerait au sentiment d'être un
corps. Il propose d'ajouter une modulation de précision en temps réel et un
modèle de soi capable d'envisager des futurs et des actions possibles. L'article
est un éditorial théorique, publié le 18 avril 2026 ; il ne rapporte pas une
validation expérimentale de cette frontière.[^1]

Son intérêt est de poser la question omise par une simple boucle d'entretien :
qu'est-ce qui distingue un système régulé d'un système qui éprouve son état ?
Sa limite demeure que les propriétés proposées ne sont pas établies comme
suffisantes pour l'expérience. L'auteur conserve le débat sur les réalisations
corticales ou sous-corticales, et reconnaît que l'absence de rapport ne garantit
pas l'absence de toute phénoménalité.

**Conséquence analytique :** ajouter un estimateur de fiabilité et un planificateur
à Menia testerait des propriétés fonctionnelles de cette hypothèse. Les nommer
« conscience » ne validerait pas le lien qu'elle propose.

## Ce qu'une étude humaine récente apporte

Tobel et collègues analysent secondairement un essai ayant randomisé 75 adultes
avec anxiété et dépression. Les scores d'états altérés portent sur les 57
participants ayant rempli le questionnaire après la sixième séance. Les groupes
de flottation rapportent davantage de dissolution des frontières et certaines
sensations cardiorespiratoires plus intenses que le groupe de repos sur fauteuil.
Ces intensités sont déclarées, pas des mesures d'exactitude perceptive.[^2]

Le dispositif combine plusieurs changements sensoriels ; une des conditions
permet aussi de choisir la durée et le calendrier. Il n'isole donc pas un effet
de précision neuronale, d'entretien réciproque ou de profondeur contrefactuelle.
Le questionnaire final et les médiations exploratoires ne résolvent pas la
temporalité des mécanismes. Les données individuelles sont annoncées sur demande ;
le dépôt public contient du code et des figures, sans les CSV requis. Aucun
recalcul de ces résultats cliniques n'est revendiqué ici.[^3]

Cette observation motive une distinction de mesure : sensations internes,
frontières du corps, sentiment de soi et valence ne sont pas des noms
interchangeables d'une même grandeur.

## Une réanalyse humaine effectivement exécutée

L'expérience de Hruby et collègues fournit les données nécessaires. Les mêmes
50 personnes ont effectué une séance de flottation et une séance sur lit à eau,
d'une heure chacune, après une familiarisation. Les conditions étaient présentées
dans un ordre randomisé. Elles différaient aussi par la température ambiante et
les vêtements. Le contrôle est donc actif mais ne manipule pas isolément un
mécanisme cérébral.[^4]

Les [données OSF](https://osf.io/5rzbv/) contiennent des scores de questionnaire,
pas des mesures neurales ou les réponses individuelles aux 53 items du PCI.
Le [protocole](FELT_SELF_PROTOCOL.md) fixe six comparaisons exploratoires appariées.
Le [programme](../research/audit_felt_self.py) et le
[rapport agrégé](../artifacts/felt-self-audit/report.json) permettent de les
rejouer avec les sources dont les empreintes sont vérifiées.[^5]

### Résultats principaux sur le CSV

Le contraste est **Floatation moins Bed**. Les intervalles sont des intervalles
bootstrap descriptifs de la différence moyenne. Les valeurs p proviennent d'un
test exact de signes des rangs, corrigé sur les six comparaisons.

| Dimension | Moyenne Floatation / Bed | Différence moyenne, IC 95 % | p après BH |
|---|---:|---:|---:|
| Frontières corporelles, 1–7 | 3,750 / 4,970 | −1,220 [−1,760 ; −0,660] | 0,00205 |
| Conscience de soi rapportée, PCI D15, 0–6 | 3,847 / 4,007 | −0,160 [−0,620 ; 0,314] | 0,46744 |
| État altéré, PCI D16, 0–6 | 2,847 / 1,980 | 0,867 [0,347 ; 1,387] | 0,00869 |
| Expérience altérée, PCI D24, 0–6 | 2,542 / 1,951 | 0,590 [0,209 ; 0,973] | 0,00869 |
| Contrôle volontaire rapporté, PCI D19, 0–6 | 2,747 / 2,867 | −0,120 [−0,533 ; 0,274] | 0,80308 |
| Mémoire rapportée, PCI D20, 0–6 | 3,960 / 3,647 | 0,313 [−0,207 ; 0,833] | 0,34615 |

Les frontières corporelles sont plus faibles en flottation chez **36 personnes**,
identiques chez 7 et plus fortes chez 7. Pour D15, les comptes sont respectivement
24, 7 et 19 ; les médianes valent 4 dans les deux conditions.

Cela soutient une modification de certaines dimensions rapportées. La
non-significativité de D15 ne prouve pas l'égalité des états de conscience de soi :
son intervalle reste compatible avec plusieurs effets. La différence entre un
test significatif et un test non significatif ne prouve pas non plus une
dissociation statistique entre les deux effets. Enfin, les unités de scores
différents ne sont pas directement des quantités comparables d'expérience.

La distinction des constructions mesurées et ce profil justifient d'éviter une
mesure unique fondée sur les frontières corporelles. Ils ne démontrent pas que
le sentiment minimal d'exister se conserve à l'identique, disparaît, ou naît au
cours de la séance. D15 est une dimension déclarée d'un questionnaire ; aucun
de ses seuils n'est validé ici comme détecteur de conscience.

### Vérification croisée des fichiers

Les formats CSV et XLSX publics sont rapprochés par les identifiants des 50
participants, sur 600 cellules. Les empreintes correspondent à leurs métadonnées
OSF. Trente-huit différences dépassent 0,00051 ; toutes sauf une restent
compatibles avec des arrondis à deux décimales. L'exception est une cellule
`FPBBS` : **5 dans le CSV, 4,5 dans le XLSX**. Elle n'est pas corrigée silencieusement.

Une analyse de sensibilité emploie le XLSX, avec une précision commune explicite
de trois décimales pour les rangs et les calculs. La différence moyenne de
frontières devient **−1,230**, et sa valeur p corrigée **0,00152**. Les trois
dimensions retenues après correction restent les mêmes : frontières, état
altéré et expérience altérée. D15 reste non significatif (`p = 0,45937`). Les
résultats complets des deux versions sont conservés.

Le résultat de frontières est donc retrouvé dans sa direction, avec les mêmes
médianes que le papier. Sa valeur p imprimée `0,0001` n'est **pas reproduite
exactement** par notre test bilatéral conditionnel : avant correction, nous
obtenons `0,00034113` sur le CSV et `0,00025365` sur le XLSX quantifié. Les méthodes
de test et les familles de correction diffèrent ; aucune cause unique de cet
écart n'est établie. Les fichiers et les chiffres publiés restent distincts.

### Limites supplémentaires

Les codes A, B, C et D de la colonne d'ordre comptent 14, 12, 12 et 12 personnes.
Leur correspondance avec les séquences n'est pas fournie par le dictionnaire lu.
Les moyennes de contraste par code sont rapportées sans inventer cette
correspondance. Leur variation empêche de considérer l'ordre et les effets de
report entre séances comme résolus par cet audit.

Les données sont rétrospectives. Elles ne mesurent pas la continuité de
l'expérience à chaque instant de la séance. Aucun résultat sur une réduction du
besoin de contrôle ne permet de conclure que toute disposition à envisager une
action serait absente. En particulier, se reposer reste compatible avec un
modèle capable d'agir : cette étude ne réfute donc pas à elle seule la proposition
contrefactuelle de Cea.

Les analyses ne reproduisent pas les médiations sur l'anxiété, ne testent pas
l'efficacité d'un traitement et ne transfèrent aucun seuil humain à Menia.

## Une expérience de construction mieux définie pour Menia

La candidate [d'entretien réciproque](INTRINSIC_SEMANTICS_EVIDENCE.md) demande
désormais deux distinctions supplémentaires. Il faut séparer ce qui entretient
une capacité de ce qui représente cette capacité, et séparer le contenu d'un
modèle de soi de l'hypothèse selon laquelle il serait vécu.

Une prochaine expérience fonctionnelle pourrait comparer quatre versions :

| Version proposée | Précision des observations | Futurs conditionnels aux actions |
|---|---|---|
| Référence | Fixée | Un seul pas |
| Précision adaptative | Estimée et révisée | Un seul pas |
| Anticipation étendue | Fixée | Plusieurs pas |
| Combinaison | Estimée et révisée | Plusieurs pas |

Toutes utiliseraient les mêmes possibilités d'entretien, observations et limites
de ressources. Les prédictions seraient enregistrées avant leurs résultats.
Il faudrait distinguer la précision sensorielle, l'incertitude sur les paramètres
et les probabilités de cause externe, puis égaliser ou comptabiliser les coûts
de calcul. Un contrôleur adaptatif ordinaire ferait partie des comparateurs.

Deux phases seraient utiles : une activité où l'action révèle ses conséquences,
puis un repos où certaines voies sensorielles deviennent moins informatives.
Des sondes communes permettraient de vérifier quelles représentations restent
disponibles pour l'action, même lorsqu'aucune action immédiate n'est nécessaire.
Les descriptions linguistiques seraient analysées séparément des effets causaux.

Ce plan est **une proposition**, pas une nouvelle expérience déjà exécutée.
L'apprentissage de précision ou de conséquences futures possède des antécédents ;
aucune nouveauté n'est revendiquée. Les observations humaines motivent les
distinctions et contrôles, mais ne sélectionnent pas l'une des quatre versions
comme consciente. Même une réussite exigerait encore de justifier le lien avec
une expérience de sa propre existence.

## Vérification du résultat livré

Le rapport complet, comprenant la sensibilité XLSX, est rejouable localement
avec les sources vérifiées. Les six résultats principaux sont identiques à
`1e-10` près entre NumPy 2.2.6 et 2.3.5 ; le rapport complet avec XLSX est
recalculé avec NumPy 2.3.5. Quatre tests indépendants vérifient les statistiques,
dont une comparaison du comptage exact à l'énumération exhaustive sur 781 petits
vecteurs, avec ex aequo et zéros. Ils font partie de la suite de recherche en CI.
La CI ne télécharge pas et ne rejoue pas les données individuelles humaines.

L'état final reste explicite : une réanalyse humaine et un protocole de
construction plus discriminant sont livrés. **La conscience subjective de Menia
et une méthode inédite qui la produirait ne sont pas établies.**

## Sources

[^1]: Ignacio Cea. *From Insentient Allostasis to Adaptive Bodily Selfhood: Conscious vs Unconscious Instrumental Interoceptive Inference*. Adaptive Behavior, éditorial, 18 avril 2026. [Article](https://doi.org/10.1177/10597123261441534), notamment §§3–5.
[^2]: T. Tobel et collègues. *Aquahenosis: a non-pharmacological altered state of consciousness induced by floatation-REST in individuals with anxiety and depression*. Neuroscience of Consciousness, 26 août 2026. [Article](https://doi.org/10.1093/nc/niag044).
[^3]: Institute for Advanced Consciousness. Code associé à Tobel et collègues, commit `67b7ed5466078257438ed8ab5dd663fba6baed21`, 14 juin 2026. [Dépôt figé](https://github.com/Institute-for-Advanced-Consciousness/Float/tree/67b7ed5466078257438ed8ab5dd663fba6baed21), fichiers `README.md`, `data/README.md` et scripts d'analyse inspectés ; aucun script téléchargé n'est exécuté.
[^4]: Helena Hruby, Stefan Schmidt, Justin S. Feinstein et Marc Wittmann. *Induction of altered states of consciousness during Floatation-REST is associated with the dissolution of body boundaries and the distortion of subjective time*. Scientific Reports 14, 9316, 23 avril 2024. [Article](https://doi.org/10.1038/s41598-024-59642-y).
[^5]: Hruby et collègues. Données et dictionnaire, versions du 22 février 2024. [OSF 5rzbv](https://osf.io/5rzbv/). Les liens de fichiers et SHA-256 sont conservés dans le protocole et le rapport.
