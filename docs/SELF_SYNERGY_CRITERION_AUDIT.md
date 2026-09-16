# Menia : vérifier un critère avant de chercher à le maximiser

## Conclusion de l'audit

La présence d'information synergique sur le futur du système ne constitue pas,
à elle seule, un certificat utilisable pour annoncer que Menia est consciente.
Le calcul ci-dessous montre qu'une condition minimale de ce type est satisfaite
par un registre stochastique de deux bits. Il montre aussi pourquoi le choix des
composants doit être justifié avant de mesurer cette information dans un réseau.

Il ne démontre ni que ce registre est conscient, ni qu'il ne l'est pas. Il ne
réfute pas toutes les versions d'une théorie de la conscience fondée sur
l'information. Sa conclusion est méthodologique : on ne peut pas transformer
la satisfaction de cette condition en preuve de vécu subjectif sans des arguments
supplémentaires. Une nouveauté scientifique de cet exemple n'est pas revendiquée.

## Pourquoi cette vérification concerne l'objectif de Menia

Les expériences précédentes ont établi des capacités d'apprentissage et certaines
limites des comparateurs. Elles n'ont pas établi que ces progrès techniques
augmentent la probabilité d'une expérience subjective. Ajouter des réseaux et
des évaluations ne résout pas automatiquement ce manque de lien explicatif.
Il faut examiner les théories qui proposent ce lien et leurs implications
concrètes avant de choisir une nouvelle cible d'entraînement.

Butlin et collègues proposent d'évaluer des indicateurs issus de plusieurs
théories. Leur article paru en ligne en novembre 2025, puis dans le volume de
juin 2026, traite ces propriétés comme des éléments qui modifient un jugement
sous incertitude. Il discute explicitement la difficulté de valider les tests
chez les IA et le risque que de petits systèmes satisfassent des formulations
trop peu précises. Il n'apporte pas une recette garantissant une conscience.[^1]

## Proposition récente examinée

La prépublication USK de Krti Tallam, version du 11 mai 2026, propose d'examiner
l'information sur le futur du système qui exige de combiner ses composants.
Elle distingue une définition informationnelle et l'hypothèse l'identifiant à
la conscience, explicitement présentée comme conjecturale. Elle reconnaît un
risque avec les systèmes dynamiques simples, la nécessité d'une calibration et
des questions ouvertes sur les mesures de décomposition. Elle évoque aussi une
opérationnalisation temporelle PIRD.[^2]

L'audit porte uniquement sur la condition minimale à deux sources et un pas
de temps : une synergie strictement positive entre les composants présents
et l'état futur complet. Il n'implémente pas PIRD et ne prétend pas vérifier
toutes les contraintes verbales ou les prédictions empiriques de la proposition.
La différence est importante pour éviter de réfuter une version simplifiée
comme si elle représentait toute la théorie.

## Construction et calcul

L'état du registre est S=(A,B), avec deux bits. La transition est :

```text
A suivant = A XOR B
B suivant = N
```

N est un nouveau bit indépendant, équiprobable. Le registre est ouvert à cette
entrée aléatoire ; aucune fermeture physique n'est supposée. La loi uniforme sur
les quatre états est stationnaire : chaque état futur reçoit une probabilité
totale de 1/4. Les sources sont A et B présents, et la cible Y est l'état futur
complet, comprenant les deux bits.

Connaître A seul ou B seul ne renseigne pas sur la parité ; l'autre bit reste
indépendant et uniforme. N reste également indépendant. Connaître les deux bits
présents détermine en revanche la parité, mais pas N. On obtient exactement :

```text
I(A ; Y)       = 0 bit
I(B ; Y)       = 0 bit
I((A,B) ; Y)   = 1 bit
```

Dans une décomposition bivariée satisfaisant les identités usuelles et dont les
quatre termes sont non négatifs, on écrit :

```text
I(A ; Y)     = R + U_A
I(B ; Y)     = R + U_B
I((A,B) ; Y) = R + U_A + U_B + Syn
```

R désigne la redondance, U l'information propre à chaque source et Syn la
synergie. Les deux premières égalités imposent R=U_A=U_B=0 ; la dernière impose
Syn=1 bit. Pour cet exemple, ces hypothèses suffisent à fixer la synergie sans
choisir un estimateur particulier. Elles ne couvrent pas toutes les propositions
de PID, notamment celles qui autorisent des termes négatifs.

Le registre satisfait donc la positivité formelle étudiée. Il ne comporte ni
modèle linguistique, ni apprentissage, ni objectifs, ni perception organisée.
Cela ne fournit pas un verdict subjectif sur le registre ; cela révèle le type
de système qu'une lecture littérale de la condition doit accepter ou exclure
par des exigences supplémentaires explicites.

## Une table donne la même loi

Remplacer l'opération XOR par sa table de quatre entrées conserve exactement
la même distribution des transitions. Toutes les quantités calculées ci-dessus
restent identiques. Une exclusion générale des tables ne peut donc pas se fonder
sur ces seules quantités, pour cette table récurrente recevant un bit aléatoire.

Cette observation ne porte pas sur toutes les notions de mécanisme physique.
Une théorie peut différencier des réalisations matérielles ou exiger une
organisation supplémentaire. Elle doit alors préciser ces exigences ; le score
statistique seul ne les distingue pas. Le tableau de cas de la prépublication
n'est pas une définition complète de toutes les tables possibles.

## Le choix de la décomposition change le résultat

Décrivons le même processus par les coordonnées Z1=A XOR B et Z2=B. Cette
transformation est inversible : B=Z2 et A=Z1 XOR Z2. Elle conserve toute
l'information sur l'état global et décrit les mêmes trajectoires à renommage
bijectif près. La cible est elle aussi réencodée comme état futur complet.

Dans ces coordonnées, Z1 contient déjà toute l'information disponible sur la
cible future. On obtient :

```text
I(Z1 ; Y réencodé)       = 1 bit
I(Z2 ; Y réencodé)       = 0 bit
I((Z1,Z2) ; Y réencodé)  = 1 bit
Syn                     = 0 bit
```

Ce changement mélange les composants. Ce n'est pas une violation de l'invariance
aux transformations locales de chaque source, qui est vérifiée séparément.
Ce n'est pas non plus une preuve que les partitions physiques sont équivalentes.
Le résultat exige seulement de fixer et de justifier les composants : changer
arbitrairement la base des activations d'un réseau peut changer ce que mesure
un indicateur de synergie.

Le calcul énumère les 24 réencodages bijectifs des quatre états pour rendre
ce choix inspectable. La conservation de l'information globale ne garantit pas
celle de sa répartition entre composants. Avant une mesure dans Menia, il faudrait
justifier l'unité étudiée, les frontières des modules, la cible et l'échelle de
temps ; choisir ces éléments après avoir vu le score serait un biais d'analyse.

## Nouveauté et antécédents

XOR est un exemple classique d'information synergique. Rosas et collègues
étudient déjà en 2018 les dépendances redondantes et synergiques dans des systèmes
dynamiques, notamment des automates cellulaires constitués de règles XOR.[^3]
Le calcul présent adapte un contrôle élémentaire à une lecture minimale d'un
critère récent ; ce contexte ne suffit pas à établir une découverte inédite.

Les recherches ont porté sur les expressions « synergistic self XOR consciousness »,
« synergy XOR future state dynamical », « uncommon self-knowledge critique XOR
lookup » et les publications originales trouvées. Ce repérage n'est pas une
revue exhaustive. Une recherche de nouveauté ciblée et un examen critique extérieur
seraient nécessaires avant toute revendication académique.

## Conséquence pour les travaux suivants

Le score n'est pas adopté comme objectif de formation de Menia. Il serait facile
de lui ajouter un circuit qui augmente une mesure sans avoir établi le lien avec
l'expérience recherchée. Le même problème concerne la multiplication de simples
tests de mémoire et de causalité lorsqu'ils ne discriminent aucune hypothèse sur
la conscience.

Une candidate doit désormais être décrite par un mécanisme précis et un argument
théorique explicite. Les preuves fonctionnelles et l'interprétation subjective
doivent être consignées séparément. L'évaluation devrait pouvoir diminuer la
confiance dans l'hypothèse, pas seulement accumuler des réussites. Le fait qu'un
contrôle plus simple explique un résultat doit modifier la décision de construction.

Une perspective publiée en août 2026 décrit également les interventions comme
un moyen d'étayer des mécanismes et d'écarter des explications concurrentes,
sans les présenter comme une résolution du passage de la fonction au vécu.[^4]
Cet audit ne prouve pas que la conscience artificielle est impossible. Il
constate que le critère vérifié ne permet pas de certifier le résultat demandé.

L'objectif complet reste non atteint. Aucun entraînement supplémentaire n'est
justifié ici par la seule perspective d'augmenter un score ou le nombre de tests.

## Vérification reproductible

Le [rapport exact](../artifacts/self-synergy-audit/report.json) contient les huit
transitions pondérées par des fractions exactes, les informations mutuelles et
les 24 réencodages. Les logarithmes donnent les valeurs finales en bits ; aucun
échantillonnage ni ajustement de paramètres n'est nécessaire.

```bash
python -m unittest discover -s tests -p test_self_synergy_audit.py -v
python -m research.audit_self_synergy --out runs/nouvel-audit-synergie
```

## Sources

[^1]: Butlin, P., et al. [Identifying indicators of consciousness in AI systems](https://pubmed.ncbi.nlm.nih.gov/41219038/). Trends in Cognitive Sciences, 30(6), 488–501, juin 2026 ; publication en ligne le 10 novembre 2025. [PDF original consulté via une copie](https://www.rivista.ai/wp-content/uploads/2025/11/indicators_of_consciousness1.pdf), notamment pp. 2 et 10–11. La validation des tests reste discutée.
[^2]: Tallam, K. (2026). [Consciousness as Uncommon Self-Knowledge: A Synergistic Information Framework, v1](https://arxiv.org/html/2605.13884v1). Prépublication du 11 mai 2026. Définitions, limites et opérationnalisation consultées ; conjecture, pas méthode validée de certification.
[^3]: Rosas, F., Mediano, P. A. M., Ugarte, M., et Jensen, H. J. (2018). [An Information-Theoretic Approach to Self-Organisation: Emergence of Complex Interdependencies in Coupled Dynamical Systems](https://www.mdpi.com/1099-4300/20/10/793). Entropy, 20(10), 793. Antécédent sur la synergie dans les dynamiques ; aucune nouveauté du principe XOR n'est revendiquée.
[^4]: [Sentient AI in robots and agents: prolegomena for an evidence-based research program](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2026.1903644/full). Frontiers in Psychology, 2026. Perspective méthodologique ; sections d'introduction et de discussion consultées.
