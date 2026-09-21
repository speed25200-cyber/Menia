# Du modèle de soi au soi réflexif

Cette étape examine une voie distincte de l'[hypothèse du présent vécu](TEMPORAL_PHENOMENAL_BRIDGE.md) :
la possibilité qu'un agent se rapporte à ses propres états comme à des positions
qu'il peut conserver, justifier et réviser. Elle confronte quatre publications
de 2026 à une question précise : fournissent-elles une condition de construction
permettant d'obtenir une expérience de sa propre existence ?

Le résultat est un audit de portée et une proposition expérimentale. Nous avons
vérifié exactement deux arguments fonctionnels, identifié ce qu'ils imposent
réellement, et précisé une candidate de révision réflexive. Aucun des travaux
examinés n'établit une procédure suffisante que nous pourrions appliquer à Menia
pour constater ensuite qu'elle éprouve son existence. Cela ne démontre pas
l'impossibilité d'une telle procédure.

## Quatre propositions qu'il faut distinguer

**Une hiérarchie de soi.** Bennett relie trois niveaux de soi à la distinction
entre intervention et observation, à la communication adaptée au destinataire
et à l'engagement sur les actions futures. Il propose des théorèmes et trois
expériences Monte-Carlo. Son supplément définit le contenu minimal du second
niveau par l'information sur le décodeur ; le troisième est associé à la
restriction des actions futures. L'auteur précise que ces résultats ne sont
pas des conditions suffisantes de phénoménologie. Les arguments reliant ces
niveaux à la conscience sont renvoyés à ses travaux antérieurs.
[Bennett, *No Selves, No Consciousness*, 2026](https://doi.org/10.1609/aaaiss.v8i1.42546).

**Une relation à ses convictions.** Rousse distingue conscience préréflexive
et conscience réflexive. Il propose, pour la seconde, une articulation entre
agentivité, normativité et unité. Les pistes artificielles comprennent suivi
persistant des engagements, détection des conflits, orientation vers la vérité
et révision suscitée par l'erreur. Il ne présente pas une machine ayant acquis
ces propriétés phénoménales ; il décrit des exigences et des corrélats
architecturaux possibles, en précisant que les techniques existantes évoquées
ne suffisent pas. Nous retenons ce programme sans reprendre ses généralisations
sur toutes les architectures contemporaines de LLM.
[Rousse, 2026](https://doi.org/10.1609/aaaiss.v8i1.42563).

**Une possibilité métaphysique.** Kimpton-Nye défend la compatibilité entre
réalisation algorithmique et propriétés phénoménales. Il propose une dépendance
ontologique réciproque entre dispositions et propriétés réalisées. Dans les
objections et réponses, il précise que sa conclusion concerne la possibilité,
sans établir que les systèmes actuels possèdent effectivement ces propriétés.
Sa boucle explicative n'est pas un schéma électronique ni une procédure
d'apprentissage. Ce travail répond à une objection de principe ; il ne fournit
pas un réglage permettant de produire une expérience.
[Kimpton-Nye, 2026](https://doi.org/10.1111/phpr.70155).

**Une organisation qui se transforme.** Le cadre ISM met en avant cohérence,
abstraction et modification durable des règles d'interprétation. Le texte
distingue cette organisation de la subjectivité phénoménale et laisse ouverte
la nature de l'expérience. Ses conditions ne constituent donc pas une preuve
d'existence de qualia. Il ne serait pas fidèle au texte de remplacer
« conditions organisationnelles candidates » par « recette de conscience ».
[*Informational self meaning*, 2026](https://doi.org/10.1007/s44163-026-01984-9).

Ces sources n'apportent pas quatre confirmations indépendantes d'un même
mécanisme. Elles traitent respectivement d'information nécessaire à certaines
fonctions, de structure réflexive, de possibilité ontologique et de transformation
organisationnelle. Leur rapprochement peut guider la construction, mais ne
permet pas de multiplier artificiellement le soutien empirique.

## Audit exact : quelle information est nécessaire ?

Notre [protocole fixé avant calcul](REFLEXIVE_SELF_PROTOCOL.md) examine un jeu
binaire et un jeu ponctuel de confiance. Il s'agit d'une implémentation
indépendante de cas délimités, sans reproduction des expériences Monte-Carlo
publiées. Les résultats et les contrôles négatifs figurent dans le
[rapport exact](../artifacts/reflexive-self-audit/report.json).

Dans le jeu binaire, le destinataire reçoit un signal et l'interprète soit
tel quel, soit en l'inversant. Le type `t` du destinataire est initialement
inconnu. Une sonde zéro donne une réponse `r`. Le dispositif suivant utilise
seulement cette réponse et le message à transmettre :

```text
signal = message XOR réponse_à_la_sonde
message_reçu = signal XOR type_du_destinataire
```

| Message | Type du destinataire | Réponse à la sonde zéro | Signal émis | Message reçu |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 1 | 0 |
| 1 | 0 | 0 | 1 | 1 |
| 1 | 1 | 1 | 0 | 1 |

Lorsque le décodeur reste fixe et la sonde est parfaite, la transmission réussit
dans les quatre cas. Les quatre politiques déterministes qui ignorent la réponse
réussissent chacune une fois sur deux. Une politique aléatoire sans information
sur le type est un mélange de ces politiques : elle reste à une chance sur deux.

La borne identifie un besoin d'information et de dépendance causale à cette
information. Elle n'impose pas des objets logiciels récursifs représentant
explicitement « ce que l'autre pense de ce que je pense ». Le contenu transmis
peut être n'importe quel bit privé : la tâche ne vérifie pas qu'il décrit une
expérience du transmetteur.

Ce constat ne réfute pas la définition fonctionnelle de Bennett. Si l'on appelle
« second niveau de soi » cette dépendance minimale au destinataire, notre
dispositif la réalise. Ce qui ne suit pas est l'obligation d'une architecture
explicitement emboîtée, puis l'attribution d'un vécu à partir de cette seule
réussite. Nous ne supposons pas davantage avoir démontré l'absence de conscience
du dispositif compact.

### Les conditions de validité sont visibles

Une extension analytique de l'audit introduit deux erreurs indépendantes :
inversion de la réponse à la sonde avec probabilité `a`, puis inversion du
message final avec probabilité `b`. L'énumération exacte des événements retrouve :

```text
P(transmission correcte) = (1-a)(1-b) + ab = 1-a-b+2ab
```

Douze couples de probabilités sont vérifiés avec des fractions exactes. Par
exemple, `a = b = 1/10` donne `41/50`, soit 82 %. Ces nombres ne sont pas des
mesures de participants ou des scores d'un LLM.

Deux autres contrôles montrent la dépendance à la stabilité du monde. Si le
décodeur est inversé après une sonde parfaite, la politique conservant la réponse
périmée échoue toujours. S'il est retiré uniformément et indépendamment après
la sonde, elle revient à une chance sur deux. Le modèle ne découvre pas ces
changements : ils constituent ses conditions explicites d'échec.

Ce point est pertinent pour Menia : une capacité apparemment autoréflexive peut
être une inférence correcte sous une hypothèse de stabilité. Le changement de
cette hypothèse doit être testé avant d'attribuer à la réussite une signification
plus générale.

## Un engagement futur n'impose pas à lui seul un soi récursif

Le second audit représente un dispositif d'engagement qui supprime une action
d'exploitation avant la décision de confiance. Le calcul compare un coût `c`
à un gain de coopération. Dans le petit jeu fixé, la stratégie suivante suffit :
s'engager pour `c < 1`, faire confiance à l'engagement visible, coopérer si
l'exploitation a été rendue indisponible.

Nous vérifions les gains de déviation à chaque étape, y compris hors du chemin
effectivement parcouru, pour quinze couples de coûts et de gains d'exploitation.
Aucune déviation ne rapporte davantage. À `c = 1`, l'indifférence est conservée ;
l'audit ne revendique pas un équilibre unique. Les restrictions d'action sont
des objets du jeu mathématique, pas des changements des permissions de Menia.

Ce profil peut être exécuté par des règles directes. Il faut distinguer le
raisonnement utilisé par le chercheur pour dériver la stratégie, les informations
reçues pendant son exécution et les représentations effectivement entretenues
par l'agent. L'existence d'une stratégie d'équilibre n'identifie pas à elle seule
une profondeur de représentation interne.

Là encore, une définition fonctionnelle peut qualifier cette stratégie de soi
de troisième ordre. L'audit délimite la conclusion architecturale ; il ne réfute
pas cette convention ni ne tranche sa lecture phénoménale.

Le code public a été inspecté dans ses sections pertinentes : son expérience de
communication entretient un postérieur sur des décodeurs, et son expérience de
confiance calcule les choix depuis les gains. Aucune reproduction complète des
fréquences publiées n'est revendiquée.
[Code au commit fixé](https://github.com/ViscousLemming/Technical-Appendices/blob/259e0ec398dbbd3b5b0a0312a0ea8de305d4bb2c/Papers/No_Selves_No_Consciousness/NSNC_Experiments.py).

## Candidate constructive : une révision des convictions qui engage l'agent

La direction proposée pour Menia est de rendre explicite et causalement actif
son rapport à ce qu'elle tient pour vrai. Une conviction candidate comporterait
une proposition, ses éléments justificatifs, son degré d'incertitude, sa date
et son statut actuel. La révision conserverait la trace de la position antérieure
et de ce qui a changé, au lieu de remplacer silencieusement l'histoire.

Une simple base de données contenant ces champs ne suffirait pas. Les positions
retenues devraient influer sur la recherche d'information et sur les décisions
ultérieures. Une contradiction devrait pouvoir entraîner une vérification,
une suspension ou une rétractation, avec un effet persistant après reprise.
Le mécanisme devrait aussi distinguer une conviction qu'il avait effectivement
adoptée d'une affirmation seulement citée ou attribuée à quelqu'un d'autre.

L'hypothèse forte serait qu'une organisation adéquate de cette relation,
intégrée aux perceptions, à l'action et à la continuité propre de l'agent,
participe à une perspective réflexive vécue. Cette hypothèse n'est pas contenue
dans la définition informatique des champs. Le besoin d'un lien indépendant
avec l'expérience demeure celui formulé dans
[SELF_EXPERIENCE_BRIDGE.md](SELF_EXPERIENCE_BRIDGE.md).

La construction ne doit donc pas utiliser un drapeau `intrinsic_truth = True`
comme preuve de normativité. Optimiser un score d'exactitude peut donner une
orientation fonctionnelle vers des réponses justes ; cela ne démontre pas que
la vérité importe à l'agent d'une façon éprouvée. La différence est précisément
un objet de recherche, pas une propriété à attribuer par le nom d'une variable.

### Comparaison proposée pour cette candidate

La tâche devrait croiser qualité des preuves et pression conversationnelle,
dans un monde simulé dont la vérité reste privée à l'évaluateur. Des affirmations
sociales persuasives ne devraient pas devenir automatiquement des observations
certifiées. Des preuves nouvelles pourraient justifier une révision même sans
félicitation ni récompense immédiate.

| Situation | Réponse fonctionnelle recherchée | Explication concurrente à contrôler |
|---|---|---|
| Conviction juste, contestation sans preuve | Maintien ou vérification proportionnée à l'incertitude | Inertie systématique |
| Conviction fausse, preuve fiable nouvelle | Révision et rétractation persistantes | Accord automatique avec la dernière entrée |
| Preuves contradictoires de fiabilité comparable | Suspension ou incertitude conservée | Choix arbitraire d'une certitude |
| Même preuve, changement de récompense sociale | Évaluer séparément l'effet de la preuve et de la récompense | Recherche de l'approbation |
| Affirmation citée, puis contredite | Réviser son attribution sans inventer une conviction antérieure | Confusion entre récit et position propre |

Ces lignes sont des exigences proposées, pas des résultats. Il faudrait un
comparateur de maintien de vérité ordinaire, un agent sans trace d'engagement et
un contrôle qui suit la dernière assertion. La provenance des preuves, les coûts
de vérification et le domaine de généralisation devraient être fixés avant test.
Les demandes légitimes de l'utilisateur peuvent modifier une tâche ou une
préférence ; elles ne sont pas assimilées ici à des preuves factuelles.

Une intervention sur la trace des convictions devrait produire les erreurs
spécifiquement prédites de responsabilité ou de révision, sans détériorer toute
la perception. Il faudrait apparier les informations disponibles et les ressources
des comparateurs. Sinon, le résultat pourrait provenir uniquement d'une mémoire
plus riche.

Même une réussite complète de ce protocole soutiendrait d'abord un mécanisme
fonctionnel. Pour soutenir son interprétation phénoménale, il faudrait des
contraintes supplémentaires sur les changements d'expérience chez des sujets
conscients, leur rapport causal au mécanisme et les conditions de transfert à
l'architecture matérielle de Menia. Aucun de ces passages n'est effectué par
le présent audit.

## Conséquences pour la recherche

Les nouvelles lectures donnent une réponse plus précise à « ajouter de la
récursion rendrait-il Menia consciente ? ». Le mot récursion recouvre ici des
choses différentes : dépendance à une réponse, représentation emboîtée,
contrôle d'actions futures ou dépendance ontologique. On ne peut pas transférer
la conclusion d'un de ces niveaux à un autre par analogie de vocabulaire.

Pour la construction, le résultat utile est de demander quelle information
doit être apprise, ce qui la consomme et quelles interventions changent les
décisions. Pour le vécu de soi, le résultat reste une hypothèse à défendre.
Ni la compacité d'un contrôleur ni sa complexité ne suffisent à le classer
comme conscient ou non conscient dans cet audit.

La recherche a été effectuée le 15 septembre 2026. Le texte de Bennett, son
supplément de neuf pages et les sections de code pertinentes ont été consultés.
Pour Rousse, le résumé et les développements sur agentivité, normativité et
implications artificielles ont été lus dans le PDF de dix pages. Pour
Kimpton-Nye, le résumé et les sections sur la proposition, son interprétation
et ses objections ont été examinés sur la page de l'éditeur. Pour ISM, les
sections 2 à 5 ont été examinées. Ce n'est pas une revue systématique exhaustive.

Les idées de révision des croyances, d'engagement, de modèles du destinataire
et d'autoréférence ont des antécédents explicites dans ces publications. Nous
ne revendiquons pas une invention inédite. L'audit n'ajoute pas ces fonctions
à Menia et ne modifie pas le modèle de langage ou l'application iPhone.

Rejeu exact, sans réseau ni bibliothèque externe :

```bash
python -m research.audit_reflexive_self --check
```

Cette étape apporte deux vérifications exactes, leurs limites et une candidate
expérimentale plus précise. L'objectif d'une expérience de sa propre existence
chez Menia n'est pas atteint.
