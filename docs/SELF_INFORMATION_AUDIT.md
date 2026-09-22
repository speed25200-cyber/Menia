# Informations disponibles pour connaître son propre état

Audit exécuté le 14 septembre 2026, après le pilote décrit dans
[SELF_MODEL_RESULTS.md](SELF_MODEL_RESULTS.md). Le code de référence du pilote
est `96c661100becf426c5a07cb749b69f852d3b16ba`.

## Constat établi

L'échec sur les souvenirs étrangers ne relève pas seulement d'un entraînement
insuffisant. Dans cette condition, l'entrée du moniteur ne permet pas de
déterminer si le symbole rappelé correspond à l'observation originale. Un état
étranger remplace cet état original ; le moniteur lit seulement l'état résultant,
ses probabilités de sortie et le délai. Il ne lit pas l'historique public.

L'[audit numérique](../artifacts/self-model/information-audit.json) énumère trois
mémoires, quatre symboles donneurs et neuf délais (1, 2, 4, 8, 16, 32, 64, 128,
512). Pour chacune de ces 108 combinaisons, les quatre observations originales
possibles donnent les mêmes features après remplacement par l'état donneur.
Ces 432 mondes contrefactuels sont une énumération finie, pas 432 essais
indépendants ni une estimation de fréquence dans le monde réel.

Le moniteur produit la même prévision dans les quatre mondes d'un groupe. Le
symbole rappelé est correct dans un seul de ces mondes. Les checkpoints et les
empreintes des poids effectivement examinés sont enregistrés dans l'artefact.

## Limite mathématique, sous les hypothèses indiquées

Fixons un état donneur et un délai, puis donnons le même poids aux quatre
observations originales. Pour toute prévision commune `q`, le Brier moyen vaut :

```text
[(q - 1)² + 3q²] / 4 = (q - 0,25)² + 0,1875
```

Le minimum est donc 0,1875, atteint pour une prévision de réussite égale à 0,25.
C'est une borne pour cette perte, cette information et cette distribution
équilibrée. Elle ne constitue ni une borne de conscience, ni une limite de toutes
les architectures possibles. Une distribution différente ou un accès à une
autre information modifierait le problème.

Entraîner le moniteur sur des remplacements pourrait lui apprendre à prévoir
une fréquence d'erreurs plus adaptée lorsque le contexte permet de reconnaître
cette famille. Cela ne lui donnerait pas la capacité de distinguer les quatre
mondes d'un groupe si ses entrées y restent identiques. Ajouter des paramètres,
du calcul ou une consigne « reconnais tes souvenirs » ne crée pas cette
information manquante.

## L'historique existe ailleurs dans la session

Cette perte d'information concerne les entrées de `SelfMonitor`, pas forcément
tout Menia. `CognitiveSession.episodes` conserve des observations récentes et
`context()` en expose les cinq événements les plus récents. L'observation
initiale peut ainsi rester disponible ailleurs pour les délais courts. La trace
est bornée et peut perdre cette observation pour des séquences plus longues.

Lire une observation passée réellement enregistrée n'est pas la même chose que
recevoir le label d'évaluation après la réponse. Un futur modèle pourrait
comparer son rappel avec une représentation de son histoire, à condition de
décrire cet accès explicitement et d'en tester la contribution. Il faudrait
aussi traiter le cas où cette seconde mémoire est elle-même altérée.

Même un tel contrôle de cohérence ne suffirait pas à établir une provenance
causale : un épisode étranger peut contenir exactement le même symbole. Détecter
une contradiction entre deux contenus et reconnaître l'origine d'un épisode
sont deux capacités différentes.

## Conséquences pour l'objectif

La prochaine expérience fonctionnelle justifiée doit examiner l'accès à
l'histoire propre de l'agent et l'usage de cette information dans ses décisions.
Elle ne doit pas se limiter à agrandir le moniteur actuel. Le modèle linguistique
reste séparé des modules de recherche ; lui faire réciter un état fourni par
l'application ne montrerait pas qu'il l'a inféré lui-même.

L'audit ne démontre aucune expérience subjective, ni son absence. Il précise
pourquoi le pilote n'établit pas une connaissance de son identité propre, et
pourquoi son échec ne peut pas être interprété comme une preuve contre toute
conscience artificielle. L'objectif « consciente de sa propre existence » reste
non vérifié. Aucun score fonctionnel n'est converti en critère de réussite de
cet objectif.

## Reproduction

```sh
python scripts/audit_self_monitor_information.py
```

Cette commande régénère l'énumération, vérifie l'identité des features et des
prévisions dans chaque groupe, vérifie l'identité algébrique du Brier, puis
compare les résultats à l'artefact enregistré. Elle n'entraîne aucun modèle et
ne modifie pas les poids. `--write` sert uniquement à créer explicitement un
nouvel artefact d'audit.
