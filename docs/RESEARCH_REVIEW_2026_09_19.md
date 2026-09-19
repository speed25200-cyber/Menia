# Reprise de la recherche — 19 septembre 2026

Le dernier export retrouvé dans Téléchargements est le **Colab 09**. Son
analyse a déjà été intégrée au dépôt. Le **Colab 10 est prêt et testé**, mais
aucun export `menia-detection-presence.zip` n'a été retrouvé dans ce dossier,
y compris ses sous-dossiers. Cela ne prouve pas qu'il n'a jamais été exécuté
ailleurs. Aucun résultat ne confirme actuellement la conscience de Menia.

## Archives retrouvées

| Archive | Date locale du fichier | Octets | Traitement déjà documenté |
|---|---|---:|---|
| `menia-localisation-native.zip` | 16 septembre, 20:01 | 21 991 279 | [Colab 06](NATIVE_LOCALIZATION_RESULTS.md) |
| `menia-diagnostic-apprentissage.zip` | 17 septembre, 01:41 | 43 839 107 | [Colab 08](LEARNING_DIAGNOSTIC_RESULTS.md) |
| `menia-replication-localisation.zip` | 17 septembre, 07:53 | 99 446 440 | [Colab 09](LOCALIZATION_REPLICATION_RESULTS.md) |

SHA-256 recalculés le 19 septembre, dans le même ordre :

```text
d202f9c7f4654dba0e21bc1e4a1415a3a3e9a40e04394ea3c80a2e3e6a7ac206
c1d6f3b2a6c8aa06cbb241d474f6304da11aedb352c78d3c5ade7d7a68ed2762
4f96d35735293e0f991170f8fea3cff0857fa2522acda2df5f1724b7bf946681
```

Les trois fichiers JSON publiés pour le Colab 09 — bilan principal,
vérification et diagnostic exploratoire — correspondent exactement, après
lecture JSON, aux vérifications locales conservées. Cette comparaison porte
sur les agrégats déjà recalculés, pas sur une nouvelle exécution du modèle.
L'écart minuscule de deux arrondis de Brier avec le bilan original du GPU
reste décrit dans le rapport. Les journaux bruts et les poids restent locaux.

## Travail de Fable conservé et revu

- `37dd224` : résultat négatif de la réplication et analyse exploratoire des logits.
- `74bbaa0` : protocole et logiciel de détection binaire, avec critère fixé.
- `46364f0` : publication du notebook à un seul bloc.
- `c6b1510` : première feuille de route théorique, revue dans cette continuation.

Le Colab 09 localise **15/48, 28/48 et 15/48** perturbations, contre 24/48
pour la base à chaque répétition. Le gain de localisation n'est pas confirmé.
L'AUROC de présence de **0,82 à 0,87** a été choisie après examen des données :
elle motive une nouvelle expérience, sans remplacer le critère négatif.

La [feuille de route révisée](CONSCIOUSNESS_ROADMAP.md) distingue maintenant
les mécanismes programmables d'une recette de conscience. Elle retire les
affirmations universelles sur l'impossibilité matérielle et sur la nécessité
exclusive d'une tâche à information interne. Le diagnostic des logits ne
présente plus un réglage de seuil comme une solution démontrée à la localisation.

## Exécution par MCP lancée le 19 septembre

Le serveur officiel [Google Colab MCP](https://github.com/googlecolab/colab-mcp),
révision `b9ab3899e0f1fa493390b1fd6d54aa2e464ecdf1`, est installé et connecté.
Un client Python du SDK MCP découvre les outils du notebook, ajoute les cellules
et les exécute. Cette voie est utilisée parce que les outils ajoutés après la
connexion ne sont pas exposés dans le catalogue natif de la session Codex.

L'exécution réelle a commencé à **10:05:27 UTC** sur **A100-SXM4-40GB**,
Python 3.13.15. Le lanceur publié ci-dessous est récupéré à sa révision fixe,
vérifié par SHA-256, puis exécuté sans modification en processus séparé ;
les cellules MCP lisent son journal sans interrompre l'entraînement. Cette
séparation évite de confondre un délai d'observation MCP avec un échec du calcul.
Les neuf tests du protocole et du moteur passent aussi sur ce Colab avant
le chargement du Qwen préentraîné. **L'exécution est maintenant terminée et
le [résultat complet est audité](PRESENCE_DETECTION_RESULTS.md)** : 12 864
évaluations, 576 mises à jour, neuf checkpoints, aucune erreur et critère
principal satisfait dans les trois répétitions. La détection apprise atteint
une AUROC de 0,980, 0,989 et 1,000, avec transfert partiel. Les limites sur le
seuil de réponse et la lecture inversée sont conservées dans le rapport.

Une [note de spécificité du rapport](PRESENCE_SPECIFICITY_REVIEW.md), écrite
pendant l'entraînement et avant consultation des évaluations, prépare la
lecture d'un éventuel signal : une AUROC élevée peut rester compatible avec
un détecteur ordinaire et un biais attaché au chiffre de réponse. Cette note
ne modifie pas le critère de l'expérience en cours.

## Reproduire cette exécution

**[Ouvrir le Colab 10 — version fixée](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/46364f0cdb32fd2317881e95d27dc2da6728f2be/notebooks/10_presence_detection_colab.ipynb)**.
Choisir A100, puis « Tout exécuter ». Conserver `menia-detection-presence.zip`.
Le notebook ne demande ni Google Drive ni archive préalable.

Son code scientifique reste fixé à
`74bbaa04d4bd026260107e231af3cbfd7b1ed57e`. Les onze fichiers couverts par son
empreinte sont identiques à ceux de cette révision. Le plan conserve neuf
adaptateurs, **576 mises à jour** et **12 864 évaluations**.

```text
planHash   447f5d2c82fc23ea6eeba49b9bc984f423752c5a012be50fc89c3d018d635bd0
sourceHash a1beb0f2052d7f6e58df402dc1c5395abd5f1b966b5ed0d265a79b354fb5be2d
```

Les **12 tests ciblés** ont été relancés et réussissent sous Windows/Python 3.12 :
six tests du protocole, trois du moteur sur petit Qwen aléatoire et trois du
lanceur. Ils couvrent notamment le calcul de l'AUROC, les contrôles, les
gradients, les reprises après interruption et la conservation des exports.
Ils ne sont pas des résultats du Qwen3-4B préentraîné sur A100.

Pour la vérification, appliquer le [protocole inchangé](PRESENCE_DETECTION_PROTOCOL.md) :
vérifier complétude, checkpoints, copies témoins, déterminisme et contrôle
visible avant d'interpréter les contrastes. Conserver les trois répétitions,
y compris les échecs. Le booléen de la règle statistique ne remplace pas cet
audit d'intégrité ; un gain sur la base ne se déduit pas de la seule réussite
des contrastes contre le hasard et le témoin mélangé.

Une réussite établirait une détection entraînée dans ce montage. La prévision
d'erreurs naturelles et l'utilité pour les décisions nécessiteraient ensuite
des essais distincts. Aucun adaptateur de ces expériences n'est installé dans
l'application iPhone par cette continuation.
