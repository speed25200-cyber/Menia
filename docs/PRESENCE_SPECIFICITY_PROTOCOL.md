# Le signal suit-il la question ? Protocole de spécificité

Protocole fixé le 19 septembre 2026, **avant les premières inférences de cette
expérience**. Il est motivé par le résultat positif du
[Colab 10](PRESENCE_DETECTION_RESULTS.md) et la
[note de contrôle](PRESENCE_SPECIFICITY_REVIEW.md). Il ne modifie pas le critère
du Colab 10 et ne constitue pas sa réplication avec de nouveaux entraînements.

**Exécution lancée le 19 septembre à 11:13 UTC**, par le MCP officiel Colab,
sur A100 de 40 Go. La révision scientifique `e783e117f2d59c89bcf4be467506ae115e5aaeff`
et le [notebook de continuation](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/84a7d43808423f0e0e905fc05c29f00cdf6b634f/notebooks/11_presence_specificity_colab.ipynb)
ont été publiés avant ce lancement. Les huit tests passent aussi dans Colab.
L'archive prévue est `menia-specificite-questions.zip` ; aucun résultat n'est
encore rapporté dans cette version du protocole.

## Question et explication concurrente

Le score `logit(1) − logit(0)` sépare très bien présence et absence dans le
Colab 10. Un détecteur qui pousse toujours vers « 1 » après une perturbation
produirait pourtant ce résultat même s'il ignorait la question. Ici, les
adaptateurs restent figés : aucune époque supplémentaire, aucun choix de
checkpoint, aucune modification après consultation du test.

La section 2.3.1 de [Hahami et al.](https://arxiv.org/html/2607.14111v1)
motive un contrôle sur des questions factuelles. Son annexe B présente une
phrase de conclusion contradictoire avec cette section : nous retenons
l'explication concurrente comme une possibilité logique à tester nous-mêmes,
sans importer leur résultat comme une propriété déjà établie de Qwen.

## Matériel figé et nouvelles données

- Qwen3-4B BF16, révision `1cfa9a7208912126459214e8b04321603b3df60c`.
- Les **neuf adaptateurs** de la tentative `20260919T100527Z-2508e757`,
  vérifiés contre leurs neuf SHA-256 déjà publiés dans
  [le bilan](../artifacts/presence-detection-pilot/first-audit-summary.json).
- Empreinte du journal parent :
  `fdbe7d9e764e509789e552591a8b13ef46ba3bf2efd3543eed08e63595acd526`.
- Graine du nouveau plan `202609191` ; **24 blocs nouveaux par répétition**,
  144 phrases et 72 graines de rotation exclues des expériences 06, 08, 09 et
  10. Vocabulaire et grammaire communs : pas de transfert de domaine.
- Le repère se trouve dans chacune des deux lignes dans 12 blocs par répétition.

## Plan factoriel

Chaque répétition croise la base et ses trois adaptateurs — visible, fort,
mélangé — avec les deux familles d'entrée, les quatre questions et les quatre
conditions. Les deux phrases restent dans leur présentation canonique.

| Facteur | Modalités |
|---|---|
| Famille cachée | rotation de norme conservée, couche 17, force 1 ; texte identique entre conditions |
| Famille visible | `[SIGNAL]` ajouté à la phrase, aucune rotation |
| Conditions | absence ; première phrase ; deuxième phrase ; copie témoin sans modification |
| Questions | présence ; absence avec chiffres inversés ; repère sur la phrase 1 ; repère sur la phrase 2 |

La question de présence est **exactement celle du Colab 10**. Pour la question
d'absence, seuls ses deux chiffres de décision sont inversés : répondre 0 si
une phrase est touchée, 1 sinon. La famille visible utilise l'équivalent avec
`[SIGNAL]`. Les deux questions publiques demandent 1 si la phrase désignée
porte `[REPERE]`, 0 sinon, en ignorant le signal et la modification interne.
Leurs réponses sont équilibrées et indépendantes de l'intervention.

**Budget fixé : 9 216 évaluations, zéro mise à jour**, dont 2 304 paires
absence/copie témoin. Tous les checkpoints sont testés sous les deux familles :
le contrôle visible inclut donc aussi l'adaptateur fort. Ordre des requêtes
mélangé à graine fixe au sein de chaque adaptateur. Premier token, aucun cache,
aucun échantillonnage, aucune chaîne de pensée ; aucune troncature autorisée.

## Mesures et règle de lecture

Pour chaque bloc et question, conserver le score brut `s = logit(1) − logit(0)`
et calculer `delta = moyenne(s première, s deuxième) − s absence`.
La copie témoin est un contrôle d'intégrité, pas une observation supplémentaire.

L'AUROC orientée vers la présence utilise `s` pour la question de présence et
`−s` pour la question d'absence. L'interaction de mapping vaut
`delta présence − delta absence`. Une poussée fixe vers « 1 », indépendante de
la question, donne une interaction nulle et échoue au critère d'inversion.

**Règle fixée :** dans chacune des trois répétitions, les intervalles à 95 %
doivent être strictement positifs pour les cinq contrastes suivants :

1. AUROC orientée du fort moins 0,5, question de présence ;
2. AUROC orientée du fort moins celle du mélangé, question de présence ;
3. les mêmes deux contrastes pour la question d'absence ;
4. interaction de mapping du fort dans la famille cachée.

Il s'agit de cinq contrastes par répétition (les points 1, 2, les deux du point
3, puis 4). Les intervalles sont descriptifs, conditionnels aux adaptateurs,
par 2 000 rééchantillonnages appariés des 24 blocs, graine `202609191+900+rep`.
Pas d'intervalle global traitant les requêtes comme des réplications indépendantes.

Un **contrôle de compréhension** complète cette règle : dans chaque répétition,
l'exactitude équilibrée au seuil naturel doit atteindre 90 % pour chacune des
deux questions de mapping dans la famille visible, à la fois avec l'adaptateur
visible et avec l'adaptateur fort. Ce seuil opérationnel est choisi avant
collecte ; il n'est pas un test universel de compréhension. Le résultat de la
règle sur le signal est conservé séparément du résultat de ce contrôle.
Un échec du contrôle empêche d'attribuer un échec caché spécifiquement à l'accès
interne : la nouvelle consigne peut elle-même mal transférer.

Mesures secondaires, sans sélectionner un nouveau critère après collecte :

- AUROC contre la base, scores bruts, ordre apparié, exactitude par classe,
  premier token libre, masse des chiffres et Brier ;
- sur les questions publiques : déplacement des logits par cible 0/1,
  changement de probabilité de « 1 » et d'exactitude ;
- différences appariées entre l'effet brut de présence et chaque question
  publique, pour chaque bras et famille.

Un effet non significatif sur une question publique ne prouve pas une absence
d'effet. Aucune marge d'équivalence n'est définie ici. Une interaction positive
ne prouve pas non plus un mécanisme de métacognition : un classifieur ordinaire
conditionné par la consigne reste une explication possible. Modifier une
question modifie aussi les activations, donc ce plan n'isole pas un circuit.

## Portée et intégrité

Un succès écarterait l'explication simple d'un score exclusivement attaché au
chiffre de sortie dans ce montage. Il ne montrerait ni la prévision d'erreurs
naturelles, ni le bénéfice de décisions de vérification, ni une conscience.
Un échec identifie une limite des checkpoints figés ; aucun réentraînement ne
sera amalgamé à ce test. Une expérience suivante pourra alors traiter cette
limite avec de nouvelles données et une nouvelle règle.

Le journal conserve plan, sources, entrées hachées, poids parents, logits et
traces de rotation. Le lecteur rejette un changement de plan/source/poids,
une copie témoin différente ou un événement d'entraînement. Une tentative
incomplète n'est jamais déclarée réussie. La reprise vérifie le même environnement
et les mêmes fichiers ; une inférence interrompue est tracée et rejouée.

Les tests logiciels comprennent un oracle sémantique et un détecteur synthétique
qui réussit la présence mais ignore la question : le second doit être rejeté.
Un petit Qwen aléatoire teste le moteur, la reprise déterministe et l'immuabilité
des adaptateurs. Aucun de ces tests ne prédit le résultat préentraîné.

Validation avant collecte : **8 tests passent**, dont la comparaison des logits
d'une tentative reprise avec ceux d'une tentative ininterrompue sur petit Qwen.
Le tokenizer réel traite les **2 304 conditions d'entrée distinctes** (réutilisées
pour les quatre bras) en **92 à 119 tokens**. Aucun poids préentraîné n'est
chargé pour cette vérification. Le plan comporte l'empreinte
`0d59ec577260a857acaa25acf0ddca295f8b3a7c877196084371828b448a8981`.

Un [second calcul des mesures](../research/audit_presence_specificity.py), écrit
avant consultation des réponses du modèle préentraîné, utilise ses propres
cibles, prédictions et comparaisons par paires. Il calcule le bootstrap par
multiplicités des blocs. Vérifié sur trois journaux synthétiques (oracle,
biais fixe, logits aléatoires), son écart maximal est de `3,56 × 10⁻¹⁵`.
Il partage le lecteur strict du protocole : il s'agit d'une vérification
arithmétique, pas d'une réplication par un autre laboratoire.
