# Prévoir une erreur réelle : le comparateur de confiance manquant

19 septembre 2026. **Instrumentation vérifiée sur un Qwen miniature sur CPU,
puis sur quatre cas courts de Qwen3-4B/A100. Aucun gain de prévision des erreurs
ni nouvel entraînement dans cette étape.** Le Colab 16 poursuit séparément son
protocole figé.

## Lacune constatée dans les données existantes

Le [premier moniteur d'activations](ACTIVATION_MONITOR_RESULTS.md) contient
672 réponses, dont 192 au test, avec des projections de dimension 128 pour
l'entrée, la couche centrale et la sortie de `model.norm`. Le journal reçu
porte l'empreinte SHA-256
`f78dcb3a93d1b0a9b4d7e4e60b50d050484b3cac464a7a2b6428188775e34ac8`.
Ses 672 événements de résultat conservent le texte, la durée, les nombres de
tokens, le dispositif, le type numérique et la configuration de génération.
Ils ne conservent ni logits, ni probabilités de tokens, ni entropie.

La lecture du collecteur confirme cette limite : `generate_text` retourne le
texte décodé et ces métadonnées ; le moniteur projette les activations avant la
tête de langage. La projection finale ne permet pas de reconstruire exactement
le vecteur complet qui entre dans cette tête. Retokeniser le texte ne restitue
pas nécessairement les IDs produits, notamment les tokens d'arrêt masqués.
Un calcul de confiance exact exige donc une nouvelle exécution enregistrée,
pas l'invention de colonnes absentes dans le premier export.

Le Brier interne déjà mesuré, 0,117064, ne bat pas la référence par catégorie,
0,113239. Ce résultat reste inchangé. Il manque cependant un comparateur pour
une prochaine tentative : ce que les probabilités de sortie permettent déjà
de prévoir, sans moniteur supplémentaire des activations.

## Ce que précise la littérature

Kumaran et al. séparent confiance, exactitude et abstention. Leurs interventions
affectent l'abstention ; leurs analyses distinguent probabilités de sortie,
rapports verbaux et activations. Dans leur examen de Gemma, le signal résiduel
d'exactitude ne prédit pas l'abstention au-delà des mesures de confiance, alors
qu'il reste de l'information sur l'abstention dans les activations. Prévoir
un choix de prudence n'équivaut donc pas à mieux prévoir une erreur. C'est un
antécédent direct, déjà cité dans Menia, dont cette lecture précise les contrôles.
[Texte complet, §4.5.1 et §7.9](https://arxiv.org/html/2603.22161v2).

La prépublication MIRROR rapporte aussi l'autonomie, la réussite globale et la
fréquence d'escalade. Sa réussite sous résolution externe parfaite est une
hypothèse favorable ; le texte présente séparément une sensibilité à un
résolveur faillible. Ces résultats rapportés, non reproduits ici, motivent de
mesurer le service rendu et le coût des vérifications, en plus des erreurs
parmi les seuls cas encore traités directement.
[MIRROR, annexes U et V](https://arxiv.org/html/2604.19809v1).

La conséquence pour Menia est notre proposition méthodologique : comparer
l'information interne à une confiance de sortie calibrée, puis examiner son
utilité décisionnelle à budget égal. La boucle générale n'est pas inédite.

## Capture ajoutée

[`research.output_confidence_trace`](../research/output_confidence_trace.py)
appelle le générateur existant sans changer sa configuration. Un hook de lecture
copie sur CPU le dernier vecteur de logits de chaque passage dans la tête de
langage. Un relais du tokenizer conserve les IDs effectivement décodés, avant
que le décodage masque les tokens spéciaux. Les copies complètes sont temporaires ;
la trace retournée contient seulement les statistiques et les IDs nécessaires.

Deux moments sont explicitement séparés :

| Information | Disponibilité | Utilisation possible |
|---|---|---|
| Maximum, marge des deux premiers tokens et entropie du vocabulaire complet | Après le premier passage dans la tête, avant le tirage du premier token | Prévision prospective sur cette même génération |
| Log-probabilité des tokens effectivement produits, somme, moyenne et entropies successives | Après les tokens correspondants ; trace complète à la fin | Vérifier ou accepter une réponse déjà rédigée |

La distribution enregistrée est le softmax des logits bruts à température 1,
avant top-k, top-p et température d'échantillonnage. Elle diffère de la
distribution effectivement utilisée pour tirer les tokens. Les scores de
séquence incluent les tokens d'arrêt ; ils ne sont ni la probabilité de
l'entier correct, ni celle de toutes ses formulations possibles. Les scores
bruts doivent être calibrés sur des résultats observés pour prévoir une réussite.

Le calcul en double précision s'effectue sur les copies CPU. Il ajoute du temps
et du transfert : une mesure de coût devra l'inclure. La capture ne change ni les
poids, ni la consigne, ni le générateur original. Le callback facultatif
`on_prefill` permet un enregistrement avant le premier tirage ; il doit rester
une écriture sans mutation du modèle ni consommation de nombres aléatoires.

## Vérifications exécutées

Quatre nouveaux tests couvrent le calcul manuel d'une distribution de trois
valeurs, l'identité des IDs produits, du texte, des métadonnées et de l'état du
générateur aléatoire, la concordance avec les logits de chaque préfixe recalculé
sans cache,
l'inclusion d'EOS et le retrait du hook si l'enregistrement échoue. Le cas EOS
distingue notamment le score brut de la distribution filtrée pour le tirage.
Les six tests existants du générateur et du moniteur restent verts : **dix tests
réussis** au total. Il s'agit de contrôles logiciels sur un petit modèle
aléatoire. Le [contrôle préentraîné désormais reçu](OUTPUT_CONFIDENCE_VALIDATION.md)
ajoute quatre cas courts et quarante vérifications sur Qwen3-4B/A100 : les
tokens et états aléatoires sont identiques, et le recalcul depuis les logits
natifs concorde à moins de `7,75e-14`. Dix tokens sont enregistrés dans ces
quatre cas ; cela ne garantit pas l'absence d'effet pour toutes les générations.

```sh
python -m unittest tests_language.test_output_confidence_trace tests_language.test_cross_model_gpu tests_language.test_activation_monitor_gpu -v
```

## Expérience à fixer après le diagnostic en cours

Cette section est une spécification de travail, pas un protocole enregistré
ni un résultat. Les tâches, effectifs, répétitions et contrastes devront être
figés avant une nouvelle collecte confirmatoire.

1. Le contrôle technique Qwen3-4B est exécuté sur quatre cas fixes. Conserver
   leur exclusion des nouveaux jeux réservés ; étendre la vérification si le
   modèle, les adaptateurs ou le chemin de génération changent. Toute divergence
   doit être comprise avant de comparer des prévisions.
2. Enregistrer séparément les prévisions avant réponse et les décisions après
   rédaction. Une probabilité du token finalement choisi n'est pas une variable
   disponible avant son choix ; elle ne doit pas être glissée dans le premier test.
3. Ajuster sur l'apprentissage, choisir sur la validation, puis figer : taux par
   famille/difficulté, lecteur d'entrée, confiance de sortie calibrée, combinaison
   entrée–confiance, puis la même combinaison avec états internes. Conserver un
   contrôle d'états permutés au sein des catégories et comparer les contrastes
   appariés sur les mêmes réponses. Aucun solveur ne fournit la cible au prédicteur.
4. Tester sur des questions nouvelles : Brier, classement au sein des catégories
   quand les deux issues existent, calibration et réponses hors format. Un
   avantage global porté par la difficulté entre catégories ne suffit pas.
   Si les poids changent, collecter les erreurs du checkpoint final et conserver
   celles du parent séparément ; voir le [contrôle de dérive de cible](SELF_PREDICTION_CONTROLS.md).
5. Si un gain d'information se reproduit, tester l'action à coût égal : répondre,
   vérifier ou s'abstenir. Rapporter réussite globale, coût, couverture et erreurs,
   avec une référence qui utilise la seule confiance de sortie. La fiabilité du
   vérificateur doit être mesurée ou son caractère exact limité explicitement à
   ces tâches. Les lésions et restaurations doivent préserver les capacités de
   base suffisamment pour interpréter l'effet sur les décisions.

Un rejeu éventuel des 672 anciennes questions serait une analyse exploratoire,
puisque leurs réponses et leur ancien test sont connus. Il ne deviendrait pas
un nouveau test réservé par simple changement de fichier. Un bénéfice fonctionnel
ne réglerait pas, à lui seul, la question de l'expérience subjective demandée.

La [préparation logicielle suivante](PROSPECTIVE_CAPTURE_AND_CONTROLS.md) ajoute
une capture commune, des comparateurs emboîtés incluant la représentation avant
sortie, 3 456 questions disjointes des anciens essais et des lecteurs ajustés
sans accès au test. Quatorze tests passent, dont trois contrôles existants du
générateur. Le collecteur prospectif et son analyse restent à finaliser ; ce
travail ne fournit aucune nouvelle performance prédictive de Qwen3-4B.
