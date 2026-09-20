# Contrôle numérique de la continuité sur Qwen3-4B

20 septembre 2026. Protocole technique fixé avant l'exécution. Le
[bilan sur le modèle préentraîné](GENERATION_NUMERICS_RESULTS.md) est désormais
disponible et audité. Neuf tests CPU passent sur un petit Qwen aléatoire, en
0,585 seconde de temps de tests. Ce contrôle précède une expérience sur les
conséquences d'un état propre ; il ne remplace pas cette expérience.

## Pourquoi ce contrôle est nécessaire

Le [module de capture](GENERATION_STATE_CONTROLS.md) conserve les tokens bruts
et le cache réellement produit. Sans intervention, des poids fixes et les
mêmes tokens doivent permettre de reconstruire ce cache en reproduisant les
mêmes appels déterministes. Cet état n'est alors pas une information
inaccessible par principe à une relecture équivalente.

En revanche, traiter le préfixe entier en un appel change la forme des calculs
par rapport à une génération incrémentale. En précision BF16, un écart peut
venir de cette différence numérique. Comparer seulement ces deux chemins
risquerait de l'attribuer à tort à une propriété de l'historique interne.
Le protocole sépare donc le contrôle de capture de la mesure de cet écart.

## Essai fixé avant exécution

Le [plan enregistré](../artifacts/generation-numerics-preparation/design.json)
contient six questions techniques fixes, deux graines identiques par paire
de générations et deux demandes ultérieures. Les questions servent à produire
des préfixes de longueurs diverses ; ce petit ensemble n'est pas un test de
performance, de calibration ou de généralisation. La correction des réponses
n'entre dans aucun critère.

- Qwen3-4B de base, révision `1cfa9a7208912126459214e8b04321603b3df60c`,
  BF16, SDPA, A100, versions et déterminisme du profil existant.
- Six paires de générations, avec les paramètres d'échantillonnage existants
  et un plafond explicite de 64 nouveaux tokens. Une réponse sans arrêt ou
  terminée par un EOS qui ne ferme pas le tour de dialogue arrête le contrôle.
- Trois caches à tokens identiques : génération effective ; relecture du
  prompt puis de chaque token émis séparément ; relecture du préfixe entier.
  Le token d'arrêt effectivement émis est traité une seule fois lors de la
  fermeture du cache de génération.
- Deux branches indépendantes demandent un code de jugement `0/1` ou un code
  d'action `2/3`. Elles emploient deux tokens de décodage glouton dans le
  vocabulaire complet, sans restriction aux codes ni recours automatique.
  Le jugement n'est jamais donné à la branche d'action.
- Chaque branche est aussi décodée depuis la séquence complète, sans cache,
  puis les deux branches du cache réel sont redécodées en ordre inverse.
  Cela représente douze générations et soixante décodages de branche.

Les contrôles stricts exigent l'égalité exacte du cache entre générations
répétées et entre génération et relecture au même rythme, l'égalité des sorties
de branches à ce rythme, l'indépendance à l'ordre des branches et la stabilité
des versions des paramètres. Un échec arrête l'essai et conserve ses traces.
Les erreurs ne déclenchent ni nouvelle graine, ni autre modèle, ni modification
de tolérance. La garde sur les paramètres reste un contrôle de versions,
sans prétendre détecter toute manipulation arbitraire du processus.

La relecture entière constitue une **mesure descriptive**, sans seuil ajouté
après coup : erreur maximale absolue et norme relative par couche et par K/V,
écarts de logits des codes et de probabilités conditionnelles, masse totale
des codes, validité du code suivi d'EOS et accord des tokens natifs. Deux
sorties invalides ne sont pas comptées comme deux actions valides. Un écart
de probabilité n'est pas présenté comme un changement d'action.

Il n'y a ni adaptation, ni perturbation, ni choix d'un état caché à apprendre
dans cet essai. Même une réussite complète ne montrerait pas que Menia lit
son état, prévoit ses capacités ou éprouve sa propre existence. Elle établirait
que les outils peuvent distinguer les deux causes techniques d'un écart dans
ces conditions précises, avec un état de référence reconstructible.

## Exécution et conservation

Le [lanceur](../scripts/colab_generation_numerics_launcher.py) exige la fin du
diagnostic de budget et un A100 libre. Il récupère une révision Git immuable,
exécute les neuf tests, puis vérifie les empreintes du plan et des sources.
Il conserve le journal et les erreurs de l'unique tentative. Il ne modifie
ni l'expérience de budget ni les poids de l'application iPhone.

Le [collecteur](../research/confidence_generation_numerics_gpu.py) consigne les
entrées, tokens, caches sous forme d'empreintes et d'écarts, comparaisons,
décodages et contrôles. Les caches complets restent en mémoire : ce diagnostic
ne les exporte pas et ne permet donc pas une recomparaison indépendante de
leurs tenseurs depuis la seule archive. Les poids de base ne sont pas exportés.
Une future étude causale demandera une provenance et des contrôles adaptés
aux interventions effectives. Aucun chronométrage de performance n'est annoncé.

L'[auditeur d'archive](../research/audit_generation_numerics.py), préparé avant
cette exécution, reconstruit le journal et les entrées avec le tokenizer fixé.
Il recalcule les décisions natives, les probabilités binaires et les comparaisons
à partir des sorties conservées, y compris lorsque la tentative échoue. Trois
tests sur sorties construites passent, notamment le refus de compter comme
action valide un code sans EOS. Ces tests sont distincts des neuf contrôles du
lanceur. L'auditeur indique explicitement ce qu'il ne peut pas refaire :
comparaison des tenseurs de cache, seconde génération et décodages en ordre
inverse, faute de leurs traces complètes dans l'archive.

```sh
python -m unittest tests_language.test_confidence_generation_numerics tests_language.test_confidence_generation_continuity -v
python -m research.confidence_generation_numerics_gpu --journal NOUVEAU_JOURNAL.jsonl
python -m research.audit_generation_numerics JOURNAL_RECU.jsonl --output AUDIT.json
```
