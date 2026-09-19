# Contrôle de la capture de confiance sur le modèle préentraîné

19 septembre 2026. **Contrôle technique préparé avant exécution GPU.** Il
complète la [revue du comparateur de confiance](NATURAL_ERROR_CONFIDENCE_REVIEW.md).
Il ne mesure ni la capacité à prévoir une erreur ni la conscience.

Le script `research/output_confidence_validation.py` fixe deux questions
élémentaires de comptage et de calcul, deux graines (421 et 972), Qwen3-4B
dans sa révision déjà utilisée et les paramètres de génération existants,
avec une limite de 32 nouveaux tokens. Il n'ajuste aucun poids et n'évalue pas
l'exactitude des réponses. Ces quatre cas techniques seront exclus de tout
futur test réservé de prévision des erreurs.

Pour chaque cas, trois chemins sont comparés :

1. Le générateur original, avec conservation des IDs réellement décodés.
2. Le même générateur avec le hook de capture des probabilités.
3. La génération Transformers retournant ses logits bruts, sans ce hook.

Les tokens, le texte, les métadonnées et les états des générateurs aléatoires
CPU/CUDA doivent être identiques entre les deux premiers chemins. Le callback
doit s'exécuter une seule fois avant le premier tirage et le hook doit être
retiré. Le troisième chemin doit reproduire les mêmes tokens et états aléatoires.
À partir de ses logits, un calcul NumPy/math séparé retrouve les probabilités,
entropies et log-probabilités de tous les tokens, y compris les tokens d'arrêt.
La tolérance numérique absolue est fixée à `1e-10` en double précision CPU.
Tous les cas et tout échec sont conservés ; aucune graine n'est remplacée.

Le code de Transformers 4.56.2 conserve dans sa boucle d'échantillonnage les
logits avant filtrage pour ce réglage. Les trois chemins partagent le modèle
et cette bibliothèque : la concordance ne constitue pas une implémentation
indépendante du Transformer. Elle peut néanmoins révéler un décalage de tokens,
une capture de scores filtrés ou un effet de l'instrumentation.

Le contrôle passe d'abord sur le Qwen miniature aléatoire utilisé par les tests,
avec huit nouveaux tokens maximum : quatre cas, quarante vérifications, écart
numérique maximal `7,11e-15`. Il ne remplace pas l'exécution sur Qwen3-4B.

L'exécution GPU utilise un processus et un répertoire séparés. Elle ne modifie
ni les sources figées ni les poids du Colab 16. Elle vérifie au moins 14 Gio
libres avant chargement. Les durées sont conservées mais ne constituent pas
un benchmark : échauffement et partage de l'A100 empêchent de leur attribuer
un coût causal de capture.

```sh
python -m research.output_confidence_validation resultat.json
```

Le résultat technique est nécessaire avant une collecte prospective, mais
insuffisant pour choisir un prédicteur. Il restera à comparer entrée et
confiance de sortie à l'apport supplémentaire d'états internes, sur des
questions nouvelles et sans utiliser la réponse finale dans une prévision
censée la précéder.
