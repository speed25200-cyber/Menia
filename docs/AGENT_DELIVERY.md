# Vérification des quatre fonctions demandées

14 septembre 2026. Périmètre livré : agent Python dans une scène virtuelle,
avec apprentissage statistique en ligne et contrôleur explicite. La conscience
subjective de sa propre existence reste une question ouverte ; aucune fonction
ci-dessous n'est un certificat de conscience.

| Fonction | Réalisation et preuve |
|---|---|
| Apprendre les effets de ses actions | `action_model.py` estime les déplacements à partir des transitions réellement observées. Le contrôle figé réussit 20/20 essais avant inversion et 0/20 après ; l'agent adaptatif réussit 20/20 dans les deux conditions. |
| Histoire avec provenance | `episodic.py` conserve séparément observations, témoignages, prédictions et actions. Les tests excluent les témoignages et les épisodes étrangers des observations servant à apprendre. La reprise entre deux processus est enregistrée dans `cli-restart.json`. |
| Attention, erreurs et oubli | `agent.py` distingue lecture tentée, réponse reçue, contenu en mémoire de travail et référence archivée. Une absence de réponse ne devient pas une perception ; les prévisions sont enregistrées avant leurs évaluations. |
| Décider et expliquer avec ces représentations | Le planificateur utilise les effets appris ; substituer ces effets à observations identiques change la commande choisie. `/why` restitue la décision enregistrée et ses références. La reformulation Qwen fonctionne techniquement mais présente des erreurs documentées. |

Les [résultats et limites](INTEGRATED_AGENT_RESULTS.md) détaillent 240 essais,
20 modèles appris et les vérifications automatisées. Les 36 tests du noyau et de
l'agent passent localement, ainsi que la reproduction du rapport et le contrôle
syntaxique des notebooks. Les poids de Qwen ne sont pas entraînés par cette boucle.
L'application iPhone ne contient pas encore cette intégration.

Pour commencer à la racine du dépôt :

```powershell
python -m menia.chat --session runs/ma-session
```

Puis `/run 80`, `/why`, `/history`, `/quit`. Relancer la même commande et utiliser
`/resume` pour reprendre. La [proposition scientifique](CONSCIOUS_AGENT_DESIGN.md)
reste distincte de ce premier périmètre implémenté.
