# Évaluation de la description linguistique de Menia

14 septembre 2026. Ces quatre cas exploratoires utilisent l'agent réel et son
journal. Ils ne constituent pas un test de conscience, ni une mesure générale
de qualité linguistique. Les commandes et `/why` fonctionnent indépendamment du LLM.

## Exécution locale vérifiée

- Serveur officiel [llama.cpp b10809](https://github.com/ggml-org/llama.cpp/releases/tag/b10809), Windows CPU, quatre threads.
- [Qwen3-1.7B-GGUF officiel](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF), révision `90862c4b9d2787eaed51d12237eafdfe7c5f6077`.
- Fichier `Qwen3-1.7B-Q8_0.gguf`, 1 834 426 016 octets.
- SHA-256 vérifié : `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`.
- Contexte serveur 4 096 tokens, sortie limitée à 192 tokens, température 0,
  graine 17, mode de raisonnement désactivé. Aucun entraînement de Qwen.

Les [sorties brutes](../artifacts/integrated-agent/language-gguf/responses.jsonl)
contiennent le contexte envoyé, la réponse et l'explication canonique. Le manifeste
enregistre les empreintes des sources. Les quatre réponses ont pris environ
8,6, 11,3, 22,2 et 17,0 secondes sur le CPU de cette machine.

| Cas | Lecture qualitative de la sortie |
|---|---|
| Aucune observation | Signale correctement l'absence de données observées. |
| Témoignage non vérifié | Identifie `other_agent` et l'absence de vérification, mais omet les coordonnées reçues et formule une localisation incohérente. |
| Décision apprise | Reprend le déplacement prévu, mais le confond avec le nom de la commande ; la limite invoquée est incorrecte et la sortie est tronquée. |
| Sortie de la mémoire de travail | Distingue partiellement archive et mémoire de travail, mais invente une incertitude sur la position de la référence ; sortie tronquée. |

**Conclusion : le transport du contexte et l'inférence fonctionnent ; une
description libre fidèle n'est pas validée.** Une réponse vraisemblable ne remplace
pas l'événement enregistré. Le programme démarre donc en mode structuré par défaut.

## Reproduire et utiliser

Télécharger le binaire officiel et le fichier GGUF aux versions ci-dessus, puis
ouvrir un terminal pour le serveur (remplacer les deux chemins) :

```powershell
& 'CHEMIN/llama-server.exe' -m 'CHEMIN/Qwen3-1.7B-Q8_0.gguf' --host 127.0.0.1 --port 8766 -c 4096 -t 4 -ngl 0 --jinja --reasoning-budget 0 --alias menia-qwen3
```

Dans un autre terminal, à la racine du dépôt :

```powershell
python -m menia.chat --session runs/ma-session --llama-url http://127.0.0.1:8766
python -m research.evaluate_agent_server --url http://127.0.0.1:8766 --out runs/nouvel-essai-langage
```

Le client Python utilise uniquement la bibliothèque standard. Il accepte une URL
locale et n'envoie aucun contexte à un service distant. Arrêter le serveur avec
Ctrl+C après utilisation. Pour les explications vérifiables : `/run 80`, puis `/why`.

## Tentatives antérieures conservées

Le chargement BF16 avec PyTorch 2.4.1 était trop lent sur ce CPU et a été arrêté.
Deux tentatives de conversion en pleine précision ont terminé avec un code
d'erreur ; l'une correspond à un crash `c10.dll` dans le journal Windows.
L'essai de quantification dynamique int8 a terminé ses quatre cas mais produit
des réponses hors sujet : voir [ses sorties](../artifacts/integrated-agent/language-layerwise-int8/responses.jsonl).
Le chargement avec PyTorch 2.8.0 a également échoué avant toute réponse.
Les manifestes marqués `initializing` sont les dernières écritures de processus
terminés ; ils ne désignent pas des calculs encore actifs.

Ces échecs ne prouvent pas une cause unique : le backend, le format de poids et
la conversion numérique diffèrent. `--hf` conserve la voie Hugging Face à titre
expérimental ; elle n'est pas le parcours validé pour ce CPU Windows. Les anciens
poids safetensors téléchargés pour ces essais ont été retirés pour libérer le disque.
