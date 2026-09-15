# Menia sur iPhone 17 Pro — version 0.2

Menia embarque un modèle préentraîné Qwen3-4B en 4 bits, sa mémoire locale et un
module qui mesure ses réponses à de petits tests de calcul. Aucun entraînement
sur Colab n’est nécessaire pour démarrer. La construction d’une application ne
démontre ni conscience subjective ni nouveauté scientifique.

## Installation avec Codemagic, sans Mac personnel

Le fichier [`codemagic.yaml`](../codemagic.yaml) contient deux workflows :

| Workflow | Résultat |
|---|---|
| `menia-check` | Tests Swift et application iPhone **non signée**, pour vérifier la compilation |
| `menia-iphone` | IPA **Ad Hoc signée**, installable sur les iPhone inclus dans le profil Apple |

1. Dans Codemagic, connecter le dépôt `speed25200-cyber/Menia` et choisir la
   branche **`codex/recall-reliability`**. Le travail reste dans la PR de recherche,
   il n’est pas encore sur `main`.
2. Dans **Team settings → Code signing identities**, disposer du certificat
   Apple Distribution et du profil **Ad Hoc** correspondant. Le profil doit
   inclure l’UDID de ton iPhone 17 Pro.
3. Le bundle identifier est actuellement **`ch.menia.local`**. S’il diffère dans
   ton compte Apple, modifier la même valeur dans `ios/project.yml` et
   `codemagic.yaml`. Le profil doit autoriser la capacité **Increased Memory Limit**,
   activée sur l’App ID Apple avant de générer/renouveler le profil.
4. Lancer **`menia-iphone`**. Codemagic exécute les tests, génère le projet Xcode,
   applique les profils disponibles et produit `build/ios/ipa/*.ipa`.
5. Ouvrir le lien d’installation de l’artefact depuis **Safari sur l’iPhone**.
   Une archive `.app.zip` non signée ne peut pas remplacer cette IPA. Si l’iPhone
   n’est pas inclus dans le profil, l’installation Ad Hoc est refusée.
6. Ouvrir Menia et toucher **Télécharger Qwen3-4B · 2,15 Go** sur Wi-Fi. Garder
   l’application ouverte pendant cette étape. Puis toucher **Charger** et **Envoyer**.

Les secrets Apple restent dans Codemagic. Aucun certificat, clé privée ou profil
n’est inclus dans le dépôt. Le workflow n’envoie pas l’application sur l’App Store
ou TestFlight. L’adhésion Apple Developer et le profil restent nécessaires pour
cette distribution Ad Hoc, même si Codemagic effectue la compilation à distance.

## Modèle fourni et mode hors ligne

Le bouton télécharge les fichiers officiels de
[`Qwen/Qwen3-4B-MLX-4bit`](https://huggingface.co/Qwen/Qwen3-4B-MLX-4bit)
à la révision `52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495` :

- **2 153 298 402 octets**, licence Apache 2.0 incluse ;
- manifeste de taille et SHA-256 de chaque fichier dans
  [`qwen3-4b.json`](../ios/MeniaCore/Sources/MeniaCore/Resources/qwen3-4b.json) ;
- téléchargement sur Wi-Fi, dans un dossier temporaire, puis contrôle des hashes ;
- remplacement du modèle précédent seulement après vérification ;
- empreinte du contenu du modèle et du tokenizer pour séparer les mesures de capacités.

Prévoir plusieurs Go de stockage libres pour les fichiers temporaires et, lors
d’un remplacement, le modèle précédent. La taille des fichiers n’est pas la RAM
nécessaire à l’inférence. Un téléchargement annulé est recommencé ; la reprise
partielle et le téléchargement en arrière-plan ne sont pas implémentés.

Seule cette action explicite contacte Hugging Face et ses serveurs de fichiers.
Les messages, notes et résultats de tests ne sont pas envoyés à un serveur.
Après installation, le chargement et la génération utilisent exclusivement les
fichiers locaux. Vérifier ce fonctionnement en mode avion sur l’appareil.

**Importer un dossier** reste disponible pour des poids MLX 4 bits de type
`qwen3`, `qwen3_5` ou `qwen3_5_text`, avec `config.json`, `tokenizer.json`,
`tokenizer_config.json` et tous les fichiers `.safetensors` référencés par l’index.
La vérification de structure et d’empreinte ne garantit pas que toute conversion
communautaire soit compatible : MLX doit encore charger et exécuter ces poids.
Qwen3.5 et les variantes 8/9B ne sont pas validés sur cet iPhone.

## Ce que Menia conserve

- Jusqu’à **50 notes explicites**, de 2 000 caractères chacune.
- Les **20 derniers échanges terminés** ; les réponses interrompues ne deviennent
  pas des souvenirs. Les cinq derniers échanges sont affichés.
- Les **128 derniers tests terminés**, avec question, prévision antérieure,
  réponse, référence et résultat, associés à l’empreinte du modèle.

L’application fournit au LLM un extrait de notes et d’échanges, avec le bilan des
tests du modèle chargé. Le moteur compte les tokens du prompt formaté et réduit
l’extrait si nécessaire, en conservant la question et le bilan. Le contexte total
est limité à **2 048 tokens**, dont **256 au maximum pour la sortie**. Le mode
`thinking` est désactivé. Les réponses peuvent donc être tronquées à cette limite.

Le fichier `session.json` est sauvegardé atomiquement avec protection iOS complète.
Le dossier privé est exclu des sauvegardes automatiques. Les anciennes notes de
la version 0.1 sont migrées ; une sauvegarde illisible est signalée et n’est pas
remplacée silencieusement. Le bouton **Effacer la mémoire et les tests** permet
un nouveau départ sans supprimer les poids. Les copies de rapports déjà partagées
ou les dossiers de modèles externes restent sous le contrôle de l’utilisateur.

## Mesurer les capacités du LLM installé

**Tester mes capacités** exécute cinq additions/soustractions choisies aléatoirement
avec des opérandes entiers entre 0 et 999. Chaque test suit cet ordre :

1. L’application sauvegarde la question et une prévision de réussite.
2. Le LLM reçoit uniquement la consigne et le calcul, sans résultat attendu,
   mémoire, estimation de fiabilité ou réponse précédente.
3. Le contrôleur compare la réponse à l’entier exact demandé. Une explication
   supplémentaire constitue un échec de ce contrat strict, même si elle contient
   le bon nombre ; ce n’est donc pas un diagnostic isolé de raisonnement arithmétique.
4. Le résultat met à jour l’estimation fournie aux conversations suivantes.

L’estimateur emploie un a priori Beta(1,1). Pour les `n` observations conservées
du même modèle, dont `s` réussites, la prévision est `(s+1)/(n+2)`. Le rapport
contient aussi le score de Brier des **prévisions faites avant les réponses**.
Zéro observation signifie « inconnu » ; 50 % est alors l’a priori du calcul,
pas une mesure. Un arrêt ne compte pas comme une erreur mathématique.

Cette nouvelle mesure porte sur le LLM exécuté. Elle est distincte du pilote
Python de sélection entre deux hypothèses d’entretien de capteur simulé : les
taux 0,93/0,57 de ce pilote ne sont pas attribués à Qwen. L’estimation Beta est
classique, conditionnelle à ce protocole, et sa calibration empirique reste à
évaluer. La fenêtre de rétention ne remplace pas un détecteur de rupture ; les
changements de distribution et l’évaluation des propositions d’action restent
des travaux futurs.

Le LLM peut proposer un test dans sa réponse. L’utilisateur lance le test avec
le bouton dédié ; aucun texte généré n’est interprété comme une commande exécutable.
Les modules de navigation, d’entretien anticipé et de mémoire récurrente de la
recherche Python ne sont pas tous portés dans cette version.

## Vérification sur appareil

Tester en Release : conversation en français, souvenir après relance, lancement
des tests, changement du bilan, export du rapport, arrêt pendant une réponse,
effacement puis relance, import invalide, téléchargement annulé et mode avion.
Mesurer le pic mémoire dans Instruments, la chauffe, le délai du premier texte,
la durée totale, la latence d’arrêt et la batterie sur une session de dix minutes.
L’application interrompt le travail en arrière-plan, en cas de température
sérieuse ou d’avertissement mémoire. La génération utilise MLX/Metal.

L’interface affiche des durées. Un morceau de streaming n’étant pas un token,
aucun débit en tokens/s n’est déduit du nombre de morceaux reçus.

## Construire sur un Mac personnel

Avec Xcode 26.2 et les outils de ligne de commande sélectionnés :

```bash
brew install xcodegen
swift test --package-path ios/MeniaCore
bash ios/prepare.sh
open ios/Menia.xcodeproj
```

Choisir l’équipe de signature et l’iPhone comme destination. Les dépendances
directes sont fixées : MLX Swift LM 3.31.3, MLX Swift 0.31.3 et Swift Transformers
1.3.0. Le verrou `ios/Package.resolved`, issu de la compilation réussie, fixe aussi les dépendances transitives. Les bibliothèques et les poids ne sont pas inclus dans le dépôt.

## Références de construction

- [Signature iOS dans Codemagic](https://docs.codemagic.io/yaml-code-signing/signing-ios/).
- [Construction d’IPA avec les outils Codemagic](https://github.com/codemagic-ci-cd/cli-tools/blob/master/docs/xcode-project/build-ipa.md).
- [Exemples iOS MLX](https://github.com/ml-explore/mlx-swift-examples).
- [Capacité Apple Increased Memory Limit](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit).
