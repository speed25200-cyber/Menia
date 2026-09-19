# Vérification indépendante du Colab 15

Préparée le 19 septembre 2026 pendant l'inférence, avant lecture des scores
comportementaux. Le [protocole scientifique](STATE_INTERCHANGE_PROTOCOL.md)
et ses sources restent figés à `0a86c18796c788400492f0c45036595057760e60`.

`research/audit_state_interchange.py` recalcule les 72 tableaux de transfert,
48 tableaux de compétence initiale et 144 contrastes, dont six principaux.
Il réimplémente les trois cibles sémantiques, les pertes dans le vocabulaire
complet, la copie du token effectivement produit, la conservation du token
destinataire, les sorties autorisées et les normes de remplacement. Les
intervalles rééchantillonnent les mêmes paires et quatre strates fixées par
le protocole. Les prérequis de compétence sont recalculés sans supprimer
les paires qui échouent.

Le code n'appelle ni l'analyse principale ni ses fonctions de codage des
hypothèses. Il partage le plan, le lecteur d'intégrité et les bibliothèques
numériques. Cette séparation peut révéler une erreur arithmétique ; elle ne
remplace ni un audit indépendant du lecteur, ni une réplication extérieure,
ni une validation de l'interprétation scientifique.

Deux tests passent localement. Le premier construit des mécanismes connus :
conservation du destinataire à la couche 17, transfert d'état à la couche 23,
copie du donneur à la couche 35, pour l'état caché et le marqueur public. Les
contrastes principaux valent exactement 0,5 dans ce jeu synthétique. Le
second introduit des réponses initiales fausses, des sorties hors options,
des logits de grande amplitude et des paires hétérogènes. Il vérifie les
intervalles non dégénérés et le rejet d'un contraste ou d'un prérequis modifié.
Ces valeurs synthétiques ne sont pas des résultats de Qwen3-4B.

Après réception du journal complet, l'audit se lance avec :

```powershell
.venv-agent\Scripts\python.exe -m research.audit_state_interchange JOURNAL.jsonl --summary RESUME.json --output VERIFICATION.json
```

Le journal doit être complet et conforme aux sources figées. Aucun verdict
sur le transfert, la conscience ou la nouveauté n'est ajouté par l'auditeur.
