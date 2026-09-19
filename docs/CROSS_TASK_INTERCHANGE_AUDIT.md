# Vérification arithmétique du Colab 16

Préparée le 19 septembre 2026 pendant l'inférence, avant lecture des scores
comportementaux. Le protocole et les sources scientifiques restent figés à
`b97753c1cca1f936e7319f971e1c1a4058df0a01`.

`research/audit_cross_task_interchange.py` recalcule les 144 tableaux de
transfert, les 48 tableaux intacts et les 432 contrastes, dont 18 principaux.
Il reconstruit les quatre prédictions à partir des faits du donneur et du
destinataire, de leurs questions et de leurs codes. Les pertes utilisent la
masse des quatre chiffres dans le vocabulaire complet ; elles ne renormalisent
pas simplement la probabilité sur les deux réponses autorisées. La copie du
token réellement produit et la conservation de la réponse destinataire sont
mesurées séparément des cibles théoriques.

L'auditeur vérifie aussi les 4 608 témoins identiques, les 12 288 copies de
dernière couche et les 18 432 transferts entre questions. Les intervalles
rééchantillonnent les mêmes paires dans les quatre strates fixées. Les deux
familles de prérequis sont recalculées : compétence intacte et transfert au
sein de chaque tâche. Les essais qui échouent restent tous dans le bilan.

Deux tests passent localement en 56,323 secondes. Des journaux synthétiques
complets réalisent les quatre mécanismes connus dans les deux sens. Un second
cas introduit des réponses fausses, des sorties hors options, des logits de
grande amplitude, des normes différentes et des paires hétérogènes. Il vérifie
les intervalles non dégénérés et le rejet de contrastes et prérequis altérés.
Ces jeux synthétiques ne sont pas des résultats du modèle préentraîné.

Le code ne réutilise ni l'analyse principale ni ses fonctions de prédiction.
Il partage cependant le plan, le lecteur d'intégrité et les bibliothèques
numériques. C'est un contrôle arithmétique supplémentaire, pas une réplication
extérieure, un audit indépendant du lecteur ou une validation de la conscience.

`scripts/plot_cross_task_interchange.py` fixe la présentation des deux sens
de transfert, des trois sites, des trois répétitions, de la base et des dix-huit
contrastes principaux. La figure a été exécutée et inspectée sur le journal
synthétique hétérogène, avec un titre explicite. Cette image de contrôle reste
locale. La figure scientifique est ensuite produite après réception du journal
complet et réussite des deux recalculs, puis inspectée visuellement.

Après réception et vérification de l'archive complète :

```powershell
.venv-agent\Scripts\python.exe -m research.audit_cross_task_interchange JOURNAL.jsonl --summary RESUME.json --output VERIFICATION.json
```

Le bilan doit également concorder avec le recalcul de l'analyse principale.
Les fichiers parents et les trois checkpoints sont recontrôlés sur PC. Les
résultats complets, y compris les directions défavorables et tous les tableaux
secondaires, sont conservés avant toute interprétation.

**Application au résultat complet :** les 44 544 passages sont reçus sans
erreur ni reprise. Le recalcul principal est exactement identique au résumé
GPU ; le second calcul retrouve les 144 tableaux de transfert, 48 tableaux
intacts et 432 contrastes, avec un écart maximal de 1,777 × 10⁻¹⁵. Les deux
vérifications réussissent avant la lecture des scores. Le [bilan](CROSS_TASK_INTERCHANGE_RESULTS.md)
publie toutes les directions, y compris celles défavorables au contenu général.
