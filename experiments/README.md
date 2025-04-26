# Détection de Fausses Nouvelles avec GNN

Ce projet implémente et compare différentes architectures de Graph Neural Networks (GNN) pour la détection de fausses nouvelles en utilisant le dataset LIAR.

## Dataset

Le dataset LIAR contient 12,791 déclarations courtes étiquetées manuellement provenant de PolitiFact. Chaque déclaration est évaluée selon 6 degrés de véracité :
- pants-fire (mensonge flagrant)
- false (faux)
- barely-true (à peine vrai)
- half-true (à moitié vrai)
- mostly-true (plutôt vrai)
- true (vrai)

Pour chaque déclaration, nous avons :
- Le texte de la déclaration
- L'orateur (speaker)
- Le contexte
- Des métadonnées sur l'orateur (parti politique, poste, état)
- L'historique des évaluations précédentes de l'orateur

## Approche

Nous transformons le problème en une tâche d'apprentissage sur graphe où :
1. Chaque déclaration est un nœud
2. Les arêtes sont créées entre les déclarations similaires (k plus proches voisins)
3. Les caractéristiques des nœuds combinent :
   - Features de contenu : TF-IDF sur le texte (300 dimensions)
   - Features sociales : Métadonnées de l'orateur encodées (10 dimensions)

## Modèles Implémentés

1. **GCN (Graph Convolutional Network)**
   - Architecture de base pour l'apprentissage sur graphe
   - Combine les features de contenu et sociales

2. **GAT (Graph Attention Network)**
   - Utilise des mécanismes d'attention pour pondérer les voisins
   - Meilleure capacité à identifier les connexions importantes

3. **DHGAT (Dual-Head Graph Attention Network)**
   - Architecture proposée spécifiquement pour ce problème
   - Traite séparément les features de contenu et sociales
   - Utilise deux têtes d'attention distinctes

## Prétraitement

1. **Features de Contenu**
   - TF-IDF sur les déclarations
   - Normalisation avec StandardScaler

2. **Features Sociales**
   - Encodage des variables catégorielles
   - Normalisation des compteurs historiques
   - Gestion des valeurs manquantes

3. **Construction du Graphe**
   - k-plus proches voisins (k=5)
   - Distance basée sur les features combinées

## Résultats

Les performances des modèles (accuracy sur le jeu de test) :
- GCN : 13.89%
- GAT : 18.31%
- DHGAT : 21.39%

Le DHGAT montre les meilleures performances, probablement grâce à sa capacité à traiter séparément et de manière plus appropriée les différents types de features.

## Structure du Projet

```
.
├── data/
│   └── liar_dataset/       # Dataset LIAR brut
├── models/
│   ├── gcn.py             # Implémentation GCN
│   ├── gat.py             # Implémentation GAT
│   └── dhgat.py           # Implémentation DHGAT
├── utils/
│   ├── data.py            # Prétraitement des données
│   ├── visualize.py       # Visualisation des résultats
│   └── util.py            # Fonctions utilitaires
├── results/               # Résultats et visualisations
├── config.py             # Configuration des modèles
├── train.py              # Script d'entraînement
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
python train.py
```

Les résultats et visualisations seront sauvegardés dans le dossier `results/`. 