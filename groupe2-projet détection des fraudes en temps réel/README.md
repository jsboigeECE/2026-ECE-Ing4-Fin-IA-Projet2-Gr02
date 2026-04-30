# C.7 - Détection de Fraude en Temps Réel

**Difficulté:** 3/5 | **Domaine:** Machine Learning

## 📋 Description

La détection de fraude est un problème de **classification fortement déséquilibré** (< 0.1% de fraudes) avec des contraintes de latence. Ce projet explore et compare des approches complémentaires :

- **Isolation Forest** : Détection d'anomalies sans supervision
- **Autoencoders** : Apprentissage de représentations normales
- **Graph Neural Networks (GNN)** : Détection en exploitant les graphes de transactions

La gestion du déséquilibre (SMOTE, class weights, focal loss) et l'évaluation avec des métriques adaptées (AUPRC, coût financier) sont centrales.

## 🎯 Objectifs Graduées

### Minimum 🟢
- [x] Entrainement : Isolation Forest + Autoencoder sur le dataset Kaggle Credit Card Fraud
- [x] Évaluation : Métriques AUPRC et analyse de base
- [x] Dataset : Kaggle Credit Card Fraud structuré et testé

### Bon 🟡
- [x] Comparaison de 3+ méthodes (Forest, AE, SMOTE, class weights, focal loss)
- [x] Gestion du déséquilibre : implémentation et comparaison
- [x] Analyse des faux positifs par coût financier
- [x] Visualisations claires des résultats

### Excellent 🔴
- [x] GNN sur graphe de transactions (PyTorch Geometric)
- [x] Détection **en streaming** (simulation temps réel)
- [x] Pipeline complet avec seuils adaptatifs
- [x] Rapport détaillé : compromis métrique/coût/latence

## 📚 Ressources de Référence

### Notebooks Recommandés
| Notebook | Sujet |
|----------|-------|
| [QC-Py-24](../groupe-02-Graphe%20de%20connaissances%20pour%20la%20gestion%20des%20risques%20financiers/notebooks) | Autoencoders et détection d'anomalies |
| [ML-4](../groupe-02-Graphe%20de%20connaissances%20pour%20la%20gestion%20des%20risques%20financiers/notebooks) | Évaluation de modèles (métriques AUPRC, courbes PR) |

### Références Externes
- **[Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** - Dataset de référence (284K transactions, 0.17% fraudes)
- **[PyOD](https://pyod.readthedocs.io/)** - 40+ algorithmes de détection d'anomalies
- **[PyOD Fraud Detection Example](https://pyod.readthedocs.io/en/latest/example.html)** - Tutoriel complet
- **[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)** - GNN pour graphes de transactions
- **[imbalanced-learn (SMOTE)](https://imbalanced-learn.org/)** - Gestion du déséquilibre

## 🚀 Installation & Configuration

### Dépendances Python
```bash
pip install -r requirements.txt
```

### Structure du Projet
```
Projet 2/
├── README.md                 # Ce fichier
├── requirements.txt          # Dépendances Python
├── dataset/
│   └── creditcard.csv       # Dataset Kaggle (à télécharger/utiliser)
├── notebooks/               # Notebooks Jupyter (à créer)
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_isolation_forest.ipynb
│   ├── 04_autoencoders.ipynb
│   └── 05_gnn.ipynb
├── src/                     # Code réutilisable
│   ├── models/              # Définitions des modèles
│   ├── preprocessing/       # Pipelines de prétraitement
│   └── evaluation/          # Métriques et visualisations
├── results/                 # Résultats, modèles sauvegardés
└── reports/                 # Visualisations, rapports
```

## 📖 Plan de Réalisation

### Phase 1 : Préparation (2-3 jours)
- [x] Download/Charger le dataset Kaggle
- [x] Exploration des données (EDA)
- [x] Analyse du déséquilibre de classes
- [x] Preprocessing et normalisation
- [x] Train/test split

### Phase 2 : Modèles de Base (3-4 jours)
- [x] **Isolation Forest** : entraînement et tuning
- [x] **Autoencoder** : architecture simple (reconstruction anomalies)
- [x] Comparaison préliminaire
- [x] Analyse des faux positifs

### Phase 3 : Gestion du Déséquilibre (2-3 jours)
- [x] Implémentation SMOTE
- [x] Class weights dans les modèles
- [x] Focal loss pour autoencoders
- [x] Benchmarking des approches

### Phase 4 : Modèles Avancés (4-5 jours)  
- [x] Construction du graphe de transactions (nœuds = transactions, arêtes = similarité temporelle/montant)
- [x] GNN (Graph Convolutional Network, PyTorch Geometric)
- [x] Détection anomalies sur le graphe
- [x] Comparaison avec Phase 2

### Phase 5 : Temps Réel & Intégration (3-4 jours)
- [x] Simulation streaming (mini-batchs séquentiels)
- [x] Adaptation seuils en temps réel
- [x] Pipeline de décision avec latence
- [x] Tests de performance

### Phase 6 : Rapport & Présentation (2 jours)
- [x] Visualisations : courbes PR, confusion matrices, réseau de fraude
- [x] Analyse ROI : coût faux positifs vs coût faux négatifs
- [x] Recommandations de déploiement
- [x] Présentation des résultats

## 🔧 Utilisation

### Entraîner un modèle
```python
from src.models import IsolationForestFraudDetector
from src.preprocessing import preprocess_creditcard

# Charger et préparer les données
X_train, X_test, y_test = preprocess_creditcard('dataset/creditcard.csv')

# Entraîner
detector = IsolationForestFraudDetector(contamination=0.001)
detector.fit(X_train)

# Évaluer
metrics = detector.evaluate(X_test, y_test)
print(metrics)
```

### Lancer les notebooks
```bash
jupyter notebook notebooks/
```

## 📊 Métriques d'Évaluation

- **AUPRC** (Area Under Precision-Recall Curve) - meilleure métrique pour déséquilibre extrême
- **ROC-AUC** - pour comparaison
- **Confusion Matrix** - par type de fraude
- **Cost Matrix** : 
  - FP (signaler légitime) : perte client
  - FN (miss fraude) : perte financière
  - TP (détecter fraude réelle) : prévention

## 💡 Tips & Bonnes Pratiques

1. **Toujours réfléchir au métier** : un faux positif peut perdre un client
2. **AUPRC >> accuracy** sur données déséquilibrées
3. **Validation cross-fold** (stratifiée)
4. **Tuning via F1 ou F2** si coûts pas exactement connus
5. **Monitoring en production** : ajuster seuils selon feedback
6. **Graphes de fraude** : même clients/commerçants = structure exploitable

## ❓ Questions Fréquentes

**Q: Où télécharger le dataset?**  
A: Kaggle (compte requis) : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**Q: Pourquoi 3+ modèles?**  
A: Ensemble methods performent mieux + comparaison juste apporte insights

**Q: GNN obligatoire?**  
A: Non - objectif "Excellent" seulement (très advanced)

**Q: Coût du faux positif vs faux négatif?**  
A: À définir (ex: FP = -5€, FN = -200€)

---

**Dernier update:** Mars 2026 (livrables complétés) | **Contact:** Manuela
