# Plan de Realisation - Avancement Realise (jusqu'a aujourd'hui)

## Objectif (rappel)
Mettre en place une base de detection de fraude sur le dataset carte bancaire, puis mesurer sa performance avec Isolation Forest.

---

## Ce qui a ete fait

### 0) Mise en place du projet
- README cree et structure projet posee.
- Dependances definies dans requirements.
- Arborescence source creee (preprocessing, models, streaming).

Fichiers principaux:
- README.md
- requirements.txt
- PLAN_DE_REALISATION.md

### 1) Chargement et verification des donnees (Phase 1.1)
- Verification du fichier dataset/creditcard.csv.
- Controle taille, colonnes, types, valeurs manquantes.

Fichier:
- check_dataset.py

Resultat:
- Dataset charge et exploitable pour la suite.

### 2) Pretraitement (Phase 1.2)
- Separation des variables: X (features) et y (classe fraude).
- Normalisation de Amount et Time avec StandardScaler.
- Split train/test en 70/30 avec stratification sur y.
- Sauvegarde des donnees preparees et du scaler.

Fichiers:
- src/preprocessing/data_loader.py
- run_preprocessing.py
- simple_preprocessing.py
- verify_preprocessing.py

Artefacts produits:
- results/preprocessed_data.npz
- results/scaler.pkl

Resultat:
- Pipeline de preparation stable et reutilisable.

### 3) Analyse du desequilibre (Phase 1.3)
- Calcul du taux de fraude et du ratio classes.
- Verification de la preservation du ratio sur train/test.
- Analyse descriptive simple (montants, temps).

Fichier:
- analyze_imbalance.py

Resultat:
- Desequilibre confirme (~0.17% fraude).

### 4) Entrainement Isolation Forest (Phase 2.1)
- Entrainement du modele sur transactions normales du train.
- Generation des predictions et scores d'anomalie sur le test.
- Sauvegarde du modele et des sorties test.

Fichiers:
- src/models/isolation_forest.py
- run_isolation_forest.py

Artefacts produits:
- results/isolation_forest_model.pkl
- results/isolation_forest_test_outputs.npz

Resultat:
- Baseline de detection d'anomalies disponible.

### 5) Evaluation du modele sur test (Phase 2.2)
- Calcul des metriques: precision, recall, AUPRC.
- Trace de la courbe Precision-Recall.
- Export des metriques et du graphe.

Fichier:
- run_isolation_forest_evaluation.py

Artefacts produits:
- results/isolation_forest_metrics.json
- results/isolation_forest_pr_curve.png

Resultats obtenus:
- precision: 0.2081
- recall: 0.2432
- auprc: 0.1181

---

## Resume rapide
Le projet a deja couvert:
- toute la preparation des donnees (Phase 1 complete),
- l'entrainement du premier modele (Phase 2.1),
- la premiere evaluation quantitative et visuelle (Phase 2.2).
│       └── monitoring.py            (Étape 6.3)
└── notebooks/
    ├── 01_data_exploration.py       (Étape 1.1)
    ├── 02_class_imbalance_analysis.py
    ├── 03_isolation_forest_evaluation.py
    ├── 04_cost_analysis.py
    ├── 05_autoencoder_detection.py
    ├── 06_method_comparison.py
    ├── 07_class_weights_impact.py
    ├── 08_focal_loss_experiment.py
    ├── 09_gnn_detection.py
    ├── 10_comprehensive_comparison.py
    └── 11_false_positive_cost_analysis.py
```

---

## 🎓 Ressources à Consulter

| Phase | Ressource | Lien |
|-------|-----------|------|
| 1.1-1.3 | Dataset Kaggle | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| 2 | PyOD Documentation | https://pyod.readthedocs.io/ |
| 3 | QC-Py-24 Notebook | Autoencoders et détection d'anomalies |
| 4 | imbalanced-learn | https://imbalanced-learn.org/ |
| 5 | PyTorch Geometric | https://pytorch-geometric.readthedocs.io/ |
| Tous | ML-4 Notebook | Évaluation de modèles et métriques AUPRC |

---

## ⏰ Chronologie Estimée

| Semaine | Tâches |
|---------|--------|
| **Semaine 1** | Phases 1 + 2.1-2.2 (Données + Isolation Forest) |
| **Semaine 2** | Phase 2.3 + Phase 3 (Coût + Autoencoders) |
| **Semaine 3** | Phase 4 (Gestion déséquilibre) + Phase 3.3 (Comparaison) |
| **Semaine 4** | Phase 5 + 6 + 7 (GNN, Streaming, Évaluation finale) |

---

## ✅ Checklist de Validation

- [ ] Environnement configuré et dépendances installées
- [ ] Dataset téléchargé et exploré
- [ ] Isolation Forest entraîné et évalué
- [ ] Autoencoder entraîné et évalué
- [ ] Déséquilibre géré (SMOTE + Class Weights)
- [x] GNN implémenté (optionnel Excellent)
- [ ] Pipeline streaming fonctionnel (optionnel Excellent)
- [ ] Rapport final avec comparaisons
- [ ] Toutes les visualisations générées

---

## 💡 Conseils de Développement

1. **Progressif** : Commencez par Isolation Forest avant les Autoencoders
2. **Test** : Validez chaque étape avant de passer à la suivante
3. **Documentation** : Documentez les décisions et résultats au fur et à mesure
4. **Performance** : Mesurer les latences pour le streaming dès maintenant
5. **Notebooks** : Utilisez les notebooks pour l'exploration, les scripts pour la production

---

**Bon courage ! 🚀**
