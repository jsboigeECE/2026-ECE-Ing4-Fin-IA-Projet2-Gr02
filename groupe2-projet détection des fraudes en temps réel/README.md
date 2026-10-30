
# Rapport : le rapport intitulé **"documentation_projet_détection_fraude"** est disponible au format PDF dans le dossier **"documentation en pdf"**.

# Détection de Fraude en Temps Réel

Projet ECE Ing4 Finance IA (Groupe 02) basé sur le dataset Kaggle Credit Card Fraud Detection.

## Objectif

Comparer plusieurs approches de détection de fraude sous contraintes de:
- qualité de détection (Precision, Recall, AUPRC),
- coût métier (FP x 1 + FN x 25),
- latence en streaming.

## Technologies

- Python 3.12
- scikit-learn (Isolation Forest, métriques)
- TensorFlow/Keras (Autoencoder, Focal Loss)
- PyTorch + PyTorch Geometric (GNN)
- imbalanced-learn (SMOTE)
- pandas, numpy, scipy, matplotlib


## Livrables principaux

- Phase 7.1: `notebooks/10_comprehensive_comparison.py`
- Phase 7.2: `notebooks/11_false_positive_cost_analysis.py`
- Rapport final : le rapport intitulé **"documentation_projet_détection_fraude"** est disponible au format PDF dans le dossier **"documentation en pdf"**.

Sorties clés :
- `results/comprehensive_comparison.json`
- `results/comprehensive_roc_comparison.png`
- `results/comprehensive_pr_comparison.png`
- `results/comprehensive_bar_comparison.png`
- `results/comprehensive_tradeoff.png`
- `results/fp_cost_analysis.json`
- `results/fp_cost_cartography.png`
- `results/fp_threshold_analysis.png`

## Exécution recommandée (une seule commande)

> **IMPORTANT — À lire avant toute exécution**
>
> Ce projet est situé dans un dossier à chemin long. Sur Windows, cela empêche `.venv` de fonctionner correctement (erreur WinError 206 lors de l'installation des dépendances).
>
> **Ne jamais exécuter les scripts avec `.venv\Scripts\python.exe`.**  
> **Utiliser exclusivement la méthode ci-dessous.**

### Méthode recommandée (automatique, zéro configuration)

Depuis la racine du projet, double-cliquer sur `run_all.bat` ou lancer dans un terminal:

```bat
run_all.bat
```

Ou en PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

Ce script fait tout automatiquement:
1. Crée un environnement Python fiable à chemin court (`C:\venvs\fraud-rt`) si absent
2. Installe toutes les dépendances (`requirements.txt`)
3. Exécute tous les scripts du projet dans l'ordre
4. Affiche `OK` ou `FAILED` pour chaque script
5. Sauvegarde les résultats dans `results/`

### Méthode manuelle (si besoin d'exécuter un script seul)

Ouvrir un terminal PowerShell dans le dossier du projet, puis:

```powershell
$py = "C:\venvs\fraud-rt\Scripts\python.exe"
& $py check_dataset.py
& $py analyze_imbalance.py
& $py run_data_loader.py
& $py run_isolation_forest.py
& $py run_isolation_forest_evaluation.py
& $py run_isolation_forest_threshold_analysis.py
& $py notebooks/05_autoencoder_detection.py
& $py notebooks/06_method_comparison.py
& $py notebooks/07_class_weights_impact.py
& $py notebooks/08_focal_loss_experiment.py
& $py notebooks/09_gnn_detection.py
& $py notebooks/10_comprehensive_comparison.py
& $py notebooks/11_false_positive_cost_analysis.py
```

### Règle absolue

| ❌ Ne pas faire | ✅ À faire |
|---|---|
| `.venv\Scripts\python.exe script.py` | `C:\venvs\fraud-rt\Scripts\python.exe script.py` |
| `python script.py` (Python système) | `$py = "C:\venvs\fraud-rt\Scripts\python.exe"` puis `& $py script.py` |
| Double-clic sur un `.py` | Toujours passer par `run_all.bat` ou `$py` |

## Résumé de statut

- Le code fonctionne sur un environnement Python complet.
- Les scripts de comparaison globale (Phase 7.1) et de coût des faux positifs (Phase 7.2) passent avec succès.
- Le projet inclut bien la simulation temps réel (streaming), pas un déploiement production réel.
