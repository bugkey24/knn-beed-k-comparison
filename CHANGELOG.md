# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-05

### Added
- **Core Pipeline Implementation (`src/`)**:
  - `config.py`: Dataclass and YAML parser loading runtime parameters.
  - `data/`: Ingestion (`loader.py`), schema assertion (`validator.py`), and stratified splitting (`splitter.py`).
  - `preprocessing/`: Zero-leakage `FeatureScaler` wrapper (`scaler.py`).
  - `models/`: Encapsulated `KNNClassifierModel` (`knn.py`) and multi-K execution iterator (`trainer.py`).
  - `evaluation/`: Multi-class metrics calculator (`metrics.py`) and comparative tabular aggregator (`comparator.py`).
  - `visualization/`: EDA plotting (`eda.py`) and results evaluation charts (`results.py`).
  - `pipeline.py` & `main.py`: Fully automated pipeline orchestrator and CLI entrypoint.
- **Testing Suite (`tests/`)**:
  - 12 comprehensive unit and integration smoke tests covering data validation, anti-leakage scaling, KNN inference, metrics, and pipeline execution. 100% test pass rate.
- **Dual Runtime Notebooks (`notebooks/`)**:
  - `01_exploration.ipynb`: Interactive Exploratory Data Analysis.
  - `02_knn_experiment.ipynb`: Interactive Colab-ready pipeline orchestrator with inline figures.
- **Documentation & Reporting (`docs/`)**:
  - `REPORT.md`: Comprehensive final academic report detailing methodology, quantitative tables, bias-variance analysis, clinical confusion matrix findings, and Section 5.4 Standardization Trade-off Analysis.
  - `experiments/EXP_001_k_comparison.md`: Formal log of baseline hyperparameter evaluation.
  - Official dataset repository citation links (UCI Machine Learning Repository #1134 and Kaggle Mirror).
  - Standardized Mermaid diagrams across `ARCHITECTURE.md`, `ROADMAP.md`, and `METHODOLOGY.md`.
- **Root Documentation (`README.md`)**:
  - Fully relative, portable document navigation links.
  - Embedded visual analytics: class distribution, channel amplitude spread, correlation matrix, accuracy vs $K$, and confusion matrix heatmap.
  - Dual runtime execution guide (local CLI and Google Colab).
- **Generated Outputs (`outputs/`)**:
  - Results CSVs: `outputs/results/k_comparison.csv` and `outputs/results/scaling_comparison.csv`.
  - Production models: `outputs/models/best_knn_k1.pkl` and `outputs/models/scaler.pkl`.
  - High-resolution figures: `accuracy_vs_k.png`, `confusion_matrix_k1.png`, `class_distribution.png`, `feature_distribution.png`, `correlation_heatmap.png`, and `scaling_comparison.png`.

### Removed
- Retired initial monolithic `PID.md` following clean decomposition into modular documentation files.
