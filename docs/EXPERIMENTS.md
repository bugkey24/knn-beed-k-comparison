# Experimentation Protocol & Guidelines

> **Purpose**: Standardized procedure for executing, recording, and evaluating experiments with K-Nearest Neighbors on the BEED dataset.

---

## 1. Experimentation Principles

1. **Strict Reproducibility**:
   - Seed must remain `42` across all runs unless explicitly running seed-sensitivity benchmarks.
   - Train/test split ratio is fixed at 80:20 with stratification (`stratify=y`).
2. **Anti-Data Leakage**:
   - Feature scalers must only fit on the training portion.
   - Test data serves exclusively for final model evaluation per $K$.
3. **Hyperparameter Grid**:
   - Default $K$ grid: `[1, 3, 5, 7, 9, 11, 15, 21, 25, 31]`.
   - Secondary experiments (bonus) may test distance metrics (`euclidean`, `manhattan`, `minkowski`) and weights (`uniform` vs `distance`).

---

## 2. Metric Reporting Standard

Every experiment evaluates:
- **Accuracy**: Overall fraction of correct predictions.
- **Macro Precision**: Average precision across all 4 classes without class-weight bias.
- **Macro Recall**: Average sensitivity across all 4 classes.
- **Macro F1-Score**: Harmonic mean of macro precision and recall.
- **Train vs Test Accuracy**: Explicitly tracks gap to identify overfitting regimes (small $K$) and underfitting regimes (large $K$).
- **Confusion Matrix**: Required for the best-performing $K$ checkpoint.

---

## 3. Experiment Log Template

Every formal run should be documented as a markdown file inside `docs/experiments/` adhering to the template below.
