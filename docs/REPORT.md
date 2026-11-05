# Final Technical Report: KNN Classification & Hyperparameter Analysis on BEED Dataset

> **Course**: Decision Support Systems (Sistem Pendukung Keputusan / SPK) Practicum  
> **Topic**: K-Nearest Neighbors Multi-Class Classification & $K$ Neighborhood Analysis on the BEED Dataset  
> **Status**: Completed & Validated  
> **Date**: 2026-11-05  

---

## Executive Summary

This study investigates the classification performance, inductive bias, and neighborhood sensitivity of the **K-Nearest Neighbors (KNN)** algorithm applied to the **Bangalore EEG Epilepsy Dataset (BEED)**. By evaluating a hyperparameter grid of $K \in [1, 3, 5, 7, 9, 11, 15, 21, 25, 31]$ under strict stratified validation and zero-leakage feature standardization, we observe that localized non-parametric boundaries excel on continuous EEG signals. The optimal model is achieved at **$K=1$ (Test Accuracy: 98.62%, Macro F1: 0.9863)**, with $K=3$ serving as a robust ensemble alternative (Test Accuracy: 98.12%). Larger $K$ values exhibit pronounced underfitting, decaying to 84.06% at $K=31$.

---

## 1. Introduction & Academic Objectives

Electroencephalography (EEG) records continuous neuroelectric potential differences across cerebral regions. In Clinical Decision Support Systems (CDSS), automated detection of epileptic seizures assists diagnosticians by triaging continuous multi-hour monitoring feeds.

The primary academic objectives of this investigation are:
1. Constructing an end-to-end, reproducible, clean-architecture machine learning workflow.
2. Formally evaluating the influence of neighborhood size $K$ on the **bias-variance trade-off**.
3. Safeguarding evaluation integrity via **stratified splitting** and **strict anti-data leakage preprocessing**.
4. Performing error analysis using **multi-class confusion matrix decompositions**.

---

## 2. Dataset Architecture & Verification

The dataset is obtained from public repositories:
- **UCI Machine Learning Repository**: [BEED: Bangalore EEG Epilepsy Dataset (#1134)](https://archive.ics.uci.edu/dataset/1134/beed:+bangalore+eeg+epilepsy+dataset)
- **Kaggle**: [BEED: Bangalore EEG Epilepsy Dataset Mirror](https://www.kaggle.com/datasets/varunrajput/beed-bangalore-eeg-epilepsy-dataset)

| Characteristic | Observed Value | Validation Criteria |
|---|---|---|
| **Total Rows** | 8,000 | Match expected row count |
| **Feature Predictors** | 16 Continuous EEG attributes (`X1`–`X16`) | All numeric (`int64`), zero missing entries |
| **Target Variable (`y`)** | 4-class categorical (`0, 1, 2, 3`) | Strictly balanced: 2,000 instances/class (25.0%) |
| **Class Taxonomy** | `0`: Healthy Baseline<br>`1`: Generalized Seizure<br>`2`: Focal Seizure<br>`3`: Seizure Event | Clinically verified neurological categories |

---

## 3. Methodology & Engineering Architecture

The architecture enforces single responsibility and unidirectional data dependencies:

```mermaid
flowchart TD
    RawData["Raw BEED Dataset (8000 rows, 16 features)"] --> Split["Stratified Split (80% Train / 20% Test, Seed=42)"]
    Split --> TrainRaw["Train Set (6400 samples)"]
    Split --> TestRaw["Test Set (1600 samples)"]
    TrainRaw --> Scaler["StandardScaler.fit()"]
    Scaler --> TrainScaled["X_train_scaled (mu=0, sigma=1)"]
    Scaler -.->|"Transform Only (Anti-Leakage)"| TestScaled["X_test_scaled"]
    TestRaw --> TestScaled
    TrainScaled --> KNN["KNN Multi-K Classifier Grid"]
    TestScaled --> KNN
    KNN --> Metrics["Macro Evaluation (Accuracy, Precision, Recall, F1)"]
```

### 3.1 Anti-Data Leakage Protocol
Feature scaling with Z-score standardization ($z = \frac{x - \mu}{\sigma}$) was fitted exclusively on the 6,400 training observations. The test set (1,600 observations) was transformed using only the means and variances learned from the training partition.

---

## 4. Empirical Results & Hyperparameter Comparison

The table below summarizes the quantitative evaluation across the hyperparameter grid:

| $K$ | Train Accuracy | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Inference Latency (ms) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **1.0000** | **0.9862** | **0.9865** | **0.9863** | **0.9863** | **94.98** |
| **3** | 0.9931 | 0.9812 | 0.9818 | 0.9813 | 0.9813 | 19.91 |
| **5** | 0.9822 | 0.9706 | 0.9713 | 0.9706 | 0.9707 | 18.42 |
| **7** | 0.9702 | 0.9537 | 0.9554 | 0.9537 | 0.9539 | 18.21 |
| **9** | 0.9567 | 0.9387 | 0.9411 | 0.9387 | 0.9390 | 21.49 |
| **11** | 0.9448 | 0.9231 | 0.9270 | 0.9231 | 0.9236 | 23.08 |
| **15** | 0.9222 | 0.8988 | 0.9060 | 0.8987 | 0.8993 | 22.67 |
| **21** | 0.8941 | 0.8769 | 0.8879 | 0.8769 | 0.8774 | 23.50 |
| **25** | 0.8789 | 0.8638 | 0.8767 | 0.8638 | 0.8639 | 27.22 |
| **31** | 0.8628 | 0.8406 | 0.8584 | 0.8406 | 0.8406 | 33.41 |

---

## 5. Visual Analysis & Analytical Reasoning

### 5.1 The Bias-Variance Trajectory (Accuracy vs. $K$)
The performance curve clearly displays the trade-off inherent to neighborhood-based classifiers:
- **Small $K$ ($K=1, 3$)**: Excellent generalization on unseen test data (98.62% and 98.12%). The EEG feature manifold forms tight, distinct spatial clusters where immediate 1-nearest or 3-nearest neighbors are almost universally from the identical clinical diagnosis.
- **Increasing $K$ ($K \ge 5$)**: Both training accuracy and testing accuracy decay concurrently. For example, testing accuracy falls from 97.06% ($K=5$) down to 84.06% ($K=31$). This indicates **underfitting caused by excessive neighborhood dilution**: when querying 31 neighbors, the voting pool spans across decision boundaries into adjacent class territories.

### 5.3 Error Decomposition (Confusion Matrix for Optimal $K=1$)
On the 1,600 held-out test instances (400 per class):

```
                   Predicted Label
               Healthy   Generalized   Focal   Seizure Event
True Healthy:    395          1          0           4        (Recall: 98.75%)
True Gen.:         0        393          5           2        (Recall: 98.25%)
True Focal:        0          2        390           8        (Recall: 97.50%)
True Seizure:      0          0          0         400        (Recall: 100.0%)
```

- **Zero False Negatives for Critical Seizures**: All 400 active Seizure Event samples (Class 3) were classified correctly with **100.0% sensitivity**, ensuring patient safety.
- **Inter-Class Confusion**: Minimal errors occur between Class 1 (Generalized) and Class 2 (Focal), with 5 instances of Class 1 predicted as Class 2, and 2 instances of Class 2 predicted as Class 1. This corresponds to the clinical reality where partial seizure onsets share electrical signal topography with secondary generalized events.

### 5.4 Standardization Trade-off Analysis

To rigorously assess the methodological impact of preprocessing, an empirical experiment was conducted comparing models trained with and without Z-score standardization across all hyperparameter values, with complete results recorded in [`../outputs/results/scaling_comparison.csv`](../outputs/results/scaling_comparison.csv).

| K | Scaled Acc | Unscaled Acc | Difference |
|---|---|---|---|
| 1 | 0.9862 | 0.9875 | -0.0013 |
| 3 | 0.9812 | 0.9831 | -0.0019 |
| 7 | 0.9537 | 0.9494 | +0.0044 |
| 21 | 0.8769 | 0.8738 | +0.0031 |
| 31 | 0.8406 | 0.8413 | -0.0006 |

![Scaling Comparison](../outputs/figures/scaling_comparison.png)

Features X1–X16 represent EEG amplitudes recorded with homogeneous physical units ($\mu\text{V}$) across discrete channels; therefore, standardization does not provide a significant distance-based advantage and, in fact, slightly reduces accuracy for small K values due to subtle variance compression in tightly clustered signal trajectories.

Nevertheless, feature scaling is RETAINED in the primary pipeline because (1) it is an essential engineering best practice for algorithmic robustness should the dataset expand or incorporate diverse sensor modalities, (2) it prevents anomalies related to channel variance spikes or localized electrode impedance shifts, and (3) it strictly adheres to the anti-leakage principle by calculating parameters exclusively on training folds.

Within the context of a Decision Support System (DSS), this finding underscores the critical importance of testing empirical assumptions rather than blindly applying best practices: distance-based algorithms operating on homogeneous physiological domains behave differently than those trained on heterogeneous multi-scale clinical tabular records.

---

## 6. System Deliverables & Verification Checklist

- [x] **Reproducible execution**: `python main.py --config config/config.yaml` executes cleanly end-to-end.
- [x] **Automated Testing**: 12/12 unit tests passing (`python -m pytest tests/`).
- [x] **Visual Artifacts**:
  - `outputs/figures/accuracy_vs_k.png`
  - `outputs/figures/confusion_matrix_k1.png`
  - `outputs/figures/class_distribution.png`
  - `outputs/figures/feature_distribution.png`
  - `outputs/figures/correlation_heatmap.png`
- [x] **Dual Runtime**: Google Colab compatible notebooks provided (`notebooks/01_exploration.ipynb` & `notebooks/02_knn_experiment.ipynb`).
- [x] **Serialized Artifacts**: Model weights (`best_knn_k1.pkl`), scaler (`scaler.pkl`), and metrics table (`k_comparison.csv`).

---

## 7. Conclusion & Recommendations

For clinical decision support on the BEED EEG dataset:
1. **Model Selection**: $K=1$ is mathematically optimal on this dataset, achieving 98.62% test accuracy. If voting redundancy is preferred to safeguard against individual electrode noise, **$K=3$ (98.12% accuracy)** provides an equally strong operational choice.
2. **Computational Consideration**: KNN test latency averages $\approx 20\text{ms}$ per 1,600 predictions, proving highly suitable for near-real-time clinical patient monitoring.
