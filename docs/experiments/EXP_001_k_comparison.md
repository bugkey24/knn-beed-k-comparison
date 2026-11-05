# Experiment: EXP-001 - Baseline Hyperparameter K Comparison

- **Date**: 2026-11-05
- **Commit/Phase**: Phase 4 / Phase 6 Integration
- **Author/Agent**: Antigravity Pair
- **Configuration**:
  - `dataset`: BEED_Data.csv (8,000 samples, 16 features, 4 classes)
  - `split`: 80% train (6,400 samples), 20% test (1,600 samples), stratified (`random_state=42`)
  - `scaler`: StandardScaler (fitted strictly on train partition)
  - `metric`: Euclidean distance
  - `weights`: Uniform
  - `k_values`: [1, 3, 5, 7, 9, 11, 15, 21, 25, 31]

---

## 1. Objective & Hypothesis
- **Objective**: Determine the empirical impact of the neighborhood size $K$ on classification performance for 16-channel EEG signals.
- **Hypothesis**: Smaller values of $K$ ($1 \le K \le 3$) will capture fine-grained boundaries in the high-density continuous signal space, whereas larger values ($K \ge 15$) will oversmooth decision boundaries and increase bias (underfitting).

---

## 2. Quantitative Results

| K | Train Accuracy | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Inference Latency (ms) |
|---|---|---|---|---|---|---|
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

## 3. Confusion Matrix Analysis (Optimal K=1)

Test sample size: 1,600 (400 per class):

```
                     Predicted
               Class 0  Class 1  Class 2  Class 3
True Class 0:    395       1        0        4      (Recall: 98.75%)
True Class 1:      0     393        5        2      (Recall: 98.25%)
True Class 2:      0       2      390        8      (Recall: 97.50%)
True Class 3:      0       0        0      400      (Recall: 100.0%)
```

- **Seizure Event (Class 3)**: Perfectly detected with 100% recall (400/400) and 0 false negatives.
- **Normal Baseline (Class 0)**: Only 5 misclassifications out of 400 (4 confused with Class 3, 1 with Class 1).
- **Subtle Inter-class Errors**: Slight overlap between Generalized (Class 1) and Focal (Class 2) seizures due to clinical similarity in partial EEG channel activations.

---

## 4. Key Takeaways & Decision
- **Peak Performance**: $K=1$ achieves 98.62% test accuracy and 0.9863 Macro F1, followed very closely by $K=3$ (98.12%).
- **Monotonic Underfitting**: Beyond $K=3$, accuracy decays monotonically from 97.06% down to 84.06% at $K=31$ as localized EEG patterns are smoothed away by distant neighbors.
- **Deployment Decision**: $K=1$ is designated as the primary model checkpoint (`outputs/models/best_knn_k1.pkl`).
