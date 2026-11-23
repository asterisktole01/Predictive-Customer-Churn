# Predictive Customer Churn - Telco

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

Binary classification of customer churn on the IBM Telco Customer Churn dataset.
Compares Logistic Regression, Random Forest, and XGBoost under class-weighted
vs. unweighted training, evaluated with stratified 5-fold cross-validation and
a held-out test set. Primary metric: **ROC-AUC** (target > 0.80).

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Key Findings](#key-findings)


---

## Overview

Customer churn — a subscriber leaving the service — is one of the highest-value
prediction problems in telecom. This project builds and compares three classifiers
for churn prediction, with explicit attention to **class imbalance** (~26.5% churners)
and **threshold-independent evaluation** (ROC-AUC).

**Questions this notebook answers:**

1. How much does class weighting improve recall without hurting ROC-AUC?
2. Does XGBoost meaningfully beat a logistic-regression baseline on tabular data?
3. Which customer attributes drive churn most strongly?

---

## Dataset

- **Source:** [IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) (public, loaded directly from the URL)
- **Size:** 7,043 customers × 21 columns
- **Target:** `Churn` (binary: Yes/No → 1/0), **26.5% positive**
- **Features:** demographics (`gender`, `SeniorCitizen`, `Partner`, `Dependents`),
  account info (`tenure`, `Contract`, `PaymentMethod`, `PaperlessBilling`),
  services (`PhoneService`, `InternetService`, `OnlineSecurity`, …),
  billing (`MonthlyCharges`, `TotalCharges`)

**Data-quality notes handled:**

- `TotalCharges` is delivered as a string with 11 blank values (customers with
  `tenure = 0`). Coerced to numeric; blanks filled to 0 (new customers, not missing-at-random).
- `customerID` is an identifier and is dropped before modeling.
- `No internet service` / `No phone service` are structural states, not missing values.
- No numeric outliers flagged by IQR fences.

---

## Approach

| Step | What was done |
|---|---|
| **Split** | 80/20 stratified train/test, `random_state=42` |
| **Preprocessing** | `ColumnTransformer`: median-impute + standard-scale numeric features; most-frequent-impute + `OneHotEncoder(drop='if_binary', handle_unknown='ignore')` for categoricals — all inside a `Pipeline` so it is fit per fold |
| **Imbalance handling** | Class weighting: `class_weight='balanced'` (LR), `'balanced_subsample'` (RF), `scale_pos_weight≈2.77` (XGB). Chosen over SMOTE because the 2.8:1 ratio is moderate and ROC-AUC is threshold-independent — documented in the notebook |
| **Models** | Logistic Regression (baseline), Random Forest (300 trees), XGBoost (300 trees, lr=0.05, depth=4) — each in **both** weighted and unweighted variants |
| **Cross-validation** | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, same folds for all 6 pipelines |
| **Metrics** | ROC-AUC (primary), F1, precision, recall |
| **Evaluation** | ROC curves, confusion matrices, top-10 feature importances |

---

## Results


### Cross-validation (5-fold, mean ± std)

| Model | Variant | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | default | [0.8xx ± 0.0xx] | [0.xx] | [0.xx] | [0.xx] |
| Logistic Regression | weighted | [0.8xx ± 0.0xx] | [0.xx] | [0.xx] | [0.xx] |
| Random Forest | default | [0.8xx ± 0.0xx] | [0.xx] | [0.xx] | [0.xx] |
| Random Forest | weighted | [0.8xx ± 0.0xx] | [0.xx] | [0.xx] | [0.xx] |
| XGBoost | default | [0.8xx ± 0.0xx] | [0.xx] | [0.xx] | [0.xx] |
| XGBoost | weighted | [0.8xx ± 0.0xx] | [0.xx] | [0.xx] | [0.xx] |

### Held-out test set (primary variant: [XGBoost | weighted])

| Metric | Value |
|---|---|
| ROC-AUC | [0.8xx] |
| F1 | [0.xx] |
| Precision | [0.xx] |
| Recall | [0.xx] |
| Accuracy | [0.xx] |

**Target met:** ROC-AUC [above / not yet above] the 0.80 threshold.

### Figures

| Figure | Description |
|---|---|
| `figures/roc_curves.png` | ROC curves for all models (weighted variant) |
| `figures/confusion_matrix_<model>.png` | Confusion matrix for the selected model |
| `figures/feature_importance_<model>.png` | Top-10 feature importances (RF / XGB) |

---

## Project Structure

Predictive Customer Churn/
├── README.md
├── notebook/
│ └── Predictive Customer Churn Classifier.ipynb
├── figures/
│ ├── roc_curves.png
│ ├── confusion_matrix_xgboost.png
│ └── feature_importance_xgboost.png
├── requirements.txt