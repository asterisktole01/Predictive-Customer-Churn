# Predictive Customer Churn - Telco

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

An end-to-end machine learning pipeline designed to predict customer churn and identify the key drivers of customer attrition. 

Rather than just optimizing for raw accuracy (which fails on imbalanced data), this project focuses on **catching actual churners** to maximize retention ROI. We compare Logistic Regression, Random Forest, and XGBoost under class-weighted vs. unweighted training, evaluated with stratified 5-fold cross-validation. Primary metric: **ROC-AUC** (target > 0.80).

---

## Table of Contents

- [Business Context](#business-context)
- [Key Findings & Business Recommendations](#key-findings--business-recommendations)
- [Dataset](#dataset)
- [Approach & Methodology](#approach--methodology)
- [Results](#results)
- [Project Structure](#project-structure)

---

## Business Context

Customer churn—a subscriber leaving the service—is one of the highest-cost problems in the telecom industry. Acquiring a new customer costs 5x to 7x more than retaining an existing one. 

However, churn is highly imbalanced: only **26.5%** of customers in this dataset actually churn. A naive model can achieve 73.5% "accuracy" simply by predicting that *everyone* will stay, but it would catch zero churners. This project solves that problem by applying **class weighting** to force the models to prioritize identifying at-risk customers.

**Business Questions Answered:**
1. How can we maximize the number of churners we catch (Recall) without overwhelming the retention team with too many false alarms?
2. Which specific customer behaviors and service combinations are driving the most revenue loss?
3. What immediate policy changes can the retention team implement to stop the bleeding?

---
## Key Findings & Business Recommendations
Class weighting dramatically improves Retention ROI. While ROC-AUC stayed flat (~0.84), class weighting boosted our Recall from 54% to 80%. We traded a higher number of "False Alarms" for fewer "Missed Churners."

Linear Models are highly competitive here. Logistic Regression slightly beat XGBoost. This means the data is largely linearly separable, and a simpler, easily explainable model is the safest choice for production deployment.

**Month-to-Month:** Feature importance revealed that Month-to-month contracts are by far the biggest predictor of churn. Customers without long-term commitment are massive flight risks.

**Service Gaps Drive Attrition:** Customers with Fiber Optic internet who lack Online Security or Tech Support are highly likely to leave. This suggests a product-market fit issue or frustration with premium tiers.

**Actionable Next Steps for the Business:**

**Incentivize Annual Contracts:** Launch a targeted campaign offering a 10% discount or a free streaming upgrade to month-to-month customers who upgrade to a 1-year contract.

**Bundle Support Services:** Proactively offer free 3-month trials of Online Security and Tech Support to all new Fiber Optic customers to increase product stickiness.


## Dataset

- **Source:** [IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
- **Size:** 7,043 customers × 21 columns
- **Target:** `Churn` (Binary: Yes/No → 1/0), 26.5% positive class.
- **Data Cleaning:** 
  - `TotalCharges` was delivered as a string with 11 blank values (new customers with `tenure = 0`). It was coerced to numeric and blanks were filled to 0.
  - `customerID` was dropped to prevent overfitting. 
  - Structural states (like `No internet service`) were preserved rather than treated as missing data.

---

## Approach & Methodology

| Step | Technical Implementation |
|---|---|
| **Data Split** | 80/20 stratified train/test split to preserve the 26.5% churn ratio. |
| **Preprocessing** | `ColumnTransformer` + `Pipeline`: Standard-scaling for numerics; `OneHotEncoder` for categoricals. Fitted strictly within CV folds to prevent data leakage. |
| **Imbalance Handling** | Class weighting (`balanced` for LR/RF, `scale_pos_weight≈2.77` for XGB) rather than SMOTE. Chosen to directly optimize threshold-independent metrics (ROC-AUC) and reflect the business cost of missing a churner. |
| **Models** | Logistic Regression (baseline), Random Forest, and XGBoost (300 trees each). Both weighted and unweighted variants were trained. |
| **Evaluation** | Stratified 5-fold Cross-Validation, ROC-AUC, F1, Precision, Recall, Confusion Matrices, and Feature Importances. |

---

## Results

*Target Metric: ROC-AUC > 0.80 — Achieved by all models.*

### Cross-Validation (5-Fold Mean)

| Model Variant | ROC-AUC | F1-Score | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| **LogReg \| weighted** | **0.8459** | 0.629 | 0.517 | **0.801** |
| LogReg \| default | 0.8461 | 0.592 | 0.652 | 0.544 |
| XGBoost \| default | 0.8445 | 0.580 | 0.656 | 0.520 |
| XGBoost \| weighted | 0.8443 | 0.631 | 0.532 | 0.777 |
| RandomForest \| weighted | 0.8239 | 0.542 | 0.634 | 0.473 |
| RandomForest \| default | 0.8231 | 0.548 | 0.634 | 0.483 |

### Held-out Test Set (Primary Model: Logistic Regression | Weighted)
On unseen data, our champion model achieved an **ROC-AUC of 0.8415**, proving it generalizes well without overfitting.

### The Business Impact of Class Weighting (Confusion Matrix)
To understand the ROI of this model, we look at the Confusion Matrix on the test set (1,409 customers):
* **Default Model:** Missed **164** churners (False Negatives). 
* **Weighted Model:** Missed only **81** churners. 
By applying class weights, the model successfully identified **83 additional at-risk customers**. While this caused False Positives to rise (from 110 to 288), the cost of sending a retention offer to a loyal customer is drastically lower than the lost revenue of a churned customer.

### Visualizing the Results

**ROC Curves**

![ROC Curves](figures/roc_curve.png)

**Confusion Matrices (Default vs Weighted Logistic Regression)**

![Confusion Matrices](figures/confusion_matrix_logreg.png)

**Top 10 Churn Drivers (XGBoost Feature Importance)**

![Feature Importance](figures/feature_importance_xgboost.png)

**Top Features Extracted by XGBoost:**
1. `Contract_Month-to-month` (Dominant driver)
2. `InternetService_Fiber optic`
3. `OnlineSecurity_No`
4. `TechSupport_No`
5. `InternetService_No`
6. `StreamingMovies_Yes`
7. `Contract_Two year`
8. `InternetService_DSL`
9. `Contract_One year`
10. `PaymentMethod_Electronic check`

**Note on Methodology:** *While Logistic Regression was our best-performing model for classification (highest ROC-AUC), we used XGBoost to extract Feature Importances. Logistic Regression coefficients are highly sensitive to multicollinearity (specifically the 'No internet service' dummy variables), which creates misleading coefficients. XGBoost's tree-based architecture handles these structural states cleanly, providing a much more reliable ranking of true business drivers.*

---

## Project Structure

```text
Predictive Customer Churn/
├── README.md
├── notebook/
│   └── Predictive Customer Churn Classifier.ipynb
├── figures/
│   ├── roc_curve.png
│   ├── confusion_matrix_logreg.png
│   └── feature_importance_xgboost.png
├── requirements.txt
└── LICENSE
