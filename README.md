# ML-Main_Project
#  Credit Card Default Prediction (UCI Dataset)

A machine learning project to predict whether a credit card client will default on their next payment using financial and demographic data. The project follows a complete ML workflow: data cleaning, feature engineering, model tuning, evaluation, and deployment using pipelines.

---

##  Dataset Overview

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- **Instances**: 30,000
- **Features**: 23 (including demographic & financial information)
- **Target Variable**: `DEFAULT` (1 = default, 0 = no default)

---

##  Objective

To build an accurate and interpretable classification model that predicts the likelihood of a credit card customer defaulting on the next payment, based on historical data.

---

##  Tools & Libraries

- Python (Jupyter Notebook)
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `joblib`

---

##  Project Workflow

### 1.  Exploratory Data Analysis (EDA)
- Data types and null value checks
- Summary statistics and target balance
- Visualizations: distributions, correlations, default rates

### 2.  Data Cleaning
- Removed invalid/unknown categories
- Checked outliers and value ranges
- Ensured correct data types

### 3.  Feature Engineering
- Separated features (`X`) and target (`y`)
- Applied `SelectKBest` (ANOVA F-test) to evaluate feature relevance
- Retained **all features** (`k='all'`) based on scores

### 4.  Train-Test Split
- Performed an 80/20 split with stratification on the `DEFAULT` column

### 5. Best Model
- **Model:** GradientBoostingClassifier
- **Accuracy:** ~82% (after tuning)
- **ROC AUC:** ~0.77

### 6.  Hyperparameter Tuning
- Used `GridSearchCV` with 5-fold cross-validation
- Tuned parameters like `n_estimators`, `learning_rate`, `max_depth`, etc.
- Chose the best model based on cross-validated accuracy

### 7.  ML Pipeline
Built a `Pipeline` that handles:
- **Missing values**: `SimpleImputer(strategy='mean')`
- **Scaling**: `StandardScaler()`
- **Model**: `GradientBoostingClassifier()`

### 8.  Model Evaluation
Evaluated the tuned model on the test set using:
- Accuracy score
- Confusion matrix (visualized with `seaborn`)
- Classification report (precision, recall, F1)
- ROC Curve and AUC score

### 9. Model Saving & Deployment
- Saved the final pipeline with `joblib`
- Reloaded pipeline and predicted on a sample row
- Confirmed the pipeline handles end-to-end prediction

---

##  Results Summary

| Metric           | Value (Tuned Model) |
|------------------|---------------------|
| Accuracy         |  High (evaluated on test set) |
| ROC-AUC Score    |  Excellent separation |
| Default Handling |  Binary classification (0 = No Default, 1 = Default) |
| Pipeline Tested  |  End-to-end prediction on new row |

---


