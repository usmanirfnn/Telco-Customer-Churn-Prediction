# Telco Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn using the Telco Customer Churn dataset.

The project covers exploratory data analysis, data cleaning, feature engineering, preprocessing, class-imbalance handling, model comparison, cross-validation, hyperparameter tuning, model evaluation, feature importance, and model serialization.

---

## Overview

Customer churn is a significant business problem for subscription-based companies. The objective of this project is to identify customers who are more likely to churn so that businesses can take targeted retention actions.

The target variable is **`Churn Value`**:

* `0` → Customer stayed
* `1` → Customer churned

The dataset contains an imbalanced target distribution, with approximately **73.5% non-churned customers** and **26.5% churned customers**.

Because of this imbalance, ROC-AUC was used as the primary metric for comparing models.

---

## Project Workflow

The project follows an end-to-end machine learning workflow:

1. Data Loading
2. Exploratory Data Analysis
3. Target Distribution Analysis
4. Missing Value Investigation
5. Data Cleaning
6. Data Leakage Prevention
7. Feature Selection
8. Train/Test Split
9. Feature Preprocessing
10. Baseline Model Training
11. Class Imbalance Handling
12. 5-Fold Stratified Cross-Validation
13. Model Comparison
14. Hyperparameter Tuning
15. Model Evaluation
16. Feature Importance Analysis
17. Model Serialization

---

## Exploratory Data Analysis

The analysis investigates relationships between customer characteristics and churn.

Key variables explored include:

* Contract Type
* Tenure
* Monthly Charges
* Internet Service
* Churn Distribution

### Key Observations

* Month-to-month customers show substantially higher churn than customers on longer-term contracts.
* Customers with lower tenure tend to have higher churn.
* Churned customers tend to have higher monthly charges.
* Fiber optic customers show a noticeably higher churn rate compared with DSL and customers without internet service.

These observations were identified through the exploratory analysis and visualizations performed in the notebook.

---

## Data Cleaning

Several preprocessing steps were performed before model training.

### Total Charges

`Total Charges` was initially stored as a string and converted to numeric values.

Blank values were identified among customers with zero tenure and replaced with `0`.

### Data Leakage Prevention

The following columns were removed to prevent target leakage or duplicated target information:

* `Churn Reason`
* `Churn Score`
* `CLTV`
* `Churn Label`

### Identifier and Low-Signal Features

The following columns were removed:

* `CustomerID`
* `Count`
* `Country`
* `State`

Location-related variables were also removed:

* `City`
* `Zip Code`
* `Lat Long`
* `Latitude`
* `Longitude`

Duplicate rows were also checked and removed during preprocessing.

---

## Feature Preprocessing

The project uses Scikit-learn preprocessing pipelines to ensure that transformations are applied consistently.

### Numerical Features

Numerical features are standardized using:

```python
StandardScaler()
```

### Categorical Features

Categorical variables are encoded using:

```python
OneHotEncoder(
    drop="first",
    handle_unknown="ignore"
)
```

Using a preprocessing pipeline helps prevent inconsistent transformations between training and evaluation data.

---

## Machine Learning Models

Four classification algorithms were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost

Two approaches were used to address class imbalance:

1. Class weighting
2. SMOTE (Synthetic Minority Over-sampling Technique)

---

## Model Comparison

Models were evaluated using **5-fold Stratified Cross-Validation** with **ROC-AUC** as the primary metric.

| Model                   | Class Weight ROC-AUC | SMOTE ROC-AUC |
| ----------------------- | -------------------: | ------------: |
| **Logistic Regression** |         **0.859174** |      0.858748 |
| Random Forest           |             0.841556 |      0.842337 |
| XGBoost                 |             0.836245 |      0.837543 |
| Decision Tree           |             0.667266 |      0.682621 |

### Best Cross-Validated Model

**Logistic Regression with class weighting achieved the highest ROC-AUC of 0.859174.**

This was the strongest result among the evaluated model and imbalance-handling combinations.

Interestingly, applying SMOTE did not improve Logistic Regression:

* Class Weight: **0.859174**
* SMOTE: **0.858748**

Random Forest and XGBoost showed small improvements with SMOTE, while Decision Tree improved more noticeably but remained significantly below the other models.

---

## Model Insights

The results demonstrate that model complexity does not necessarily lead to better predictive performance.

In this dataset, **Logistic Regression outperformed the tree-based models** in cross-validated ROC-AUC.

This makes Logistic Regression an important benchmark because it provides:

* Strong predictive discrimination
* Relatively simple architecture
* High interpretability
* Low computational complexity

---

## Hyperparameter Tuning

The project also includes hyperparameter optimization using `RandomizedSearchCV`.

The tuning process uses:

* 25 randomized parameter combinations
* 5-fold Stratified Cross-Validation
* ROC-AUC scoring
* Fixed random state for reproducibility

The XGBoost search space includes parameters such as:

* `n_estimators`
* `max_depth`
* `learning_rate`
* `subsample`
* `colsample_bytree`

---

## Model Evaluation

The notebook evaluates model performance using:

* Accuracy
* ROC-AUC
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC Curve

ROC-AUC is emphasized because the target variable is imbalanced.

The cross-validation results provide the primary model comparison, while the held-out test set is used for final performance evaluation.

---

## Feature Importance

Feature importance analysis is performed on the XGBoost model after preprocessing.

The notebook visualizes the **Top 15 features** contributing to the model's predictions.

This provides additional insight into which customer characteristics are most influential when predicting churn.

---

## Business Insights

The exploratory analysis provides several potential business implications.

### Contract Type

Month-to-month customers have considerably higher churn.

**Potential strategy:** Encourage customers to move toward longer-term contracts through appropriate retention incentives.

### Customer Tenure

Customers with lower tenure tend to churn more frequently.

**Potential strategy:** Strengthen onboarding and early-stage customer retention programs.

### Monthly Charges

Churned customers tend to have higher monthly charges.

**Potential strategy:** Investigate pricing sensitivity, service bundles, and customer-value alignment.

### Internet Service

Fiber optic customers show a noticeably higher churn rate.

**Potential strategy:** Investigate service quality, pricing, technical support, and customer expectations for fiber customers.

These observations should be combined with model-based analysis before being used for operational decisions.

---

## Model Serialization

The trained model pipeline is saved using Joblib:

```python
joblib.dump(final_model, "churn_pipeline.joblib")
```

Saving the complete pipeline allows preprocessing and prediction steps to be reused consistently when deploying the model.

The resulting model can be integrated into applications such as:

* Streamlit
* FastAPI
* Python-based prediction services

---

## Project Structure

```text
Telco-Customer-Churn-Prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── Telco_Customer_Churn_Prediction.ipynb
│
├── data/
│   └── Telco_customer_churn.xlsx
│
└── models/
    └── churn_pipeline.joblib
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Imbalanced-learn
* Joblib
* Jupyter Notebook

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Telco-Customer-Churn-Prediction.git
cd Telco-Customer-Churn-Prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Make sure the dataset is located at:

```text
data/Telco_customer_churn.xlsx
```

Then open:

```text
Telco_Customer_Churn_Prediction.ipynb
```

Run the notebook from beginning to end to reproduce the analysis and model training workflow.

---

## Future Improvements

Potential extensions to this project include:

* SHAP-based model explainability
* Probability-threshold optimization
* Cost-sensitive churn prediction
* Customer segmentation
* Additional hyperparameter optimization
* Automated prediction pipelines
* Streamlit deployment
* FastAPI deployment
* Customer retention recommendation system

---

## Author

**Usman**

Business Analytics student focused on Data Analytics, Machine Learning, and AI.


