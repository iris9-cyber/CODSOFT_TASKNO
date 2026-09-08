# CODSOFT_2 — Customer Churn Prediction 🏦

Machine Learning internship project for **CodSoft** — Task 2.
This project predicts whether a bank customer will **churn** (leave the bank) based on
their demographic and account information.

---

## 📌 Problem Statement

Develop a model to predict customer churn for a subscription-based service or business.
Use historical customer data, including features like usage behavior and customer
demographics, and try algorithms like **Logistic Regression**, **Random Forest**, or
**Gradient Boosting** to predict churn.

---

## 📂 Dataset

Dataset used: [Bank Customer Churn Modelling — Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)

`data/Churn_Modelling.csv` — 10,000 customers, 14 columns, target = `Exited` (1 = churned, 0 = stayed)

| Column | Description |
|---|---|
| CreditScore | Customer's credit score |
| Geography | Country (France / Germany / Spain) |
| Gender | Male / Female |
| Age | Customer age |
| Tenure | Years as a bank customer |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Owns a credit card (0/1) |
| IsActiveMember | Actively using the account (0/1) |
| EstimatedSalary | Estimated annual salary |
| **Exited** | **Target** — did the customer churn? |

Class balance: **20.4% churned / 79.6% stayed** (moderate imbalance).
No missing values.

---

## ⚙️ Project Workflow (Step by Step)

### 1. Load & explore the data
Checked shape, data types, missing values, and churn rate.

### 2. Drop non-predictive columns
Removed `RowNumber`, `CustomerId`, `Surname` — pure identifiers with no genuine
behavioral signal, which would only encourage overfitting.

### 3. Encode categorical features
- `Geography` (France/Germany/Spain) → label encoded
- `Gender` (Male/Female) → label encoded

### 4. Train/test split
Stratified 80/20 split to preserve the churn ratio in both sets.

### 5. Feature scaling
Applied `StandardScaler` to all features (important for Logistic Regression).

### 6. Model training & comparison
Trained and compared three classifiers:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 70.8% | 0.383 | 0.717 | 0.500 | 0.774 |
| Random Forest | 84.2% | 0.602 | 0.654 | 0.627 | 0.856 |
| **Gradient Boosting** | **86.7%** | **0.793** | 0.469 | 0.590 | **0.867** |

### 7. Model selection & evaluation
Selected **Gradient Boosting** (highest ROC-AUC, most robust metric under class
imbalance). Generated a full classification report, confusion matrix, and ROC curve
comparison across all three models.

### 8. Feature importance analysis
Identified the strongest churn drivers: **Age**, **Number of Products**, and
**Active Membership status** dominate the model's decisions — older, less-engaged
customers with fewer bank products are the highest churn risk.

### 9. Final predictions & artifacts
Saved predictions with churn probability for every test customer, plus the trained
model, scaler, and label encoders for reuse.

---

## 🗂️ Repository Structure

```
CODSOFT_2/
├── data/
│   └── Churn_Modelling.csv
├── churn_prediction.py
├── test_predictions.csv
├── evaluation_report.txt
├── confusion_matrix.png
├── roc_curve.png
├── feature_importance.png
├── feature_importance.csv
├── churn_model.joblib
├── scaler.joblib
├── label_encoders.joblib
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/iris9-cyber/CODSOFT_TASKNO.git
cd CODSOFT_TASKNO/CODSOFT_2

# 2. Install dependencies
pip install pandas scikit-learn matplotlib joblib

# 3. Run the pipeline
python churn_prediction.py
```

### Predict churn for a new customer
The script exposes a `predict_churn()` function:

```python
from churn_prediction import predict_churn

predict_churn({
    "CreditScore": 600, "Geography": "Germany", "Gender": "Male", "Age": 45,
    "Tenure": 3, "Balance": 120000, "NumOfProducts": 1, "HasCrCard": 1,
    "IsActiveMember": 0, "EstimatedSalary": 80000,
})
# -> {'will_churn': True, 'churn_probability': 0.6642}
```

---

## 🛠️ Tech Stack

- **Python 3**
- **pandas / numpy** — data loading & manipulation
- **scikit-learn** — preprocessing, Logistic Regression, Random Forest, Gradient Boosting, metrics
- **matplotlib** — confusion matrix, ROC curve, feature importance plots
- **joblib** — model persistence

---

## 📊 Key Results

- Best model: **Gradient Boosting Classifier**
- Test accuracy: **86.7%**, ROC-AUC: **0.867**
- Top churn drivers: Age, Number of Products, Active Membership, Balance
- Precision on churners is high (79%) — when the model flags a customer as a churn
  risk, it's usually right, making it useful for targeted retention campaigns.

---

## 🔮 Possible Improvements

- Address class imbalance further with SMOTE or class-weighted thresholds tuned for recall
- Hyperparameter tuning via `GridSearchCV` / `RandomizedSearchCV`
- Try XGBoost / LightGBM for potentially stronger performance
- Add SHAP values for per-customer explainability
- Build a simple Streamlit/Flask app for live churn-risk scoring

---

## 🙌 Acknowledgements

Internship task provided by **[CodSoft](https://www.codsoft.in)**.

#codsoft #internship #machinelearning
