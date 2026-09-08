"""
CODSOFT Machine Learning Internship - Task 2
Customer Churn Prediction
---------------------------------------------------------------
Predicts whether a bank customer will churn (leave the bank) based on
their demographic and account information.

Dataset: Kaggle "Bank Customer Churn Modelling"
  data/Churn_Modelling.csv -> 10,000 rows, 14 columns, target = 'Exited'

Pipeline:
  1. Load & explore data
  2. Clean / drop non-predictive identifier columns
  3. Encode categorical features (Geography, Gender)
  4. Scale numeric features
  5. Train/test split (stratified)
  6. Train & compare 3 classifiers: Logistic Regression, Random Forest, Gradient Boosting
  7. Evaluate (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
  8. Pick best model, save it + scaler + encoders
  9. Feature importance analysis
  10. Predict churn probability for new/sample customers
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
DATA_PATH = "data/Churn_Modelling.csv"
df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {df['Exited'].mean():.2%}")

# ----------------------------------------------------------------------
# 2. DROP NON-PREDICTIVE IDENTIFIER COLUMNS
# ----------------------------------------------------------------------
# RowNumber, CustomerId, Surname carry no genuine predictive signal about
# behavior and would just let the model memorize/overfit on IDs/names.
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

# ----------------------------------------------------------------------
# 3. ENCODE CATEGORICAL FEATURES
# ----------------------------------------------------------------------
le_geo = LabelEncoder()
le_gender = LabelEncoder()

df["Geography"] = le_geo.fit_transform(df["Geography"])   # France/Germany/Spain -> 0/1/2
df["Gender"] = le_gender.fit_transform(df["Gender"])        # Female/Male -> 0/1

# ----------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ----------------------------------------------------------------------
X = df.drop(columns=["Exited"])
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------------------------
# 5. SCALE NUMERIC FEATURES
# ----------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------------------
# 6. TRAIN & COMPARE MODELS
# ----------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    ),
}

results = {}
trained_models = {}
probas = {}

for name, model in models.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    results[name] = metrics
    trained_models[name] = model
    probas[name] = proba

    print(f"{name}: acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
          f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} roc_auc={metrics['roc_auc']:.4f}")

# ----------------------------------------------------------------------
# 7. PICK BEST MODEL (by ROC-AUC - robust to class imbalance)
# ----------------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["roc_auc"])
best_model = trained_models[best_name]
print(f"\nBest model: {best_name}")

best_preds = best_model.predict(X_test_scaled)
report_text = classification_report(y_test, best_preds, target_names=["Stayed", "Churned"])
print("\nClassification report (best model):")
print(report_text)

# ----------------------------------------------------------------------
# 8. SAVE EVALUATION REPORT
# ----------------------------------------------------------------------
with open("evaluation_report.txt", "w") as f:
    f.write("CUSTOMER CHURN PREDICTION - MODEL COMPARISON\n")
    f.write("=" * 55 + "\n\n")
    for name, m in results.items():
        f.write(f"{name}:\n")
        for k, v in m.items():
            f.write(f"  {k:10s}: {v:.4f}\n")
        f.write("\n")
    f.write(f"BEST MODEL: {best_name}\n\n")
    f.write("Classification Report (test set, best model):\n")
    f.write(report_text)

# ----------------------------------------------------------------------
# 9. CONFUSION MATRIX PLOT
# ----------------------------------------------------------------------
cm = confusion_matrix(y_test, best_preds)
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(cm, cmap="Blues")
labels = ["Stayed", "Churned"]
ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix - {best_name}")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 10. ROC CURVE PLOT (all models)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
for name, proba in probas.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 11. FEATURE IMPORTANCE (best model, if tree-based; else coefficients)
# ----------------------------------------------------------------------
feature_names = X.columns.tolist()

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
else:
    importances = np.abs(best_model.coef_[0])

imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(imp_df["feature"], imp_df["importance"], color="#4C72B0")
ax.set_title(f"Feature Importance - {best_name}")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

imp_df.sort_values("importance", ascending=False).to_csv("feature_importance.csv", index=False)

# ----------------------------------------------------------------------
# 12. SAVE PREDICTIONS FOR THE TEST SET
# ----------------------------------------------------------------------
test_output = X_test.copy()
test_output["Actual_Exited"] = y_test.values
test_output["Predicted_Exited"] = best_preds
test_output["Churn_Probability"] = best_model.predict_proba(X_test_scaled)[:, 1]
test_output.to_csv("test_predictions.csv", index=False)

# ----------------------------------------------------------------------
# 13. SAVE MODEL, SCALER, ENCODERS
# ----------------------------------------------------------------------
joblib.dump(best_model, "churn_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump({"Geography": le_geo, "Gender": le_gender}, "label_encoders.joblib")

print("\nSaved: churn_model.joblib, scaler.joblib, label_encoders.joblib,")
print("       test_predictions.csv, feature_importance.csv/.png,")
print("       confusion_matrix.png, roc_curve.png, evaluation_report.txt")


# ----------------------------------------------------------------------
# 14. DEMO FUNCTION - predict churn for a new customer
# ----------------------------------------------------------------------
def predict_churn(customer: dict) -> dict:
    """
    customer keys: CreditScore, Geography, Gender, Age, Tenure, Balance,
                   NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
    """
    row = pd.DataFrame([customer])
    row["Geography"] = le_geo.transform(row["Geography"])
    row["Gender"] = le_gender.transform(row["Gender"])
    row = row[feature_names]
    row_scaled = scaler.transform(row)
    pred = best_model.predict(row_scaled)[0]
    prob = best_model.predict_proba(row_scaled)[0, 1]
    return {"will_churn": bool(pred), "churn_probability": round(float(prob), 4)}


if __name__ == "__main__":
    sample_customer = {
        "CreditScore": 600, "Geography": "Germany", "Gender": "Male", "Age": 45,
        "Tenure": 3, "Balance": 120000, "NumOfProducts": 1, "HasCrCard": 1,
        "IsActiveMember": 0, "EstimatedSalary": 80000,
    }
    print(f"\nDemo prediction for a sample customer:\n{sample_customer}")
    print(f"-> {predict_churn(sample_customer)}")
