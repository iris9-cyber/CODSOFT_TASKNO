import re
import string
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


# 1. LOAD DATA

DATA_PATH = "data/spam.csv"
df = pd.read_csv(DATA_PATH, encoding="latin-1")

# 2. CLEAN UP STRAY COLUMNS

extra_cols = [c for c in df.columns if c.startswith("Unnamed")]
df["v2"] = df["v2"].fillna("") + " " + df[extra_cols].fillna("").agg(" ".join, axis=1)
df = df[["v1", "v2"]]
df.columns = ["label", "message"]
df["message"] = df["message"].str.strip()

print(f"Raw rows: {len(df)}")


# 3. REMOVE DUPLICATES

before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {before - len(df)} duplicate rows -> {len(df)} rows remain")
print(df["label"].value_counts())
print(f"Spam rate: {(df['label'] == 'spam').mean():.2%}")


# 4. TEXT CLEANING

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_message"] = df["message"].apply(clean_text)


# 5. TRAIN / TEST SPLIT

X = df["clean_message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 6. TF-IDF VECTORIZATION

vectorizer = TfidfVectorizer(
    max_features=5000, ngram_range=(1, 2), min_df=2, stop_words="english"
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 7. TRAIN & COMPARE MODELS

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Linear SVM": LinearSVC(class_weight="balanced", random_state=42),
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, pos_label="spam"),
        "recall": recall_score(y_test, preds, pos_label="spam"),
        "f1": f1_score(y_test, preds, pos_label="spam"),
    }
    results[name] = metrics
    trained_models[name] = model

    print(f"{name}: acc={metrics['accuracy']:.4f} "
          f"precision(spam)={metrics['precision']:.4f} "
          f"recall(spam)={metrics['recall']:.4f} f1(spam)={metrics['f1']:.4f}")


# 8. PICK BEST MODEL (by F1 on the spam class - the class that matters)

best_name = max(results, key=lambda k: results[k]["f1"])
best_model = trained_models[best_name]
print(f"\nBest model: {best_name}")

best_preds = best_model.predict(X_test_tfidf)
report_text = classification_report(y_test, best_preds, target_names=["ham", "spam"])
print("\nClassification report (best model):")
print(report_text)


# 9. SAVE EVALUATION REPORT

with open("evaluation_report.txt", "w") as f:
    f.write("SPAM SMS DETECTION - MODEL COMPARISON\n")
    f.write("=" * 55 + "\n\n")
    for name, m in results.items():
        f.write(f"{name}:\n")
        for k, v in m.items():
            f.write(f"  {k:10s}: {v:.4f}\n")
        f.write("\n")
    f.write(f"BEST MODEL: {best_name}\n\n")
    f.write("Classification Report (test set, best model):\n")
    f.write(report_text)


# 10. CONFUSION MATRIX PLOT

cm = confusion_matrix(y_test, best_preds, labels=["ham", "spam"])
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(cm, cmap="Blues")
labels = ["ham", "spam"]
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


# 11. RETRAIN ON ALL DATA & SAVE FINAL ARTIFACTS

full_vectorizer = TfidfVectorizer(
    max_features=5000, ngram_range=(1, 2), min_df=2, stop_words="english"
)
X_full_tfidf = full_vectorizer.fit_transform(X)

final_model = models[best_name].__class__(**best_model.get_params())
final_model.fit(X_full_tfidf, y)

joblib.dump(final_model, "spam_classifier_model.joblib")
joblib.dump(full_vectorizer, "tfidf_vectorizer.joblib")

# Save test predictions for inspection
test_output = pd.DataFrame({
    "message": X_test.values,
    "actual_label": y_test.values,
    "predicted_label": best_preds,
})
test_output.to_csv("test_predictions.csv", index=False)

print("\nSaved: spam_classifier_model.joblib, tfidf_vectorizer.joblib,")
print("       test_predictions.csv, evaluation_report.txt, confusion_matrix.png")


# 12. DEMO FUNCTION - classify a custom SMS message

def predict_message(message: str) -> str:
    """
    Classify a custom SMS message as 'spam' or 'ham'.
    """
    cleaned = clean_text(message)
    vec = full_vectorizer.transform([cleaned])
    return final_model.predict(vec)[0]


if __name__ == "__main__":
    samples = [
        "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!",
        "Hey, are we still meeting for lunch tomorrow at 1pm?",
        "URGENT: Your account has been suspended. Verify your details immediately to avoid charges.",
    ]
    print("\nDemo predictions:")
    for msg in samples:
        pred = predict_message(msg)
        print(f"  '{msg[:60]}...' -> {pred}")
