import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, plot_confusion_matrix

# 1. LOAD DATA

TRAIN_PATH = "/content/train_data.txt"
TEST_PATH = "/content/test_data.txt"

train_cols = ["ID", "TITLE", "GENRE", "DESCRIPTION"]
test_cols = ["ID", "TITLE", "DESCRIPTION"]

train_df = pd.read_csv(TRAIN_PATH, sep=":"":: ", engine="python", names=train_cols)
test_df = pd.read_csv(TEST_PATH, sep=":"":: ", engine="python", names=test_cols)

print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
print(f"Number of genres: {train_df['GENRE'].nunique()}")


# 2. TEXT CLEANING

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\\S+|www\\.\\S+", " ", text)    
    text = re.sub(r"\W+", " ", text)    
    text = re.sub(r"\\s+", " ", text).strip()
    return text

train_df["CLEAN_DESC"] = train_df["DESCRIPTION"].astype(str).apply(clean_text)

test_df["CLEAN_DESC"] = test_df["DESCRIPTION"].astype(str).apply(clean_text)

# 3. TRAIN / VALIDATION SPLIT (stratified, so rare genres appear in both)

X = train_df["CLEAN_DESC"]
y = train_df["GENRE"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. TF-IDF VECTORIZATION

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    stop_words="english",
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# 5. TRAIN / COMPARE MODELS

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1),
    "Linear SVM": LinearSVC(class_weight="balanced")
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_val_tfidf)
    acc = accuracy_score(y_val, preds)
    f1_macro = f1_score(y_val, preds, average="macro")
    f1_weighted = f1_score(y_val, preds, average="weighted")

    results[name] = {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}
    trained_models[name] = model

    print(f"{name}: accuracy={acc:.4f}, f1_macro={f1_macro:.4f}")

# 6. PICK BEST MODEL (by weighted F1)

best_name = max(results.keys(), key=lambda k: results[k]["f1_weighted"])
best_model = trained_models[best_name]

print(f"\nBest model: {best_name}")

# 7. FINAL BEST MODEL ON *ALL* LABELED DATASET

print("\nRetraining best model on the full labeled dataset ...")
full_vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=2, sublinear_tf=True, stop_words="english")
full_tfidf = full_vectorizer.fit_transform(train_df["CLEAN_DESC"]) 

final_model = models[best_name].__class__(**best_model.get_params())
final_model.fit(full_tfidf, train_df["GENRE"]) 

# 8. PREDICT ON THE UNLABELED TEST SET

X_test = test_df["CLEAN_DESC"]
X_test_tfidf = full_vectorizer.transform(X_test)

test_preds = final_model.predict(X_test_tfidf)

# Save outputs
joblib.dump(final_model, "models/final_model.joblib")
joblib.dump(full_vectorizer, "models/tfidf_vectorizer.joblib")

# create outputs folder if needed
import os
os.makedirs("outputs", exist_ok=True)

# evaluation on validation set
val_preds = best_model.predict(X_val_tfidf)
report = classification_report(y_val, val_preds)
cm = confusion_matrix(y_val, val_preds)

with open("outputs/evaluation_report.txt", "w") as f:
    f.write(f"Best model: {best_name}\n")
    f.write(f"Results on validation set:\n")
    for k, v in results.items():
        f.write(f"{k}: accuracy={v['accuracy']:.4f}, f1_macro={v['f1_macro']:.4f}, f1_weighted={v['f1_weighted']:.4f}\n")
    f.write("\nClassification report (validation):\n")
    f.write(report)

# save confusion matrix plot
plt.figure(figsize=(8,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png')

# save sample predictions
out_df = test_df.copy()
out_df['PRED_GENRE'] = test_preds
out_df.to_csv('outputs/test_predictions.csv', index=False)

print("Saved outputs to outputs/ and models/")
