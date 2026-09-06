# CODSOFT_TASKN
Task 1. This project predicts a movie's genre from its plot summary using classic NLP + ML techniques.
Machine Learning internship project for **CodSoft**.
This project predicts a movie's **genre** from its **plot summary** using classic NLP + ML techniques.

---

# Problem Statement

Build a machine learning model that can predict the genre of a movie based on its plot
summary or other textual information, using techniques like **TF-IDF** or word embeddings
with classifiers such as **Naive Bayes**, **Logistic Regression**, or **Support Vector Machines

# Dataset

Dataset used: [Genre Classification Dataset (IMDb) — Kaggle](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)

| File | Rows | Format |
|------|------|--------|
| `data/train_data.txt` | 54,214 | `ID ::: TITLE ::: GENRE ::: DESCRIPTION` |
| `data/test_data.txt`  | 54,200 | `ID ::: TITLE ::: DESCRIPTION` (unlabeled) |

The training data covers **27 genres** (drama, documentary, comedy, horror, action, romance,
sci-fi, thriller, etc.), with a natural class imbalance (drama/documentary/comedy are the
most common).

---

##  Project Workflow ()

### 1. Load the data
Parsed the `:::`-delimited `.txt` files into pandas DataFrames.

## 2. Text cleaning
For every plot summary:
- Lowercased text
- Removed URLs and numbers
- Stripped punctuation
- Collapsed extra whitespace

### 3. Train/validation split
Used a **stratified 80/20 split** on the labeled data so rare genres are represented in
both sets, giving an honest accuracy estimate.

### 4. Feature extraction — TF-IDF
Converted cleaned text into numeric features using `TfidfVectorizer`:
- Unigrams + bigrams (`ngram_range=(1,2)`)
- English stop words removed
- Max 50,000 features
- `sublinear_tf=True` to dampen very frequent terms

### 5. Model training & comparison
Trained and compared three classifiers on the same TF-IDF features:

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Multinomial Naive Bayes | 46.9% | 0.348 |
| Logistic Regression | 50.4% | 0.523 |
| **Linear SVM** | **56.3%** | **0.563** |

### 6. Model selection & evaluation
Selected the **Linear SVM** (best weighted F1) and generated a full
`classification_report` (precision/recall/F1 per genre) plus a confusion matrix
heatmap for the 10 most common genres.

### 7. Final training & prediction
Retrained the winning model on **all** labeled data (not just the 80% split), then
generated genre predictions for every row in the unlabeled `test_data.txt`.

### 8. Artifacts saved
- `genre_classifier_model.joblib` — trained Linear SVM model
- `tfidf_vectorizer.joblib` — fitted TF-IDF vectorizer
- `test_predictions.csv` — predicted genre for each test movie
- `evaluation_report.txt` — full metrics for all 3 models
- `confusion_matrix.png` — visual error analysis

---

## 🗂️ Repository Structure

```
CODSOFT_1/
├── data/
│   ├── train_data.txt
│   └── test_data.txt
├── movie_genre_classification.py
├── test_predictions.csv
├── evaluation_report.txt
├── confusion_matrix.png
├── genre_classifier_model.joblib
├── tfidf_vectorizer.joblib
└── README.md
```

##  Acknowledgements

Internship task provided by **[CodSoft](https://www.codsoft.in)**.

#codsoft #internship #machinelearning
