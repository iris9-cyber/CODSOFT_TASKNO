# CODSOFT_1 — Movie Genre Classification 🎬

Machine Learning internship project for **CodSoft** — Task 1.
This project predicts a movie's genre from its plot summary using NLP and classical ML techniques.

---

## 📌 Problem Statement

Develop a model to classify movies into genres based on their plot summaries.
Use text preprocessing, feature extraction (TF-IDF), and classic ML classifiers like
**Logistic Regression**, **Naive Bayes**, or **Support Vector Machines** to predict genre labels.

---

## 📂 Dataset

Movie plot summaries with genre labels — multi-class text classification problem.

Features:
- Plot summary text
- Genre labels (Drama, Action, Comedy, Horror, Romance, Thriller, etc.)

---

## ⚙️ Project Workflow

### 1. Load & explore the data
Loaded plot summaries and genre labels.

### 2. Text preprocessing
- Lowercased text
- Removed URLs, numbers, punctuation
- Removed stop words
- Tokenization

### 3. Feature extraction — TF-IDF
Converted text into numeric features using `TfidfVectorizer`.

### 4. Train/test split
Stratified split to preserve genre distribution.

### 5. Model training & comparison
Trained and compared classifiers:

| Model | Accuracy |
|---|---|
| Logistic Regression | 82.5% |
| Naive Bayes | 80.1% |
| **Random Forest** | **88.9%** |

### 6. Model selection & evaluation
Selected **Random Forest** (highest accuracy). Generated classification report and confusion matrix.

### 7. Feature importance analysis
Identified top keywords predictive of each genre.

### 8. Final predictions & artifacts
Saved model, vectorizer, and test predictions.

---

## 🗂️ Repository Structure

```
CODSOFT_1/
├── data/
│   └── [movie plots dataset]
├── movie_genre_classification.py
├── test_predictions.csv
├── evaluation_report.txt
├── confusion_matrix.png
├── feature_importance.png
├── genre_model.joblib
├── tfidf_vectorizer.joblib
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Navigate to task directory
cd CODSOFT_1

# 2. Install dependencies
pip install pandas scikit-learn matplotlib joblib nltk

# 3. Run the pipeline
python movie_genre_classification.py
```

### Classify a custom plot
```python
from movie_genre_classification import predict_genre

predict_genre("A young wizard discovers his magical powers...")
# -> 'Fantasy'
```

---

## 🛠️ Tech Stack

- **Python 3**
- **pandas / numpy** — data manipulation
- **scikit-learn** — TF-IDF, ML classifiers, metrics
- **matplotlib** — visualization
- **joblib** — model persistence
- **nltk** — text processing

---

## 📊 Key Results

- Best model: **Random Forest**
- Test accuracy: **88.9%**
- Effective for multi-class genre prediction
- Clear genre-specific keywords identified

---

## 🙌 Acknowledgements

Internship task provided by **[CodSoft](https://www.codsoft.in)**.

#codsoft #internship #machinelearning
