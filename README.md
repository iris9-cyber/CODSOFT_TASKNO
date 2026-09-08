# CODSOFT Machine Learning Internship 🎯

A collection of machine learning projects completed as part of the **CodSoft** internship program.

---

## 📂 Projects Overview

### [CODSOFT_1 — Movie Genre Classification 🎬](./CODSOFT_1/)
Predicts movie genres from plot summaries using NLP and classical ML algorithms.

**Key Features:**
- Text preprocessing & TF-IDF vectorization
- Multi-class classification (Drama, Action, Comedy, Horror, Romance, Thriller)
- Models: Logistic Regression, Naive Bayes, SVM, Random Forest
- Best Model: Random Forest (88.9% accuracy)

**Quick Start:**
```bash
cd CODSOFT_1
python movie_genre_classification.py
```

---

### [CODSOFT_2 — Customer Churn Prediction 🏦](./CODSOFT_2/)
Predicts whether a bank customer will churn based on demographic and account information.

**Key Features:**
- Binary classification (Churned / Stayed)
- Feature engineering & scaling
- Models: Logistic Regression, Random Forest, Gradient Boosting
- Best Model: Gradient Boosting (86.7% accuracy, 0.867 ROC-AUC)

**Quick Start:**
```bash
cd CODSOFT_2
python churn_prediction.py
```

---

## 🛠️ Tech Stack (All Projects)

- **Python 3**
- **pandas / numpy** — data manipulation
- **scikit-learn** — ML algorithms & metrics
- **matplotlib / seaborn** — visualization
- **joblib** — model persistence
- **nltk / sklearn** — NLP & text processing

---

## 📋 Repository Structure

```
CODSOFT_TASKNO/
├── CODSOFT_1/
│   ├── README.md
│   ├── movie_genre_classification.py
│   ├── data/
│   ├── models/
│   └── outputs/
│
├── CODSOFT_2/
│   ├── README.md
│   ├── churn_prediction.py
│   ├── data/
│   └── models/
│
└── README.md (this file)
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/iris9-cyber/CODSOFT_TASKNO.git
cd CODSOFT_TASKNO
```

### 2. Install Dependencies
```bash
pip install pandas scikit-learn matplotlib seaborn joblib nltk
```

### 3. Run Individual Projects
Each task has its own directory with complete documentation:

```bash
# Task 1: Movie Genre Classification
cd CODSOFT_1
python movie_genre_classification.py

# Task 2: Customer Churn Prediction
cd CODSOFT_2
python churn_prediction.py
```

---

## 📊 Project Results Summary

| Task | Problem | Best Model | Accuracy | Key Insight |
|---|---|---|---|---|
| **CODSOFT_1** | Movie Genre Classification | Random Forest | 88.9% | Certain genres (Drama, Thriller) more predictable from plot |
| **CODSOFT_2** | Customer Churn Prediction | Gradient Boosting | 86.7% | Age, Product Count, & Active Status are top churn drivers |

---

## 📚 Learning Outcomes

- End-to-end ML pipeline development
- Data preprocessing & feature engineering
- Model selection & hyperparameter tuning
- Classification metrics & evaluation
- Model persistence & deployment
- Visualization & reporting

---

## 🙌 Acknowledgements

Internship program provided by **[CodSoft](https://www.codsoft.in)**.

All projects follow best practices for ML development, including:
- Clean, documented code
- Proper train/test splits
- Multiple model comparisons
- Comprehensive evaluation metrics
- Saved models for inference

---

## 📄 License

This project is provided as-is for educational purposes.

#codsoft #internship #machinelearning #python
