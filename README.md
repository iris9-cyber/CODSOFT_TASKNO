# CODSOFT_TASKNO

Quick start
-----------

This repository contains a movie genre classification project (training script + README). The main training script is `movie_genre_classification.py`.

1) Install dependencies:

   pip install -r requirements.txt

2) Provide the dataset files

   - Place your training file as `data/train_data.txt` and test file as `data/test_data.txt` (or edit the constants `TRAIN_PATH` and `TEST_PATH` at the top of `movie_genre_classification.py`).

   Note: The original script used `/content/train_data.txt` and `/content/test_data.txt` (for Google Colab). If you run locally, either move files to those locations or update the paths in the script.

3) Run training:

   python movie_genre_classification.py

4) Outputs

   After running, the script saves:
   - models/final_model.joblib
   - models/tfidf_vectorizer.joblib
   - outputs/evaluation_report.txt
   - outputs/confusion_matrix.png
   - outputs/test_predictions.csv

Files
-----

- movie_genre_classification.py   -- Main training script (TF-IDF + models + output saving)
- README.md                       -- This file
- requirements.txt                 -- Python dependencies
- .gitignore                       -- Typical Python ignores
- data/                            -- Place dataset files here
- models/                          -- Saved model artifacts (created by script)
- outputs/                         -- Generated outputs (created by script)

Notes & next steps
------------------
- Consider updating `movie_genre_classification.py` to accept command-line arguments (argparse) and use relative paths — I can do that in a follow-up commit.
- If your dataset contains multi-label genres (multiple genres per movie), we should convert labels using MultiLabelBinarizer and use a multi-label strategy.
- If you want, I can also open a PR with more refactoring (split into src/, add tests, CI).
