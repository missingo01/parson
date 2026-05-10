import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DATA_PATH = "data/intent_dataset.csv"
MODEL_PATH = "models/intent_classifier.pkl"

def main():
    df = pd.read_csv(DATA_PATH)

    X = df["query"]
    y = df["intent"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            stop_words="english"
        )),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    print("Intent classifier trained and saved.")

if __name__ == "__main__":
    main()