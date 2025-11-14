import re
import html
import joblib
import numpy as np
import pandas as pd

# from bs4 import BeautifulSoup        # for HTML stripping (pip install beautifulsoup4)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score)
df = pd.read_csv("emails.csv")
df
df['spam'].value_counts()
df.isna().sum()

df['Label_num'] = df['spam'].map({0:'Ham',1:'Spam'})
df.head()
def simple_normalize(text):
   
    # lower
    text = text.lower()
    # remove URLs and emails
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)                # emails
    text = re.sub(r'http\S+|www\.\S+', ' ', text)            # urls
    # remove non-alphanumeric except spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # collapse spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text
#### 3) Prepare data and split
def prepare_data(df, text_col="text"):
    df['clean_text'] = df[text_col].apply(simple_normalize)
    X = df['clean_text'].values
    y = df['spam'].values
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
### 4) Build pipelines for models
def build_pipelines():
    # TF-IDF vectorizer common config
    tfidf = TfidfVectorizer(max_df=0.9, min_df=3, ngram_range=(1,2), stop_words=None)
    # note: if you want to remove stopwords using NLTK, pass stop_words=STOPWORDS (converted to list) but ensure it's installed.
    pipe_nb = Pipeline([("tfidf", tfidf), ("clf", MultinomialNB())])
    pipe_lr = Pipeline([("tfidf", tfidf), ("clf", LogisticRegression(max_iter=1000))])
    pipe_rf = Pipeline([("tfidf", tfidf), ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1))])
    return {"nb": pipe_nb, "lr": pipe_lr, "rf": pipe_rf}
### 5) Train and evaluate
def evaluate_model(pipe, X_train, X_test, y_train, y_test):
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = None
    try:
        probs = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        pass

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    roc = roc_auc_score(y_test, probs) if probs is not None else None

    print("Accuracy: {:.4f}".format(acc))
    print("Precision: {:.4f}".format(prec))
    print("Recall: {:.4f}".format(rec))
    print("F1: {:.4f}".format(f1))
    if roc is not None:
        print("ROC AUC: {:.4f}".format(roc))
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, preds, digits=4))
    return pipe, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}
#### 6) Save / load model
def save_model(pipe, path="spam_model.joblib"):
    joblib.dump(pipe, path)

def load_model(path="spam_model.joblib"):
    return joblib.load(path)
#### 7) Example usage
if __name__ == "__main__":
    # path to your CSV
    # csv_path = "spam_emails.csv"   # adjust
    # df = load_csv(csv_path)
    X_train, X_test, y_train, y_test = prepare_data(df, text_col="text")

    pipelines = build_pipelines()

    # Train and evaluate each
    scores = {}
    for name, pipe in pipelines.items():
        print("\n>>> Training:", name)
        model, metrics = evaluate_model(pipe, X_train, X_test, y_train, y_test)
        scores[name] = metrics

    # Pick best by F1 (example)
    best_name = max(scores, key=lambda k: scores[k]['f1'])
    print("\nBest model:", best_name, "metrics:", scores[best_name])

    # Save the best model
    best_pipe = pipelines[best_name]
    save_model(best_pipe, path=f"best_spam_model_{best_name}.joblib")
    print("Saved model to:", f"best_spam_model_{best_name}.joblib")

    # Example predict on new text
    example_texts = [
        "Congratulations! You've won a $1000 gift card. Click here to claim now.",
        "i was meet the friend yesterday"
        
    ]
    cleaned = [simple_normalize(t) for t in example_texts]
    preds = best_pipe.predict(cleaned)
    probs = best_pipe.predict_proba(cleaned)[:,1] if hasattr(best_pipe, "predict_proba") else None
    for t, p, prob in zip(example_texts, preds, probs if probs is not None else [None]*len(preds)):
        print(f"\nText: {t}\nPredicted label: {'spam' if p==1 else 'ham'}, probability: {prob}")


