
import argparse, json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_json(args.input, lines=True)
    n = int(len(df)*0.8)
    tr, va = df.iloc[:n], df.iloc[n:]
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    Xtr, Xva = tfidf.fit_transform(tr.text), tfidf.transform(va.text)
    clf = LogisticRegression(max_iter=200)
    clf.fit(Xtr, tr.label)
    pred = clf.predict(Xva)
    f1 = f1_score(va.label, pred)
    json.dump({"f1": float(f1)}, open(args.out, "w"))
    print("baseline F1:", f1)
