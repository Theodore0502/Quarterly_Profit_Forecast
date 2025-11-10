# classify.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/uci_iris.csv")
X, y = df.drop(columns=["label"]).values, df["label"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=1000)
clf.fit(Xtr, ytr)
pred = clf.predict(Xte)
print(f"Accuracy = {accuracy_score(yte, pred):.4f}")
