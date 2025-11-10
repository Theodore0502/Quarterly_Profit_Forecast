# cluster.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("data/uci_iris.csv")
X = df.drop(columns=["label"]).values
km = KMeans(n_clusters=3, n_init="auto", random_state=42)
lab = km.fit_predict(X)
print(f"silhouette = {silhouette_score(X, lab):.4f}")
