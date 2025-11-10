from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True, parents=True)

def main():
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "label"})
    feats = [c for c in df.columns if c != "label"]
    scaler = StandardScaler()
    df[feats] = scaler.fit_transform(df[feats])
    out = DATA_DIR / "uci_iris.csv"
    df.to_csv(out, index=False)
    print(f"Saved -> {out.resolve()}")

if __name__ == "__main__":
    main()
