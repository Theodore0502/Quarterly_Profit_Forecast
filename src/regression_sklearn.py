# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

INP = "out/processed_quarterly.csv"

def main():
    if not os.path.exists(INP):
        raise SystemExit("❌ Thiếu out/processed_quarterly.csv — hãy chạy prepare_data trước.")
    df = pd.read_csv(INP, parse_dates=["date"]).sort_values("date")
    X_cols = [c for c in ["revenue","marketing_cost","cpi"] if c in df.columns]
    if not X_cols:
        raise SystemExit("❌ Không có biến độc lập (revenue/marketing_cost/cpi).")
    X = df[X_cols].values
    y = df["profit"].values
    tscv = TimeSeriesSplit(n_splits=3)
    rows, y_pred_all = [], np.full_like(y, np.nan, dtype=float)
    for fold, (tr, te) in enumerate(tscv.split(X), start=1):
        model = LinearRegression().fit(X[tr], y[tr])
        pred = model.predict(X[te])
        y_pred_all[te] = pred
        rmse = np.sqrt(mean_squared_error(y[te], pred))
        mae  = mean_absolute_error(y[te], pred)
        rows.append({"fold":fold, "rmse":rmse, "mae":mae, "features":"+".join(X_cols)})
    pd.DataFrame(rows).to_csv("out/regression_eval.csv", index=False)
    pd.DataFrame({"date": df["date"], "y_true": y, "y_pred": y_pred_all}).to_csv("out/regression_oof_predictions.csv", index=False)
    print("✅ out/regression_eval.csv, out/regression_oof_predictions.csv")

if __name__ == "__main__":
    main()
