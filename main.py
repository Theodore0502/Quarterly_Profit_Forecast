# main.py
import argparse, pandas as pd
from pathlib import Path
from prophet import Prophet   # pip install prophet
OUT_DIR = Path("out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def forecast(csv, date_col="Date", y_col="Net_income", freq="QE", horizon=4):
    df = pd.read_csv(csv)
    df[date_col] = pd.to_datetime(df[date_col])

    # Dùng QE (quarter-end).
    resample_rule = "QE"
    df = (df.set_index(date_col).sort_index()
            .resample(resample_rule)[[y_col]].sum()
            .reset_index().rename(columns={date_col:"ds", y_col:"y"}))

    # model
    m = Prophet(seasonality_mode="additive", yearly_seasonality=True)
    m.fit(df[["ds","y"]])

    # Đồng bộ tần suất trong forecast
    fcst = m.predict(m.make_future_dataframe(periods=horizon, freq=resample_rule))

    fcst[["ds","yhat","yhat_lower","yhat_upper"]].to_csv(OUT_DIR/"forecast_prophet.csv", index=False)
    try:
        import matplotlib.pyplot as plt
        m.plot(fcst); plt.savefig(OUT_DIR/"forecast_plot.png", dpi=150, bbox_inches="tight"); plt.close()
        m.plot_components(fcst); plt.savefig(OUT_DIR/"forecast_components.png", dpi=150, bbox_inches="tight"); plt.close()
    except Exception as e:
        print("Plot skipped:", e)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Quarterly profit forecast")
    ap.add_argument("--csv", required=True, help="Ví dụ: data/Apple_DB.csv hoặc CSV Kaggle")
    ap.add_argument("--date-col", default="Date")
    ap.add_argument("--y-col", default="Net_income")
    ap.add_argument("--freq", default="Q")
    ap.add_argument("--horizon", type=int, default=4)
    args = ap.parse_args()
    forecast(args.csv, args.date_col, args.y_col, args.freq, args.horizon)
