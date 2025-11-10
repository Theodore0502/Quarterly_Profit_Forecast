# main.py
import argparse, pandas as pd
from pathlib import Path
from prophet import Prophet

OUT_DIR = Path("out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def forecast(csv, date_col="Date", y_col="Net_income", freq="QE", horizon=4,
             mult=False, quarterly=False, logistic=False):
    df = pd.read_csv(csv)
    df[date_col] = pd.to_datetime(df[date_col])

    # Map tần suất; 'Q' cũ => dùng 'QE' (quarter-end)
    resample_rule = "QE" if freq.upper().startswith("Q") else freq

    # Resample về quý và rename cột
    df = (df.set_index(date_col).sort_index()
            .resample(resample_rule)[[y_col]].sum()
            .reset_index()
            .rename(columns={date_col: "ds", y_col: "y"}))

    # Chọn mode seasonality & growth
    season_mode = "multiplicative" if mult else "additive"
    if logistic:
        df["floor"] = 0
        df["cap"] = df["y"].max() * 1.5
        m = Prophet(growth="logistic", yearly_seasonality=True,
                    seasonality_mode=season_mode)
    else:
        m = Prophet(yearly_seasonality=True, seasonality_mode=season_mode)

    if quarterly:
        m.add_seasonality(name="quarterly", period=365.25/4, fourier_order=5)

    m.fit(df[["ds", "y", "floor", "cap"]] if logistic else df[["ds", "y"]])

    future = m.make_future_dataframe(periods=horizon, freq=resample_rule)
    if logistic:
        future["floor"] = 0
        future["cap"]   = df["cap"].iloc[0]

    fcst = m.predict(future)
    fcst[["ds","yhat","yhat_lower","yhat_upper"]].to_csv(OUT_DIR/"forecast_prophet.csv", index=False)

    try:
        import matplotlib.pyplot as plt
        m.plot(fcst); plt.savefig(OUT_DIR/"forecast_plot.png", dpi=150, bbox_inches="tight"); plt.close()
        m.plot_components(fcst); plt.savefig(OUT_DIR/"forecast_components.png", dpi=150, bbox_inches="tight"); plt.close()
    except Exception as e:
        print("Plot skipped:", e)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Quarterly profit forecast")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--date-col", default="Date")
    ap.add_argument("--y-col", default="Net_income")
    ap.add_argument("--freq", default="QE")              # dùng 'QE' mặc định
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--multiplicative", action="store_true", help="seasonality_mode=multiplicative")
    ap.add_argument("--quarterly", action="store_true", help="thêm seasonality 4 quý/năm")
    ap.add_argument("--logistic", action="store_true", help="ràng buộc floor=0 (growth=logistic)")
    args = ap.parse_args()
    forecast(args.csv, args.date_col, args.y_col, args.freq, args.horizon,
             mult=args.multiplicative, quarterly=args.quarterly, logistic=args.logistic)
