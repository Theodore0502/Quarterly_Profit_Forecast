# vẽ lại biểu đồ forecast (đẹp + rõ phần tương lai + note quý)
import pandas as pd
import matplotlib.pyplot as plt

act = pd.read_csv("out/processed_quarterly.csv", parse_dates=["ds"])
fc  = pd.read_csv("out/forecast_prophet.csv", parse_dates=["ds"])

last_actual_date = act["ds"].max()
fc["is_future"]  = fc["ds"] > last_actual_date

# gộp để tiện vẽ/ghi chú
df = pd.merge(fc, act[["ds","y"]], on="ds", how="left")

fig, ax = plt.subplots(figsize=(12, 7))

# 1) Thực tế (quá khứ)
past = df[df["ds"] <= last_actual_date]
ax.plot(past["ds"], past["y"], linewidth=2, label="Thực tế")

# 2) Fit trong quá khứ (tùy chọn, nét đứt cho đỡ gây nhầm)
ax.plot(past["ds"], past["yhat"], linewidth=1, linestyle="--", label="Phần khớp trong quá khứ (yhat)")

# 3) Dự báo tương lai (chỉ phần future)
fut = df[df["is_future"]]
ax.plot(fut["ds"], fut["yhat"], linewidth=2, label="Dự báo")

# 4) Dải tin cậy cho tương lai
ax.fill_between(fut["ds"], fut["yhat_lower"], fut["yhat_upper"], alpha=0.2, label="Khoảng tin cậy")

# 5) Ghi chú quý: gắn nhãn cho 8 điểm gần nhất (4 quá khứ + 4 dự báo)
def qlabel(dt):
    p = dt.to_period("Q")
    return f"{p.year}Q{p.quarter}"

annot_df = pd.concat([past.tail(4), fut.head(4)])
for _, r in annot_df.iterrows():
    ax.annotate(qlabel(r["ds"]),
                (r["ds"], r["y"] if pd.notna(r["y"]) else r["yhat"]),
                textcoords="offset points", xytext=(0,8), ha="center", fontsize=9)

ax.set_title("Lợi nhuận theo quý — Thực tế vs Dự báo (Prophet)")
ax.set_xlabel("Thời gian (quý)")
ax.set_ylabel("Lợi nhuận")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
