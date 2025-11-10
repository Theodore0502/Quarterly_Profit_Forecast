import pandas as pd
import matplotlib.pyplot as plt

# === Đọc dữ liệu dự báo ===
fcst = pd.read_csv("out/forecast_prophet.csv", parse_dates=["ds"])

# === Tìm mốc chia giữa dữ liệu thật và phần dự báo ===
split_date = fcst["ds"].iloc[-5]  # 4 quý cuối là forecast

plt.figure(figsize=(11,6))

# === Vẽ vùng tin cậy ===
plt.fill_between(
    fcst["ds"], fcst["yhat_lower"], fcst["yhat_upper"],
    color="skyblue", alpha=0.3, label="Khoảng tin cậy (80%)"
)

# === Vẽ đường dự báo trung vị ===
plt.plot(fcst["ds"], fcst["yhat"], color="darkorange", linewidth=2.2, label="Dự báo (Prophet)")

# === Đường kẻ dọc ngăn cách vùng forecast ===
plt.axvline(split_date, color="gray", linestyle="--", linewidth=1.3)
plt.text(split_date, plt.ylim()[1]*0.9, "Bắt đầu dự báo", color="gray", fontsize=9, rotation=90, va="top", ha="right")

# === Ghi chú giá trị dự báo (tính bằng nghìn USD) ===
for x, y in zip(fcst["ds"], fcst["yhat"]):
    plt.text(x, y + 800, f"{int(y/1000)}k", ha="center", va="bottom", fontsize=8, color="darkorange", rotation=45)

# === Tiêu đề và trục ===
plt.title("📈 Dự báo lợi nhuận theo quý của Apple (Prophet)\nĐường cam: Dự báo trung vị  |  Vùng xanh: Khoảng tin cậy 80%", fontsize=13, pad=15)
plt.xlabel("Thời gian (Quý)")
plt.ylabel("Lợi nhuận (triệu USD)")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig("out/plots/01_du_bao_prophet.png", dpi=160)
plt.show()
