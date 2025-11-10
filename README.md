# quarterly_profit_forecast

Profit_forecast/
├─ data/
│  ├─ Apple_DB.csv                  ---Bộ dữ liệu tài chính Apple (nguồn Kaggle) – đầu vào chính cho bài toán dự báo lợi nhuận quý.
│  └─ uci_iris.csv                  ---Dữ liệu UCI Iris đã chuẩn hóa (tạo bởi fetch_uci.py) – tải & chuẩn hóa dữ liệu, đồng thời làm input nhanh cho phân lớp/ phân cụm
├─ out/
│  ├─ forecast_prophet.csv           ---Bảng dự báo từ Prophet: các cột ds, yhat, yhat_lower, yhat_upper.
│  ├─ forecast_plot.png              ---Đồ thị thực tế vs dự báo (kèm khoảng tin cậy).   
│  ├─ forecast_components.png        ---Biểu đồ thành phần của dự báo: trend & seasonality. → Dùng để giải thích mô hình.
│  ├─ regression_eval.csv            ---Bảng chỉ số MAE/RMSE của mô hình hồi quy tuyến tính (baseline) – để so sánh tham chiếu với Prophet.
│  ├─ regression_oof_predictions.csv ---Dự đoán OOF (out-of-fold) của hồi quy tuyến tính – phục vụ vẽ hình & kiểm chứng.
│  ├─ processed_quarterly.csv        ---Dữ liệu sau tiền xử lý & resample về quý (QE), giúp trace pipeline (log trung gian).
│  ├─ quarterly_profit.csv           ---Bản rút trích chuỗi lợi nhuận theo quý (đầu vào trực tiếp cho training Prophet).
│  └─ plots/
│     ├─ 01_thuc_te_vs_du_bao_prophet.png   ---Đồ thị thực tế vs dự báo (Prophet)
│     └─ 02_hoi_quy_oof_vs_thuc_te.png      ---Đồ thị Linear Regression (OOF) vs thực tế – dùng để nhìn nhanh baseline.
├─ src/
│  ├─ prepare_data.py           ---Hàm/tiện ích tiền xử lý: làm sạch số, đổi tên cột Date→ds, Net_income→y, resample QE, xuất file trung gian (processed_quarterly.csv, quarterly_profit.csv).
│  └─ regression_sklearn.py     ---Cài đặt hồi quy tuyến tính (sklearn) theo kiểu time-series (thường tạo lag features, tính OOF, xuất regression_eval.csv và 02_hoi_quy_oof_vs_thuc_te.png). → Dùng làm baseline tham chiếu.
├─ fetch_uci.py                 ---Tải/khởi tạo dữ liệu UCI (Iris) & chuẩn hóa (StandardScaler) → xuất data/uci_iris.csv → Download & chuẩn hóa dữ liệu học thuật.
├─ classify.py                  ---Demo phân lớp (vd Logistic Regression) trên uci_iris.csv. In ra Accuracy (≈0.93) → Dùng để minh họa khai phá dữ liệu – phân lớp (đủ điểm mục mở rộng).
├─ cluster.py                   ---Demo phân cụm (vd KMeans) trên uci_iris.csv. In ra Silhouette (≈0.48) → Minh họa khai phá dữ liệu – phân cụm (có thể chọn chạy 1 trong 2: classify hoặc cluster).      
└─ main.py                      ---Đọc data/Apple_DB.csv → tiền xử lý (làm sạch số, Date→ds, Net_income→y, resample QE).
                                ---Huấn luyện Prophet và dự báo 4 quý.
                                ---Xuất toàn bộ CSV/Hình vào out/        


python main.py --csv data/Apple_DB.csv --date-col Date --y-col Net_income --freq Q --horizon 4

- Prophet cho dự báo (file src/forecast_prophet.py)

- Hồi quy tuyến tính cho baseline/so sánh (file src/regression_sklearn.py) và có pipeline gọi lần lượt trong src/cli.py.

# Thuật toán
- Prophet: src/forecast_prophet.py
    + Tải out/quarterly_profit.csv → đổi cột date→ds, profit→y → Prophet(...).fit(df) → make_future_dataframe(periods=4, freq="QE-DEC") → predict → xuất out/forecast_prophet.csv.

- Linear Regression: src/regression_sklearn.py
    + Đọc out/processed_quarterly.csv → chọn đặc trưng ["revenue","marketing_cost","cpi"] (nếu có) → TimeSeriesSplit → LinearRegression().fit(...) → tính MAE/RMSE, regression_eval.csv.

- Tiền xử lý/chuẩn hoá: src/prepare_data.py
    + Hàm _to_float loại $, dấu phẩy, ký tự lạ → ép số,
    + Chuẩn hoá cột, đổi tên (Date→date, Net_income→profit, …),
    + Sắp xếp theo date, gộp/quy đổi theo quý, sinh out/processed_quarterly.csv và out/quarterly_profit.csv.

# Chuẩn hóa
- src/prepare_data.py có:
    + Hàm _to_float(...) làm sạch chuỗi tiền tệ → float.
    + Chuẩn hoá tên cột (Date→date, Net_income→profit, …).
    + Ép date về datetime, sort, resample/gộp theo quý.
    + Ghi out/processed_quarterly.csv (đủ biến) và out/quarterly_profit.csv (2 cột date, profit) để Prophet dùng trực tiếp.

EDA