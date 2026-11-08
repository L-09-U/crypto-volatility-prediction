# Dự đoán biến động giá BTC/ETH bằng XGBoost

**Demo:** https://crypto-volatility-prediction-ncfntsfeappua4qz8oapjla.streamlit.app/

Dự án thực tập: dự đoán **biên độ nến ngày kế tiếp** `(High − Low) / Close` của BTC/USDT và ETH/USDT,
kèm website Streamlit để xem biểu đồ giá, dự đoán và đánh giá mô hình.

> Chỉ phục vụ học tập và nghiên cứu. Mô hình dự đoán **mức biến động**, không dự đoán hướng giá và
> không phải lời khuyên đầu tư.

## Bài toán và dữ liệu
- Dữ liệu: 3.229 nến ngày (UTC) mỗi mã, từ 01/01/2018 đến 03/11/2026, lấy từ API dữ liệu công khai của Binance, chỉ dùng nến đã đóng.
- Target: biên độ nến ngày t+1. Đặc trưng chỉ dùng thông tin đến hết ngày t (biên độ các ngày trước,
  trung bình trượt, |return|, độ lệch chuẩn return 5/20 ngày, thay đổi volume, thứ trong tuần).
- Mô hình: XGBoost học `log(biên độ)`, hiệu chỉnh smearing (Duan) khi đổi về thang gốc.
  Hệ số smearing hiện tính trên phần dư của chính tập huấn luyện, nên chỉ coi là phương pháp đang thử nghiệm.
- Baseline: biên độ hôm nay, trung bình 5 ngày, trung bình 20 ngày.
- Đánh giá: walk-forward (expanding window), 5 fold × 180 ngày, `gap=1`, từ 18/05/2024 đến 03/11/2026.
  Đây là kiểm thử khả năng dự báo trên dữ liệu quá khứ, chưa phải backtest giao dịch có phí và trượt giá.

## Kết quả (ngoài mẫu, gộp 5 fold; lần chạy ngày 04/11/2026)
| Tài sản | Mô hình | MAE | RMSE |
|---|---|---|---|
| BTCUSDT | XGBoost | 1,236% | 1,760% |
| BTCUSDT | Baseline TB 5 ngày | 1,484% | 2,079% |
| ETHUSDT | XGBoost | 1,912% | 2,809% |
| ETHUSDT | Baseline TB 5 ngày | 2,165% | 3,170% |

XGBoost thắng baseline TB 5 ngày ở 5/5 fold cho cả hai tài sản, theo cả MAE và RMSE
(MAE giảm khoảng 16,7% với BTC, 11,7% với ETH).

Phân tích chuỗi thời gian: giá (log) không dừng (ADF p ≈ 0,84 với BTC, 0,72 với ETH), còn log return và biên độ nến thì dừng.
Log return gần như không có tự tương quan (ACF lag 1 ≈ −0,06), trong khi biên độ nến có tự tương quan lớn và kéo dài
(ACF lag 1 ≈ 0,49), tức biến động tụ cụm. Thử nghiệm ban đầu với return/hướng giá ngày kế tiếp cho độ chính xác hướng ≈ 50%,
không vượt baseline, nên dự án tập trung vào dự đoán biến động.

## Cấu trúc thư mục
```
crypto-volatility-prediction/
├── notebooks/model_training.ipynb   # toàn bộ quy trình: tải dữ liệu, EDA, baseline, XGBoost, walk-forward
├── data/                            # BTCUSDT.csv, ETHUSDT.csv (nến ngày)
├── artifacts/                       # metrics.csv, folds.csv, predictions.csv, forecast.json, meta.json
├── models/                          # BTCUSDT_model.json, ETHUSDT_model.json
├── app.py                           # website Streamlit
└── requirements.txt
```

## Chạy website
```bash
pip install -r requirements.txt
streamlit run app.py
```
App chỉ đọc `data/` và `artifacts/`, không huấn luyện lại. Để cập nhật dự đoán mới, chạy lại notebook rồi thay hai thư mục này.
Ô "Giá trực tiếp" chỉ để hiển thị và không dùng cho mô hình.

## Hạn chế
- Mức cải thiện so với baseline là vừa phải (khoảng 12–17% MAE) và đo trên một cấu hình cố định chưa tinh chỉnh.
- Cách xử lý log (gốc / log+exp / log+smearing) được chọn dựa trên chính các fold walk-forward, nên có thể hơi lạc quan.
- Khoảng 80% là phân vị thực nghiệm của tỷ lệ thực tế/dự đoán, không phải đảm bảo thống kê.
- Dữ liệu chỉ gồm giá và volume; chưa có tin tức hay dữ liệu vĩ mô; chưa có backtest giao dịch.

## Hướng mở rộng
Baseline GARCH/HAR, LSTM (so sánh công bằng trên cùng target và 5 fold), backtest giao dịch có phí và trượt giá,
tự động cập nhật dữ liệu mỗi ngày.
