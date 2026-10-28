# Dự đoán biến động giá BTC/ETH bằng XGBoost

Dự án học tập: dự đoán **biên độ nến ngày kế tiếp** `(High − Low) / Close` của BTC/USDT và ETH/USDT,
kèm website Streamlit để xem biểu đồ giá, dự đoán và đánh giá mô hình.

> Chỉ phục vụ học tập và nghiên cứu. Mô hình dự đoán **mức biến động**, không dự đoán hướng giá và
> không phải lời khuyên đầu tư.

## Bài toán và dữ liệu
- Dữ liệu: nến ngày (UTC) từ 2018-01-01 đến 2026-10-25, lấy từ API dữ liệu công khai của Binance, chỉ dùng nến đã đóng.
- Target: biên độ nến ngày t+1. Đặc trưng chỉ dùng thông tin đến hết ngày t (biên độ các ngày trước,
  trung bình trượt, |return|, độ lệch chuẩn return 5/20 ngày, thay đổi volume, thứ trong tuần).
- Mô hình: XGBoost học `log(biên độ)`, hiệu chỉnh smearing (Duan) khi đổi về thang gốc.
- Baseline: biên độ hôm nay, trung bình 5 ngày, trung bình 20 ngày.
- Đánh giá: walk-forward (expanding window), 5 fold × 180 ngày, `gap=1`, từ 2024-05-09 đến 2026-10-25.

## Kết quả (ngoài mẫu, gộp 5 fold)
| Tài sản | Mô hình | MAE | RMSE |
|---|---|---|---|
| BTCUSDT | XGBoost | 1.238% | 1.772% |
| BTCUSDT | Baseline TB 5 ngày | 1.488% | 2.084% |
| ETHUSDT | XGBoost | 1.902% | 2.805% |
| ETHUSDT | Baseline TB 5 ngày | 2.167% | 3.170% |

XGBoost thắng baseline TB 5 ngày ở 5/5 fold cho cả hai tài sản (MAE giảm khoảng 17% với BTC, 12% với ETH).
Thử nghiệm ban đầu với return/hướng giá ngày kế tiếp cho độ chính xác hướng ≈ 50%, không vượt baseline,
nên dự án tập trung vào dự đoán biến động.

## Cấu trúc thư mục
```
crypto-volatility-prediction/
├── notebooks/model_training.ipynb   # toàn bộ quy trình: tải dữ liệu, feature, baseline, XGBoost, walk-forward
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

## Hạn chế
- Mức cải thiện so với baseline là vừa phải (khoảng 12–17% MAE) và đo trên một cấu hình cố định chưa tinh chỉnh.
- Cách xử lý log (gốc / log+exp / log+smearing) được chọn dựa trên chính các fold walk-forward, nên có thể hơi lạc quan.
- Khoảng 80% là phân vị thực nghiệm của tỷ lệ thực tế/dự đoán, không phải đảm bảo thống kê.
- Dữ liệu chỉ gồm giá và volume; chưa có tin tức hay dữ liệu vĩ mô.

## Hướng mở rộng
LSTM (so sánh công bằng trên cùng target và 5 fold), triển khai Streamlit Community Cloud, cập nhật dữ liệu tự động mỗi ngày.
