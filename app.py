"""AI Crypto Volatility Forecast – Streamlit app (BTC/ETH, nến ngày).
Chạy: streamlit run app.py   (cần thư mục data/ và artifacts/ cạnh file này, tạo bởi Cell 4 + Cell 5)
"""
import json
from pathlib import Path

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
DATA = ROOT / "data"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
PAGES = ["Biểu đồ giá", "Dự đoán ngày mai", "Đánh giá mô hình", "Giới thiệu"]
DISCLAIMER = ("Chỉ phục vụ học tập và nghiên cứu. Mô hình dự đoán **mức biến động** (biên độ nến), "
              "không dự đoán hướng giá và không phải lời khuyên đầu tư.")

st.set_page_config(page_title="AI Crypto Volatility Forecast", page_icon="📈", layout="wide")

if not (ART / "forecast.json").exists():
    st.error("Chưa có thư mục `artifacts/` và `data/`. Hãy chạy Cell 4 + Cell 5 trong Colab "
             "và đặt hai thư mục này cạnh `app.py`.")
    st.stop()


@st.cache_data
def load_raw(sym):
    return pd.read_csv(DATA / f"{sym}.csv", parse_dates=["date"])


@st.cache_data
def load_oos(sym):
    p = pd.read_csv(ART / "predictions.csv", parse_dates=["date"])
    return p[p["symbol"] == sym].reset_index(drop=True)


@st.cache_data
def load_json(name):
    return json.loads((ART / name).read_text(encoding="utf-8"))


@st.cache_data
def load_csv(name):
    return pd.read_csv(ART / name)


# ---------------------------------------------------------------- Giá trực tiếp
COINBASE_IDS = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD"}


@st.cache_data(ttl=4, show_spinner=False)
def fetch_live(sym):
    """Giá trực tiếp từ API công khai (miễn phí, không cần key). Binance trước, Coinbase dự phòng."""
    try:
        r = requests.get("https://data-api.binance.vision/api/v3/ticker/24hr",
                         params={"symbol": sym}, timeout=4)
        r.raise_for_status()
        j = r.json()
        return {"price": float(j["lastPrice"]), "chg": float(j["priceChangePercent"]),
                "high": float(j["highPrice"]), "low": float(j["lowPrice"]),
                "src": "Binance", "unit": "USDT"}
    except Exception:
        pass
    try:
        r = requests.get(f"https://api.exchange.coinbase.com/products/{COINBASE_IDS[sym]}/stats", timeout=4)
        r.raise_for_status()
        j = r.json()
        last, opn = float(j["last"]), float(j["open"])
        return {"price": last, "chg": (last / opn - 1) * 100, "high": float(j["high"]),
                "low": float(j["low"]), "src": "Coinbase (dự phòng)", "unit": "USD"}
    except Exception:
        return None


@st.fragment(run_every=5)
def live_box(sym):
    d = fetch_live(sym)
    st.markdown("**Giá trực tiếp** (tự làm mới mỗi 5 giây)")
    if d is None:
        st.info("Không lấy được giá trực tiếp lúc này. Các biểu đồ và dự đoán bên dưới vẫn dùng dữ liệu nến ngày đã đóng.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Giá hiện tại ({d['unit']})", f"{d['price']:,.2f}", f"{d['chg']:+.2f}% (24h)")
    c2.metric("Cao nhất 24h", f"{d['high']:,.2f}")
    c3.metric("Thấp nhất 24h", f"{d['low']:,.2f}")
    now = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S")
    st.caption(f"Nguồn: {d['src']} · cập nhật {now} UTC. Đây chỉ là giá hiển thị trực tiếp, "
               "không dùng để huấn luyện hay tạo dự đoán (mô hình dùng nến ngày đã đóng).")


# ---------------------------------------------------------------- Trang 1
def page_prices(sym):
    df = load_raw(sym).copy()
    for w in (5, 20, 50):
        df[f"sma{w}"] = df["close"].rolling(w).mean()
    days = st.sidebar.select_slider("Khoảng hiển thị (ngày)", options=[60, 90, 180, 365, 730, 1500], value=180)
    smas = st.sidebar.multiselect("Đường SMA", [5, 20, 50], default=[20])
    d = df.tail(days)

    live_box(sym)
    st.divider()
    st.markdown("**Dữ liệu nến ngày đã đóng cửa** (dùng cho biểu đồ và mô hình)")
    last, prev = df.iloc[-1], df.iloc[-2]
    c1, c2, c3 = st.columns(3)
    c1.metric("Giá đóng cửa", f"{last['close']:,.2f} USDT", f"{(last['close'] / prev['close'] - 1) * 100:+.2f}%")
    c2.metric("Biên độ nến gần nhất", f"{(last['high'] - last['low']) / last['close'] * 100:.2f}%")
    c3.metric("Nến đóng cửa cuối (UTC)", str(last["date"].date()))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d["date"], open=d["open"], high=d["high"], low=d["low"],
                                 close=d["close"], name="Giá"), row=1, col=1)
    for w in smas:
        fig.add_trace(go.Scatter(x=d["date"], y=d[f"sma{w}"], mode="lines", name=f"SMA{w}"), row=1, col=1)
    fig.add_trace(go.Bar(x=d["date"], y=d["volume"], name="Volume", opacity=0.6), row=2, col=1)
    fig.update_layout(height=650, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.04))
    st.plotly_chart(fig)
    st.caption("Màu nến chỉ là mã hóa trực quan, không phải tín hiệu mua/bán. Dữ liệu nến ngày, múi giờ UTC.")


# ---------------------------------------------------------------- Trang 2
def page_forecast(sym):
    f = load_json("forecast.json")[sym]
    st.subheader(f"Biên độ biến động dự đoán cho ngày {f['forecast_for_date']} (UTC)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Biên độ dự đoán", f"{f['pred_range'] * 100:.2f}%")
    c2.metric("Khoảng 80% (thực nghiệm)", f"{f['band_low'] * 100:.2f}% – {f['band_high'] * 100:.2f}%")
    c3.metric("Ước tính ≈ USDT", f"{f['pred_range'] * f['last_close']:,.0f}")
    st.caption(
        f"Biên độ = (Cao − Thấp) / Đóng cửa của nến ngày dự đoán. Dữ liệu cuối: nến {f['last_candle_date']} "
        f"(đóng cửa {f['last_close']:,.2f}). Mô hình {f['model_version']}, huấn luyện trên {f['n_train']} ngày. "
        "Khoảng 80% lấy từ phân vị 10%–90% của tỷ lệ thực tế/dự đoán trong walk-forward, "
        "không phải đảm bảo. Số USDT chỉ là quy đổi gần đúng (biên độ × giá đóng cửa hôm nay).")

    o = load_oos(sym)
    n = st.sidebar.select_slider("Số ngày hiển thị (lịch sử dự đoán)", options=[30, 60, 120, 240, 900], value=120)
    t = o.tail(n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t["date"], y=t["actual"] * 100, mode="lines", name="Thực tế",
                             line=dict(color="#E8EAED", width=2)))
    fig.add_trace(go.Scatter(x=t["date"], y=t["pred"] * 100, mode="lines", name="XGBoost",
                             line=dict(color="#FF9F1C", width=2.5)))
    fig.add_trace(go.Scatter(x=t["date"], y=t["ma5"] * 100, mode="lines", name="Baseline TB 5 ngày",
                             line=dict(color="#7C8DA6", dash="dot", width=1.5)))
    fig.update_layout(height=420, yaxis_title="Biên độ nến (%)", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.1))
    st.markdown("**Lịch sử dự đoán ngoài mẫu (walk-forward)**")
    st.plotly_chart(fig)

    tbl = t.tail(10).iloc[::-1].copy()
    tbl["Ngày"] = tbl["date"].dt.date
    tbl["Thực tế (%)"] = tbl["actual"] * 100
    tbl["Dự đoán (%)"] = tbl["pred"] * 100
    tbl["Sai số (điểm %)"] = tbl["Dự đoán (%)"] - tbl["Thực tế (%)"]
    st.markdown("**10 ngày gần nhất trong lịch sử OOS**")
    st.dataframe(tbl[["Ngày", "Thực tế (%)", "Dự đoán (%)", "Sai số (điểm %)"]].style.format(
        {"Thực tế (%)": "{:.2f}", "Dự đoán (%)": "{:.2f}", "Sai số (điểm %)": "{:+.2f}"}), hide_index=True)


# ---------------------------------------------------------------- Trang 3
def page_metrics(sym):
    m = load_csv("metrics.csv")
    m = m[m["symbol"] == sym].copy()
    ref = m.loc[m["model"] == "Baseline: TB 5 ngày", "MAE"].iloc[0]
    ref_r = m.loc[m["model"] == "Baseline: TB 5 ngày", "RMSE"].iloc[0]
    m["MAE (% giá)"] = m["MAE"] * 100
    m["RMSE (% giá)"] = m["RMSE"] * 100
    m["MAE giảm vs TB5 (%)"] = (1 - m["MAE"] / ref) * 100
    m["RMSE giảm vs TB5 (%)"] = (1 - m["RMSE"] / ref_r) * 100
    st.subheader("So sánh mô hình trên dữ liệu ngoài mẫu (walk-forward, 5 fold × 180 ngày)")
    st.dataframe(m[["model", "MAE (% giá)", "RMSE (% giá)", "bias_%", "MAE giảm vs TB5 (%)",
                    "RMSE giảm vs TB5 (%)"]].rename(columns={"model": "Mô hình", "bias_%": "Bias (%)"}).style.format(
        {"MAE (% giá)": "{:.3f}", "RMSE (% giá)": "{:.3f}", "Bias (%)": "{:+.2f}",
         "MAE giảm vs TB5 (%)": "{:+.1f}", "RMSE giảm vs TB5 (%)": "{:+.1f}"}), hide_index=True)
    st.caption("MAE/RMSE tính bằng điểm phần trăm của biên độ nến. Bias > 0: dự đoán trung bình cao hơn thực tế. "
               "Baseline TB 5 ngày = trung bình biên độ 5 nến gần nhất.")

    fo = load_csv("folds.csv")
    fo = fo[fo["symbol"] == sym]
    wins = int((fo["MAE_xgb"] < fo["MAE_ma5"]).sum())
    st.markdown(f"**XGBoost thắng baseline TB 5 ngày (theo MAE) ở {wins}/{len(fo)} fold.**")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"Fold {k}\n{a}" for k, a in zip(fo["fold"], fo["from"])], y=fo["MAE_xgb"] * 100,
                         name="XGBoost", marker_color="#FF9F1C"))
    fig.add_trace(go.Bar(x=[f"Fold {k}\n{a}" for k, a in zip(fo["fold"], fo["from"])], y=fo["MAE_ma5"] * 100,
                         name="Baseline TB 5 ngày", marker_color="#7C8DA6"))
    fig.update_layout(barmode="group", height=380, yaxis_title="MAE (điểm %)", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig)
    st.dataframe(fo[["fold", "from", "to", "MAE_xgb", "MAE_ma5", "RMSE_xgb", "RMSE_ma5"]].style.format(
        {c: "{:.5f}" for c in ["MAE_xgb", "MAE_ma5", "RMSE_xgb", "RMSE_ma5"]}), hide_index=True)


# ---------------------------------------------------------------- Trang 4
def page_about():
    meta = load_json("meta.json")
    st.subheader("Giới thiệu & phương pháp")
    st.markdown(f"""
**Bài toán.** Dự đoán *biên độ nến ngày kế tiếp* của BTC/USDT và ETH/USDT:
`(High − Low) / Close`. Đây là thước đo mức biến động, **không** phải hướng tăng/giảm.

**Dữ liệu.** Nến ngày (UTC) từ API công khai của Binance, chỉ dùng nến đã đóng cửa.

**Mô hình.** XGBoost học `log(biên độ)`, có hiệu chỉnh smearing (Duan) khi đổi về thang gốc.
Baseline: biên độ hôm nay, trung bình 5 ngày, trung bình 20 ngày.

**Đặc trưng.** Biên độ các ngày trước, trung bình trượt biên độ, |return|, độ lệch chuẩn return 5/20 ngày,
thay đổi volume, thứ trong tuần.

**Đánh giá.** {meta['validation']}; mọi bước fit chỉ dùng dữ liệu trước thời điểm kiểm tra.

**Tại sao không dự đoán hướng giá?** Thử nghiệm ban đầu cho thấy return ngày kế tiếp gần như không dự đoán được
từ dữ liệu giá/volume (độ chính xác hướng ≈ 50%). Biến động thì có tính tụ cụm nên dự đoán được tốt hơn baseline.

**Phiên bản mô hình:** {meta['model_version']} · tạo lúc {meta['generated_at_utc']} (UTC).
""")
    st.warning(DISCLAIMER)


# ---------------------------------------------------------------- Main
st.title("📈 AI Crypto Volatility Forecast")
sym = st.sidebar.selectbox("Tài sản", SYMBOLS)
page = st.sidebar.radio("Trang", PAGES)

if page == PAGES[0]:
    page_prices(sym)
elif page == PAGES[1]:
    page_forecast(sym)
elif page == PAGES[2]:
    page_metrics(sym)
else:
    page_about()

st.divider()
st.caption(DISCLAIMER)
