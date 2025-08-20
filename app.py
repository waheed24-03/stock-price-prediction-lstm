import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# -------------------- App Config --------------------
st.set_page_config(page_title="Indian Stock Price Predictor", page_icon="📈", layout="wide")
st.title("📈 Indian Stock Price Predictor")
st.markdown(
    """
This app uses trained LSTM neural networks to predict the next closing price for 20 major NSE stocks.
After making a prediction, a graph will show the model's historical performance on the test dataset.

**Disclaimer:** Educational use only — not financial advice.
"""
)

# -------------------- Constants --------------------
N_STEPS = 20

# -------------------- Stock List --------------------
top_20_indian_stocks = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "SBIN.NS": "State Bank of India",
    "BAJFINANCE.NS": "Bajaj Finance",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "WIPRO.NS": "Wipro",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ITC.NS": "ITC Limited",
    "HCLTECH.NS": "HCL Technologies",
    "ASIANPAINT.NS": "Asian Paints",
    "LT.NS": "Larsen & Toubro",
    "AXISBANK.NS": "Axis Bank",
    "MARUTI.NS": "Maruti Suzuki",
    "DMART.NS": "Avenue Supermarts",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "TITAN.NS": "Titan Company",
}

# -------------------- Helpers --------------------
@st.cache_resource
def load_keras_model(model_path: str):
    """Load a Keras model if present, else return None."""
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

@st.cache_data(ttl=60 * 30)  # cache 30 minutes
def download_history_single(ticker: str, start: str = "2015-01-01") -> pd.DataFrame:
    """
    Robust single-ticker downloader to avoid Yahoo rate limits for multi-ticker.
    Returns a dataframe with columns: date, open, high, low, close, volume, Name
    """
    try:
        # Using Ticker().history is more reliable than multi-ticker download
        end_date = (datetime.utcnow() + timedelta(days=1)).date()
        hist = (
            yf.Ticker(ticker)
            .history(start=start, end=end_date, interval="1d", auto_adjust=False, actions=False)
            .rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
        )
        if hist.empty:
            return pd.DataFrame()
        hist = hist.reset_index()
        if "Date" in hist.columns:
            hist = hist.rename(columns={"Date": "date"})
        hist["Name"] = ticker
        return hist[["date", "open", "high", "low", "close", "volume", "Name"]]
    except Exception:
        return pd.DataFrame()

def make_prediction_from_series(model, close_series: np.ndarray, n_steps: int = N_STEPS):
    """Scale last n_steps and predict next close; return predicted float."""
    if len(close_series) < n_steps:
        raise ValueError(f"Not enough data to predict (need {n_steps}, have {len(close_series)})")
    last_n = close_series[-n_steps:].astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(last_n.reshape(-1, 1))
    x = scaled.reshape(1, n_steps, 1)
    y_scaled = model.predict(x, verbose=0)
    y = scaler.inverse_transform(y_scaled)[0][0]
    return float(last_n[-1]), float(y)

# -------------------- Sidebar --------------------
st.sidebar.header("Make a Prediction")
selected_stock_name = st.sidebar.selectbox("Select a Stock", options=list(top_20_indian_stocks.values()))
selected_ticker = [t for t, n in top_20_indian_stocks.items() if n == selected_stock_name][0]

# -------------------- Main --------------------
if st.sidebar.button("Predict & Show Performance"):
    model_path = f"trained_models/lstm_model_{selected_ticker}.keras"
    model = load_keras_model(model_path)

    if model is None:
        st.error(
            f"Model for {selected_stock_name} not found.\n"
            f"Expected file: `{model_path}`"
        )
        st.stop()

    with st.spinner(f"Fetching {selected_stock_name} history..."):
        df = download_history_single(selected_ticker, start="2015-01-01")

    if df.empty:
        st.error(
            "Could not download data from Yahoo Finance (may be rate-limited or blocked).\n"
            "Please try again later."
        )
        st.stop()

    # ----- Prediction using the same history (no extra live call) -----
    try:
        last_close, predicted = make_prediction_from_series(model, df["close"].values, n_steps=N_STEPS)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.success(f"Prediction for {selected_stock_name} ({selected_ticker})")

    c1, c2, c3 = st.columns(3)
    c1.metric("Last Close Price", f"₹{last_close:,.2f}")
    c2.metric("Predicted Next Close", f"₹{predicted:,.2f}")
    change = predicted - last_close
    pct = (change / last_close) * 100 if last_close else 0.0
    c3.metric("Predicted Change", f"₹{change:,.2f} ({pct:.2f}%)")

    # ----- Historical Performance Chart -----
    st.markdown("---")
    st.subheader("Historical Model Performance")

    with st.spinner("Generating performance graph..."):
        df = df.sort_values("date").set_index("date")
        split = int(len(df) * 0.8)
        train, test = df.iloc[:split], df.iloc[split:]

        scaler_hist = MinMaxScaler(feature_range=(0, 1))
        scaler_hist.fit(train["close"].values.reshape(-1, 1))

        test_prices = test["close"].values.reshape(-1, 1)
        test_scaled = scaler_hist.transform(test_prices)

        X_test = []
        for i in range(N_STEPS, len(test_scaled)):
            X_test.append(test_scaled[i - N_STEPS : i, 0])
        X_test = np.array(X_test).reshape(-1, N_STEPS, 1)

        if len(X_test) == 0:
            st.warning("Not enough test data to evaluate performance.")
        else:
            yhat_scaled = model.predict(X_test, verbose=0)
            yhat = scaler_hist.inverse_transform(yhat_scaled).ravel()
            test_with_preds = test.iloc[N_STEPS:].copy()
            test_with_preds["Prediction"] = yhat

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=train.index, y=train["close"], mode="lines", name="Actual Training Price"
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=test_with_preds.index, y=test_with_preds["close"], mode="lines", name="Actual Test Price"
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=test_with_preds.index,
                    y=test_with_preds["Prediction"],
                    mode="lines",
                    name="Predicted Price",
                )
            )
            fig.update_layout(
                title=f"Model Performance for {selected_stock_name}",
                xaxis_title="Date",
                yaxis_title="Close Price (₹)",
                legend_title="Legend",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.write("By Syed Abdul Waheed")
