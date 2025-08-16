import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import os
import plotly.graph_objects as go

# App Configuration 
st.set_page_config(
    page_title="Indian Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# App Title and Description 
st.title("📈 Indian Stock Price Predictor")
st.markdown("""
This app uses trained LSTM neural networks to predict the next closing price for 20 major stocks from the Indian National Stock Exchange (NSE).
After making a prediction, a graph will show the model's historical performance on the test dataset.
**Disclaimer:** This is for educational purposes only and should not be considered financial advice.
""")

# Stock List 
top_20_indian_stocks = {
    'RELIANCE.NS': 'Reliance Industries', 'TCS.NS': 'Tata Consultancy Services',
    'HDFCBANK.NS': 'HDFC Bank', 'INFY.NS': 'Infosys', 'ICICIBANK.NS': 'ICICI Bank',
    'HINDUNILVR.NS': 'Hindustan Unilever', 'SBIN.NS': 'State Bank of India',
    'BAJFINANCE.NS': 'Bajaj Finance', 'KOTAKBANK.NS': 'Kotak Mahindra Bank',
    'WIPRO.NS': 'Wipro', 'BHARTIARTL.NS': 'Bharti Airtel', 'ITC.NS': 'ITC Limited',
    'HCLTECH.NS': 'HCL Technologies', 'ASIANPAINT.NS': 'Asian Paints', 'LT.NS': 'Larsen & Toubro',
    'AXISBANK.NS': 'Axis Bank', 'MARUTI.NS': 'Maruti Suzuki', 'DMART.NS': 'Avenue Supermarts',
    'ULTRACEMCO.NS': 'UltraTech Cement', 'TITAN.NS': 'Titan Company'
}

#  Model and Data Loading (with Caching for speed) 
@st.cache_resource
def load_keras_model(model_path):
    """Loads a Keras model from the specified path."""
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

@st.cache_data
def load_full_stock_data(tickers):
    """Downloads and caches the full historical stock data from Yahoo Finance."""
    print("Downloading historical data for performance graphs...")
    start_date = "2015-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    df = data.stack().reset_index()
    df.rename(columns={
        'Date': 'date',
        'Ticker': 'Name',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    
    print("Download complete.")
    return df

#  Sidebar for User Input 
st.sidebar.header("Make a Prediction")
selected_stock_name = st.sidebar.selectbox(
    "Select a Stock",
    options=list(top_20_indian_stocks.values())
)
selected_ticker = [ticker for ticker, name in top_20_indian_stocks.items() if name == selected_stock_name][0]

#  Main App Logic 
# Load the historical data for all stocks needed for the graphs
full_df = load_full_stock_data(list(top_20_indian_stocks.keys()))

if full_df.empty:
    st.error("Could not download historical data from Yahoo Finance. Please check your internet connection.")
else:
    if st.sidebar.button("Predict & Show Performance"):
        model_path = f'trained_models/lstm_model_{selected_ticker}.keras'
        model = load_keras_model(model_path)

        if model is None:
            st.error(f"Model for {selected_stock_name} not found. Please ensure it has been trained and is in the 'trained_models' folder.")
        else:
            # --- Prediction for Next Day ---
            with st.spinner(f"Predicting for {selected_stock_name}..."):
                n_steps = 20
                # Fetch live data for the prediction
                stock_data_live = yf.download(selected_ticker, start=datetime.now() - timedelta(days=60), end=datetime.now(), progress=False)
                
                last_n_days = stock_data_live['Close'].values[-n_steps:]
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(last_n_days.reshape(-1, 1))
                input_data = np.reshape(scaled_data, (1, n_steps, 1))
                
                prediction_scaled = model.predict(input_data)
                predicted_price = scaler.inverse_transform(prediction_scaled)[0][0]
                
               
                # Explicitly convert the numpy float to a standard Python float
                last_close_price = float(last_n_days[-1])
                
                st.success(f"Prediction for {selected_stock_name} ({selected_ticker})")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Last Close Price", f"₹{last_close_price:.2f}")
                col2.metric("Predicted Next Close", f"₹{predicted_price:.2f}")
                
                change = predicted_price - last_close_price
                percent_change = (change / last_close_price) * 100
                col3.metric("Predicted Change", f"₹{change:.2f} ({percent_change:.2f}%)")

            # Historical Performance Graph 
            st.markdown("---")
            st.subheader("Historical Model Performance")
            with st.spinner("Generating performance graph..."):
                stock_df = full_df[full_df['Name'] == selected_ticker].copy()
                stock_df.set_index('date', inplace=True)
                
                split_point = int(len(stock_df) * 0.8)
                training_data = stock_df.iloc[:split_point]
                testing_data = stock_df.iloc[split_point:]

                scaler_hist = MinMaxScaler(feature_range=(0, 1))
                scaler_hist.fit(training_data['close'].values.reshape(-1, 1))
                
                test_prices = testing_data['close'].values.reshape(-1, 1)
                test_data_scaled = scaler_hist.transform(test_prices)

                X_test = []
                for i in range(n_steps, len(test_data_scaled)):
                    X_test.append(test_data_scaled[i-n_steps:i, 0])
                X_test = np.array(X_test)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                hist_predictions_scaled = model.predict(X_test)
                hist_predicted_prices = scaler_hist.inverse_transform(hist_predictions_scaled)

                testing_data_with_preds = testing_data.iloc[n_steps:].copy()
                testing_data_with_preds['Prediction'] = hist_predicted_prices

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=training_data.index, y=training_data['close'], mode='lines', name='Actual Training Price', line=dict(color='royalblue')))
                fig.add_trace(go.Scatter(x=testing_data_with_preds.index, y=testing_data_with_preds['close'], mode='lines', name='Actual Test Price', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=testing_data_with_preds.index, y=testing_data_with_preds['Prediction'], mode='lines', name='Predicted Price', line=dict(color='red', dash='dash')))
                
                fig.update_layout(
                    title=f'Model Performance for {selected_stock_name}',
                    xaxis_title='Date', yaxis_title='Close Price (₹)',
                    legend_title='Legend', height=500
                )
                st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.write("By Syed Abdul Waheed")
