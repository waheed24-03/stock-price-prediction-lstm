# 📈 Stock Market Prediction using LSTM

This project addresses the challenging task of **stock market prediction** by leveraging the power of **deep learning**.
We use a **Long Short-Term Memory (LSTM)** model (a type of Recurrent Neural Network) trained on historical stock data to identify temporal patterns and forecast the next day’s closing price.

The entire pipeline — from **data acquisition** to **model training** to **deployment** — is automated.
The final product is a **Streamlit web app** where users can select a stock, view predictions, and explore model performance.

---

## 🚀 Live Demo
🔗 [https://waheed-stock-prediction.streamlit.app/]()



---

## ✨ Key Features
- 🧠 **Deep Learning Core**: Robust LSTM neural network for time-series forecasting.
- 🤖 **Automated Training**: Single script handles fetching, preprocessing, and training models for 20+ stocks.
- 🌐 **Interactive Web Interface**: Clean and responsive Streamlit dashboard.
- 📊 **Instant Predictions**: Select a stock to get the next closing price prediction.
- 📈 **Performance Visualization**: Compare predicted vs. actual values with interactive graphs.
- 🔁 **Reproducibility**: Random seeds ensure consistent, reproducible training results.

---

## 🛠 Technology Stack
| Technology | Purpose |
|-----------------|----------|
| **Python** | Core programming language |
| **TensorFlow / Keras** | Building & training LSTM models |
| **Streamlit** | Web application interface |
| **Pandas / NumPy** | Data manipulation & preprocessing |
| **Scikit-learn** | Scaling & metrics |
| **yfinance** | Fetching real-time stock data |
| **Plotly / Matplotlib** | Data visualizations |



## 📊 Model Performance Results

The models were trained on stock data (2015–Present) with an **80/20 train-test split**.
Here is the performance summary across 20 stocks:

| Rank | Stock Ticker | R-squared (R²) | RMSE (₹) | Model Performance |
|------|--------------|----------------|----------|-------------------|
| 1 | WIPRO.NS | **94.37%** | ₹7.38 | 🟢 Excellent |
| 2 | ICICIBANK.NS | **94.19%** | ₹40.52 | 🟢 Excellent |
| 3 | ASIANPAINT.NS| **94.11%** | ₹88.09 | 🟢 Excellent |
| 4 | INFY.NS | 92.88% | ₹51.28 | 🟢 Excellent |
| 5 | LT.NS | 92.61% | ₹79.49 | 🟢 Excellent |
| 6 | HCLTECH.NS | 90.98% | ₹66.22 | 🟢 Very Good |
| 7 | BAJFINANCE.NS| 90.88% | ₹27.04 | 🟢 Very Good |
| 8 | TCS.NS | 90.54% | ₹112.43 | 🟢 Very Good |
| 9 | HINDUNILVR.NS| 89.85% | ₹52.49 | 🟢 Very Good |
| 10 | DMART.NS | 89.64% | ₹163.88 | 🟢 Very Good |
| 11 | SBIN.NS | 88.48% | ₹33.15 | 🟡 Good |
| 12 | KOTAKBANK.NS | 88.01% | ₹52.00 | 🟡 Good |
| 13 | AXISBANK.NS | 87.87% | ₹30.05 | 🟡 Good |
| 14 | MARUTI.NS | 87.63% | ₹358.22 | 🟡 Good |
| 15 | HDFCBANK.NS | 86.10% | ₹64.62 | 🟡 Good |
| 16 | BHARTIARTL.NS| 84.78% | ₹132.57 | 🟡 Decent |
| 17 | ITC.NS | 84.13% | ₹11.30 | 🟡 Decent |
| 18 | RELIANCE.NS | 80.20% | ₹53.19 | 🟡 Decent |
| 19 | TITAN.NS | 77.09% | ₹99.99 | 🟠 Fair |
| 20 | ULTRACEMCO.NS| 73.44% | ₹652.85 | 🟠 Fair |

---



🔮 Future Enhancements
- 📊 Multivariate Analysis: Add volume, RSI, MACD, and sentiment data.
- ⚡ Hyperparameter Tuning: Automated search (KerasTuner).
- ☁️ Cloud Deployment: Deploy on AWS/GCP/Heroku for permanent hosting.


 Acknowledgements

- Yahoo Finance API (via yfinance)
- Streamlit community





