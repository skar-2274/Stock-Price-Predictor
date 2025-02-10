import streamlit as st
import pickle
import numpy as np
import yfinance as yf
import subprocess

st.title("Stock Price Prediction App")

# Selects a stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT)", value="AAPL")

# Train model for chosen ticker
if st.button("Train Model For This Ticker"):
    st.write(f"Training model for {ticker}... (This may take a while)")
    subprocess.run(["python", "main.py", ticker])

# Load trained model
try:
    with open(f"stock_price_model_{ticker}.pkl", "rb") as f:
        model = pickle.load(f)

    # SMA_200 for the selected ticker
    def get_latest_sma_200(ticker):
        df = yf.download(ticker, period="1y")
        df["SMA_200"] = df["Close"].rolling(window=200).mean()
        latest_sma_200 = df["SMA_200"].dropna().iloc[-1]
        return latest_sma_200

    sma_200_value = get_latest_sma_200(ticker)
    st.write(f"Latest SMA 200 for {ticker}: {sma_200_value:.2f}")

    # Prediction
    if st.button("Predict"):
        prediction = model.predict(np.array([[sma_200_value]]))
        st.write(f"**Predicted Stock Price for {ticker}: ${prediction[0]:.2f}**")

except FileNotFoundError:
    st.write(f"No trained model found for {ticker}. Train one first.")
