import streamlit as st
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
from textblob import TextBlob
from GoogleNews import GoogleNews
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Configuration
st.set_page_config(
    page_title="SentiStock: AI-Powered US Stock Analysis and Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define class labels
CLASSES = ["Sell", "Buy"]

# Load models
@st.cache_resource
def load_models():
    try:
        model = load_model("resnl_stock_sentiment_model.h5")
        with open("scaler_resnl.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

model, scaler = load_models()

# Helper functions (updated as described above)

# UI Components
st.title("SentiStock: AI-Powered US Stock Insights 📊")
st.markdown("---")

# Main content
col1, col2 = st.columns([1, 3])

with col1:
    ticker = st.text_input("Enter US Stock Ticker", value="AAPL")
    stock_data = pd.DataFrame()  # Initialize stock_data
    latest_data = None
    recommendation = None
    confidence = None
    probs = {}

    if model is not None and scaler is not None:
        try:
            with st.spinner("Fetching stock data..."):
                stock_data = get_stock_data(ticker)
            if stock_data.empty:
                st.warning("No stock data available.")
            else:
                with st.spinner("Fetching news sentiment..."):
                    news_features = get_news_sentiment(ticker)
                with st.spinner("Calculating technical indicators..."):
                    processed_data = calculate_technical_indicators(stock_data)
                    latest_data = processed_data.iloc[-1]
                with st.spinner("Preparing features and making predictions..."):
                    features = prepare_features(processed_data, news_features)
                    scaled_data = scaler.transform(features)
                    pred_probs = model.predict(scaled_data)[0]
                    recommendation, confidence, probs = get_recommendation(pred_probs, CLASSES)
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")

with col2:
    if not stock_data.empty and latest_data is not None:
        st.subheader(f"{ticker} Technical Analysis")
        st.line_chart(stock_data[['Close', 'Moving_Avg']])
        
        col2_1, col2_2, col2_3 = st.columns(3)
        with col2_1:
            st.metric("Current Price", f"${latest_data['Close']:.2f}")
            st.metric("RSI", f"{latest_data['RSI']:.2f}")
        with col2_2:
            st.metric("14-Day EMA", f"${latest_data['EMA']:.2f}")
            st.metric("Daily Volume", f"{latest_data['Volume']:,.0f}")
        with col2_3:
            st.metric("News Sentiment", f"{news_features['Sentiment_Score']:.2f}")
        
        st.markdown("---")
        st.subheader("Recent News Headlines")
        for headline, sentiment in zip(news_features['Headlines'], news_features['Sentiments']):
            st.write(f"Headline: {headline}")
            st.write(f"Sentiment: {'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'}")
            st.write("---")
    else:
        st.warning("No stock data available to display technical analysis.")

st.markdown("---")
st.subheader("Investment Recommendation")
if recommendation is not None:
    col3_1, col3_2 = st.columns([1, 2])
    with col3_1:
        st.metric("Recommendation", recommendation)
        st.progress(confidence)
        st.caption(f"Confidence Level: {confidence*100:.1f}%")
        st.markdown("**Probabilities:**")
        for label, prob in probs.items():
            st.write(f"**{label}:** {prob*100:.1f}%")
    with col3_2:
        if recommendation == "Buy":
            st.success("**Analysis:** Strong positive indicators detected. Consider adding to your portfolio.")
        elif recommendation == "Sell":
            st.error("**Analysis:** Negative trends detected. Consider reducing your position.")
    st.markdown("---")
    st.subheader("Recent News Analysis")
    gn = GoogleNews()
    gn.search(f"{ticker} stock news")
    results = gn.results()[:5]
    for news in results:
        with st.expander(news['title']):
            st.caption(news['media'])
            st.write(news['desc'])
            st.caption(news['date'])
else:
    st.warning("Enter a valid stock ticker to see analysis.")

st.markdown("---")
st.caption("© 2025 US Stock Analyzer. For educational purposes only.")
st.caption("Disclaimer: This is subject to risk read all market regulations and rules clearly before investing. No legal binding shall be attached to us.")
