import streamlit as st
import requests
import pandas as pd
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt

API_URL = "http://localhost:8000"

st.set_page_config(page_title="SentiFlow Dashboard", layout="wide")

st.title("SentiFlow - Real-Time Sentiment Intelligence")

with st.sidebar:
    st.header("Actions")
    mode = st.radio("Mode:", ["Manual Text", "Batch CSV"])
    autorefresh = st.checkbox("Auto-refresh predictions", value=False)
    refresh_interval = st.slider("Auto-refresh interval (seconds)", min_value=5, max_value=60, value=15)

@st.cache_data
def fetch_prediction(text):
    r = requests.post(f"{API_URL}/predict", json={"text": text})
    r.raise_for_status()
    return r.json()

@st.cache_data
def fetch_batch(texts):
    r = requests.post(f"{API_URL}/predict_batch", json={"texts": texts})
    r.raise_for_status()
    return r.json()

@st.cache_data
def fetch_csv(csv_path, text_column):
    r = requests.post(f"{API_URL}/predict_csv", json={"csv_path": csv_path, "text_column": text_column})
    r.raise_for_status()
    return r.json()

if mode == "Manual Text":
    text = st.text_area("Enter text to analyze", height=150)
    if st.button("Analyze"):
        if text.strip() == "":
            st.warning("Please type a non-empty text")
        else:
            pred = fetch_prediction(text)
            st.metric("Sentiment", pred["label"], delta=f"{pred['score']:.2f}")
            st.write(pred)

            df = pd.DataFrame([pred])
            counts = df["label"].value_counts()
            st.subheader("Sentiment Distribution")
            cols = st.columns(2)
            with cols[0]:
                st.bar_chart(counts)
            with cols[1]:
                st.image(WordCloud(width=400, height=250).generate(" ".join([text])).to_image(), use_column_width=True)

            st.subheader("Recent Predictions")
            st.dataframe(df)

            if autorefresh:
                for i in range(1, 10):
                    time.sleep(refresh_interval)
                    pred = fetch_prediction(text)
                    st.write(f"Refresh {i}: {pred}")

else:
    csv_file = st.file_uploader("Upload CSV file", type=["csv"])
    colname = st.text_input("Text column name", value="text")
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        st.write(df.head())
        if st.button("Analyze Batch"):
            texts = df[colname].astype(str).tolist()
            pred = fetch_batch(texts)
            pd_result = pd.DataFrame(pred["items"])
            st.dataframe(pd_result)

            st.subheader("Batch Sentiment Distribution")
            dist = pd_result["label"].value_counts()
            st.bar_chart(dist)
            st.image(WordCloud(width=800, height=400).generate(" ".join(df[colname].astype(str).tolist())).to_image(), use_column_width=True)

            st.subheader("Timeline Trend (simulated by row order)")
            trend = pd.DataFrame({"score": pd_result["score"], "label": pd_result["label"]})
            st.line_chart(trend["score"])

            if autorefresh:
                while True:
                    time.sleep(refresh_interval)
                    pred = fetch_batch(texts)
                    st.write("Auto-refreshed batch", pred["items"][0:3])
                    st.experimental_rerun()
