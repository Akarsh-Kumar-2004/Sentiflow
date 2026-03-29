import streamlit as st
import requests
import pandas as pd
import time
import re
from wordcloud import WordCloud

API_URL = "http://localhost:8000"

st.set_page_config(page_title="SentiFlow Dashboard", layout="wide")

st.title("SentiFlow - Real-Time Sentiment Intelligence")

# Sidebar
with st.sidebar:
    st.header("Actions")
    mode = st.radio("Mode:", ["Manual Text", "Batch CSV"])
    autorefresh = st.checkbox("Auto-refresh predictions", value=False)
    refresh_interval = st.slider("Auto-refresh interval (seconds)", 5, 60, 15)


# API functions
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


# =========================
# 🔹 MANUAL MODE
# =========================
if mode == "Manual Text":
    text = st.text_area("Enter text to analyze", height=150)

    if st.button("Analyze"):
        if text.strip() == "":
            st.warning("Please type a non-empty text")

        else:
            # 🔹 Single prediction
            pred = fetch_prediction(text)

            st.metric("Sentiment", pred["label"])
            st.write(f"Confidence: {pred['score']:.2f}")
            st.write(pred)

            df = pd.DataFrame([pred])

            # 🔥 Smart chunk splitting
            chunks = [c.strip() for c in re.split(r"[,.!?]", text) if c.strip()]

            # 🔥 Batch prediction (FAST)
            try:
                batch_res = fetch_batch(chunks)
                chunk_labels = [item["label"] for item in batch_res["items"]]
            except:
                chunk_labels = ["UNKNOWN"] * len(chunks)

            chunk_df = pd.DataFrame({
                "chunk": chunks,
                "label": chunk_labels
            })

            counts = chunk_df["label"].value_counts()

            st.subheader("Sentiment Distribution (by text chunks)")

            cols = st.columns(2)

            with cols[0]:
                st.bar_chart(counts)

            with cols[1]:
                st.write("Chunk-wise Sentiment:")
                st.dataframe(chunk_df)

            # 🔹 Wordcloud
            st.subheader("Word Cloud")
            st.image(
                WordCloud(width=400, height=250)
                .generate(text)
                .to_image(),
                use_container_width=True
            )

            st.subheader("Recent Predictions")
            st.dataframe(df)

            # 🔄 Auto refresh
            if autorefresh:
                for i in range(1, 10):
                    time.sleep(refresh_interval)
                    pred = fetch_prediction(text)
                    st.write(f"Refresh {i}: {pred}")


# =========================
# 🔹 BATCH MODE
# =========================
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

            st.subheader("Word Cloud")
            st.image(
                WordCloud(width=800, height=400)
                .generate(" ".join(df[colname].astype(str).tolist()))
                .to_image(),
                use_container_width=True
            )

            st.subheader("Timeline Trend")
            st.line_chart(pd_result["score"])

            if autorefresh:
                while True:
                    time.sleep(refresh_interval)
                    pred = fetch_batch(texts)
                    st.write("Auto-refreshed batch", pred["items"][:3])
                    st.experimental_rerun()