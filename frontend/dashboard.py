import streamlit as st
#import requests
import pandas as pd
import time
#from requests import HTTPError
from wordcloud import WordCloud
#import os

#API_URL = os.getenv("API_URL", "http://localhost:8000")
from model import SentimentModel
from data_sources import TopicDataService
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def load_model():
    return SentimentModel()

@st.cache_resource
def load_topic_service():
    return TopicDataService()

model = load_model()
topic_data = load_topic_service()

st.set_page_config(page_title="SentiFlow Dashboard", layout="wide")

st.title("SentiFlow - Real-Time Sentiment Intelligence")

# Sidebar
with st.sidebar:
    st.header("Actions")
    mode = st.radio("Mode:", ["Manual Text", "Topic Tracking"])
    autorefresh = st.checkbox("Auto-refresh predictions", value=False)
    refresh_interval = st.slider("Auto-refresh interval (seconds)", 5, 60, 15)


# API functions
# @st.cache_data
# def fetch_prediction(text):
#     r = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=60)
#     r.raise_for_status()
#     return r.json()
@st.cache_data
def fetch_prediction(text):
    return model.predict(text)

# @st.cache_data(show_spinner=False)
# def fetch_topic(keyword, source, limit):
#     r = requests.post(
#         f"{API_URL}/analyze_topic",
#         json={"keyword": keyword, "source": source, "limit": limit},
#         timeout=60,
#     )
#     r.raise_for_status()
#     return r.json()
@st.cache_data(show_spinner=False)
def fetch_topic(keyword, source, limit):
    fetch_result = topic_data.fetch(
        keyword=keyword,
        source=source,
        limit=limit
    )

    items = fetch_result["items"]
    errors = fetch_result["errors"]

    if not items:
        return {
            "keyword": keyword,
            "source": source,
            "count": 0,
            "sentiment_label": "NEUTRAL",
            "dominant_emotion": "NEUTRAL",
            "emotion_breakdown": {},
            "source_errors": errors,
            "items": [],
        }

    texts = [item["text"] for item in items]

    predictions = model.predict_batch(texts)

    enriched_items = []
    emotion_counter = Counter()
    label_counter = Counter()

    for item, prediction in zip(items, predictions):

        emotion = prediction.get("emotion", "NEUTRAL")
        label = prediction.get("label", "NEUTRAL")

        emotion_counter[emotion] += 1
        label_counter[label] += 1

        enriched_items.append({
            **item,
            "label": label,
            "score": float(prediction.get("score", 0.0)),
            "sentiment_value": float(
                prediction.get("sentiment_value", 0.0)
            ),
            "emotion": emotion,
            "emotion_score": float(
                prediction.get("emotion_score", 0.0)
            ),
        })

    sentiment_label = (
        label_counter.most_common(1)[0][0]
        if label_counter
        else "NEUTRAL"
    )

    top_emotion = (
        emotion_counter.most_common(1)[0][0]
        if emotion_counter
        else "NEUTRAL"
    )

    return {
        "keyword": keyword,
        "source": source,
        "count": len(enriched_items),
        "sentiment_label": sentiment_label,
        "dominant_emotion": top_emotion,
        "emotion_breakdown": dict(emotion_counter),
        "source_errors": errors,
        "items": enriched_items,
    }

def render_wordcloud(texts):
    joined = " ".join(str(text) for text in texts if str(text).strip()).strip()
    if not joined:
        st.info("No text available for the word cloud.")
        return
    st.image(
        WordCloud(width=800, height=400).generate(joined).to_image(),
        use_container_width=True,
    )


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
            with st.spinner("Analyzing text..."):
                pred = fetch_prediction(text)

            metrics = st.columns(4)
            metrics[0].metric("Sentiment", pred["label"])
            metrics[1].metric("Confidence", f"{pred['score']:.2f}")
            metrics[2].metric("Sentiment Score", f"{pred.get('sentiment_value', 0.0):.2f}")
            metrics[3].metric("Emotion", pred.get("emotion", "NEUTRAL"))

            # 🔥 Smart chunk splitting
            chunk_df = pd.DataFrame(
                [
                    {
                        "chunk": item.get("text"),
                        "label": item.get("sentiment_label", item.get("label")),
                        "emotion": item.get("emotion_label", item.get("emotion")),
                        "score": item.get("sentiment_score", item.get("score")),
                        "sentiment_value": item.get("sentiment_value"),
                    }
                    for item in pred.get("chunks", [])
                ]
            )

            # 🔥 Batch prediction (FAST)
            if not chunk_df.empty:

                st.subheader("Chunk-Level Breakdown")

                cols = st.columns(2)

                with cols[0]:
                    st.bar_chart(chunk_df["label"].value_counts())

                with cols[1]:
                    st.bar_chart(chunk_df["emotion"].value_counts())

                st.dataframe(chunk_df, use_container_width=True)

            # 🔹 Wordcloud
            st.subheader("Word Cloud")
            render_wordcloud([text])

            st.subheader("Analysis Snapshot")
            st.dataframe(pd.DataFrame([pred]), use_container_width=True)

            # 🔄 Auto refresh
            if autorefresh:
                for i in range(1, 10):
                    time.sleep(refresh_interval)
                    pred = fetch_prediction(text)
                    st.write(f"Refresh {i}: {pred}")



elif mode == "Topic Tracking":
    controls = st.columns([3, 1, 1])
    keyword = controls[0].text_input("Search keyword", placeholder="e.g. Tesla, IPL, iPhone")
    # source = controls[1].selectbox("Source", ["all", "newsapi", "gnews", "twitter"])
    source = controls[1].selectbox("Source",["gnews"])
    limit = controls[2].slider("Texts", 50, 100, 50, step=10)

    if st.button("Track Topic"):
        if not keyword.strip():
            st.warning("Enter a keyword to fetch and analyze real texts.")
        else:
            with st.spinner(
                f"Fetching and analyzing {limit} texts for '{keyword}'..."
            ):
                topic = fetch_topic(keyword, source, limit)

            source_errors = topic.get("source_errors", {})

            if source_errors:
                st.warning("Some sources failed.")
                st.json(source_errors)

            metrics = st.columns(4)

            sentiment_label = topic.get("sentiment_label", "NEUTRAL")

            metrics[0].metric(
                "Texts analyzed",
                topic.get("count", 0)
            )

            metrics[1].metric(
                "Sentiment",
                sentiment_label
            )

            metrics[2].metric(
                "Dominant emotion",
                topic.get("dominant_emotion", "NEUTRAL")
            )

            metrics[3].metric(
                "Source",
                topic.get("source", "gnews")
            )

            emotion_df = pd.DataFrame(
                list(topic.get("emotion_breakdown", {}).items()),
                columns=["emotion", "count"],
            )

            if not emotion_df.empty:
                st.subheader("Emotion Breakdown")
                st.bar_chart(
                    emotion_df.set_index("emotion")
                )

            items_df = pd.DataFrame(
                topic.get("items", [])
            )

            st.subheader("Fetched Texts")

            if items_df.empty:
                st.info(
                    "No texts were found for this topic."
                )
            else:
                display_columns = [
                    "published_at",
                    "source",
                    "author",
                    "label",
                    "emotion",
                    "score",
                    "sentiment_value",
                    "text",
                ]

                display_columns = [
                    column
                    for column in display_columns
                    if column in items_df.columns
                ]

                st.dataframe(
                    items_df[display_columns],
                    use_container_width=True,
                )

                st.subheader("Topic Word Cloud")

                render_wordcloud(
                    items_df["text"].tolist()
                )