import streamlit as st
import requests
import pandas as pd
import time
from requests import HTTPError
from wordcloud import WordCloud

API_URL = "http://localhost:8000"

st.set_page_config(page_title="SentiFlow Dashboard", layout="wide")

st.title("SentiFlow - Real-Time Sentiment Intelligence")

# Sidebar
with st.sidebar:
    st.header("Actions")
    mode = st.radio("Mode:", ["Manual Text", "Topic Tracking"])
    autorefresh = st.checkbox("Auto-refresh predictions", value=False)
    refresh_interval = st.slider("Auto-refresh interval (seconds)", 5, 60, 15)


# API functions
@st.cache_data
def fetch_prediction(text):
    r = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=60)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False)
def fetch_topic(keyword, source, limit):
    r = requests.post(
        f"{API_URL}/analyze_topic",
        json={"keyword": keyword, "source": source, "limit": limit},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


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
    source = controls[1].selectbox("Source", ["all", "newsapi", "gnews", "twitter"])
    limit = controls[2].slider("Texts", 50, 100, 50, step=10)

    if st.button("Track Topic"):
        if not keyword.strip():
            st.warning("Enter a keyword to fetch and analyze real texts.")
        else:
            try:
                with st.spinner(f"Fetching and analyzing {limit} texts for '{keyword}'..."):
                    topic = fetch_topic(keyword, source, limit)
            except HTTPError as exc:
                try:
                    detail = exc.response.json().get("detail", str(exc))
                except ValueError:
                    detail = str(exc)
                if isinstance(detail, dict):
                    st.error(detail.get("message", "Topic tracking failed."))
                    if detail.get("sources"):
                        st.json(detail["sources"])
                else:
                    st.error(detail)
            else:
                source_errors = topic.get("source_errors", {})
                if source_errors:
                    st.warning("Some sources failed, but partial results are shown below.")
                    st.json(source_errors)

                metrics = st.columns(4)
                sentiment_label = topic.get("sentiment_label")
                if not sentiment_label:
                    avg_sentiment = float(topic.get("avg_sentiment", 0.0))
                    if avg_sentiment > 0.2:
                        sentiment_label = "POSITIVE"
                    elif avg_sentiment < -0.2:
                        sentiment_label = "NEGATIVE"
                    else:
                        sentiment_label = "NEUTRAL"
                metrics[0].metric("Texts analyzed", topic["count"])
                metrics[1].metric("Sentiment", sentiment_label)
                metrics[2].metric("Dominant emotion", topic["dominant_emotion"])
                metrics[3].metric("Source", topic["source"])

                emotion_df = pd.DataFrame(
                    list(topic["emotion_breakdown"].items()),
                    columns=["emotion", "count"],
                )
                if not emotion_df.empty:
                    st.subheader("Emotion Breakdown")
                    st.bar_chart(emotion_df.set_index("emotion"))

                items_df = pd.DataFrame(topic["items"])
                st.subheader("Fetched Texts")
                st.dataframe(
                    items_df[
                        [
                            "published_at",
                            "source",
                            "author",
                            "label",
                            "emotion",
                            "score",
                            "sentiment_value",
                            "text",
                        ]
                    ],
                    use_container_width=True,
                )

                st.subheader("Topic Word Cloud")
                render_wordcloud(items_df["text"].tolist())
