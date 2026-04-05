from dotenv import load_dotenv
load_dotenv()
from collections import Counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data_sources import TopicDataService
from .model import SentimentModel

app = FastAPI(title="SentiFlow API")
model = SentimentModel()
topic_data = TopicDataService()

class PredictionRequest(BaseModel):
    text: str

class TopicRequest(BaseModel):
    keyword: str
    source: str = Field(default="all", pattern="^(all|newsapi|gnews|twitter)$")
    limit: int = Field(default=50, ge=10, le=100)

@app.get("/")
def root():
    return {"message": "SentiFlow FastAPI is running"}

@app.post("/predict")
def predict(req: PredictionRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")
    res = model.predict(req.text)
    return {
        "text": req.text,
        "label": res.get("label"),
        "score": float(res.get("score", 0.0)),
        "sentiment_value": float(res.get("sentiment_value", 0.0)),
        "emotion": res.get("emotion"),
        "emotion_score": float(res.get("emotion_score", 0.0)),
        "chunks": res.get("chunks", []),
    }


@app.post("/analyze_topic")
def analyze_topic(req: TopicRequest):
    keyword = req.keyword.strip()
    if not keyword:
        raise HTTPException(status_code=400, detail="Keyword is empty")

    fetch_result = topic_data.fetch(keyword=keyword, source=req.source, limit=req.limit)
    items = fetch_result["items"]
    errors = fetch_result["errors"]

    if not items and errors:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "All selected sources failed.",
                "sources": errors,
            },
        )

    if not items:
        raise HTTPException(
            status_code=404,
            detail=(
                "No texts found. Add NEWSAPI_KEY or GNEWS_API_KEY for news, "
                "or install snscrape for Twitter scraping."
            ),
        )

    texts = [item["text"] for item in items]
    predictions = model.predict_batch(texts)

    enriched_items = []
    emotion_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()

    for item, prediction in zip(items, predictions):
        emotion = prediction.get("emotion", "NEUTRAL")
        label = prediction.get("label", "NEUTRAL")
        emotion_counter[emotion] += 1
        label_counter[label] += 1
        enriched_items.append(
            {
                **item,
                "label": label,
                "score": float(prediction.get("score", 0.0)),
                "sentiment_value": float(prediction.get("sentiment_value", 0.0)),
                "emotion": emotion,
                "emotion_score": float(prediction.get("emotion_score", 0.0)),
            }
        )

    sentiment_label = label_counter.most_common(1)[0][0] if label_counter else "NEUTRAL"
    top_emotion = emotion_counter.most_common(1)[0][0] if emotion_counter else "NEUTRAL"

    return {
        "keyword": keyword,
        "source": req.source,
        "count": len(enriched_items),
        "sentiment_label": sentiment_label,
        "dominant_emotion": top_emotion,
        "emotion_breakdown": dict(emotion_counter),
        "source_errors": errors,
        "items": enriched_items,
    }
