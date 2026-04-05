from collections import Counter
from datetime import datetime
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data_sources import TopicDataService
from .model import SentimentModel

app = FastAPI(title="SentiFlow API")
model = SentimentModel()
topic_data = TopicDataService()

class PredictionRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]

class FileBatchRequest(BaseModel):
    csv_path: str
    text_column: Optional[str] = "text"


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
#fixing the predict_batch endpoint to handle empty list and return proper error message
#also added error handling for the predict_csv endpoint to handle cases where the CSV file cannot be loaded or the specified text column is not found.
@app.post("/predict_batch")
def predict_batch(req: BatchRequest):
    if len(req.texts) == 0:
        raise HTTPException(status_code=400, detail="texts list is empty")
    results = model.predict_batch(req.texts)
    output = []
    for text, res in zip(req.texts, results):
        output.append({
            "text": text,
            "label": res.get("label"),
            "score": float(res.get("score", 0.0)),
            "sentiment_value": float(res.get("sentiment_value", 0.0)),
            "emotion": res.get("emotion"),
            "emotion_score": float(res.get("emotion_score", 0.0)),
        })
    return {"items": output}

@app.post("/predict_csv")
def predict_csv(req: FileBatchRequest):
    try:
        df = pd.read_csv(req.csv_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load CSV: {e}")
    if req.text_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.text_column}' not found")
    texts = df[req.text_column].astype(str).tolist()
    results = model.predict_batch(texts)
    df_out = df.copy()
    df_out["sentiment_label"] = [r.get("label") for r in results]
    df_out["sentiment_score"] = [float(r.get("score", 0.0)) for r in results]
    df_out["sentiment_value"] = [float(r.get("sentiment_value", 0.0)) for r in results]
    df_out["emotion"] = [r.get("emotion") for r in results]
    df_out["emotion_score"] = [float(r.get("emotion_score", 0.0)) for r in results]
    return {"predictions": df_out.to_dict(orient="records")}  


@app.post("/analyze_topic")
def analyze_topic(req: TopicRequest):
    keyword = req.keyword.strip()
    if not keyword:
        raise HTTPException(status_code=400, detail="Keyword is empty")

    items = topic_data.fetch(keyword=keyword, source=req.source, limit=req.limit)
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
    timeline_map: dict[str, list[float]] = {}
    emotion_counter: Counter[str] = Counter()

    for item, prediction in zip(items, predictions):
        published_at = item.get("published_at")
        date_key = _normalize_date(published_at)
        emotion = prediction.get("emotion", "NEUTRAL")
        emotion_counter[emotion] += 1

        timeline_map.setdefault(date_key, []).append(float(prediction.get("sentiment_value", 0.0)))
        enriched_items.append(
            {
                **item,
                "label": prediction.get("label"),
                "score": float(prediction.get("score", 0.0)),
                "sentiment_value": float(prediction.get("sentiment_value", 0.0)),
                "emotion": emotion,
                "emotion_score": float(prediction.get("emotion_score", 0.0)),
            }
        )

    timeline = [
        {
            "date": date,
            "avg_sentiment": round(sum(values) / len(values), 4),
            "count": len(values),
        }
        for date, values in sorted(timeline_map.items())
    ]
    avg_sentiment = sum(item["sentiment_value"] for item in enriched_items) / len(enriched_items)
    top_emotion = emotion_counter.most_common(1)[0][0] if emotion_counter else "NEUTRAL"

    return {
        "keyword": keyword,
        "source": req.source,
        "count": len(enriched_items),
        "avg_sentiment": round(avg_sentiment, 4),
        "dominant_emotion": top_emotion,
        "emotion_breakdown": dict(emotion_counter),
        "timeline": timeline,
        "items": enriched_items,
    }


def _normalize_date(value: str | None) -> str:
    if not value:
        return "Unknown"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return str(value)[:10]
